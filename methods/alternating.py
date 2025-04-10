import torch
import random
import json

from torch import nn
from transformers.utils import is_apex_available
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict, concatenate_datasets
from typing import Dict, Union, Any

if is_apex_available():
    from apex import amp

class GradientAscentTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        loss = -loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


class AlternatingTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        data_collator,
        train_args_forgetting: TrainingArguments,
        train_args_annealing: TrainingArguments,
        retain_dataset,
        forget_dataset,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        chunk_size=16,
        interleaving_factor=0.5,
        perform_final_annealing=True,
        eval_after_annealing=True,
        annealing_fraction=1.0  # Fraction of retain set for annealing
    ):
        """
        Initializes the Alternating Trainer.

        Args:
            model: Pre-trained model.
            tokenizer: Tokenizer for the model.
            data_collator: Data collator for batching.
            train_args_forgetting (TrainingArguments): Training arguments for the forgetting phase.
            train_args_annealing (TrainingArguments): Training arguments for the annealing phase.
            retain_dataset: Dataset to retain during training.
            forget_dataset: Dataset to unlearn sequentially.
            compute_metrics: Metric computation function (optional).
            preprocess_logits_for_metrics: Preprocessing function for logits (optional).
            interleaving_factor (float): Fraction determining annealing frequency.
            perform_final_annealing (bool): Whether to perform a final annealing phase.
            eval_after_annealing (bool): Whether to evaluate after annealing.
            annealing_fraction (float): Fraction of retain set to use for annealing (1.0 means full set).
            logging_dir (str): Directory for logging results.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_args_forgetting = train_args_forgetting
        self.train_args_annealing = train_args_annealing
        self.retain_dataset = retain_dataset
        self.forget_dataset = forget_dataset
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.chunk_size = chunk_size
        self.interleaving_factor = interleaving_factor
        self.perform_final_annealing = perform_final_annealing
        self.eval_after_annealing = eval_after_annealing
        self.annealing_fraction = annealing_fraction

        self.global_log_history = []
        self.total_runtime = 0
        self.total_flos = 0
        self.n_chunks = len(self.forget_dataset) // self.chunk_size
        self.rem_samples = len(self.forget_dataset) % self.chunk_size

    def train(self):
        """
        Executes the alternating training process with interleaved unlearning and annealing phases.
        """
        chunk_size = self.chunk_size
        chunk_count = 0

        train_dataset = {"retain": self.retain_dataset}
        eval_dataset = {"retain": self.retain_dataset}

        for i in range(self.n_chunks):
            # Select forget dataset chunk
            start_idx = i * chunk_size
            partial_forget_set = self.forget_dataset.select(range(start_idx, start_idx + chunk_size))
            train_dataset['forget'] = partial_forget_set

            if i == 0:
                eval_dataset['forget'] = partial_forget_set
            else:
                eval_dataset['forget'] = concatenate_datasets([eval_dataset['forget'], partial_forget_set])

            # Forgetting Phase
            print(f"Unlearning Chunk {i+1}/{self.n_chunks}")
            self._run_phase(
                phase_name="Forgetting",
                train_args=self.train_args_forgetting,
                train_dataset=train_dataset["forget"],
                eval_dataset=self._split_eval_dataset(eval_dataset)
            )

            # Annealing Phase (Based on Interleaving Factor)
            chunk_count += 1
            if self.interleaving_factor > 0 and chunk_count >= (1 / self.interleaving_factor):  # Trigger annealing
                print(f"Annealing after Chunk {i+1}")
                self._run_phase(
                    phase_name="Annealing",
                    train_args=self.train_args_annealing,
                    train_dataset=self._get_annealing_subset(train_dataset["retain"]),
                    eval_dataset=self._split_eval_dataset(eval_dataset) if self.eval_after_annealing else None
                )
                chunk_count = 0  # Reset chunk counter

        # Final Forgetting Phase for Remaining Samples
        if self.rem_samples > 0:
            print("Unlearning Remaining Samples")
            start_idx += chunk_size
            partial_forget_set = self.forget_dataset.select(range(start_idx, start_idx + self.rem_samples))
            train_dataset['forget'] = partial_forget_set
            eval_dataset['forget'] = concatenate_datasets([eval_dataset['forget'], partial_forget_set])
            self._run_phase(
                phase_name="Final Forgetting",
                train_args=self.train_args_forgetting,
                train_dataset=train_dataset["forget"],
                eval_dataset=self._split_eval_dataset(eval_dataset)
            )

        # Final Annealing Phase (if enabled)
        if self.perform_final_annealing:
            print("Performing Final Annealing Phase")
            self._run_phase(
                phase_name="Final Annealing",
                train_args=self.train_args_annealing,
                train_dataset=train_dataset["retain"],
                eval_dataset=self._split_eval_dataset(eval_dataset) if self.eval_after_annealing else None
            )

    def _get_annealing_subset(self, retain_dataset):
        """
        Samples a subset of the retain dataset for annealing, based on the annealing_fraction.

        Args:
            retain_dataset: The full retain dataset.

        Returns:
            Dataset: A subset of the retain dataset.
        """
        if self.annealing_fraction == 1.0:
            return retain_dataset  # Use full dataset
        else:
            sample_size = int(len(retain_dataset) * self.annealing_fraction)
            indices = random.sample(range(len(retain_dataset)), sample_size)
            return retain_dataset.select(indices)

    def _run_phase(self, phase_name, train_args, train_dataset, eval_dataset):
        """
        Executes a training phase (either forgetting or annealing).
        """

        print(f'\nTrain: {train_dataset}\n\nEval: {eval_dataset}')
        
        if phase_name.endswith("Forgetting"):
            trainer = GradientAscentTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=train_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
            )
        else:
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=train_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
            )

        trainer.train()

        # Logging phase results
        print(f"Completed {phase_name} Phase")
        self.global_log_history.extend(trainer.state.log_history[:-1])
        self.total_runtime += trainer.state.log_history[-1]["train_runtime"]
        self.total_flos += trainer.state.log_history[-1]["total_flos"]

    def _split_eval_dataset(self, eval_dataset_dict):
        split_eval_dataset_dict = DatasetDict()
        for subset in eval_dataset_dict.keys():
            split_eval_dataset_dict[f"{subset}_1"] = eval_dataset_dict[subset].filter(lambda example: example["task"] == "Task1")
            split_eval_dataset_dict[f"{subset}_2"] = eval_dataset_dict[subset].filter(lambda example: example["task"] == "Task2")
            split_eval_dataset_dict[f"{subset}_3"] = eval_dataset_dict[subset].filter(lambda example: example["task"] == "Task3")

        return split_eval_dataset_dict

    def save_model(self, output_dir):
        """
        Saves the model and tokenizer to the specified directory.
        """
        #output_dir = self.train_args_forgetting.output_dir
        print(f"Saving final model to {output_dir}/final_model/")
        self.model.save_pretrained(f"{output_dir}/final_model/")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model/")

    def save_summary(self, output_dir):
        """
        Get a summary of the training process.
        """
        log = {
            "log_history": self.global_log_history,
            "total_runtime": self.total_runtime,
            "total_flos": self.total_flos
        }
        with open(f'{output_dir}/log_history.json', 'w') as file:
            file.write(json.dumps(log, indent=4))
        
        return log