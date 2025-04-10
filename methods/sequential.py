import datasets
import torch
import inspect
import json

from torch.utils.data import Dataset, SequentialSampler
from transformers import DataCollatorForSeq2Seq, Trainer
from datasets import DatasetDict, concatenate_datasets
from typing import Optional
from itertools import cycle

class AdvSupervisedDataset(Dataset):
    """Dataset for adv supervised fine-tuning."""

    def __init__(
        self,
        negative_data_dict,
        positive_data_dict,
        positive_ratio,
        positive_factor,
        add_remaining=True
    ):
        super(AdvSupervisedDataset, self).__init__()

        negative_data_dict = negative_data_dict.to_dict()
        positive_data_dict = positive_data_dict.to_dict()
        len_positive = len(positive_data_dict["input_ids"])
        len_negative = len(negative_data_dict["input_ids"])
        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        self.factor = []
        for i in range(len_negative):
            # Add one negative example at a time
            self.input_ids.append(negative_data_dict["input_ids"][i])
            self.labels.append(negative_data_dict["labels"][i])
            self.attention_mask.append(negative_data_dict["attention_mask"][i])
            self.factor.append(-1)

            # Add positive examples
            pos_index_list = list(range(i * positive_ratio, min((i + 1) * positive_ratio, len_positive)))
            pos_index_range = slice(i * positive_ratio, min((i + 1) * positive_ratio, len_positive))
        
            self.input_ids.extend(positive_data_dict["input_ids"][pos_index_range])
            self.labels.extend(positive_data_dict["labels"][pos_index_range])
            self.attention_mask.extend(positive_data_dict["attention_mask"][pos_index_range])
        
            if pos_index_list:
                self.factor.extend([positive_factor] * len(pos_index_list))

        # Add the remaining positive samples if applicable 
        if positive_ratio is not None and len_positive > positive_ratio * len_negative and add_remaining:
            print(f"{len_positive - positive_ratio * len_negative} remaining positive samples were added")
            self.input_ids.extend(positive_data_dict["input_ids"][positive_ratio * len_negative:])
            self.labels.extend(positive_data_dict["labels"][positive_ratio * len_negative:])
            self.attention_mask.extend(positive_data_dict["attention_mask"][positive_ratio * len_negative:])
            self.factor.extend([positive_factor] * (len_positive - positive_ratio * len_negative))
        
        # Add all positive samples if they are less than the forget ones
        if positive_ratio is not None and positive_ratio == 0 and not add_remaining:  
            self.input_ids.extend(positive_data_dict["input_ids"])
            self.labels.extend(positive_data_dict["labels"])
            self.attention_mask.extend(positive_data_dict["attention_mask"])
            self.factor.extend([positive_factor] * len(positive_data_dict["input_ids"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            factor=self.factor[i],
        )
    
    def summary(self):
        summary = {
            "retain_samples" : sum([1 for f in self.factor if f > 0]),
            "forget_samples" : sum([1 for f in self.factor if f < 0]),
            "total_samples" : len(self)
        }
        return summary

    def select(self, selection_range):
        # Create a new instance of AdvSupervisedDataset with the same data dictionaries and args,
        # but do not fill the attributes with data yet.
        new_dataset = AdvSupervisedDataset(datasets.Dataset.from_dict({"input_ids": []}), datasets.Dataset.from_dict({"input_ids": []}), None, None)

        # Manually set the selected items for each attribute
        new_dataset.input_ids = [self.input_ids[i] for i in selection_range]
        new_dataset.labels = [self.labels[i] for i in selection_range]
        new_dataset.attention_mask = [self.attention_mask[i] for i in selection_range]
        new_dataset.factor = [self.factor[i] for i in selection_range]

        return new_dataset


class AscentPlusDescentDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        # print([f["factor"] for f in features])
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features])
        return batch


class AscentPlusDescentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "factor" not in inputs.keys():
            return super().compute_loss(model, inputs, return_outputs)
        factors = inputs.pop("factor")
        #print(factors)
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        valid_counts = (shift_labels != -100).sum(dim=-1).float()

        loss = loss.view(shift_logits.size(0), -1)

        loss = loss.sum(dim=-1) / valid_counts
        adjusted_loss = (loss * factors).mean()
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')


class SequentialUnlearning:
    def __init__(
            self,
            model,
            tokenizer,
            data_collator,
            training_args,
            forget_dataset,
            retain_dataset,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
            sequential=True,
            chunk_size=16, 
            positive_ratio=1,
            positive_factor=1,
            retain_loss="CE",
            pretrain_model=None
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.training_args = training_args
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.chunk_size = chunk_size
        self.sequential = sequential
        self.positive_ratio = positive_ratio
        self.positive_factor = positive_factor
        self.retain_loss = retain_loss

        self.eval_ready = False
        self.retain_next_index = 0
        self.log_history = []
        self.total_runtime = 0
        self.total_flos = 0

        assert self.retain_loss == "CE" or self.retain_loss == "KL"
        self.pretrain_model = pretrain_model

    def train(self, split_retain=False):
        """
        Method to train the model either sequentially or with all data at once.

        Parameters:
        - split_retain (bool): If True, will split the retain dataset into chunks matching the forget chunks.
        """

        # Prepare the train dataset
        if self.sequential:
            self._train_sequential(split_retain)
        else:
            self._train_all_at_once()

    def _train_all_at_once(self):
        """
        Train the model with the entire retain and forget datasets at once.
        """
        # Create the combined training dataset
        train_ds = DatasetDict()
        train_ds['retain'] = self.retain_dataset
        train_ds['forget'] = self.forget_dataset

        # Set up the dataset
        positive_ratio = (len(train_ds['retain']) // len(train_ds['forget'])) * self.positive_ratio
        train_dataset = AdvSupervisedDataset(
            train_ds['forget'],
            train_ds['retain'],
            positive_ratio=positive_ratio,
            positive_factor=self.positive_factor,
            add_remaining=True
        )

        # Create eval dataset (all previous data seen during training)
        eval_dataset = DatasetDict({'retain': self.retain_dataset, 'forget': self.forget_dataset})

        # Split eval datasets to task specific
        eval_dataset = self._split_eval_dataset(eval_dataset)

        #print(f'\nTrain: {train_dataset.summary()}\n\nEval: {eval_dataset}')

        # Create the trainer and train
        trainer = AscentPlusDescentTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=self.training_args,
            train_dataset=train_dataset,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )

        trainer.train()
        
        self.log_history.extend(trainer.state.log_history[:-1])
        self.total_runtime += trainer.state.log_history[-1]["train_runtime"]
        self.total_flos += trainer.state.log_history[-1]["total_flos"]

    def _train_sequential(self, split_retain):
        """
        Train the model sequentially in chunks of the forget dataset.
        """
        n_chunks = len(self.forget_dataset) // self.chunk_size
        rem_samples = len(self.forget_dataset) % self.chunk_size

        print(f"Number of chunks: {n_chunks}")
        print(f"Remaining forget samples: {rem_samples}")

        # Initialize the eval dataset (starts empty)
        eval_dataset = DatasetDict({'retain': Dataset(), 'forget': Dataset()})

        # Iterate over chunks of the forget dataset
        for i in range(n_chunks):           
            start_idx = i * self.chunk_size
            partial_forget_set = self.forget_dataset.select(range(start_idx, start_idx + self.chunk_size))

            # Update eval dataset by concatenating the forget chunk
            if i == 0:
                eval_dataset['forget'] = partial_forget_set
            else:
                eval_dataset['forget'] = concatenate_datasets([eval_dataset['forget'], partial_forget_set])

            # Call the helper method to handle the training loop
            print(f"\nTraining on chunk {i+1} ...")
            self._inner_training_loop(i, partial_forget_set, split_retain, eval_dataset)

        # Handle remaining samples
        if rem_samples > 0:
            start_idx = n_chunks * self.chunk_size
            partial_forget_set = self.forget_dataset.select(range(start_idx, start_idx + rem_samples))

            eval_dataset['forget'] = concatenate_datasets([eval_dataset['forget'], partial_forget_set])

            # Call the helper method for remaining samples
            self._inner_training_loop(n_chunks, partial_forget_set, split_retain, eval_dataset)

    def _inner_training_loop(self, chunk_index, partial_forget_set, split_retain, eval_dataset):
        """
        Handles the repeated part of the training loop for both regular chunks and remaining samples.

        Parameters:
        - chunk_index (int): The index of the current chunk.
        - partial_forget_set (Dataset): The chunk of the forget dataset being used for training.
        - split_retain (bool): Whether to split the retain dataset to match the size of the forget chunk.
        - eval_dataset (DatasetDict): The evaluation dataset that accumulates seen data.
        """

        # Split retain dataset if requested
        if split_retain:
            retain_chunk_size = len(partial_forget_set) * self.positive_ratio 
            partial_retain_set, self.retain_next_index = self._get_cycled_retain_samples(
                self.retain_dataset,
                retain_chunk_size, 
                self.retain_next_index
            )
                     
            train_ds = DatasetDict({
                'retain': partial_retain_set,
                'forget': partial_forget_set
            })

            if chunk_index == 0:
                eval_dataset['retain'] = partial_retain_set
            else:
                if not self.eval_ready:
                    eval_dataset['retain'] = concatenate_datasets([eval_dataset['retain'], partial_retain_set])
                    if len(eval_dataset['retain']) > len(self.retain_dataset):
                        eval_dataset['retain'] = eval_dataset['retain'].take(len(self.retain_dataset))
                        self.eval_ready = True
        else:
            train_ds = DatasetDict({
                'retain': self.retain_dataset.shuffle(),
                'forget': partial_forget_set
            })
            eval_dataset['retain'] = self.retain_dataset

        # Positive ratio calculation
        positive_ratio = len(train_ds['retain']) // len(train_ds['forget'])
        train_dataset = AdvSupervisedDataset(
            train_ds['forget'],
            train_ds['retain'],
            positive_ratio=positive_ratio,
            positive_factor=self.positive_factor
        )

        # Split eval dataset per task
        eval_dataset = self._split_eval_dataset(eval_dataset)
        
        # Print out the chunk details
        #print(f'\nChunk {chunk_index + 1}:\nTrain: {train_dataset.summary()}\n\nEval: {eval_dataset}')
        
        # Create and train with the trainer
        trainer = AscentPlusDescentTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=self.training_args,
            train_dataset=train_dataset,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )

        trainer.train()
        
        self.log_history.extend(trainer.state.log_history[:-1])
        self.total_runtime += trainer.state.log_history[-1]["train_runtime"]
        self.total_flos += trainer.state.log_history[-1]["total_flos"]

    def save_model(self):
        """
        Save the model and tokenizer after training.
        """
        output_dir = self.training_args.output_dir
        print(f"Saving final model to {output_dir}/final_model")
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")

    def save_summary(self, output_dir):
        """
        Get a summary of the training process.
        """
        log = {
            "log_history": self.log_history,
            "total_runtime": self.total_runtime,
            "total_flos": self.total_flos
        }
        with open(f'{output_dir}/log_history.json', 'w') as file:
            file.write(json.dumps(log, indent=4))
        
        return log

    def _get_cycled_retain_samples(self, retain_dataset, required_size, current_index):
        """
        Cycle through retain samples to match a required size, returning a Dataset.

        Parameters:
        - retain_dataset (Dataset): The retain dataset to cycle through.
        - required_size (int): The total number of retain samples needed for the current chunk.
        - current_index (int): The starting index for cycling.

        Returns:
        - cycled_retain (Dataset): A Dataset containing the retain samples for the current chunk.
        - next_index (int): The updated index for the next chunk.
        """
        total_retain_samples = len(retain_dataset)

        # Convert Dataset to a list of dictionaries (rows)
        retain_list = retain_dataset.to_list()
        #print(retain_list)

        # Cycle through retain samples starting from current_index
        retain_cycle = cycle(retain_list)
        cycled_samples = []

        # Skip to the current index
        for _ in range(current_index):
            next(retain_cycle)

        # Collect the required samples
        for _ in range(required_size):
            cycled_samples.append(next(retain_cycle))

        # Update the index (wrap around if needed)
        next_index = (current_index + required_size) % total_retain_samples

        # Convert the list of samples (list of dicts) back to a Dataset
        # Transforming list of dicts into a dict of lists for Dataset.from_dict
        cycled_samples_dict = {key: [row[key] for row in cycled_samples] for key in cycled_samples[0]}
        cycled_retain = datasets.Dataset.from_dict(cycled_samples_dict)

        return cycled_retain, next_index

    def _split_eval_dataset(self, eval_dataset_dict):
        split_eval_dataset_dict = DatasetDict()
        for subset in sorted(eval_dataset_dict.keys(), reverse=True):
            if "id" in eval_dataset_dict[subset].column_names:
                eval_dataset_dict[subset] = eval_dataset_dict[subset].remove_columns(['id', 'input', 'output', 'split'])

            split_eval_dataset_dict[f"{subset}_1"] = eval_dataset_dict[subset].filter(lambda example: example["task"] == "Task1")
            split_eval_dataset_dict[f"{subset}_2"] = eval_dataset_dict[subset].filter(lambda example: example["task"] == "Task2")
            split_eval_dataset_dict[f"{subset}_3"] = eval_dataset_dict[subset].filter(lambda example: example["task"] == "Task3")

            if 'task' in split_eval_dataset_dict[f"{subset}_1"].column_names:
                split_eval_dataset_dict[f"{subset}_1"] = split_eval_dataset_dict[f"{subset}_1"].remove_columns('task')
                split_eval_dataset_dict[f"{subset}_2"] = split_eval_dataset_dict[f"{subset}_2"].remove_columns('task')
                split_eval_dataset_dict[f"{subset}_3"] = split_eval_dataset_dict[f"{subset}_3"].remove_columns('task')
            
        return split_eval_dataset_dict