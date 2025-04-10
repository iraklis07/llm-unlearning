import numpy as np
import pandas as pd
import torch
import random
import json
import cohere
import os
import glob
import shutil
import datasets
import gc

from datasets import load_dataset
from statistics import harmonic_mean, mean
from rouge_score import rouge_scorer
from tqdm import tqdm, trange
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM


class QuantitativeEvaluation:
    """A class for final quantitative evaluation of the unlearned model
    based on the official evaluation script provided by the task organizers.

    The official script can be found in: ```semeval-unlearning-data/evaluate_generations.py```.
    """
    def __init__(self, args):
        self.args = args
        self.seed = args["seed"]
        self.debug = args["debug"]
        self.checkpoint_path = args["checkpoint_path"]
        self.data_path = args["data_path"]
        self.split = args["split"]
        self.output_dir = args["output_dir"]
        self.mia_data_path = args["mia_data_path"]
        self.mmlu_metrics_file_path = args["mmlu_metrics_file_path"]
        self.max_new_tokens = args["max_new_tokens"]
        self.batch_size = args["batch_size"]
        self.compute_metrics_only = args["compute_metrics_only"]
        self.keep_files = args["keep_files"]

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def inference(self, model, tokenizer):
        accelerator = Accelerator()
        model.to(accelerator.device)

        forget_file = self.data_path + f"forget_{self.split}-00000-of-00001.parquet"
        retain_file = self.data_path + f"retain_{self.split}-00000-of-00001.parquet"

        for split, train_file in [('retain', retain_file), ('forget', forget_file)]:
            data_files = {}
            dataset_args = {}
            if train_file is not None:
                data_files["train"] = train_file
            raw_datasets = datasets.load_dataset(
                "parquet",
                data_files=data_files,
                **dataset_args,
            )
            train_dataset = raw_datasets["train"]

            output_dic = defaultdict(lambda: {'id': [], 'task': [], 'input': [], 'expected_output': [], 'model_output': [], 'nll': []})

            with accelerator.split_between_processes(train_dataset, apply_padding=True) as data:
                for idx in tqdm(range(len(data['input']))):
                    question, answer = data["input"][idx], data["output"][idx]
                    output_dic[accelerator.process_index]['id'].append(data["id"][idx])
                    output_dic[accelerator.process_index]['task'].append(data["task"][idx])
                    output_dic[accelerator.process_index]['input'].append(data["input"][idx])
                    output_dic[accelerator.process_index]['expected_output'].append(data["output"][idx])
                    input_ids = tokenizer(
                        question,
                        return_tensors='pt'
                    ).input_ids.to(model.device)

                    combined_input_ids = tokenizer(
                        question + answer,
                        return_tensors='pt'
                    ).input_ids.to(model.device)
                    combined_target_ids = combined_input_ids.clone()
                    combined_target_ids[:, :len(input_ids[0])] = -100

                    with torch.no_grad():
                        out = model.generate(input_ids, 
                                             max_new_tokens=self.max_new_tokens,
                                             do_sample=False,
                                             use_cache=True,
                                             pad_token_id=tokenizer.eos_token_id)
                        output_ids = out[:, len(input_ids[0]):]
                        output = tokenizer.batch_decode(
                            output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)[0]
                        output_dic[accelerator.process_index]['model_output'].append(output)

                        # For Perplexity
                        out = model(combined_input_ids, labels=combined_target_ids)
                        neg_log_likelihood = out.loss.item()
                        output_dic[accelerator.process_index]['nll'].append(neg_log_likelihood)

            accelerator.wait_for_everyone()

            if self.debug:
                print([len(value) for value in output_dic[accelerator.process_index].values()])
            output_df = pd.DataFrame.from_dict(output_dic[accelerator.process_index])

            output_file_name = f"{self.output_dir}/{split}_{accelerator.process_index}.csv"
            if self.debug:
                print('Saving to: ', output_file_name)
            output_df.to_csv(output_file_name, index=False)

    def mia_attacks(self, model, tokenizer):
        member_file = self.mia_data_path + 'member.jsonl'
        nonmember_file = self.mia_data_path + 'nonmember.jsonl'

        accelerator = Accelerator()
        model.to(accelerator.device)

        for dataset, train_file in [('member', member_file), ('nonmember', nonmember_file)]:
            data_files = {}
            dataset_args = {}
            if train_file is not None:
                data_files["train"] = train_file
            raw_datasets = datasets.load_dataset(
                "json",
                data_files=data_files,
                **dataset_args,
            )
            train_dataset = raw_datasets["train"]

            output_dic = defaultdict(lambda: {'id': [], 'nll': []})

            with accelerator.split_between_processes(train_dataset, apply_padding=True) as data:
                for idx in tqdm(range(len(data['document']))):
                    document = data["document"][idx]
                    output_dic[accelerator.process_index]['id'].append(data["id"][idx])
                    input_ids = tokenizer(
                        document,
                        return_tensors='pt'
                    ).input_ids.to(model.device)

                    target_ids = input_ids.clone()

                    with torch.no_grad():
                        out = model(input_ids, labels=target_ids)
                        neg_log_likelihood = out.loss.item()
                        output_dic[accelerator.process_index]['nll'].append(neg_log_likelihood)

            accelerator.wait_for_everyone()

            output_df = pd.DataFrame.from_dict(output_dic[accelerator.process_index])

            results_dir = os.path.join(self.output_dir, 'mia_results')
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            output_file_name = f"{results_dir}/{dataset}_{accelerator.process_index}.csv"
            if self.debug:
                print('Saving to: ', output_file_name)
            output_df.to_csv(output_file_name, index=False)

    def compute_auc(self, member_loss, nonmember_loss):
        assert not np.any(np.isnan(member_loss))
        assert not np.any(np.isnan(nonmember_loss))
        combined_loss = member_loss + nonmember_loss 
        combined_loss = -1 * np.array(combined_loss)
        combined_labels = len(member_loss) * [1] + len(nonmember_loss) * [0]
        fp, tp, _ = roc_curve(combined_labels, combined_loss)

        auc_score = float(auc(fp, tp))

        return auc_score
    
    def compute_metrics(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        results = {}
        aggregate_scores_list = []
        for split in ['forget', 'retain']:
            files = glob.glob(self.output_dir + '/{}_*.csv'.format(split))
            if len(files) == 0:
                print("[ERROR] Missing inference files, rerun script with inference first")
                return  # sys.exit(1) throws a long traceback so just return for now
            df_list = [pd.read_csv(f) for f in files]
            if not self.keep_files:
                _ = [os.remove(f) for f in files]
            df = pd.concat(df_list, ignore_index=True)

            df['regurgitation-score-rouge-1'] = None
            df['regurgitation-score'] = None
            df['knowledge-score'] = None
            ground_truths = df['expected_output'].tolist()
            gen_outputs = df['model_output'].tolist()

            for i, (gen, gt) in enumerate(zip(gen_outputs, ground_truths)):
                if df.loc[i, 'id'][:-1].endswith('sc'):
                    rouge_scores = scorer.score(str(gt), str(gen))
                    df.loc[i, 'regurgitation-score-rouge-1'] = rouge_scores['rouge1'].recall
                    df.loc[i, 'regurgitation-score'] = rouge_scores['rougeL'].recall
                elif df.loc[i, 'id'][:-1].endswith('qa'):
                    df.loc[i, 'knowledge-score'] = int(str(gt).strip().lower() == str(gen).strip().lower())

            results[split+'-set'] = {'overall-regurgitation-score': np.mean(df['regurgitation-score']),
                                     'overall-knowledge-score': np.mean(df['knowledge-score'])}
            split_aggregate_scores_dict = df.groupby('task')[['regurgitation-score', 'knowledge-score']].mean().to_dict(orient='index')
            results[split+'-set'].update(split_aggregate_scores_dict)
            split_aggregate_score_values = [float(val) for inner in split_aggregate_scores_dict.values() for val in inner.values()]
            if split == 'forget':
                split_aggregate_score_values = [(1 - val) for val in split_aggregate_score_values]

            aggregate_scores_list.extend(split_aggregate_score_values)

        if self.mia_data_path is not None:
            mia_results_dir = os.path.join(self.output_dir, 'mia_results')
            mia_results = {}
            for dataset in ['member', 'nonmember']:
                files = glob.glob(mia_results_dir + '/{}_*.csv'.format(dataset))
                if len(files) == 0:
                    print("[ERROR] Missing mia files, rerun script with inference first")
                    return  # sys.exit(1) throws a long traceback so just return for no
                df_list = [pd.read_csv(f) for f in files]
                df = pd.concat(df_list, ignore_index=True)
                mia_results[dataset] = df['nll'].tolist()
            
            if not self.keep_files:
                shutil.rmtree(mia_results_dir)

            auc = self.compute_auc(mia_results['member'], mia_results['nonmember'])
            # Best MIA rates we can get are ~0.5. 1 implies model still remembers the forget set
            results['mia_loss_acc'] = auc
    #        aggregate_scores_list.append(1 - auc) 

        if self.mmlu_metrics_file_path is not None:
            with open(self.mmlu_metrics_file_path) as inptr:
                mmlu_scores = json.loads(inptr.read())
            results['mmlu_average'] = mmlu_scores['average_acc']
    #        aggregate_scores_list.append(mmlu_scores['average_acc'])
        
        results['aggregated-terms'] = aggregate_scores_list

        task_aggregate = harmonic_mean(aggregate_scores_list)
        results['harmonic-mean-task-aggregate'] = task_aggregate

        results['aggregate-score'] = -1
        
        # Need MMLU and MIA scores to compute the aggregate
        if 'mmlu_average' in results and 'mia_loss_acc' in results:
            if results['mmlu_average'] < 0.371:
                # MMLU score should not drop below 75% of pre-unlearning preformance
                print("[WARNING] The MMLU average for the provided checkpoint is below threshold. If this happens your model may not be considered in final challenge ranking.")
    
            mia_final_score = 1 - abs(results['mia_loss_acc'] - 0.5)*2
            results['mia_final_score'] = mia_final_score
            results['aggregate-score'] = mean([task_aggregate, results['mmlu_average'], mia_final_score])

        metrics_file = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(metrics_file, 'w') as outptr:
            outptr.write(json.dumps(results, indent=4))

    def run(self):
        # Set random seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        if self.debug:
            print('Evaluating Checkpoint at {}'.format(self.checkpoint_path))

        # Set up accelerator
        accelerator = Accelerator()
        if not self.compute_metrics_only:
            model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
            tokenizer.pad_token = tokenizer.eos_token

            self.inference(model, tokenizer)

            if self.mia_data_path is not None:
                self.mia_attacks(model, tokenizer)

        if accelerator.is_main_process:
            print("Computing final metrics...")
            self.compute_metrics()


class QualitativeEvaluation:
    def __init__(self, checkpoint_path, path_to_predictions, path_to_gqa, output_dir, n_samples=5):
        """
        Initialize the QualitativeEvaluation class.

        Args:
            model: The trained PEFT model to evaluate.
            tokenizer: Tokenizer for encoding/decoding text.
            path_to_predictions: Path to directory containing predictions for retain/forget datasets.
            path_to_gqa: Path to JSON file containing general questions and answers.
            output_dir: Directory where CSV files will be saved.
            n_samples: Number of samples to evaluate per task/subset/evaluation type.
        """
        self.checkpoint_path = checkpoint_path
        self.path_to_predictions = path_to_predictions
        self.path_to_gqa = path_to_gqa
        self.output_dir = output_dir
        self.n_samples = n_samples

        # Insert a COHERE_API_KEY (free for up to a certain amount of requests per minute) if 
        # you want to rephrase certain questions and evaluate the unlearned model on them. 

        self.co = None
        #self.co = cohere.ClientV2(COHERE_API_KEY)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def prepare_input_ids(self, tokenizer, input, apply_chat_template=False):
        if apply_chat_template:
            chat = [{"role": "user", "content": input}]
            input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(input, add_special_tokens=False, return_tensors="pt")

    def rephrase_question(self, question):
        prompt = f"I want to rephrase the following question without adding more information:\n{question} Return the rephrased question only."
        response = self.co.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content[0].text

    def evaluate_rephrased_questions(self, model, tokenizer, examples, output_file):
        accelerator = Accelerator()
        model.to(accelerator.device)

        results = []
        for example in examples:
            try:
                rephrased_question = self.rephrase_question(example["input"])
            except Exception as e:
                rephrased_question = example["input"]
                print(f"Error in rephrasing: {str(e)}")

            input_ids = self.prepare_input_ids(tokenizer, rephrased_question)
            input_seq_len = len(rephrased_question)

            try:
                outputs = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = text_output[input_seq_len + 1:]
            except Exception as e:
                generated_text = f"Error in generation: {str(e)}"

            results.append({
                "ID": example["id"],
                "Original": example["input"],
                "Rephrased": rephrased_question,
                "Expected Output": example["expected_output"],
                "Original Prediction": example["model_output"].strip(),
                "Rephrased Prediction": generated_text
            })

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)

    def evaluate_general_questions(self, model, tokenizer, output_file):
        accelerator = Accelerator()
        model.to(accelerator.device)

        with open(self.path_to_gqa, 'r') as fs:
            questions = json.load(fs)

        results = []
        for k, qa in questions.items():
            input_ids = self.prepare_input_ids(tokenizer, qa['question'])
            input_seq_len = len(qa['question'])

            try:
                outputs = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = text_output[input_seq_len + 1:]
            except Exception as e:
                generated_text = f"Error in generation: {str(e)}"

            results.append({
                "Question ID": k,
                "Question": qa['question'],
                "True Answer": qa['answer'].strip(),
                "Generated Answer": generated_text
            })

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)

    def process_dataset(self, output_dir):
        questions_to_rephrase = []

        for split in ['retain', 'forget']:
            files = glob.glob(f"{self.path_to_predictions}/{split}_*.csv")
            if not files:
                print(f"Missing '{split}' prediction files.")
                continue

            df_list = [pd.read_csv(f) for f in files]
            df = pd.concat(df_list, ignore_index=True)
            unique_tasks = df['task'].unique()
            df_to_concat = []

            for task in sorted(unique_tasks):
                task_examples = df[df['task'] == task]
                sc_examples = task_examples[task_examples['id'].str[:-1].str.endswith("sc")]
                qa_examples = task_examples[~task_examples['id'].str[:-1].str.endswith("sc")]

                sc_sample = sc_examples.sample(n=min(self.n_samples, len(sc_examples)), random_state=42, ignore_index=True)
                qa_sample = qa_examples.sample(n=min(self.n_samples, len(qa_examples)), random_state=42, ignore_index=True)

                df_to_concat.append(sc_sample)
                df_to_concat.append(qa_sample)

                if split == 'forget':
                    questions_to_rephrase.extend(qa_sample.to_dict('records'))

            df = pd.concat(df_to_concat, ignore_index=True)
            df.to_csv(os.path.join(output_dir, f"{split}_samples.csv"), index=False)

        return questions_to_rephrase

    def create_excel_file(self):
        files = glob.glob(f"{self.output_dir}/*.csv")
        with pd.ExcelWriter(f"{self.output_dir}/summary.xlsx") as writer:
            for file in files:
                sheet_name = " ".join(file.split("/")[-1].split(".")[0].split("_")).title()
                df = pd.read_csv(file)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def run(self):
        accelerator = Accelerator()
        accelerator.free_memory()
        
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

        questions_to_rephrase = self.process_dataset(self.output_dir)
        #rephrased_output_file = os.path.join(self.output_dir, "rephrased_questions.csv")
        #self.evaluate_rephrased_questions(model, tokenizer, questions_to_rephrase, rephrased_output_file)

        general_questions_output_file = os.path.join(self.output_dir, "general_questions.csv")
        self.evaluate_general_questions(model, tokenizer, general_questions_output_file)

        accelerator.wait_for_everyone()

        self.create_excel_file()


class MMLU:
    def __init__(self, topics, seed=42):
        self.topics = topics
        self.seed = seed
        self.acc_scores = {}

    def get_ans(self, model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'],
                           attention_mask=inputs['attention_mask']).logits[0, -1]

        # Create a list of tuples having (logit, 'option') format
        options_list = [(logits[tokenizer(' A').input_ids[-1]], 'A'), (logits[tokenizer(' B').input_ids[-1]], 'B'), (logits[tokenizer(' C').input_ids[-1]], 'C'), (logits[tokenizer(' D').input_ids[-1]], 'D')]
        options_list = sorted(options_list, reverse=True)
        ans_list = []
        for i in range(3):
            ans_list.append(options_list[i][1])

        #print(ans_list)
        return ans_list

    def evaluate_topic(self, model, tokenizer, topic):
        accelerator = Accelerator()
        model.to(accelerator.device)

        # Load the MMLU dataset
        dataset = load_dataset("cais/mmlu", topic, split="test", trust_remote_code=True)

        answers_dict = {
            0: "A",
            1: "B",
            2: "C",
            3: "D"
        }

        correct_count = 0

        with accelerator.split_between_processes(dataset, apply_padding=True) as data:
            for i in trange(len(data)):
                input = data[i]['question']
                choices = data[i]['choices']
                A = choices[0]
                B = choices[1]
                C = choices[2]
                D = choices[3]

                prompt = f"""Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]

                Question: {input}\n
                A) {A}\n
                B) {B}\n
                C) {C}\n
                D) {D}\n

                Answer:<|end_of_turn|>"""

                #print(answers_dict[data[i]['answer']])

                if (self.get_ans(model, tokenizer, prompt)[0] == answers_dict[data[i]['answer']]):
                    #print("CORRECT!!!")
                    #correct_pred_dict[accelerator.process_index]['correct'] += 1
                    #correct_pred_dict[accelerator.process_index] += 1
                    correct_count += 1

        accelerator.wait_for_everyone()
        
        # Convert local correct count to a tensor
        correct_count_tensor = torch.tensor(correct_count, device=accelerator.device)
    
        # Aggregate results across processes
        correct_total = accelerator.reduce(correct_count_tensor, reduction="sum").item()

        acc = float(correct_total)/(len(dataset))

        return topic, acc

    def run(self, model_path, mmlu_metrics_file_path):
        # Set random seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        mmlu_dir = "/".join(mmlu_metrics_file_path.split("/")[:-1])

        if not os.path.exists(mmlu_dir):
            Path(mmlu_dir).mkdir(parents=True, exist_ok=True)

        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model.to(accelerator.device)

        for topic in self.topics:
            gc.collect()
            torch.cuda.empty_cache()

            try:
                topic, acc = self.evaluate_topic(model, tokenizer, topic)
                #print(f"{topic}: {acc}")
                self.acc_scores[topic] = acc
            except Exception as e:
                print(f"Error evaluating {topic}: {e}")

        # Avearge of acc from the dictionary
        self.acc_scores["average_acc"] = sum(self.acc_scores.values()) / len(self.acc_scores)

        accelerator.wait_for_everyone()

        # Save the results in a json file
        if accelerator.is_main_process:
            with open(mmlu_metrics_file_path, "w") as f:
                f.write(json.dumps(self.acc_scores, indent=4))

            print(f"Average accuracy: {self.acc_scores['average_acc']}")