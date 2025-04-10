import pandas as pd
from datasets import DatasetDict, Dataset, concatenate_datasets


class DatasetProcessor:
    def __init__(self, data_dir, tokenizer, n_samples_per_task=None, start_idx=0, gold_standard=False):
        """
        Initializes the DatasetProcessor with the specified data path, tokenizer, and sampling options.

        Args:
            data_path (str): Path to the dataset files.
            tokenizer: Tokenizer to use for tokenization and masking.
            num_samples (int, optional): Number of samples to select from each task/dataset. Defaults to None (no sampling).
            start_idx (int, optional): Starting index for sampling. Defaults to 0.
        """
        self.data_path = data_dir
        self.tokenizer = tokenizer
        self.num_samples = n_samples_per_task
        self.start_idx = start_idx
        self.gold_standard = gold_standard

    def create_datasets(self, split, task, split_tasks=True, split_retain=False):
        """
        Creates and returns a DatasetDict from parquet files at the specified data path.

        Args:
            split (str): Split name (e.g., "train", "test").
            task (str): Subtask name ("all", "Task1", "Task2", "Task3"), Defaults to all
            split_tasks (bool, optional): Whether to split datasets by tasks. Defaults to True.
            split_retain (bool, optional): Whether to split the retain dataset by tasks. Defaults to False.

        Returns:
            DatasetDict: A dictionary containing the loaded datasets.
        """
        
        retain_df = pd.read_parquet(f'{self.data_path}/retain_{split}-00000-of-00001.parquet', engine='pyarrow')
        forget_df = pd.read_parquet(f'{self.data_path}/forget_{split}-00000-of-00001.parquet', engine='pyarrow')

        dataset = DatasetDict()

        if task == "all":
            if not split_tasks:
                dataset['retain'] = Dataset.from_pandas(retain_df)
                dataset['forget'] = Dataset.from_pandas(forget_df)
            else:
                tasks = retain_df['task'].unique()
                tasks.sort()
    
                if split_retain:
                    for task in tasks:
                        task_retain_df = retain_df[retain_df['task'] == task]
                        dataset[f'retain_{task[-1]}'] = Dataset.from_pandas(task_retain_df, preserve_index=False)
                else:
                    dataset['retain'] = Dataset.from_pandas(retain_df)
    
                for task in tasks:
                    task_forget_df = forget_df[forget_df['task'] == task]
                    dataset[f'forget_{task[-1]}'] = Dataset.from_pandas(task_forget_df, preserve_index=False)
        else:
            dataset[f'retain_{task[-1]}'] = Dataset.from_pandas(retain_df[retain_df['task'] == task], preserve_index=False)
            dataset[f'forget_{task[-1]}'] = Dataset.from_pandas(forget_df[forget_df['task'] == task], preserve_index=False)

        if self.num_samples is not None:
            dataset = self.sample_dataset(dataset)

        return dataset
        

    def sample_dataset(self, dataset_dict):
        """
        Samples a fixed number of examples from each task in the dataset.

        Args:
            dataset_dict (DatasetDict): Dictionary containing datasets.

        Returns:
            DatasetDict: A dictionary with sampled datasets.
        """
        sampled_dataset_dict = DatasetDict()
        
        for key in dataset_dict.keys():
            # Ensure that sampling is per task, respecting self.num_samples
            dataset = dataset_dict[key]
            if "task" in dataset.column_names:
                tasks = dataset.unique('task')
                sampled_rows = []
                for task in tasks:
                    task_rows = dataset.filter(lambda x: x["task"] == task)
                    sampled_rows.append(task_rows.select(
                        range(self.start_idx, self.start_idx + self.num_samples)
                    ))
                sampled_dataset_dict[key] = concatenate_datasets(sampled_rows).shuffle()
            else:
                sampled_dataset_dict[key] = dataset  # Fallback to full dataset
        
        return sampled_dataset_dict

    def tokenize_and_mask(self, dataset_dict):
        def tokenize_and_mask_function(examples):
            if self.gold_standard:
                combined_texts = [f"{p} {c}{self.tokenizer.eos_token}" for p, c in zip(examples["input"], examples["output"])]
            else:
                combined_texts = [f"{p} {c}" for p, c in zip(examples["input"], examples["output"])]

            tokenized = self.tokenizer(combined_texts, return_tensors=None)

            labels = [ids.copy() for ids in tokenized["input_ids"]]
            for i, prompt in enumerate(examples["input"]):
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                labels[i][:len(prompt_ids)] = [-100] * len(prompt_ids)

            tokenized["labels"] = labels
            return tokenized

        return dataset_dict.map(
            tokenize_and_mask_function,
            batched=True,
            batch_size=None,
            #remove_columns=["id", "input", "output", "task", "split"]
        )

    def __call__(self, split, task='all', split_tasks=True, split_retain=False):
        """
        Combines all steps: creating, sampling, tokenizing, and masking the datasets.

        Args:
            split (str): Split name (e.g., "train", "test").
            split_tasks (bool, optional): Whether to split datasets by tasks. Defaults to True.
            split_retain (bool, optional): Whether to split the retain dataset by tasks. Defaults to False.

        Returns:
            DatasetDict: A ready-to-use DatasetDict for training.
        """
        assert task in ["all", "Task1", "Task2", "Task3"]
        
        dataset_dict = self.create_datasets(split, task, split_tasks, split_retain)
        return self.tokenize_and_mask(dataset_dict)

class DatasetProcessorAugmented:
    def __init__(
        self, 
        data_dir, 
        tokenizer,
        n_samples_per_task=None, 
        start_idx=0, 
        gold_standard=False, 
        forget_relabel=False
    ):
        """
        Dataset processor used for relabelling forget data
        Initializes the DatasetProcessor with the specified data path, tokenizer, and sampling options.

        Args:
            data_path (str): Path to the dataset files.
            tokenizer: Tokenizer to use for tokenization and masking.
            num_samples (int, optional): Number of samples to select from each task/dataset. Defaults to None (no sampling).
            start_idx (int, optional): Starting index for sampling. Defaults to 0.
        """
        self.data_path = data_dir
        self.tokenizer = tokenizer
        self.num_samples = n_samples_per_task
        self.start_idx = start_idx
        self.gold_standard = gold_standard

        self.forget_relabel = forget_relabel

    def relabel_forget_augment_retain(self, retain_df, forget_df, split="validation"):
        def replace_forget_output_with_retain(df1, df2):
            """
            Replace the 'output' column in df1 with the 'output' column from df2.
            Adjust for cases where lengths differ:
            - If df2's 'output' is smaller, duplicate values cyclically.
            - If df2's 'output' is larger, truncate to match df1's length.
            """
            n = len(df1)
            m = len(df2)
            
            # Adjust df2's output column to match the length of df1
            if m < n:
                adjusted_output = (df2['output'].tolist() * ((n // m) + 1))[:n]
            else:
                adjusted_output = df2['output'].iloc[:n].tolist()
            
            # Replace the output column in df1
            df = df1.copy()
            df['output'] = adjusted_output
            return df
            
        def circular_shift_output_column(df, n):            
            result_df = df.copy()
            
            # Perform the circular shift
            n = n % len(result_df)  # Normalize n to ensure it is within the range of the dataframe's length
            result_df['output'] = result_df['output'].iloc[-n:].tolist() + result_df['output'].iloc[:-n].tolist()
            
            return result_df

        modified_dfs = []

        def split_df(df):
            sc_df = df[df['id'].str.contains(r"sc\d$", na=False)]
            qa_df = df[df['id'].str.contains(r"qa\d$", na=False)]

            return sc_df, qa_df

        for task in sorted(retain_df['task'].unique()):
            subtask_forget = forget_df[forget_df['task'] == task]
            subtask_retain = retain_df[retain_df['task'] == task]

            forget_sc_df, forget_qa_df  = split_df(subtask_forget)
            retain_sc_df, retain_qa_df  = split_df(subtask_retain)
            
            modified_forget_sc = replace_forget_output_with_retain(forget_sc_df, retain_sc_df)
            modified_forget_qa = replace_forget_output_with_retain(forget_qa_df, retain_qa_df)

            #modified_forget_sc = circular_shift_output_column(forget_sc_df, 5)
            #modified_forget_qa = circular_shift_output_column(forget_qa_df, 5)
            
            modified_dfs.append(modified_forget_sc)
            modified_dfs.append(modified_forget_qa)

            #print(modified_forget_sc[['input', 'output']].head(5))
            #print(modified_forget_qa[['input', 'output']].head(5))

        #relabeled_forget_df = pd.concat(modified_dfs, ignore_index=True)

        relabeled_forget_df = pd.read_parquet(f'{self.data_path}/forget_{split}_relabeled.parquet', engine='pyarrow')
        #print(relabeled_forget_df[['input', 'output']].head(5))

        augmented_retain_df = pd.concat([retain_df, relabeled_forget_df], ignore_index=True)

        return augmented_retain_df, forget_df

        
    def create_datasets(self, split, task, split_tasks=True, split_retain=False):
        """
        Creates and returns a DatasetDict from parquet files at the specified data path.

        Args:
            split (str): Split name (e.g., "train", "test").
            task (str): Subtask name ("all", "Task1", "Task2", "Task3"), Defaults to all
            split_tasks (bool, optional): Whether to split datasets by tasks. Defaults to True.
            split_retain (bool, optional): Whether to split the retain dataset by tasks. Defaults to False.

        Returns:
            DatasetDict: A dictionary containing the loaded datasets.
        """
        
        retain_df = pd.read_parquet(f'{self.data_path}/retain_{split}-00000-of-00001.parquet', engine='pyarrow')
        forget_df = pd.read_parquet(f'{self.data_path}/forget_{split}-00000-of-00001.parquet', engine='pyarrow')

        if self.forget_relabel:
            retain_df, forget_df = self.relabel_forget_augment_retain(retain_df, forget_df, split=split)
            
        dataset = DatasetDict()

        if task == "all":
            if not split_tasks:
                dataset['retain'] = Dataset.from_pandas(retain_df)
                dataset['forget'] = Dataset.from_pandas(forget_df)
            else:
                tasks = retain_df['task'].unique()
                tasks.sort()
    
                if split_retain:
                    for task in tasks:
                        task_retain_df = retain_df[retain_df['task'] == task]
                        dataset[f'retain_{task[-1]}'] = Dataset.from_pandas(task_retain_df, preserve_index=False)
                else:
                    dataset['retain'] = Dataset.from_pandas(retain_df)
    
                for task in tasks:
                    task_forget_df = forget_df[forget_df['task'] == task]
                    dataset[f'forget_{task[-1]}'] = Dataset.from_pandas(task_forget_df, preserve_index=False)
        else:
            dataset[f'retain_{task[-1]}'] = Dataset.from_pandas(retain_df[retain_df['task'] == task], preserve_index=False)
            dataset[f'forget_{task[-1]}'] = Dataset.from_pandas(forget_df[forget_df['task'] == task], preserve_index=False)

        if self.num_samples is not None:
            dataset = self.sample_dataset(dataset)

        return dataset

    def sample_dataset(self, dataset_dict):
        """
        Samples a fixed number of examples from each task in the dataset.

        Args:
            dataset_dict (DatasetDict): Dictionary containing datasets.

        Returns:
            DatasetDict: A dictionary with sampled datasets.
        """
        sampled_dataset_dict = DatasetDict()
        
        for key in dataset_dict.keys():
            # Ensure that sampling is per task, respecting self.num_samples
            dataset = dataset_dict[key]
            if "task" in dataset.column_names:
                tasks = dataset.unique('task')
                sampled_rows = []
                for task in tasks:
                    task_rows = dataset.filter(lambda x: x["task"] == task)
                    sampled_rows.append(task_rows.select(
                        range(self.start_idx, self.start_idx + self.num_samples)
                    ))
                sampled_dataset_dict[key] = concatenate_datasets(sampled_rows).shuffle()
            else:
                sampled_dataset_dict[key] = dataset  # Fallback to full dataset
        
        return sampled_dataset_dict

    def tokenize_and_mask(self, dataset_dict):
        def tokenize_and_mask_function(examples):
            if self.gold_standard:
                combined_texts = [f"{p} {c}{self.tokenizer.eos_token}" for p, c in zip(examples["input"], examples["output"])]
            else:
                combined_texts = [f"{p} {c}" for p, c in zip(examples["input"], examples["output"])]

            tokenized = self.tokenizer(combined_texts, return_tensors=None)

            labels = [ids.copy() for ids in tokenized["input_ids"]]
            for i, prompt in enumerate(examples["input"]):
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                labels[i][:len(prompt_ids)] = [-100] * len(prompt_ids)

            tokenized["labels"] = labels
            return tokenized

        return dataset_dict.map(
            tokenize_and_mask_function,
            batched=True,
            batch_size=None,
            #remove_columns=["id", "input", "output", "task", "split"]
        )

    def __call__(self, split, task='all', split_tasks=True, split_retain=False):
        """
        Combines all steps: creating, sampling, tokenizing, and masking the datasets.

        Args:
            split (str): Split name (e.g., "train", "test").
            split_tasks (bool, optional): Whether to split datasets by tasks. Defaults to True.
            split_retain (bool, optional): Whether to split the retain dataset by tasks. Defaults to False.

        Returns:
            DatasetDict: A ready-to-use DatasetDict for training.
        """
        assert task in ["all", "Task1", "Task2", "Task3"]
        
        dataset_dict = self.create_datasets(split, task, split_tasks, split_retain)

        return self.tokenize_and_mask(dataset_dict)