import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from statistics import harmonic_mean
from rouge_score import rouge_scorer


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB allocated")
        print(f"GPU {i}: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB cached")


def make_compute_metrics(model, tokenizer, max_samples=32):
    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred

        # Reduce dataset size if necessary
        if inputs.shape[0] > max_samples:
            idx_to_keep = np.random.choice(inputs.shape[0], max_samples, replace=False)
            inputs = inputs[idx_to_keep]
            labels = labels[idx_to_keep]
            predictions = predictions[idx_to_keep]

        # Find the first valid positions in labels
        first_valid = np.argmax(labels != -100, axis=1) - 1
        first_valid = np.clip(first_valid, 0, labels.shape[1] - 1)  # Ensure valid indices

        # Decode labels with padding replaced
        masked_labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(masked_labels, skip_special_tokens=True)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Metrics initialization
        total_rouge_recall, exact_match_count = 0.0, 0
        num_sent_comp, num_question_answer = 0, 0

        # Loop through samples
        for i, seq in enumerate(inputs):
            trimmed_seq = torch.from_numpy(seq[:first_valid[i] + 1])

            preds = model.generate(
                input_ids=trimmed_seq.unsqueeze(0).to(model.device),
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

            preds = preds[:, first_valid[i] + 1:]
            pred_text = tokenizer.decode(
                preds[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            if i % 4 == 0:
                print(pred_text)

            # Check if task is QA or sentence completion
            if trimmed_seq[-1] == 32: 
                num_question_answer += 1
                exact_match_count += int(
                    decoded_labels[i].strip().lower() == pred_text.strip().lower()
                )
            else:
                num_sent_comp += 1
                total_rouge_recall += scorer.score(
                    decoded_labels[i], pred_text
                )['rougeL'].recall

        # Safeguard against division by zero
        rouge_mean = total_rouge_recall / max(1, num_sent_comp)
        exact_match_rate = exact_match_count / max(1, num_question_answer)

        return {
            "RougeL": round(rouge_mean, 4),
            "EM": round(exact_match_rate, 4)
        }

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def process_log_history(log_history):
    """
    Processes log_history to separate training stats and evaluation stats by task.
    """
    train_stats = []
    eval_stats = {'retain': [[], [], []], 'forget': [[], [], []]}

    for item in log_history:
        keys = item.keys()
        if 'loss' in keys:
            train_stats.append(item)
        else:
            for i in range(3):  # Three tasks
                if f'eval_retain_{i+1}_loss' in keys:
                    eval_stats['retain'][i].append(item)
                elif f'eval_forget_{i+1}_loss' in keys:
                    eval_stats['forget'][i].append(item)
    
    return train_stats, eval_stats


def format_eval_data(eval_data, subset, task_idx):
    """
    Converts evaluation data for a specific subset (retain/forget) and task into a DataFrame.
    """
    keys = [
        f"eval_{subset}_{task_idx+1}_loss",
        f"eval_{subset}_{task_idx+1}_RougeL",
        f"eval_{subset}_{task_idx+1}_EM",
        f"eval_{subset}_{task_idx+1}_runtime",
        f"eval_{subset}_{task_idx+1}_samples_per_second",
        f"eval_{subset}_{task_idx+1}_steps_per_second"
    ]
    
    df = pd.DataFrame(eval_data)
    df.rename(columns={keys[0]: "Loss", keys[1]: "RougeL", keys[2]: "EM"}, inplace=True)
    df.drop(columns=keys[3:], inplace=True)  # Drop unnecessary metrics
    return df, df["Loss"].max()


def prepare_eval_data(eval_stats):
    """
    Prepares evaluation data for all tasks and subsets (retain/forget) as DataFrames.
    """
    max_eval_loss = 0
    for subset in ['retain', 'forget']:
        for i in range(3):  # Three tasks
            eval_stats[subset][i], max_loss = format_eval_data(eval_stats[subset][i], subset, i)
            max_eval_loss = max(max_eval_loss, max_loss)
    return eval_stats, max_eval_loss


def plot_metric(axes, x_idx, retain_series, forget_series, row_idx, col_idx, avg=False):
    """
    Helper function to plot a single metric on the grid.
    """
    axes[row_idx, col_idx].plot(x_idx, retain_series, color='blue')
    axes[row_idx, col_idx].plot(x_idx, forget_series, color='red')


def plot_metrics(log_history, output_path):
    """
    Generate evaluation plots for Loss, RougeL, and Exact Match (EM).
    """
    # Step 1: Process log history
    _, eval_stats = process_log_history(log_history)
    eval_stats, max_loss = prepare_eval_data(eval_stats)
    
    # Step 2: Define constants and setup plot
    x_idx = range(1, len(eval_stats['retain'][0])+1)  # Epoch indices
    key_map = {0: "Loss", 1: "RougeL", 2: "EM"}
    fig, axes = plt.subplots(3, 4, figsize=(11, 7))  # 3 rows (metrics), 4 columns (tasks + avg)
    axes = np.atleast_2d(axes)  # Ensure axes are treated as 2D
    
    # Step 3: Plot each metric for each task and the average
    for i, metric in key_map.items():
        for j in range(3):  # Three tasks
            retain_series = eval_stats['retain'][j][metric]
            forget_series = eval_stats['forget'][j][metric]
            forget_series_adjusted = 1 - forget_series if i > 0 else forget_series  # Invert forget for RougeL/EM
            plot_metric(axes, x_idx, retain_series, forget_series_adjusted, i, j)
        
        # Average column
        retain_series = [eval_stats['retain'][j][metric] for j in range(3)]
        forget_series = [eval_stats['forget'][j][metric] for j in range(3)]
        
        if i == 0:  # Arithmetic mean for Loss
            retain_avg = [np.mean([retain_series[j][k] for j in range(3)]) for k in range(len(retain_series[0]))]
            forget_avg = [np.mean([forget_series[j][k] for j in range(3)]) for k in range(len(forget_series[0]))]
        else:  # Harmonic mean for RougeL and EM
            retain_avg = [harmonic_mean([retain_series[j][k] for j in range(3)]) for k in range(len(retain_series[0]))]
            forget_avg = [harmonic_mean([1 - forget_series[j][k] for j in range(3)]) for k in range(len(forget_series[0]))]
        
        plot_metric(axes, x_idx, retain_avg, forget_avg, i, 3, avg=True)
    
    # Step 4: Set plot labels and titles
    for j, task_title in enumerate(["Task 1", "Task 2", "Task 3", "Mean"]):
        axes[0, j].set_title(task_title, fontsize=14)
    
    for i, row_title in key_map.items():
        axes[i, 0].set_ylabel(row_title, fontsize=14)
    
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch", fontsize=12)
    
    # Enable grid and adjust limits for metrics where applicable
    for row_axes in axes:
        for ax in row_axes:
            ax.grid(True)  # Enable grid for all subplots
            if ax in axes[1:]:
                ax.set_ylim(-0.1, 1.1)  # Set limits for RougeL and EM
            else:
                ax.set_ylim(0, 1.1*max_loss)  # Set limits for Loss
    
    # Step 5: Final layout adjustments and display
    fig.legend(["Retain", "Forget"], loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)  # Adjust legend position
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for legend
    plt.show()

    fig.savefig(f"{output_path}/metrics_plots.png")


def plot_training_stats(log_history):
    """
    Processes the log history and plots learning rate and grad norm in two subplots.
    """
    # Process the log history to extract training stats
    train_stats, _ = process_log_history(log_history)
    
    # Generate cumulative step indices
    cumulative_steps = list(range(1, len(train_stats) + 1))
    learning_rates = [entry['learning_rate'] for entry in train_stats]
    grad_norms = [entry['grad_norm'] for entry in train_stats]
    
    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    
    # Plot learning rate
    axs[0].plot(cumulative_steps, learning_rates, marker='o', label='Learning Rate', color='blue')
    axs[0].set_title("Learning Rate over Cumulative Steps")
    axs[0].set_xlabel("Cumulative Steps")
    axs[0].set_ylabel("Learning Rate")
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot gradient norm
    axs[1].plot(cumulative_steps, grad_norms, marker='o', label='Gradient Norm', color='orange')
    axs[1].set_title("Gradient Norm over Cumulative Steps")
    axs[1].set_xlabel("Cumulative Steps")
    axs[1].set_ylabel("Gradient Norm")
    axs[1].grid(True)
    axs[1].legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()