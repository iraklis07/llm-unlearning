# Parameter-Efficient Unlearning for Large Language Models using Data Chunking

This repository contains the codebase accompanying the paper *AILS-NTUA at SemEval-2025 Task 4: Parameter-Efficient Unlearning for Large Language Models using Data Chunking*. We introduce two novel gradient-based unlearning methods for large language models (LLMs), leveraging parameter-efficient fine-tuning to selectively forget training data. Read the full paper on [arXiv](https://arxiv.org/abs/2503.02443).

Our method **Sequential Unlearning with Gradient Difference (SUGD)** achieved leading performance in *SemEval 2025 Task 4: Unlearning Sensitive Content from Large Language Models*, significantly outperforming all other submissions.

## 🏆 Leaderboard (7B Model)

| Method                           | Aggregate | Task Aggregate | MIA Score | MMLU Avg. |
|:--------------------------------:|:---------:|:--------------:|:---------:|:---------:|
| Gradient Ascent                  | 0.394     | 0              | 0.912     | 0.269     |
| Gradient Difference              | 0.243     | 0              | 0.382     | 0.348     |
| KL Minimization                  | 0.395     | 0              | 0.916     | 0.269     |
| Negative Preference Optimization | 0.188     | 0.021          | 0.080     | 0.463     |
| 2nd Best                         | 0.487     | 0.944          | 0.048     | 0.471     |
| **SUGD (Ours)**                  | **0.706** | 0.827          | **0.847** | 0.443     |

## Dataset and Models

- [**Dataset**](https://github.com/amazon-science/lume-llm-unlearning/tree/main) : The data used are part of the *LUME: LLM Unlearning with Multitask Evaluations* Benchmark.
- [**Models**](https://huggingface.co/llmunlearningsemeval2025organization) : The models (based on the OLMo family) are fine-tuned to memorize all data of the LUME benchmark. Two versions are released: one with 7B parameters and one with 1B.

---

## Repository Structure

- `configs/` – Example configuration files for running AGAD and SUGD.
- `methods/` – Core implementation of:
  - Alternating Gradient Ascent-Descent (AGAD)
  - Sequential Unlearning with Gradient Difference (SUGD)
- `utils/` – Helper functions for preprocessing, evaluation, and metrics computation.
- `semeval25-unlearning-data/` – Dataset as provided by the SemEval 2025 Task 4 organizers.
- `example-notebooks/` – Sample notebooks with outputs to give a feeling of the expected results.

### Notebooks

The experiments were conducted using **Jupyter Notebooks** on **AWS SageMaker Studio** to facilitate job scheduling and experimentation.

- `Gold-Standard.ipynb`: Trains the base model on retain data only.
- `AGAD.ipynb`: Implements the Alternating Gradient Ascent-Descent method.
- `SUGD.ipynb`: Implements the Sequential Unlearning with Gradient Difference method.
- `SUGD-Experiments.ipynb`: Notebook used for extensive experimentation and manual hyperparameter tuning of the SUGD method (supports multi-gpu training).

AGAD and SUGD notebooks are designed for experimentation in one GPU and would require some modifications to run on a distributed setup. See below for details on that.


### Setup Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/iraklis07/llm-unlearning.git
cd llm-unlearning
```

#### 2. Create and activate a conda environment

```bash
conda create -n unlearning python=3.11
conda activate unlearning
```

#### 3. Install PyTorch with CUDA support (via conda)

```bash
conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### 4. Install required Python packages (via pip)

```bash
pip install -r requirements.txt
```

#### 5. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name unlearning --display-name "unlearning"
```

You are now set to run the notebooks using the created kernel.

---

## Distributed Training (Multi-GPU with Accelerate)

This repository supports **multi-GPU training** using 🤗 `accelerate`. We do not use DeepSpeed in our setup. Below is an example accelerate config file for distributed training in 8 GPUs with mixed precision, leveraging the DistributedDataParallel (DDP) technique (MULTI-GPU). More details on how to configure one can be found [here](https://huggingface.co/docs/accelerate/en/package_reference/cli). 

`default_config.yaml`

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: true
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

To set up and run a notebook in a distributed setting, one needs to define a wrapper function that incorporates the whole code and launch it from a notebook cell using ```notebook_launcher(function, args, num_processes)```, always esnuring the presence of a configuration file in the system similar to the above. See SUGD-Experiments.ipynb for more details.

---

## Citation Information

If you use this codebase in your work, please cite:

```bibtex
@misc{premptis2025ailsntuasemeval2025task4,
  title={AILS-NTUA at SemEval-2025 Task 4: Parameter-Efficient Unlearning for Large Language Models using Data Chunking}, 
  author={Iraklis Premptis and Maria Lymperaiou and Giorgos Filandrianos and Orfeas Menis Mastromichalakis and Athanasios Voulodimos and Giorgos Stamou},
  year={2025},
  eprint={2503.02443},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.02443}, 
}
```

## Contact

Feel free to reach out if you have any questions. [Iraklis Premptis](mailto:h.premptis@gmail.com)
>>>>>>> 35aaa96 (Initial commit)
