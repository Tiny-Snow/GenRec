# GenRec: Unified Generative Recommendation Framework

<p align="center">
  <a href="https://opensource.org/licenses/GPL-3.0"><img src="https://img.shields.io/badge/License-GPL%203.0-blue.svg" alt="License: GPL-3.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://python-poetry.org/"><img src="https://img.shields.io/badge/dependency%20management-poetry-blue" alt="Poetry"></a>
</p>

## 📖 Introduction

**GenRec** is a unified, modular, and extensible research framework for sequential and generative recommendation systems. It is designed with the following principles:

- **Generality**: Supports both traditional sequential recommendation and modern generative recommendation paradigms
- **Standardization**: Provides consistent APIs and configuration schemas across different model families
- **Modularity**: Decouples datasets, models, trainers, and losses for flexible composition
- **Reproducibility**: Built-in experiment tracking, checkpointing, and deterministic training
- **Extensibility**: Easy to add new models, losses, and evaluation metrics
- **State-of-the-art**: Includes cutting-edge sequential recommendation models (e.g., [SASRec++](https://arxiv.org/abs/1808.09781) and [HSTU](https://arxiv.org/abs/2402.17152)), SID tokenizers (e.g., [RQ-VAE](https://arxiv.org/abs/2203.01941)), and generative recommendation models (e.g., [TIGER](https://arxiv.org/abs/2305.05065))

The framework is built on [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/transformers/), leveraging [Accelerate](https://huggingface.co/docs/accelerate/) for distributed training and [Poetry](https://python-poetry.org/) for dependency management.

> [!NOTE]
> This repository is under active development. The `seqrec` module is stable and ready for use, while `quantizer` and `genrec` modules are still being actively developed.

## 🎉 News

- **[May 16, 2026]** Our paper [Mitigating Popularity Bias Amplification in Scaling Transformer-based Sequential Recommenders](https://doi.org/10.1145/3770855.3818185), which proposes **SPRINT**, has been accepted to **KDD 2026**!

## ✨ Features

### Datasets

GenRec uses datasets from the [GenRec-Datasets](https://github.com/Tiny-Snow/GenRec-Datasets) repository. The datasets are provided in a standardized format for easy integration. Supported datasets cover a wide range of domains, including Amazon, Douban, Gowalla, MovieLens, Yelp, etc.

### Models

#### Sequential Recommendation

For sequential recommendation, we implement several state-of-the-art transformer-based architectures.

| Model | Description | Paper |
|-------|-------------|-------|
| **SASRec++** | Advanced SASRec with LlamaDecoder architecture | [Paper Link](https://arxiv.org/abs/1808.09781) |
| **HSTU** | Standard Hierarchical Sequential Transduction Units | [Paper Link](https://arxiv.org/abs/2402.17152) |

#### SID Tokenization

For SID tokenization, we implement the standard Residual-Quantized Variational AutoEncoder (RQ-VAE) architecture. We plan to add more advanced tokenizers in the future.

| Model | Description | Paper |
|-------|-------------|-------|
| **RQ-VAE** | Standard Residual-Quantized Variational AutoEncoder | [Paper Link](https://arxiv.org/abs/2203.01941) |

#### Generative Recommendation

For generative recommendation, we implement the standard T5-based architecture as proposed in the TIGER paper. We plan to add more advanced generative models in the future.

| Model | Description | Paper |
|-------|-------------|-------|
| **TIGER** | Standard T5-based generative recommendation model | [Paper Link](https://arxiv.org/abs/2305.05065) |

### Loss Functions

#### Sequential Recommendation

For sequential recommendation, we implement various loss functions, debiasing methods, and regularization techniques to improve model performance and mitigate biases.

| Loss | Description | Paper |
|------|-------------|-------|
| **BCE** | Standard Binary Cross-Entropy loss | [Paper Link](https://arxiv.org/abs/1808.09781) |
| **SL** | Standard Sampled Softmax Loss | [Paper Link](https://arxiv.org/abs/2402.17152) |
| **DROS** | Distributionally Robust Optimization for sequential recommendation | [Paper Link](https://dl.acm.org/doi/abs/10.1145/3539618.3591624) |
| **LogDet** | Redundancy reduction-based regularization | [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d5753be6f71fbfefaf47aa27ec41279c-Abstract-Conference.html) |
| **D2LR** | IPS-based debiasing method | [Paper Link](https://dl.acm.org/doi/abs/10.1145/3726302.3730181) |
| **R2Rec** | Popularity-aware reweighting method | [Paper Link](https://dl.acm.org/doi/abs/10.1145/3696410.3714572) |
| **ReSN** | Spectral regularization method in collaborative filtering | [Paper Link](https://dl.acm.org/doi/abs/10.1145/3701551.3703579) |
| **SPRINT** | Scalable Popularity Regularization IN Transformers | [Paper Link](https://doi.org/10.1145/3770855.3818185) |

#### SID Tokenization

For SID tokenization, we implement the standard VAE loss function, which consists of a reconstruction term and a commitment term. We plan to add more advanced loss functions for tokenization in the future.

| Loss | Description | Paper |
|------|-------------|-------|
| **VAE Loss** | Standard VAE loss with reconstruction and commitment terms | [Paper Link](https://arxiv.org/abs/2203.01941) |

#### Generative Recommendation

For generative recommendation, we implement the standard cross-entropy loss for sequence generation. We plan to add more advanced loss functions for generative recommendation in the future.

| Loss | Description | Paper |
|------|-------------|-------|
| **Cross-Entropy** | Standard CE loss for sequence generation | [Paper Link](https://arxiv.org/abs/2305.05065) |

## 📦 Installation

GenRec uses [Poetry](https://python-poetry.org/) for dependency management. Ensure you have Python 3.13+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/Tiny-Snow/GenRec.git
cd GenRec
```

### 2. Install dependencies

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 3. Activate the virtual environment

```bash
poetry shell
```

## 🚀 Quick Start

### Basic Usage with CLI

GenRec provides command-line interfaces for different tasks. Example configuration templates are available in [`scripts/configs/`](scripts/configs/). For instance, to run a sequential recommendation experiment with `SASRec++` + `SPRINT` and `BCE` loss on the `Amazon-Beauty` dataset, you can use:

```bash

poetry run python -m genrec.main_seqrec --config scripts/configs/seqrec/sasrec_sprint_bce_amazon-beauty.yaml
```

### Configuration Files

All experiments are configured via YAML files with easy-to-understand hierarchical structures for global settings, dataset parameters, model architecture, training arguments, and evaluation metrics. Here's a [template yaml](scripts/configs/seqrec/template.yaml) for sequential recommendation:

```yaml
# A template config file for sequence recommendation tasks
# If you want to add hyperparameter search space, use the "search__" prefix before the parameter name.

# global settings
seed: 42
output_dir: null  # TODO: output directory to save model checkpoints, logs, and results
pretrained_ckpt: null  # optional path to a pretrained checkpoint to load
test_eval: false  # whether to run evaluation on the test set instead of validation set
save_predictions: false  # whether to save the predictions on the test set

# dataset settings
dataset:
    type: seqrec

    interaction_data_path: null  # TODO: path to interaction data file
    max_seq_length: 50
    min_seq_length: 1

# collator settings
collator:
    type: seqrec

    num_negative_samples: 16
    negative_sampling_strategy: uniform

# model settings
model:
    type: sasrec

    config:
        # base model parameters
        hidden_size: 256
        num_attention_heads: 4
        num_hidden_layers: 2

        # subclass model parameters
        attention_dropout: 0

# trainer settings
trainer:
    type: bce

    config:
        # training arguments - Run control
        do_train: true
        do_eval: true
        do_predict: true
        overwrite_output_dir: true
        remove_unused_columns: false

        # training arguments - Optimization & schedule
        num_train_epochs: 200
        per_device_train_batch_size: 512
        per_device_eval_batch_size: 1024
        gradient_accumulation_steps: 1  # batch_size = per_device_train_batch_size * num_devices * gradient_accumulation_steps
        learning_rate: 1.0e-3
        weight_decay: 0.1
        max_grad_norm: 1.0
        optim: adamw_torch
        lr_scheduler_type: linear
        warmup_ratio: 0.05

        # training arguments - Evaluation & checkpointing
        eval_strategy: epoch
        save_strategy: epoch
        eval_delay: 0  # skip warmup
        eval_accumulation_steps: 1
        save_total_limit: 1  # keep only the best checkpoint
        load_best_model_at_end: true  # load the best model when finished training
        metric_for_best_model: ndcg@5  # should exist in the metrics
        greater_is_better: true
        prediction_loss_only: false
        save_safetensors: true

        # training arguments - Parallelism & precision
        dataloader_num_workers: 4
        dataloader_pin_memory: true
        dataloader_drop_last: true
        ddp_find_unused_parameters: true
        ddp_broadcast_buffers: false
        gradient_checkpointing: false
        bf16: false
        tf32: true

        # training arguments - Logging / tracking
        logging_strategy: epoch
        report_to: ["tensorboard"]

        # base trainer parameters
        norm_embeddings: false  # whether to L2-normalize user and item embeddings
        eval_interval: 5  # run metrics every epoch
        train_stop_epoch: -1  # by default, do not stop training early
        metrics:
        - ["hr", {}]
        - ["ndcg", {}]
        - ["popularity", {p: [0.1, 0.2]}]
        - ["unpopularity", {p: [0.2, 0.4]}]
        model_loss_weight: 0.0
        top_k: [1, 5, 10]

        # subclass trainer parameters
```

### Grid Search for Hyperparameter Tuning

GenRec provides a simple yet powerful grid search script for hyperparameter optimization. You can use the provided shell script:

```bash
# Edit scripts/grid_search.sh to configure GPU groups and paths, then run:
bash scripts/grid_search.sh
```

Or use the Python script directly:

```bash
poetry run python scripts/grid_search.py \
  --template scripts/configs/seqrec/template.yaml \
  --search scripts/configs/seqrec/sasrec_sprint_bce_amazon-beauty.yaml \
  --main genrec.main_seqrec \
  --gpu_groups "4,5" --gpu_groups "6,7" \
  --output_root ./outputs/seqrec/sasrec_sprint
```

This allows you to easily specify a search configuration file with `search__` prefixed parameters, and the template configurations in `template` will be automatically filled if it is not specified in the search config. An example search configuration file looks like this:

```yaml
# search_config.yaml
model:
  type: sasrec_sprint
  config:
    hidden_size: 256
    search__num_attention_heads: [1, 2, 4, 8]
    search__num_hidden_layers: [2, 4, 6, 8]
    search__attention_dropout: [0, 0.2, 0.4]
    search__sprint_attention_weight: [5.0, 2.5, 1.0, 0.1]

trainer:
  type: bce
  config:
    learning_rate: 1.0e-3
    search__weight_decay: [0, 0.1]
```

The grid search script will:
- Generate all hyperparameter combinations from `search__` prefixed parameters
- Schedule trials across available GPU groups
- Save results and logs for each trial
- Aggregate metrics into a CSV file for analysis

## 📁 Project Structure

```
GenRec/
├── src/genrec/
│   ├── main_seqrec.py          # Entry point for sequential recommendation
│   ├── main_quantizer.py       # Entry point for SID tokenizer training
│   ├── main_genrec.py          # Entry point for generative recommendation
│   ├── datasets/               # Dataset loaders and collators
│   │   ├── dataset_seqrec.py
│   │   ├── dataset_quantizer.py
│   │   └── dataset_genrec.py
│   ├── models/                 # Model implementations
│   │   ├── model_seqrec/       # Sequential recommendation models
│   │   │   ├── sasrec.py
│   │   │   ├── sasrec_sprint.py
│   │   │   ├── hstu.py
│   │   │   └── hstu_sprint.py
│   │   ├── model_quantizer/    # Item tokenizers
│   │   │   └── rqvae.py
│   │   └── model_genrec/       # Generative recommendation models
│   │       └── tiger.py
│   ├── trainers/               # Training loops and loss functions
│   │   ├── trainer_seqrec/
│   │   │   ├── bce.py          # Binary cross-entropy loss
│   │   │   ├── sl.py           # Sampled softmax loss
│   │   │   ├── bce_dros.py     # BCE + DROS regularization
│   │   │   ├── bce_logdet.py   # BCE + LogDet regularization
│   │   │   ├── bce_resn.py     # BCE + ReSN (SPRINT)
│   │   │   └── sl_*.py         # SL variants with regularizations
│   │   ├── trainer_quantizer/
│   │   └── trainer_genrec/
│   └── __init__.py
├── scripts/
│   ├── grid_search.py          # Grid search script
│   ├── grid_search.sh          # Shell wrapper for grid search
│   └── configs/                # Configuration templates
│       ├── seqrec/
│       ├── quantizer/
│       └── genrec/
├── tests/                      # Unit tests
├── pyproject.toml              # Poetry configuration
└── README.md
```

## ⚖️ License

This software is provided under the [GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/) © 2026 [Tiny Snow](https://tiny-snow.github.io/). All rights reserved.

## 💭 Feedback

This repository is initially built by [Tiny Snow](https://tiny-snow.github.io/) and [Wentworth1028](https://github.com/Wentworth1028) for research purpose. If you find any bugs or want to contribute to this repository, please feel free to open an issue or pull request.

## 📝 Citation

If you find GenRec useful in your research, please consider citing:

```bibtex
@inproceedings{yang2026mitigating,
  title={Mitigating Popularity Bias Amplification in Scaling Transformer-based Sequential Recommenders},
  author={Yang, Weiqin and Pan, Yue and Gao, Chongming and Zhou, Sheng and Wang, Xiang and and Wang, Can and Chen, Jiawei},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  year={2026}
}
@inproceedings{yang2026bear,
  title={BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models},
  author={Yang, Weiqin and Wang, Bohao and Xu, Zhenxiang and Chen, Jiawei and Zhang, Shengjia and Chen, Jingbang and Jin, Canghong and Wang, Can},
  booktitle={Proceedings of the 49th international ACM SIGIR conference on Research and development in Information Retrieval},  
  year={2026}
}
```
