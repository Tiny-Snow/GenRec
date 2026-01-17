"""Entry point for sequential recommendation experiments.

Expected configuration schema::

    output_dir: /path/to/trial_output  # directory to save model checkpoints, logs, and results
    seed: 42
    pretrained_ckpt: /optional/pretrained/run  # optional path to a pretrained checkpoint to load
    test_eval: false  # whether to run evaluation on the test set instead of validation set
    save_predictions: false  # whether to save the predictions on the test set
    dataset:
        type: seqrec
        interaction_data_path: data/movielens/train.parquet
        ...  # dataset-specific parameters
    collator:
        type: seqrec
        ...  # collator-specific parameters
    model:
        type: sasrec
        config:
            hidden_size: 128
            ...  # model hyper-parameters
    trainer:
        type: bce
        config:
            num_train_epochs: 10
            ...  # trainer hyper-parameters
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Dict, cast

import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from rich import print_json
from transformers.utils import logging

from .datasets import DatasetSplitLiteral, SeqRecCollator, SeqRecCollatorConfig, SeqRecDataset
from .models import SeqRecModel, SeqRecModelConfigFactory, SeqRecModelFactory
from .trainers import SeqRecTrainerFactory, SeqRecTrainingArgumentsFactory

__all__ = [
    "main",
]


logger = logging.get_logger(__name__)


def load_config(config_path: Path, *, is_main_process: bool) -> Dict[str, Any]:
    """Loads and prints a configuration file in JSON/YAML format."""
    if not config_path.exists():  # pragma: no cover - defensive check
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix in {".yaml", ".yml"}:  # pragma: no cover - not default format
        with open(config_path, "r", encoding="utf-8") as f:
            configs: Dict[str, Any] = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            configs: Dict[str, Any] = json.load(f)
    else:  # pragma: no cover - defensive check
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    if is_main_process:
        logger.info(f"Loaded configuration from {config_path}:")
        print_json(data=configs)

    return configs


def save_experiment_config(
    configs: Dict[str, Any],
    *,
    output_dir: Path,
    config_path: Path,
    is_main_process: bool,
) -> None:
    """Persists the resolved configuration alongside useful metadata."""

    if not is_main_process:
        return

    serialisable_cfg = copy.deepcopy(configs)
    metadata = serialisable_cfg.setdefault("_meta", {})
    metadata["config_path"] = str(config_path.resolve())

    output_config_path = output_dir / "experiment_config.json"
    if output_config_path.exists():  # pragma: no cover - defensive check
        logger.warning(f"Overwriting existing configuration file at: {output_config_path}")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(serialisable_cfg, f, indent=4)


def main():
    """Main function for sequential recommendation experiments."""

    accelerator = Accelerator()

    # Set up logging (only master process should emit verbose logs)
    if accelerator.is_main_process:
        logging.set_verbosity_info()  # set logging level to INFO
    else:
        logging.set_verbosity_error()
    if accelerator.is_local_main_process:
        logging.enable_progress_bar()  # enable tqdm progress bar for local main process
    else:
        logging.disable_progress_bar()

    # Parse args and load configuration
    parser = argparse.ArgumentParser(description="Sequential Recommendation Experiment")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration file",
    )
    args = parser.parse_args()
    if accelerator.is_main_process:
        logger.info(f"Parsed command-line arguments: config={args.config}")
    cfg = load_config(args.config, is_main_process=accelerator.is_main_process)
    raw_cfg = copy.deepcopy(cfg)

    # Extract experiment-level parameters from the configuration
    try:
        output_dir = Path(cfg.pop("output_dir")).expanduser()
    except KeyError as exc:
        raise KeyError("`output_dir` must be specified in the configuration file.") from exc

    seed = int(cfg.pop("seed", 42))
    pretrained_ckpt_value = cfg.pop("pretrained_ckpt", None)
    pretrained_ckpt = Path(pretrained_ckpt_value).expanduser() if pretrained_ckpt_value is not None else None
    test_eval = bool(cfg.pop("test_eval", False))
    save_predictions = bool(cfg.pop("save_predictions", False))

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    save_experiment_config(
        raw_cfg,
        output_dir=output_dir,
        config_path=args.config,
        is_main_process=accelerator.is_main_process,
    )
    accelerator.wait_for_everyone()

    if pretrained_ckpt is not None:
        if not pretrained_ckpt.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")
        if not pretrained_ckpt.is_dir():
            raise ValueError(f"`pretrained_ckpt` should be a directory: {pretrained_ckpt}")

    # Set up seed
    set_seed(seed)

    # Builds datasets. Refer to the constructor of `SeqRecDataset`.
    assert "dataset" in cfg, "`dataset` configuration section is missing."
    dataset_cfg: Dict[str, Any] = cfg["dataset"]
    dataset_type = dataset_cfg.pop("type", None)
    assert dataset_type == "seqrec", f"Unsupported dataset type: {dataset_type}"

    train_dataset = SeqRecDataset(**dataset_cfg, split=DatasetSplitLiteral.TRAIN)
    valid_dataset = SeqRecDataset(**dataset_cfg, split=DatasetSplitLiteral.VALIDATION)
    test_dataset = SeqRecDataset(**dataset_cfg, split=DatasetSplitLiteral.TEST)

    if accelerator.is_main_process:
        logger.info(f"Loaded datasets from {dataset_cfg['interaction_data_path']}.")
        for dataset in (train_dataset, valid_dataset, test_dataset):
            logger.info(f"{dataset.split.capitalize()} dataset: {dataset.stats()}")

    # Builds collator. Refer to the constructor of `SeqRecCollator`.
    assert "collator" in cfg, "`collator` configuration section is missing."
    collator_cfg = cfg["collator"]
    collator_type: str = collator_cfg.pop("type", None)
    assert collator_type == "seqrec", f"Unsupported collator type: {collator_type}"

    collator_cfg = SeqRecCollatorConfig(**collator_cfg)
    collator = SeqRecCollator(train_dataset, collator_cfg, seed=seed)

    # Builds model. Refer to the constructor of `SeqRecModel` and `SeqRecModelConfig`.
    assert "model" in cfg, "`model` configuration section is missing."
    model_cfg = cfg["model"]
    model_type: str = model_cfg.pop("type", None)

    model_config_cfg = model_cfg.pop("config", {})
    item_size = train_dataset.item_size
    model_config = SeqRecModelConfigFactory.create(model_type, item_size=item_size, **model_config_cfg)

    # Load pretrained checkpoint if provided
    if pretrained_ckpt is not None:
        model = SeqRecModelFactory.from_pretrained(model_type, pretrained_ckpt, config=model_config)
        assert isinstance(model, SeqRecModel), f"Pretrained model is not an instance of SeqRecModel: {type(model)}"
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model {model_type} checkpoint from {pretrained_ckpt}.")
    else:
        model = SeqRecModelFactory.create(model_type, config=model_config, **model_cfg)
        if accelerator.is_main_process:
            logger.info(f"Initialized model {model_type}.")

    # Builds trainer. Refer to the constructor of `SeqRecTrainer`.
    assert "trainer" in cfg, "`trainer` configuration section is missing."
    trainer_cfg = cfg["trainer"]
    trainer_type: str = trainer_cfg.pop("type", None)

    training_args_cfg = trainer_cfg.pop("config", {})
    training_args = SeqRecTrainingArgumentsFactory.create(
        trainer_type,
        output_dir=output_dir,
        logging_dir=output_dir / "runs",
        seed=seed,
        **training_args_cfg,
    )

    trainer = SeqRecTrainerFactory.create(
        trainer_type,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if not test_eval else test_dataset,
        **trainer_cfg,
    )
    if accelerator.is_main_process:
        logger.info(f"Initialized trainer {trainer_type}.")

    # Training
    if training_args.do_train:
        trainer.train()

    # Predicting, save results and metrics
    if training_args.do_predict:
        pred = trainer.predict(test_dataset)

        if save_predictions and accelerator.is_main_process:
            save_path = output_dir / "test_predictions.pkl.gz"
            with cast(BinaryIO, gzip.open(save_path, "wb")) as f:
                pickle.dump(pred, f)
            logger.info(f"Saved test predictions to {save_path}.")

        if accelerator.is_main_process:
            logger.info("Test set metrics:")
            print_json(data=pred.metrics)
            save_path = output_dir / "test_metrics.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(pred.metrics, f, indent=4)
            logger.info(f"Saved test metrics to {save_path}.")

    # Exit
    accelerator.end_training()


if __name__ == "__main__":
    main()
