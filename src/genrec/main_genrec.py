"""Entry point for generative recommendation experiments.

Expected configuration schema::

    output_dir: /path/to/trial_output  # directory to save model checkpoints, logs, and results
    seed: 42
    pretrained_ckpt: /optional/pretrained/run  # optional path to a pretrained checkpoint to load
    test_eval: false  # whether to run evaluation on the test set instead of validation set
    save_predictions: false  # whether to save the predictions on the test set
    dataset:
        type: genrec
        interaction_data_path: /path/to/movielens-1m/proc/user2item.pkl
        textual_data_path: /path/to/movielens-1m/proc/item2title.pkl
        sid_cache_path: /path/to/outputs_quantizer/movielens-1m/test_processed_semantic_ids.npy
        ...  # dataset-specific parameters
    collator:
        type: genrec
        ...  # collator-specific parameters
    model:
        type: tiger
        config:
            hidden_size: 128
            num_heads: 4
            num_encoder_layers: 4
            num_decoder_layers: 4
            ...  # model hyper-parameters
    trainer:
        type: ce
        config:
            num_train_epochs: 200
            ...  # trainer hyper-parameters
"""

# TODO: Update the expected configuration schema

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Dict, cast

from accelerate import Accelerator
from accelerate.utils import set_seed
from jaxtyping import Int
import numpy as np
from rich import print_json
from transformers.utils import logging
import yaml

from .datasets import DatasetSplitLiteral, GenRecCollator, GenRecCollatorConfig, GenRecDataset
from .models import GenRecModel, GenRecModelConfigFactory, GenRecModelFactory
from .trainers import GenRecTrainerFactory, GenRecTrainingArgumentsFactory

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
    """Main function for generative recommendation experiments."""

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
    parser = argparse.ArgumentParser(description="Generative Recommendation Experiment")
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

    # Loads SID cache and builds datasets. Refer to the constructor of `GenRecDataset`.
    assert "dataset" in cfg, "`dataset` configuration section is missing."
    dataset_cfg: Dict[str, Any] = cfg["dataset"]
    dataset_type = dataset_cfg.pop("type", None)
    assert dataset_type == "genrec", f"Unsupported dataset type: {dataset_type}"

    sid_cache_path = dataset_cfg.pop("sid_cache_path", None)
    assert sid_cache_path is not None, "`sid_cache_path` must be provided in the dataset configuration."
    loaded_sid_cache: Int[np.ndarray, "I C"] = np.load(sid_cache_path)
    sid_cache: Int[np.ndarray, "I+1 C"]  # pad a zero vector for padding token
    sid_cache = np.vstack([np.zeros((1, loaded_sid_cache.shape[1]), dtype=loaded_sid_cache.dtype), loaded_sid_cache])

    train_dataset = GenRecDataset(**dataset_cfg, split=DatasetSplitLiteral.TRAIN, sid_cache=sid_cache)
    valid_dataset = GenRecDataset(**dataset_cfg, split=DatasetSplitLiteral.VALIDATION, sid_cache=sid_cache)
    test_dataset = GenRecDataset(**dataset_cfg, split=DatasetSplitLiteral.TEST, sid_cache=sid_cache)

    assert (
        int(sid_cache.shape[0]) == train_dataset.item_size + 1
    ), f"SID cache size ({sid_cache.shape[0]}) does not match dataset item size ({train_dataset.item_size + 1})."

    if accelerator.is_main_process:
        logger.info(f"Loaded datasets from {dataset_cfg['interaction_data_path']}.")
        for dataset in (train_dataset, valid_dataset, test_dataset):
            logger.info(f"{dataset.split.capitalize()} dataset: {dataset.stats()}")

    # Builds collator. Refer to the constructor of `GenRecCollator`.
    assert "collator" in cfg, "`collator` configuration section is missing."
    collator_cfg = cfg["collator"]
    collator_type: str = collator_cfg.pop("type", None)
    assert collator_type == "genrec", f"Unsupported collator type: {collator_type}"

    collator_cfg = GenRecCollatorConfig(**collator_cfg)
    collator = GenRecCollator(train_dataset, collator_cfg, seed=seed)

    # Builds model. Refer to the constructor of `GenRecModel` and `GenRecModelConfig`.
    assert "model" in cfg, "`model` configuration section is missing."
    model_cfg = cfg["model"]
    model_type: str = model_cfg.pop("type", None)

    model_config_cfg = model_cfg.pop("config", {})
    assert train_dataset.sid_cache is not None
    vocab_size = int(train_dataset.sid_cache.max()) + 1  # use dataset.sid_cache, as it may be shifted
    model_config = GenRecModelConfigFactory.create(
        model_type,
        vocab_size=vocab_size,
        eos_token_id=train_dataset.eos_token_id,
        decoder_start_token_id=train_dataset.decoder_start_token_id,
        **model_config_cfg,
    )

    # Load pretrained checkpoint if provided
    if pretrained_ckpt is not None:
        model = GenRecModelFactory.from_pretrained(model_type, pretrained_ckpt, config=model_config)
        assert isinstance(model, GenRecModel), f"Pretrained model is not an instance of GenRecModel: {type(model)}"
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model {model_type} checkpoint from {pretrained_ckpt}.")
    else:
        model = GenRecModelFactory.create(model_type, config=model_config, **model_cfg)
        if accelerator.is_main_process:
            logger.info(f"Initialized model {model_type}.")

    # Builds trainer. Refer to the constructor of `GenRecTrainer`.
    assert "trainer" in cfg, "`trainer` configuration section is missing."
    trainer_cfg = cfg["trainer"]
    trainer_type: str = trainer_cfg.pop("type", None)

    training_args_cfg = trainer_cfg.pop("config", {})
    training_args = GenRecTrainingArgumentsFactory.create(
        trainer_type,
        output_dir=output_dir,
        logging_dir=output_dir / "runs",
        seed=seed,
        **training_args_cfg,
    )

    trainer = GenRecTrainerFactory.create(
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
