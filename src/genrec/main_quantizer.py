"""Entry point for quantizer training experiments.

Expected configuration schema::

    output_dir: /path/to/trial_output  # directory to save model checkpoints, logs, and results
    seed: 42
    pretrained_ckpt: /optional/pretrained/run  # optional path to a pretrained checkpoint to load
    save_predictions: true  # whether to save the predictions on the test set
    dataset:
        type: quantizer
        interaction_data_path: /path/to/movielens-1m/proc/user2item.pkl
        textual_data_path: /path/to/movielens-1m/proc/item2title.pkl
        lm_encoder_type: sentence_t5
        lm_encoder_path: /path/to/sentence-t5-base
        ...  # dataset-specific parameters
    collator:
        type: quantizer
        ...  # collator-specific parameters
    model:
        type: rqvae
        config:
            num_codebooks: 3
            ...  # model hyper-parameters
    trainer:
        type: rqvae
        config:
            num_train_epochs: 20000
            ...  # trainer hyper-parameters
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, cast

from accelerate import Accelerator
from accelerate.utils import set_seed
from jaxtyping import Float, Int
import numpy as np
from rich import print_json
import torch
from transformers.utils import logging
import yaml

from .datasets import (
    DatasetSplitLiteral,
    LMEncoderFactory,
    QuantizerCollator,
    QuantizerCollatorConfig,
    QuantizerDataset,
)
from .models import QuantizerModel, QuantizerModelConfigFactory, QuantizerModelFactory
from .trainers import QuantizerTrainerFactory, QuantizerTrainingArgumentsFactory

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
    """Main function for quantizer training experiments."""

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
    parser = argparse.ArgumentParser(description="Quantizer Training Experiment")
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
    save_predictions = bool(cfg.pop("save_predictions", True))

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

    # Builds datasets. Refer to the constructor of `QuantizerDataset`.
    assert "dataset" in cfg, "`dataset` configuration section is missing."
    dataset_cfg: Dict[str, Any] = cfg["dataset"]
    dataset_type = dataset_cfg.pop("type", None)
    assert dataset_type == "quantizer", f"Unsupported dataset type: {dataset_type}"

    lm_encoder_type = dataset_cfg.pop("lm_encoder_type", None)
    assert lm_encoder_type is not None, "`lm_encoder_type` must be specified in the dataset configuration."
    lm_encoder_path = dataset_cfg.pop("lm_encoder_path", None)
    lm_encoder = LMEncoderFactory.create(
        name=lm_encoder_type,
        local_model_dir=lm_encoder_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    aux_item_embeddings: Optional[Float[np.ndarray, "I+1 D_aux"]] = None
    aux_item_embeddings_path = dataset_cfg.pop("aux_item_embeddings_path", None)
    if aux_item_embeddings_path is not None:
        aux_path = Path(aux_item_embeddings_path).expanduser()
        if not aux_path.exists():
            raise FileNotFoundError(f"Auxiliary item embeddings file not found: {aux_path}")
        aux_item_embeddings = np.load(aux_path)

    dataset = QuantizerDataset(
        **dataset_cfg,
        lm_encoder=lm_encoder,
        aux_item_embeddings=aux_item_embeddings,
    )

    if accelerator.is_main_process:
        logger.info(f"Loaded datasets from {dataset_cfg['interaction_data_path']}.")
        logger.info(f"Train dataset: {dataset.stats()}")

    # Builds collator. Refer to the constructor of `QuantizerCollator`.
    assert "collator" in cfg, "`collator` configuration section is missing."
    collator_cfg = cfg["collator"]
    collator_type: str = collator_cfg.pop("type", None)
    assert collator_type == "quantizer", f"Unsupported collator type: {collator_type}"

    collator_cfg = QuantizerCollatorConfig(**collator_cfg)
    collator = QuantizerCollator(dataset, collator_cfg, seed=seed)

    # Builds model. Refer to the constructor of `QuantizerModel` and `QuantizerModelConfig`.
    assert "model" in cfg, "`model` configuration section is missing."
    model_cfg = cfg["model"]
    model_type: str = model_cfg.pop("type", None)

    model_config_cfg = model_cfg.pop("config", {})
    model_config = QuantizerModelConfigFactory.create(
        model_type,
        embed_dim=lm_encoder.embedding_dim,
        **model_config_cfg,
    )

    # Load pretrained checkpoint if provided
    if pretrained_ckpt is not None:
        model = QuantizerModelFactory.from_pretrained(model_type, pretrained_ckpt, config=model_config)
        assert isinstance(
            model, QuantizerModel
        ), f"Pretrained model is not an instance of QuantizerModel: {type(model)}"
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model {model_type} checkpoint from {pretrained_ckpt}.")
    else:
        model = QuantizerModelFactory.create(model_type, config=model_config, **model_cfg)
        if accelerator.is_main_process:
            logger.info(f"Initialized model {model_type}.")

    # Builds trainer. Refer to the constructor of `QuantizerTrainer`.
    assert "trainer" in cfg, "`trainer` configuration section is missing."
    trainer_cfg = cfg["trainer"]
    trainer_type: str = trainer_cfg.pop("type", None)

    training_args_cfg = trainer_cfg.pop("config", {})
    training_args = QuantizerTrainingArgumentsFactory.create(
        trainer_type,
        output_dir=output_dir,
        logging_dir=output_dir / "runs",
        seed=seed,
        **training_args_cfg,
    )

    trainer = QuantizerTrainerFactory.create(
        trainer_type,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        **trainer_cfg,
    )
    if accelerator.is_main_process:
        logger.info(f"Initialized trainer {trainer_type}.")

    # Training
    if training_args.do_train:
        trainer.train()

    # Predicting, save results and metrics
    if training_args.do_predict:
        pred = trainer.predict(dataset)

        if save_predictions and accelerator.is_main_process:
            save_path = output_dir / "test_predictions.pkl.gz"
            with cast(BinaryIO, gzip.open(save_path, "wb")) as f:
                pickle.dump(pred, f)
            logger.info(f"Saved test predictions to {save_path}.")

            # also post-process save the semantic_ids separately for easier usage
            semantic_ids: Int[torch.Tensor, "I C"] = torch.as_tensor(pred.predictions[0])
            processed_semantic_ids: Int[np.ndarray, "I C+1"] = model.post_process_quantized_ids(semantic_ids).numpy()
            semantic_ids_save_path = output_dir / "test_processed_semantic_ids.npy"
            np.save(semantic_ids_save_path, processed_semantic_ids)
            logger.info(f"Saved test semantic IDs to {semantic_ids_save_path}.")

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
