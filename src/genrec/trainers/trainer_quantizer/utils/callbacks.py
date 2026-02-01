"""Callbacks for quantizer trainers."""

from __future__ import annotations

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

__all__ = [
    "EpochIntervalEvalCallback",
    "HardStopCallback",
]


class EpochIntervalEvalCallback(TrainerCallback):
    """Callback to perform evaluation every `eval_interval` epochs."""

    def __init__(self, eval_interval: int = 5) -> None:
        """Initializes the callback with evaluation parameters.

        Args:
            eval_interval (int): Number of epochs between evaluations.
        """
        self.eval_interval = eval_interval

    def on_epoch_end(  # type: ignore - must return TrainerControl
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Called at the end of an epoch to determine if evaluation should be performed.
        If the current epoch is not a multiple of `eval_interval`, evaluation is skipped.
        """
        assert state.epoch is not None, "EpochEvalCallback requires `state.epoch` to be not None."
        current_epoch = int(state.epoch)
        if current_epoch % self.eval_interval != 0:
            control.should_evaluate = False
            control.should_save = False
        return control


class HardStopCallback(TrainerCallback):
    """Callback to stop training at a specific epoch."""

    def __init__(self, stop_epoch: int = -1) -> None:
        """Initializes the callback with the stopping epoch.

        Args:
            stop_epoch (int): The epoch at which to stop training. If less than 0, training continues
                until the maximum number of epochs. Default is -1.
        """
        self.stop_epoch = stop_epoch

    def on_epoch_end(  # type: ignore - must return TrainerControl
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Called at the end of an epoch to determine if training should be stopped.
        If the current epoch is greater than or equal to `stop_epoch`, training is stopped.
        """
        assert state.epoch is not None, "HardStopCallback requires `state.epoch` to be not None."
        current_epoch = int(state.epoch)
        if self.stop_epoch >= 0 and current_epoch >= self.stop_epoch:
            control.should_training_stop = True
        return control
