"""Callbacks for trainers."""

from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__all__ = [
    "EpochIntervalEvalCallback",
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
