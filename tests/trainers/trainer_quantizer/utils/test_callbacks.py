from __future__ import annotations

from transformers import TrainerControl, TrainerState, TrainingArguments

from genrec.trainers.trainer_quantizer.utils.callbacks import EpochIntervalEvalCallback, HardStopCallback


def _build_state(epoch: float) -> TrainerState:
    state = TrainerState()
    state.epoch = epoch
    return state


def _build_control(should_evaluate: bool = True, should_save: bool = True) -> TrainerControl:
    control = TrainerControl()
    control.should_evaluate = should_evaluate
    control.should_save = should_save
    return control


def test_epoch_interval_eval_callback_disables_evaluation_when_not_due(tmp_path) -> None:
    callback = EpochIntervalEvalCallback(eval_interval=2)
    args = TrainingArguments(output_dir=str(tmp_path))
    state = _build_state(epoch=3.0)
    control = _build_control(should_evaluate=True, should_save=True)

    updated = callback.on_epoch_end(args=args, state=state, control=control)

    assert updated.should_evaluate is False
    assert updated.should_save is False


def test_epoch_interval_eval_callback_keeps_evaluation_when_due(tmp_path) -> None:
    callback = EpochIntervalEvalCallback(eval_interval=2)
    args = TrainingArguments(output_dir=str(tmp_path))
    state = _build_state(epoch=4.0)
    control = _build_control(should_evaluate=True, should_save=True)

    updated = callback.on_epoch_end(args=args, state=state, control=control)

    assert updated.should_evaluate is True
    assert updated.should_save is True


def test_hard_stop_callback_requests_stop_at_epoch(tmp_path) -> None:
    callback = HardStopCallback(stop_epoch=3)
    args = TrainingArguments(output_dir=str(tmp_path))
    state = _build_state(epoch=3.0)
    control = TrainerControl()

    updated = callback.on_epoch_end(args=args, state=state, control=control)

    assert updated.should_training_stop is True


def test_hard_stop_callback_ignores_negative_stop_epoch(tmp_path) -> None:
    callback = HardStopCallback(stop_epoch=-1)
    args = TrainingArguments(output_dir=str(tmp_path))
    state = _build_state(epoch=10.0)
    control = TrainerControl()

    updated = callback.on_epoch_end(args=args, state=state, control=control)

    assert updated.should_training_stop is False
