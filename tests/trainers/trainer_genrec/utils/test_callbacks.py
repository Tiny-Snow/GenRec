import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from genrec.trainers.trainer_genrec.utils.callbacks import EpochIntervalEvalCallback, HardStopCallback


def _build_state(epoch: float | None) -> TrainerState:
    state = TrainerState()
    state.epoch = epoch
    return state


def _build_control(should_evaluate: bool = True, should_training_stop: bool = False) -> TrainerControl:
    control = TrainerControl()
    control.should_evaluate = should_evaluate
    control.should_training_stop = should_training_stop
    return control


def _build_args(tmp_path) -> TrainingArguments:
    return TrainingArguments(output_dir=str(tmp_path))


def test_epoch_interval_eval_callback_disables_evaluation_when_not_multiple(tmp_path):
    callback = EpochIntervalEvalCallback(eval_interval=3)
    args = _build_args(tmp_path)
    state = _build_state(epoch=2.0)
    control = _build_control(should_evaluate=True)

    result = callback.on_epoch_end(args=args, state=state, control=control)

    assert result.should_evaluate is False


def test_epoch_interval_eval_callback_keeps_evaluation_when_multiple(tmp_path):
    callback = EpochIntervalEvalCallback(eval_interval=2)
    args = _build_args(tmp_path)
    state = _build_state(epoch=4.0)
    control = _build_control(should_evaluate=True)

    result = callback.on_epoch_end(args=args, state=state, control=control)

    assert result.should_evaluate is True


def test_epoch_interval_eval_callback_requires_epoch(tmp_path):
    callback = EpochIntervalEvalCallback(eval_interval=1)
    args = _build_args(tmp_path)
    state = TrainerState()
    control = _build_control(should_evaluate=True)

    with pytest.raises(AssertionError):
        callback.on_epoch_end(args=args, state=state, control=control)


def test_hard_stop_callback_sets_training_stop_when_epoch_reached(tmp_path):
    callback = HardStopCallback(stop_epoch=3)
    args = _build_args(tmp_path)
    state = _build_state(epoch=3.0)
    control = _build_control(should_training_stop=False)

    result = callback.on_epoch_end(args=args, state=state, control=control)

    assert result.should_training_stop is True


def test_hard_stop_callback_does_not_stop_before_epoch(tmp_path):
    callback = HardStopCallback(stop_epoch=5)
    args = _build_args(tmp_path)
    state = _build_state(epoch=4.0)
    control = _build_control(should_training_stop=False)

    result = callback.on_epoch_end(args=args, state=state, control=control)

    assert result.should_training_stop is False


def test_hard_stop_callback_requires_epoch(tmp_path):
    callback = HardStopCallback(stop_epoch=1)
    args = _build_args(tmp_path)
    state = TrainerState()
    control = _build_control(should_training_stop=False)

    with pytest.raises(AssertionError):
        callback.on_epoch_end(args=args, state=state, control=control)
