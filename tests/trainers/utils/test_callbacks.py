import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from genrec.trainers.utils.callbacks import EpochIntervalEvalCallback


def _build_state(epoch):
    state = TrainerState()
    state.epoch = epoch
    return state


def _build_control(should_evaluate=True):
    control = TrainerControl()
    control.should_evaluate = should_evaluate
    return control


def _build_args(tmp_path):
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
