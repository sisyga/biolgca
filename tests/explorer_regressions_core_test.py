"""A rule that fails while the explorer plays: the notebook user can still get its traceback."""

import time

import pytest

from lgca import explore
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec
from lgca.pipeline import InteractionPipelineSpec
from lgca.rules import interaction


@pytest.fixture
def broken_model():
    @interaction(kind="birth_death", families="classical", name="student.boom_later")
    def boom_later(state, rate=0.1):
        if state.step >= 2:
            raise RuntimeError("the rule broke at step 2")
        state.divide_cells(rate)

    return ModelSpec(space=SpaceSpec(geometry="square", dims=(8, 8)), state=StateSpec(density=1.0),
                     time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[boom_later()]))


def test_error_while_playing_is_shown_and_kept_with_its_traceback(broken_model):
    explorer = explore(broken_model, interval=0.0)
    try:
        assert explorer.last_error is None
        explorer.play()
        end = time.monotonic() + 10
        while explorer._play.value and time.monotonic() < end:
            time.sleep(0.01)
        assert "the rule broke at step 2" in explorer._status.value and ".last_error" in explorer._status.value
        error = explorer.last_error
        assert isinstance(error, RuntimeError)
        frames = []
        tb = error.__traceback__
        while tb is not None:
            frames.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        assert frames[-1] == "boom_later"  # the traceback leads to the rule
        explorer.reset()
        assert explorer.last_error is None
    finally:
        explorer.close()


def test_error_of_a_step_button_is_kept(broken_model):
    explorer = explore(broken_model, steps_per_frame=3, interval=0.0)
    try:
        explorer._guarded(explorer.advance)  # what the step button does
        assert isinstance(explorer.last_error, RuntimeError)
    finally:
        explorer.close()
