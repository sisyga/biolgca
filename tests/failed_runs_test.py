"""A run that fails keeps what it recorded; the Explorer says what happened; sweeps choose what a failed run does.

A failed step is rolled back (``rollback_test.py``). The run then keeps the
recordings of the completed steps, without the empty rows of the steps that
did not run, writes its output files, and says in the error and the model's
metadata where it stopped.
"""

import csv
import threading

import numpy as np
import pytest

from lgca import explore, interaction
from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.plugins import default_registry
from lgca.simulation import (
    DensityRecorder,
    FamilyPopulationRecorder,
    FieldRecorder,
    NodeRecorder,
    PopulationRecorder,
    ScalarTimeSeriesRecorder,
    Schedule,
)
from lgca.study import sweep

_STARTED = []
_LOCK = threading.Lock()


@pytest.fixture(autouse=True)
def _rules():
    plugins, aliases = dict(default_registry._plugins), dict(default_registry._aliases)

    @interaction(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), name="breaks_at")
    def breaks_at(state, at=3, new_family=False):
        if state.step == 1:
            with _LOCK:
                _STARTED.append(at)
        if state.identity_based:  # new families, so that their recording grows
            state.cells.divide(np.arange(len(state.cells)) % 4 == 0, new_family=new_family)
        else:
            state.divide_cells(0.2)
        if state.step == at:
            raise ValueError("broken")

    _STARTED.clear()
    yield
    default_registry._plugins, default_registry._aliases = plugins, aliases


def _spec(family="classical", steps=6, at=3, observers=(), dims=(8, 6)):
    identity = family in ("ib", "nove_ib")
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=dims),
        state=StateSpec(density=0.3 if family in ("classical", "ib") else 1.0, restchannels=1,
                        identity_based=identity, volume_exclusion=family in ("classical", "ib"),
                        fields={"u": np.ones(dims)}),
        time=TimeSpec(steps=steps, seed=2),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "breaks_at", "parameters": {"at": at, "new_family": identity}}, {"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=tuple(observers)),
    )


@pytest.mark.parametrize("family", ["classical", "nove", "ib", "nove_ib"])
def test_a_failed_run_keeps_the_recordings_and_files_of_the_completed_steps(family, tmp_path):
    identity = family in ("ib", "nove_ib")
    observers = [PopulationRecorder(), DensityRecorder(schedule=Schedule(every=2)), NodeRecorder(),
                 FieldRecorder("u"), ScalarTimeSeriesRecorder(output_path=tmp_path / "series.csv")]
    observers += [FamilyPopulationRecorder()] if identity else []
    model = build_model(_spec(family, observers=observers))

    with pytest.raises(ValueError, match="broken") as error:
        model.run(showprogress=False)

    notes = "\n".join(error.value.__notes__)
    assert "step 3 was rolled back" in notes
    assert "the run stopped at step 3 of 6; recordings and output files hold steps 0 to 2" in notes
    lgca = model.lgca
    assert lgca.n_steps.tolist() == [0, 1, 2] and len(lgca.n_t) == 3
    assert np.all(lgca.n_t > 0)  # no empty rows of steps that did not run
    assert lgca.dens_steps.tolist() == [0, 2] and len(lgca.dens_t) == 2
    assert lgca.nodes_steps.tolist() == [0, 1, 2] and len(lgca.nodes_t) == 3
    assert observers[3].steps.tolist() == [0, 1, 2] and len(observers[3].values["u"]) == 3
    if identity:
        assert lgca.fam_pop_steps.tolist() == [0, 1, 2] and len(lgca.fam_pop_t) == 3
        np.testing.assert_array_equal(np.asarray(lgca.nodes_t[-1]).shape, lgca.nodes[lgca.nonborder].shape)
    with open(tmp_path / "series.csv", encoding="utf-8") as handle:
        assert [row["step"] for row in csv.DictReader(handle)] == ["0", "1", "2"]
    runtime = model.metadata["runtime"]
    assert runtime["failed_step"] == 3 and runtime["end_step"] == 2
    assert str(tmp_path / "series.csv") in model.metadata["output_paths"]


def test_the_model_is_at_the_last_completed_step_and_can_go_on():
    model = build_model(_spec(at=3, observers=[PopulationRecorder()]))
    with pytest.raises(ValueError, match="broken"):
        model.run(showprogress=False)

    assert model._step == 2
    cells = int(model.lgca.cell_density[model.lgca.nonborder].sum())
    assert model.lgca.n_t[-1] == cells  # the state that was recorded last


def test_the_explorer_shows_that_the_failed_step_was_not_applied():
    explorer = explore(_spec(at=3))
    try:
        with pytest.raises(ValueError, match="broken"):
            explorer.advance(5)
        assert explorer.step == 2 and explorer._label.value == "step 2"  # the frame shows the applied steps
        explorer._next.click()  # as in the notebook: the error is shown, not raised
        assert "broken" in explorer._status.value and "step 3 was rolled back" in explorer._status.value
        assert explorer.step == 2
    finally:
        explorer.close()


def test_a_sweep_can_record_the_errors_of_failed_runs():
    spec = _spec(steps=3)
    with pytest.warns(UserWarning, match="2 of 4 runs failed"):
        table = sweep(spec, grid={"at": [0, 2]}, seeds=[1, 2], errors="record", showprogress=False)

    assert table.columns.tolist() == ["at", "seed", "error", "population"]
    failed = table[table["at"] == 2]
    assert failed["error"].str.startswith("ValueError: broken").all()
    assert failed["population"].isna().all()
    assert table[table["at"] == 0]["error"].isna().all() and table[table["at"] == 0]["population"].notna().all()
    with pytest.warns(UserWarning, match="1 of 2 runs failed"):
        long = sweep(spec, grid={"at": [0, 2]}, seeds=[1], errors="record", long=True,
                     measure={"population": "population"}, showprogress=False)
    assert long[long["at"] == 2]["step"].isna().all() and len(long[long["at"] == 0]) == 4


def test_a_sweep_that_raises_cancels_the_runs_that_have_not_started():
    spec = _spec(steps=40, at=1, dims=(40, 40))
    with pytest.raises(RuntimeError, match=r"the run with at=1, seed=\d failed: ValueError: broken"):
        sweep(spec, grid={"at": [1, 0]}, seeds=range(10), n_jobs=2, backend="threads", showprogress=False)

    assert len(_STARTED) < 20  # before, every run of the sweep ran before the error was raised


def test_the_errors_policy_is_checked():
    with pytest.raises(ValueError, match="errors must be"):
        sweep(_spec(steps=1), errors="ignore", showprogress=False)
    with pytest.raises(ValueError, match="conflict"):
        sweep(_spec(steps=1), measure={"error": "population"}, errors="record", showprogress=False)
