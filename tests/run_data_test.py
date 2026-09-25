"""result.data: the recorded data of a run by name, with its steps."""

import numpy as np
import pytest

from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    run_model,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import (
    DensityRecorder,
    NodeRecorder,
    PerTypeRecorder,
    PopulationRecorder,
    ScalarTimeSeriesRecorder,
    Schedule,
)

FAMILIES = [(True, False), (False, False), (True, True), (False, True)]


def _spec(ve, ib, observers, steps=6):
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(8, 8)),
        state=StateSpec(density=0.4, restchannels=1, volume_exclusion=ve, identity_based=ib),
        time=TimeSpec(steps=steps, seed=3),
        dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=observers))


def _occupied(lgca):
    return int((lgca.cell_density[lgca.nonborder] > 0).sum())


@pytest.mark.parametrize("ve, ib", FAMILIES)
def test_the_data_are_the_recorded_arrays_by_name(ve, ib, tmp_path):
    observers = [NodeRecorder(schedule=Schedule(every=2)), DensityRecorder(), PopulationRecorder(),
                 PerTypeRecorder(),
                 ScalarTimeSeriesRecorder(metrics={"occupied": _occupied}, schedule=Schedule(steps=[0, 6]),
                                          output_path=tmp_path / "series.csv")]
    result = run_model(_spec(ve, ib, observers), showprogress=False)
    data, lgca = result.data, result.lgca

    assert list(data) == ["population", "density", "nodes", "moving", "resting", "occupied"]
    assert data["population"] is lgca.n_t and data["n"] is lgca.n_t
    np.testing.assert_array_equal(data["density"], lgca.dens_t)
    assert data.steps("nodes").tolist() == [0, 2, 4, 6]
    assert len(data["nodes"]) == 4
    assert data.steps("density").tolist() == list(range(7))
    np.testing.assert_array_equal(data["moving"] + data["resting"], data["density"])
    assert data.steps("occupied").tolist() == [0, 6]
    assert data["occupied"][0] == int((data["density"][0] > 0).sum())
    assert "n" in data and "flux" not in data


def test_a_missing_name_says_what_was_recorded():
    result = run_model(_spec(True, False, [PopulationRecorder()]), showprogress=False)
    with pytest.raises(KeyError, match="'density' was not recorded; this run recorded population"):
        result.data["density"]
    empty = run_model(_spec(True, False, []), showprogress=False)
    assert len(empty.data) == 0
    with pytest.raises(KeyError, match="recorded nothing"):
        empty.data.steps("n")


def test_a_later_run_does_not_change_the_data_of_an_earlier_one():
    model = build_model(_spec(True, False, [PopulationRecorder(), DensityRecorder()], steps=3))
    first = model.run(showprogress=False)
    before = first.data["density"].copy()
    second = model.run(showprogress=False)  # continues the dynamics from the end of the first run
    np.testing.assert_array_equal(first.data["density"], before)
    np.testing.assert_array_equal(second.data["density"][0], before[-1])


@pytest.mark.parametrize("options", [{}, {"ve": False}, {"ib": True}, {"n_species": 2}])
def test_get_lgca_models_give_the_data_of_their_last_run(options):
    import warnings

    from lgca import get_lgca

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgca = get_lgca(geometry="square", dims=(8, 8), density=0.3, seed=1, **options)
    with pytest.raises(KeyError, match="timeevo"):
        lgca.data["density"]
    lgca.timeevo(timesteps=4, record=True, recordN=True, showprogress=False)
    assert list(lgca.data) == ["population", "density", "nodes"]
    assert lgca.data["n"] is lgca.n_t and lgca.data.steps("nodes").tolist() == [0, 1, 2, 3, 4]
    lgca.timeevo(timesteps=2, showprogress=False)  # only the density: the nodes of the first run are gone
    assert list(lgca.data) == ["density"] and len(lgca.data["density"]) == 3
    with pytest.raises(KeyError, match="record=True"):
        lgca.data["nodes"]
