"""CompiledModel.reconfigure changes the dynamics of a running model, atomically and on the record."""

import pytest

from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    model_spec_to_dict,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import PopulationRecorder


def _spec(steps=4, observers=(), **state):
    state.setdefault("density", 0.3)
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(10, 8)), state=StateSpec(restchannels=1, **state),
        time=TimeSpec(steps=steps, seed=5),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 0.1, "death_rate": 0.05}},
            {"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=tuple(observers)),
    )


def _state(model):
    return model.lgca.nodes.tolist(), model.lgca.rng.bit_generator.state, model._step


@pytest.mark.parametrize("state", [{}, {"identity_based": True}, {"volume_exclusion": False, "capacity": 4}])
def test_a_reconfigured_model_runs_as_a_replay_of_its_record(state):
    live = build_model(_spec(**state))
    live.run(showprogress=False)
    live.reconfigure({"birth_rate": 0.4})
    live.run(showprogress=False)

    record = live.metadata["reconfigurations"]
    assert record == [{"from_step": 5, "changes": {
        "dynamics.operators[0].parameters.birth_rate": {"old": 0.1, "new": 0.4}}}]
    assert live.initial_spec.dynamics.operators[0]["parameters"]["birth_rate"] == 0.1
    assert live.spec.dynamics.operators[0]["parameters"]["birth_rate"] == 0.4
    replay = build_model(live.initial_spec)
    for _ in range(record[0]["from_step"] - 1):
        replay.step()
    replay.reconfigure({path: change["new"] for path, change in record[0]["changes"].items()})
    for _ in range(4):
        replay.step()
    assert _state(replay) == _state(live)


def test_earlier_results_keep_their_spec_and_metadata():
    model = build_model(_spec())
    first = model.run(showprogress=False)
    model.reconfigure({"birth_rate": 0.4})
    second = model.run(showprogress=False)

    assert first.spec.dynamics.operators[0]["parameters"]["birth_rate"] == 0.1
    assert first.context.spec is first.spec and first.pipeline is not second.pipeline
    assert "reconfigurations" not in first.metadata and len(second.metadata["reconfigurations"]) == 1
    assert second.pipeline.operators[0].parameters["birth_rate"] == 0.4


@pytest.mark.parametrize("changes, error", [
    ({"birth_rate": 2.0}, "probabilit"),  # found only when the model steps: tried on a copy first
    ({"dynamics.operators[0].parameters.birth_rate": "fast"}, "names a cell trait"),  # at build
    ({"time.steps": 10}, "not part of the dynamics"),
    ({"space.dims": (4, 4)}, "not part of the dynamics"),
    ({}, "map paths"),
])
def test_a_rejected_change_leaves_the_model_as_it_was(changes, error):
    model, untouched = build_model(_spec()), build_model(_spec())
    model.step(), untouched.step()
    spec, pipeline, metadata = model.spec, model.pipeline, model_spec_to_dict(model.spec)

    with pytest.raises((ValueError, TypeError, KeyError), match=error):
        model.reconfigure(changes)

    assert model.spec is spec and model.pipeline is pipeline and model_spec_to_dict(model.spec) == metadata
    assert "reconfigurations" not in model.metadata and model._initial_spec is None
    model.step(), untouched.step()
    assert _state(model) == _state(untouched)


def test_the_model_owns_the_new_values():
    switch = {"rates": [[0.0, 0.5], [0.5, 0.0]]}
    two = build_model(ModelSpec(state=StateSpec(density=0.3, n_species=2), time=TimeSpec(seed=1),
                                dynamics=InteractionPipelineSpec(operators=[
                                    {"name": "phenotype_switch", "parameters": {"rates": [[0, 0.1], [0.1, 0]]}}])))
    two.reconfigure({"dynamics.operators[0].parameters": switch})
    switch["rates"][0][1] = 0.9  # the caller's dict changes afterwards

    assert two.spec.dynamics.operators[0]["parameters"]["rates"][0][1] == 0.5
    change = two.metadata["reconfigurations"][0]["changes"]["dynamics.operators[0].parameters"]
    assert change["new"]["rates"][0][1] == 0.5
    two.step()


def test_metadata_describe_the_new_operators():
    model = build_model(_spec())
    model.reconfigure({"dynamics.operators": [{"name": "random_walk"}]})

    assert model.metadata["operator_names"] == ["random_walk"]
    assert "birth_death" not in model.metadata["schedule"] and model.metadata["growth_capacities"] == []
    result = model.run(showprogress=False)
    assert result.metadata["operator_names"] == ["random_walk"]
    cells = model.lgca.cell_density[model.lgca.nonborder].sum()
    model.step()
    assert model.lgca.cell_density[model.lgca.nonborder].sum() == cells  # a random walk keeps the cells


def test_recorders_of_the_spec_still_record_after_a_change():
    model = build_model(_spec(observers=[PopulationRecorder()]))
    model.reconfigure({"birth_rate": 0.3})
    result = model.run(showprogress=False)
    assert len(result.data["population"]) == 5
