from dataclasses import replace

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    model_spec_from_json,
    model_spec_to_dict,
    model_spec_to_json,
    run_model,
)
from lgca.pipeline import BirthDeathSpec, InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
from lgca.plugins import ConservationLaw, InteractionOperator, PluginInfo, describe_plugin, list_plugins
from lgca.simulation import DensityRecorder, FamilyPopulationRecorder, NodeRecorder
from lgca.simulation import Observer


@pytest.mark.parametrize("geometry,dims", [("lin", [3]), ("square", [3, 4]),
    ("hex", [3, 4]), ("cubic", [3, 4, 5]), ("moore", [3, 4, 5])])
def test_plain_json_and_yaml_dimension_arrays(geometry, dims):
    import json
    from lgca.model import model_spec_from_yaml

    data = {"schema_version": 1, "model": {"space": {"geometry": geometry, "dims": dims},
            "state": {"density": 0}, "time": {"steps": 0}}}
    for spec in (model_spec_from_json(json.dumps(data)),
                 model_spec_from_yaml(f"schema_version: 1\nmodel:\n  space:\n    geometry: {geometry}\n    dims: {dims}\n  state:\n    density: 0\n  time:\n    steps: 0\n"),
                 ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims), state=StateSpec(density=0))):
        assert build_model(spec).lgca.dims == tuple(dims)


@pytest.mark.parametrize("dims", [[3], [3, 4, 5], [], [0, 3], [True, 3], [1.5, 3]])
def test_dimension_errors_identify_model_field(dims):
    with pytest.raises(ValueError, match="model.space.dims"):
        build_model(ModelSpec(space=SpaceSpec(geometry="square", dims=dims)))


def test_legacy_stepping_continues_compiled_dynamics_and_rng():
    spec = _square_spec(operators=[{"name": "random_walk"}], timesteps=4)
    spec = replace(spec, dynamics=replace(spec.dynamics, propagation=False))
    expected = run_model(spec, showprogress=False).lgca.nodes.copy()
    compiled = build_model(replace(spec, time=replace(spec.time, steps=2)))
    compiled.run(showprogress=False)
    compiled.lgca.timestep()
    compiled.lgca.timeevo(1, showprogress=False)
    np.testing.assert_array_equal(compiled.lgca.nodes, expected)
    assert compiled._step == 4
    assert compiled.lgca.enable_propagation is False


def test_live_animation_uses_compiled_death_operator():
    import matplotlib.pyplot as plt

    compiled = build_model(_square_spec(operators=[{
        "name": "birth_death", "parameters": {"death_rate": 1}
    }], timesteps=0))
    animation = compiled.lgca.live_animate_flux()
    animation._func(0)
    assert compiled.lgca.total_population() == 0
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_scalar_metric_serialization_preserves_semantics():
    from lgca.simulation import ScalarTimeSeriesRecorder, _total_population

    spec = ModelSpec(analysis=AnalysisSpec(observers=[ScalarTimeSeriesRecorder()]))
    restored = model_spec_from_json(model_spec_to_json(spec))
    assert restored.analysis.observers[0].metrics["population"] is _total_population
    custom = ScalarTimeSeriesRecorder(metrics={"population": lambda lgca: 123})
    with pytest.raises(TypeError, match="default population metric"):
        model_spec_to_dict(replace(spec, analysis=AnalysisSpec(observers=[custom])))


@pytest.mark.parametrize("n_species", [1, 2])
def test_ve_growth_capacity_configuration_and_metadata(n_species):
    shape = (1, 2) if n_species == 1 else (1, 2, 2)
    nodes = np.zeros(shape, dtype=bool)
    nodes[..., 0] = True
    capacity = n_species + 1
    spec = ModelSpec(space=SpaceSpec(geometry="lin"),
        state=StateSpec(nodes=nodes, n_species=n_species, capacity=capacity),
        time=TimeSpec(steps=1, seed=122),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 1, "crowding": False}}], propagation=False))
    result = run_model(model_spec_from_json(model_spec_to_json(spec)), showprogress=False)
    assert result.lgca.total_population() == capacity  # without crowding, capacity is a hard limit
    assert result.metadata["capacity"] == capacity
    assert result.metadata["channel_capacity"] == 2
    assert result.metadata["growth_capacities"][0]["capacity"] == capacity
    with pytest.raises(ValueError, match="unknown plugin parameter"):
        build_model(replace(spec, dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 1, "capacity": capacity}}])))


@pytest.mark.parametrize("field,value", [("beta", float("inf")), ("species", -1),
                                        ("parameters", {"betta": 3})])
def test_serialized_composed_term_contracts(field, value):
    import json
    from lgca.model import model_spec_from_yaml

    term = {"name": "persistent_walk", field: value}
    data = {"schema_version": 1, "model": {"space": {"geometry": "lin", "dims": 2},
            "dynamics": {"operators": [{"type": "reorientation", "terms": [term]}]}}}
    for loader in (model_spec_from_json, model_spec_from_yaml):
        with pytest.raises(ValueError, match=field):
            build_model(loader(json.dumps(data)))


@pytest.mark.parametrize("name", ["polar_alignment", "aggregation", "nematic_alignment"])
def test_bounded_candidate_batches_preserve_seeded_trajectory(name, monkeypatch):
    import lgca.pipeline as pipeline

    spec = _square_spec(operators=[{"name": name}], timesteps=3, seed=127)
    expected = run_model(spec, showprogress=False).lgca.nodes_t.copy()
    monkeypatch.setattr(pipeline, "_MAX_CANDIDATE_BATCH_BYTES", 480)
    actual = run_model(spec, showprogress=False).lgca.nodes_t
    np.testing.assert_array_equal(actual, expected)
    batches = list(pipeline._candidate_batches(np.ones((4, 4), dtype=bool), 10))
    assert max(len(batch[0]) for batch in batches) == 1


def test_multiline_yaml_is_not_probed_as_a_filesystem_path(monkeypatch):
    from pathlib import Path
    from lgca.model import model_spec_from_yaml, model_spec_to_yaml

    text = model_spec_to_yaml(_square_spec())

    def reject_path_probe(path):
        raise AssertionError("Inline model text must not be passed to Path.exists")

    monkeypatch.setattr(Path, "exists", reject_path_probe)
    restored = model_spec_from_yaml(text)
    assert restored.space.dims == (4, 5)


def _square_spec(*, operators=(), timesteps=3, seed=17):
    return ModelSpec(
        description=Description(title="registry driven square LGCA"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1),
        time=TimeSpec(steps=timesteps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation="default"),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )


def _legacy_square(*, interaction, timesteps=3, seed=17):
    lgca = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        interaction=interaction,
        bc="periodic",
        seed=seed,
    )
    lgca.timeevo(
        timesteps=timesteps,
        record=True,
        recorddens=True,
        showprogress=False,
    )
    return lgca


def test_registered_operator_wire_format_contains_only_stable_data_tags():
    spec = replace(
        _square_spec(),
        dynamics=InteractionPipelineSpec(
            operators=[
                BirthDeathSpec(name="birth_death", parameters={"birth_rate": 0.1}),
                ReorientationSpec(
                    terms=[ReorientationTermSpec(name="uniform", beta=0.5)]
                ),
            ]
        ),
    )

    operators = model_spec_to_dict(spec)["model"]["dynamics"]["operators"]

    assert operators[0] == {
        "name": "birth_death",
        "parameters": {"birth_rate": 0.1},
    }
    assert operators[1]["type"] == "reorientation"
    assert "BirthDeathSpec" not in model_spec_to_json(spec)
    assert "ReorientationSpec" not in model_spec_to_json(spec)


def test_initializer_declaration_round_trips_as_pure_data():
    state = replace(
        _square_spec().state,
        density=None,
        initializer={
            "name": "region",
            "parameters": {"placement": "center", "extent": [2, 3], "density": 0.5},
        },
    )
    spec = replace(_square_spec(), state=state)

    loaded = model_spec_from_json(model_spec_to_json(spec))

    assert loaded.state.initializer == state.initializer


def test_python_only_operator_and_observer_have_actionable_portability_errors():
    class PythonOnlyOperator:
        name = "external.unregistered"
        parameters = {}

    class PythonOnlyObserver(Observer):
        pass

    operator_spec = replace(
        _square_spec(),
        dynamics=InteractionPipelineSpec(operators=[PythonOnlyOperator()]),
    )
    observer_spec = replace(
        _square_spec(), analysis=AnalysisSpec(observers=[PythonOnlyObserver()])
    )

    with pytest.raises(TypeError, match=r"not portable.*registered plugin"):
        model_spec_to_dict(operator_spec)
    with pytest.raises(TypeError, match=r"not portable.*built-in observer"):
        model_spec_to_dict(observer_spec)


def test_model_spec_runs_propagation_as_separate_deterministic_phase():
    result = run_model(_square_spec(operators=()), showprogress=False)
    legacy = _legacy_square(interaction="only_propagation")

    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    assert result.metadata["operator_names"] == []
    assert result.metadata["geometry"] == "square"
    assert "Propagation" in result.pipeline.describe_schedule()


def test_classical_random_walk_plugin_is_native_and_matches_legacy():
    spec = _square_spec(operators=[{"name": "random_walk"}], timesteps=2, seed=43)
    result = run_model(spec, showprogress=False)
    legacy = _legacy_square(interaction="random_walk", timesteps=2, seed=43)
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_excitable_medium_plugin_is_native_and_matches_legacy():
    parameters = {"beta": 0.04, "alpha": 1.2, "N": 3}
    spec = ModelSpec(
        description=Description(title="native equivalence classical.excitable_medium"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=2),
        time=TimeSpec(steps=2, seed=68),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "excitable_medium", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=2,
        interaction="excitable_medium",
        bc="periodic",
        seed=68,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_ib_random_walk_plugin_is_native_and_matches_legacy():
    spec = ModelSpec(
        description=Description(title="native equivalence ib.random_walk"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1, identity_based=True),
        time=TimeSpec(steps=2, seed=53),
        dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        ib=True,
        interaction="random_walk",
        bc="periodic",
        seed=53,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_pipeline_timing_is_aggregated_in_constant_space():
    spec = _square_spec(
        operators=[{"name": "random_walk"}], timesteps=100, seed=71
    )

    result = run_model(spec, showprogress=False)
    timings = result.metadata["runtime"]["operator_timings"]

    assert len(timings) == 2
    assert {entry["name"] for entry in timings} == {"random_walk", "Propagation"}
    assert all(entry["count"] == 100 for entry in timings)


def test_pipeline_timing_trace_is_explicit_and_bounded():
    spec = _square_spec(
        operators=[{"name": "random_walk"}], timesteps=10, seed=72
    )
    spec = replace(spec, time=replace(spec.time, timing_trace=3))

    result = run_model(spec, showprogress=False)
    trace = result.metadata["runtime"]["timing_trace"]

    assert len(trace) == 3
    assert [entry["name"] for entry in trace] == [
        "random_walk",
        "Propagation",
        "random_walk",
    ]
    assert [entry["step"] for entry in trace] == [1, 1, 2]


@pytest.mark.parametrize("plugin_name", ["go_and_grow_mutations", "go_or_grow_glioblastoma", "evo_steric"])
def test_models_whose_mutants_found_families_are_recorded_as_such(plugin_name):
    from types import SimpleNamespace

    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(4, 5)),
        state=StateSpec(density=0.35, restchannels=1, identity_based=True,
                        volume_exclusion=plugin_name == "go_and_grow_mutations",
                        capacity=None if plugin_name == "go_and_grow_mutations" else 8),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name}])))
    runner = SimpleNamespace(context=model)
    assert FamilyPopulationRecorder._is_mutating_family_interaction(model.lgca, runner)


def test_build_model_exposes_compiled_pipeline_schedule():
    compiled = build_model(_square_spec(operators=[{"name": "excitable_medium"}]))

    assert compiled.metadata["operator_names"] == ["excitable_medium"]
    assert compiled.metadata["observer_names"] == ["NodeRecorder", "DensityRecorder"]
    assert compiled.metadata["reorientation_term_names"] == []
    schedule = compiled.pipeline.describe_schedule()
    assert "excitable_medium" in schedule
    assert "birth_death" in schedule
    assert "Propagation" in schedule


class _FieldSetupProbe(InteractionOperator):
    def __init__(self):
        super().__init__(
            PluginInfo(
                name="field_setup_probe",
                operator_kind="reorientation",
                backend_families=("classical",),
                conservation_law=ConservationLaw(True, True, True),
            )
        )
        self.signal_shape = None

    def setup(self, context) -> None:
        if not hasattr(context.lgca, "signal"):
            raise AssertionError("state.fields.signal was not attached before operator setup")
        self.signal_shape = context.lgca.signal.shape

    def apply(self, context, step: int) -> None:
        pass


def test_state_fields_are_attached_before_operator_setup():
    probe = _FieldSetupProbe()
    spec = ModelSpec(
        description=Description(title="field setup order"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.25, fields={"signal": np.ones((4, 5))}),
        time=TimeSpec(steps=0, seed=41),
        dynamics=InteractionPipelineSpec(operators=[probe], propagation=False),
    )

    compiled = build_model(spec)

    assert probe.signal_shape == compiled.lgca.nodes.shape[: len(compiled.lgca.dims)]


def test_vector_state_fields_are_padded_when_attached_to_lgca():
    director = np.zeros((4, 5, 2), dtype=float)
    director[..., 0] = 1.0
    spec = ModelSpec(
        description=Description(title="vector field attachment"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.25, fields={"director": director}),
        time=TimeSpec(steps=0, seed=42),
        dynamics=InteractionPipelineSpec(operators=(), propagation=False),
    )

    compiled = build_model(spec)

    assert compiled.lgca.director.shape == compiled.lgca.nodes.shape[: len(compiled.lgca.dims)] + (2,)
    np.testing.assert_allclose(compiled.lgca.director[compiled.lgca.nonborder], director)


def test_unknown_operator_reports_modelspec_path():
    with pytest.raises(ValueError, match=r"dynamics\.operators\[0\].*missing"):
        build_model(_square_spec(operators=[{"name": "missing"}]))


def test_invalid_time_spec_reports_modelspec_path():
    with pytest.raises(ValueError, match=r"time\.steps"):
        build_model(_square_spec(timesteps=-1))


def _state_for_plugin(plugin_name):
    family = plugin_name.split(".", 1)[0]
    if "." not in plugin_name and set(describe_plugin(plugin_name).backend_families) <= {"ib", "nove_ib"}:
        family = "nove_ib"  # e.g. the research models, stacks of rules for identity-based models
    if plugin_name == "phenotype_switch":
        return StateSpec(density=0.35, restchannels=1, n_species=2)
    if plugin_name == "trait_switch":
        return StateSpec(density=0.35, restchannels=1, identity_based=True, traits={"alignment": 1.0})
    if family == "go_or_grow":  # migrating cells in velocity channels, resting cells in rest channels
        nodes = np.zeros((4, 5, 2, 5), dtype=bool)
        nodes[::2, :, 0, :4] = nodes[1::2, :, 1, 4] = True
        return StateSpec(nodes=nodes, restchannels=1, n_species=2)
    if family == "ib":
        return StateSpec(density=0.35, restchannels=2, identity_based=True)
    if family == "nove":
        restchannels = 1 if plugin_name.endswith(("go_or_grow", "go_or_rest")) else 0
        return StateSpec(density=0.35, restchannels=restchannels, volume_exclusion=False)
    if family == "nove_ib":
        return StateSpec(
            density=0.35,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
        )
    if family == "multispecies":
        if plugin_name.endswith("excitable_medium_ms"):
            return StateSpec(density=0.35, restchannels=1, n_species=2)
        return StateSpec(density=0.35, restchannels=1, volume_exclusion=False, n_species=2)
    if plugin_name in ("chemotaxis", "contact_guidance", "directed_motion", "pde"):  # read a named field
        return StateSpec(density=0.35, restchannels=2, fields={
            "signal": np.arange(20.0).reshape(4, 5), "director": np.ones((4, 5, 2))})
    return StateSpec(density=0.35, restchannels=2)


_PARAMETERS_FOR_PLUGIN = {"chemotaxis": {"field": "signal"}, "directed_motion": {"field": "director"},
                          "pde": {"field": "signal", "diffusion": 1.0},
                          "trait_switch": {"switch": {"alignment": 0.1}},
                          "phenotype_switch": {"rates": [[0, 0.1], [0.2, 0]]}}


@pytest.mark.parametrize(
    "plugin_name",
    [plugin.name for plugin in list_plugins(kind="interaction") if not plugin.name.startswith("legacy.")],
)
def test_all_registered_interactions_run_one_step_through_modelspec(plugin_name):
    spec = ModelSpec(
        description=Description(title=f"run {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=_state_for_plugin(plugin_name),
        time=TimeSpec(steps=1, seed=21),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": plugin_name, "parameters": _PARAMETERS_FOR_PLUGIN.get(plugin_name, {})}]),
    )

    result = run_model(spec, showprogress=False)

    assert result.metadata["operator_names"] == [plugin_name]
    assert result.lgca.cell_density.shape == result.lgca.nodes.shape[: len(result.lgca.dims)]


@pytest.mark.parametrize("operator", ["nove_ib.birth", "nove_ib.birthdeath", "nove_ib.birthdeath_cancerdfe",
                                    "nove_ib.go_or_grow", "nove_ib.go_or_grow_kappa",
                                    "nove_ib.go_or_grow_kappa_chemo", "nove_ib.go_or_grow_glioblastoma",
                                    "nove_ib.evo_steric"])
@pytest.mark.parametrize("override", [None, 3, 7])
@pytest.mark.filterwarnings("ignore:The interaction name:FutureWarning")
def test_nove_identity_capacity_precedence(operator, override):
    parameters = {} if override is None else {"capacity": override}
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=3),
                     state=StateSpec(volume_exclusion=False, identity_based=True,
                                     restchannels=1, density=1, capacity=3),
                     time=TimeSpec(steps=1, seed=138),
                     dynamics=InteractionPipelineSpec(operators=[{"name": operator, "parameters": parameters}]))
    if override == 7:
        with pytest.raises(ValueError, match="conflicts"):
            build_model(spec)
    else:
        compiled = build_model(spec)
        assert compiled.lgca.capacity == 3
        compiled.run(False)
