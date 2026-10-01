from dataclasses import replace
from functools import partial

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

    probe = compiled.pipeline.operators[0]  # the model's own copy of the operator object
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


@pytest.mark.parametrize("shape", [(3,), (5,)])  # the lattice's nodes, and with its border nodes
def test_the_model_keeps_its_own_copy_of_a_state_field(shape):
    signal = np.ones(shape)
    compiled = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=3), time=TimeSpec(steps=0, seed=1),
                                     state=StateSpec(fields={"signal": signal})))
    signal[:] = -5.0

    np.testing.assert_array_equal(compiled.lgca.signal, np.ones(5))


def _edit_after_build(case):
    """A model, and an edit of it in place: of the caller's dicts, lists, arrays or operator objects."""
    from lgca.plugins import create_plugin

    growth = {"birth_rate": 0.5, "death_rate": 0.0}
    operators = [{"name": "birth_death", "parameters": growth}]
    state = {"density": 1.0, "restchannels": 1, "volume_exclusion": False, "capacity": 4}
    if case == "parameters":
        edit = partial(growth.update, death_rate=1.0)
    elif case == "operator object":
        operators = [create_plugin("birth_death", growth)]
        edit = partial(operators[0].parameters.update, death_rate=1.0)
    elif case == "nested rates":
        rates = [[0.0, 0.5], [0.0, 0.0]]
        operators = [{"name": "phenotype_switch", "parameters": {"rates": rates}}]
        state["n_species"] = 2

        def edit():
            rates[0][1], rates[1][0] = 0.0, 1.0
    elif case == "operator list":
        edit = partial(operators.append, {"name": "birth_death", "parameters": {"death_rate": 1.0}})
    elif case == "nodes":
        nodes = np.zeros((30, 3), dtype=int)
        nodes[:10] = 1
        state = {"nodes": nodes, "restchannels": 1, "volume_exclusion": False, "capacity": 4}
        edit = partial(nodes.fill, 0)
    elif case == "label lists":
        nodes = np.empty((30, 3), dtype=object)  # identity-based without volume exclusion: lists of labels
        for index in np.ndindex(nodes.shape):
            nodes[index] = [3 * index[0] + index[1] + 1] if index[0] < 10 else []
        state = {"nodes": nodes, "restchannels": 1, "identity_based": True, "volume_exclusion": False,
                 "capacity": 4}
        edit = partial(nodes[3, 0].extend, [98, 99])
    elif case == "traits":
        nodes = np.zeros((30, 3), dtype=int)
        nodes[:10] = np.arange(1, 31).reshape(10, 3)  # labels
        kappa = np.linspace(0.0, 1.0, 30)
        state = {"nodes": nodes, "restchannels": 1, "identity_based": True, "traits": {"kappa": kappa}}
        edit = partial(kappa.fill, 5.0)
    else:
        initializer = {"name": "region", "parameters": {"extent": 4, "density": 2}}
        state = {"initializer": initializer, "restchannels": 1, "volume_exclusion": False, "capacity": 4}
        edit = partial(initializer["parameters"].update, extent=20)
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=30), state=StateSpec(**state),
                     time=TimeSpec(steps=10, seed=3), dynamics=InteractionPipelineSpec(operators=operators))
    return spec, edit


@pytest.mark.parametrize("case", ["parameters", "operator object", "nested rates", "operator list", "nodes",
                                  "label lists", "traits", "initializer"])
def test_changes_of_the_spec_after_the_build_reach_neither_the_model_nor_its_spec(case):
    from copy import deepcopy

    spec, edit = _edit_after_build(case)
    reference = build_model(deepcopy(spec))
    model = build_model(spec)
    built = model_spec_to_json(model.spec)  # text: a dict may hold the lists of the spec
    edit()
    assert model_spec_to_json(model.spec) == built
    for _ in range(10):
        model.step()
        reference.step()
    np.testing.assert_array_equal(model.lgca.nodes, reference.lgca.nodes)


@pytest.mark.parametrize("state", [{}, {"volume_exclusion": False}, {"identity_based": True},
                                   {"identity_based": True, "volume_exclusion": False}],
                         ids=["classical", "nove", "ib", "nove_ib"])
@pytest.mark.parametrize("stacked", [False, True])
def test_models_built_from_one_operator_object_do_not_affect_each_other(state, stacked):
    from lgca import stack
    from lgca.plugins import create_plugin
    from lgca.rules import StackOperator

    # validate stores the model's capacity on the operator: a capacity-2 model ran with the capacity of a
    # capacity-50 model built from the same object later
    def template():
        growth = create_plugin("birth_death", {"birth_rate": 0.5})
        if not stacked:
            return growth, growth

        @stack(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), register=False,
               name="model_spec_test.growth_stack")
        def growth_stack(state):
            """Growth by an operator object."""
            return [growth]

        return StackOperator(growth_stack), growth  # a stack object whose rules include an operator object

    def build(capacity, operator):
        return build_model(ModelSpec(
            space=SpaceSpec(geometry="lin", dims=30), time=TimeSpec(steps=30, seed=7),
            state=StateSpec(density=1.0, restchannels=1, capacity=capacity, **state),
            dynamics=InteractionPipelineSpec(operators=[operator], propagation=False)))

    alone = build(2, template()[0])
    operator, growth = template()
    first = build(2, operator)
    other = build(50, operator)
    for _ in range(30):
        alone.step()
        first.step()
    np.testing.assert_array_equal(first.lgca.nodes, alone.lgca.nodes)
    running = first.pipeline.operators[0]
    assert running is not operator and first.spec.dynamics.operators[0] is not operator
    assert (running.operators[0] if stacked else running).capacity == 2
    assert (other.pipeline.operators[0].operators[0] if stacked else other.pipeline.operators[0]).capacity == 50
    # the object in the spec is a template: it never runs
    assert growth.capacity is None and (not stacked or operator.operators == [])


def test_operator_objects_that_cannot_be_copied_are_explained():
    import threading

    from lgca import stack
    from lgca.plugins import create_plugin
    from lgca.rules import StackOperator

    locked = create_plugin("birth_death", {"birth_rate": 0.5})
    locked.lock = threading.Lock()
    with pytest.raises(ValueError, match=r"dynamics\.operators\[1\] \('birth_death'\) cannot be copied.*"
                                         r"__deepcopy__"):
        build_model(_square_spec(operators=[{"name": "random_walk"}, locked]))

    @stack(kind="birth_death", families="classical", register=False, name="model_spec_test.locked_stack")
    def locked_stack(state):
        """A stack of an operator object that cannot be copied."""
        return [{"name": "random_walk"}, locked]

    with pytest.raises(ValueError, match=r"model_spec_test\.locked_stack: operators\[1\] \('birth_death'\) cannot "
                                         r"be copied.*__deepcopy__"):
        build_model(_square_spec(operators=[StackOperator(locked_stack)]))
    spec = _square_spec(operators=[{"name": "random_walk"}])
    spec = replace(spec, state=replace(spec.state, parameters={"handle": threading.Lock()}))
    with pytest.raises(ValueError, match=r"model\.state\.parameters\.handle cannot be copied"):
        build_model(spec)

    # copies also fail with other errors, e.g. of a multiprocessing.Lock or of a wrapper that forwards attributes
    class Inherited:
        def __deepcopy__(self, memo):
            raise RuntimeError("Lock objects should only be shared between processes through inheritance")

    class Forwarding(type(locked)):
        def __init__(self, helper):
            self.__dict__.update(create_plugin("birth_death", {"birth_rate": 0.5}).__dict__)
            self.helper = helper

        def __getattr__(self, name):  # a copy is made without its attributes: this recurses
            return getattr(self.helper, name)

    unshared = create_plugin("birth_death", {"birth_rate": 0.5})
    unshared.handle = Inherited()
    with pytest.raises(ValueError, match=r"dynamics\.operators\[0\] \('birth_death'\) cannot be copied \(RuntimeError: "
                                         r"Lock objects.*__deepcopy__"):
        build_model(_square_spec(operators=[unshared]))
    with pytest.raises(ValueError, match=r"dynamics\.operators\[0\] \('birth_death'\) cannot be copied "
                                         r"\(RecursionError.*__getattr__.*__deepcopy__"):
        build_model(_square_spec(operators=[Forwarding(helper=object())]))
    spec = replace(spec, state=replace(spec.state, parameters={"handle": Inherited()}))
    with pytest.raises(ValueError, match=r"model\.state\.parameters\.handle cannot be copied \(RuntimeError"):
        build_model(spec)


class _Deadly(InteractionOperator):
    """Sets the death rate of another operator to 1 at every step: it kills the cells only if it and that
    operator run as a pair."""

    def __init__(self, target):
        super().__init__(PluginInfo(name="model_spec_test.deadly", operator_kind="birth_death",
                                    backend_families=("classical",)))
        self.target = target

    def apply(self, context, step):
        self.target.parameters["death_rate"] = 1.0


@pytest.mark.parametrize("stacked", [False, True])
def test_operator_objects_that_refer_to_each_other_still_do_in_the_model(stacked):
    from lgca import stack
    from lgca.plugins import create_plugin
    from lgca.rules import StackOperator

    def pair():
        growth = create_plugin("birth_death", {"birth_rate": 0.0, "death_rate": 0.0})
        return [_Deadly(growth), growth]

    @stack(kind="birth_death", families="classical", register=False, name="model_spec_test.deadly_pair")
    def deadly_pair(state):
        """An operator and the one it drives, made for every model."""
        return pair()

    operators = [StackOperator(deadly_pair)] if stacked else pair()
    spec = replace(_square_spec(operators=operators), state=StateSpec(density=1.0, restchannels=1))
    model = build_model(spec)
    running = model.pipeline.operators[0].operators if stacked else model.pipeline.operators
    assert running[0].target is running[1]  # the copies of the pair: one drives the other
    if not stacked:
        assert operators[0].target is operators[1] and running[1] is not operators[1]
        assert model.spec.dynamics.operators[0].target is model.spec.dynamics.operators[1]
    model.step()
    assert model.lgca.cell_density[model.lgca.nonborder].sum() == 0  # the running growth killed every cell
    if not stacked:
        assert operators[1].parameters["death_rate"] == 0.0  # the template never ran


def test_operator_objects_of_registered_rules_count_as_their_mappings():
    from lgca.model import model_spec_to_dict
    from lgca.plugins import create_plugin
    from lgca.study import vary

    operators = (create_plugin("random_walk"), create_plugin("birth_death", {"birth_rate": 0.5}))
    spec = _square_spec(operators=operators)
    assert spec.dynamics.operators == ({"name": "random_walk", "parameters": {}},
                                       {"name": "birth_death", "parameters": {"birth_rate": 0.5}})
    mapped = _square_spec(operators=[{"name": "random_walk"}, {"name": "birth_death",
                                                                "parameters": {"birth_rate": 0.5}}])
    np.testing.assert_array_equal(run_model(spec, showprogress=False).lgca.nodes,
                                  run_model(mapped, showprogress=False).lgca.nodes)
    # vary changes it, also by a short name, and a model file saves the same
    assert vary(spec, {"birth_rate": 0.2}).dynamics.operators[1]["parameters"] == {"birth_rate": 0.2}
    assert vary(spec, {"dynamics.operators[0]": operators[1]}).dynamics.operators[0]["name"] == "birth_death"
    assert model_spec_to_dict(spec)["model"]["dynamics"]["operators"][1] == spec.dynamics.operators[1]


def test_operator_objects_that_hold_more_than_their_name_and_parameters_stay_objects():
    from lgca.plugins import create_plugin
    from lgca.study import vary

    marked = create_plugin("birth_death", {"birth_rate": 0.5})
    marked.note = "kept with the operator"
    changed = create_plugin("birth_death", {"birth_rate": 0.5})
    changed.capacity = 3
    other = create_plugin("birth_death", {"death_rate": 0.1})
    helped = create_plugin("birth_death", {"birth_rate": 0.5, "helper": other})  # its parameters refer to one
    ran = build_model(_square_spec(operators=[create_plugin("random_walk")])).pipeline.operators[0]
    with pytest.warns(FutureWarning, match="deprecated"):  # a deprecated name would warn at every build
        legacy = create_plugin("classical.random_walk")
    given = [marked, changed, helped, other, ran, legacy]
    spec = _square_spec(operators=given)
    assert all(entry is operator for entry, operator in zip(spec.dynamics.operators, given))
    with pytest.raises(KeyError, match=r"operators\[0\] is an operator object \('birth_death'\), which vary cannot "
                                       r"set 'birth_rate' in.*\{'name': 'birth_death', 'parameters'.*attributes set "
                                       r"on it"):
        vary(spec, {"dynamics.operators[0].birth_rate": 0.2})
    class Events(Observer):
        """Receives the steps from an operator."""

        def __init__(self):
            super().__init__()
            self.steps = []

        def setup(self, lgca, runner):
            self.steps = []

        def observe(self, lgca, step, runner):
            pass

    class Reporting(InteractionOperator):
        """Changes no cells; reports every step to an observer and records it in a list of its own."""

        def __init__(self, events, recorded):
            super().__init__(PluginInfo(name="model_spec_test.reporting", operator_kind="birth_death",
                                        backend_families=("classical",)))
            self.events, self.recorded = events, recorded

        def apply(self, context, step):
            self.events.steps.append(step)
            self.recorded.append(step)

    events, recorded = Events(), []
    spec = replace(_square_spec(operators=[Reporting(events, recorded)]), analysis=AnalysisSpec(observers=[events]))
    result = run_model(spec, showprogress=False)
    running = result.pipeline.operators[0]
    assert running.events is events and events.steps == [1, 2, 3]  # the observer is the caller's
    assert recorded == [] and running.recorded == [1, 2, 3]  # the rest of the operator is copied with it


def test_the_model_does_not_change_its_spec_when_its_label_lists_change():
    nodes = np.empty((30, 3), dtype=object)  # identity-based without volume exclusion: lists of labels
    for index in np.ndindex(nodes.shape):
        nodes[index] = [3 * index[0] + index[1] + 1] if index[0] < 10 else []
    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=30), time=TimeSpec(steps=5, seed=1),
                                  state=StateSpec(nodes=nodes, restchannels=1, identity_based=True,
                                                  volume_exclusion=False, capacity=4),
                                  dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}])))
    built = model_spec_to_json(model.spec)
    model.lgca.nodes[model.lgca.nonborder][3, 0].append(99)  # e.g. to place a cell before the run
    assert 99 in model.lgca.nodes[model.lgca.nonborder][3, 0]  # the lattice's list, which is not model.spec's
    assert model_spec_to_json(model.spec) == built


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


def _library_interactions():
    """The interactions of the library outside the zoo and the examples, whose rules need the fields of their
    models; not the legacy functions (``tests/legacy``) or the rules that tests register."""
    from lgca.plugins import default_registry

    modules = {plugin.name: getattr(default_registry.resolve(plugin.name), "__module__", None) or ""
               for plugin in list_plugins(kind="interaction")}
    return [name for name, module in modules.items()
            if module.startswith("lgca.") and not module.startswith(("lgca.zoo.", "lgca.examples."))]


@pytest.mark.parametrize("plugin_name", _library_interactions())
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
