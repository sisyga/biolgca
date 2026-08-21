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
    run_model,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.plugins import ConservationLaw, InteractionOperator, PluginInfo, describe_plugin, list_plugins
from lgca.simulation import DensityRecorder, FamilyPopulationRecorder, NodeRecorder


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


def test_model_spec_runs_propagation_as_separate_deterministic_phase():
    result = run_model(_square_spec(operators=()), showprogress=False)
    legacy = _legacy_square(interaction="only_propagation")

    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    assert result.metadata["operator_names"] == []
    assert result.metadata["geometry"] == "square"
    assert "Propagation" in result.pipeline.describe_schedule()


def test_classical_only_propagation_plugin_is_native_and_matches_legacy():
    spec = _square_spec(
        operators=[{"name": "classical.only_propagation"}],
        timesteps=2,
        seed=60,
    )
    result = run_model(spec, showprogress=False)
    legacy = _legacy_square(interaction="only_propagation", timesteps=2, seed=60)

    plugin = describe_plugin("classical.only_propagation")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_random_walk_modelspec_reproduces_existing_timestep():
    result = run_model(
        _square_spec(operators=[{"name": "classical.random_walk"}]),
        showprogress=False,
    )
    legacy = _legacy_square(interaction="random_walk")

    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_random_walk_plugin_is_native_and_matches_legacy():
    spec = _square_spec(operators=[{"name": "classical.random_walk"}], timesteps=2, seed=43)
    result = run_model(spec, showprogress=False)
    legacy = _legacy_square(interaction="random_walk", timesteps=2, seed=43)

    plugin = describe_plugin("classical.random_walk")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
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
            operators=[{"name": "classical.excitable_medium", "parameters": parameters}]
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

    plugin = describe_plugin("classical.excitable_medium")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_ib_random_walk_plugin_is_native_and_matches_legacy():
    spec = ModelSpec(
        description=Description(title="native equivalence ib.random_walk"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1, identity_based=True),
        time=TimeSpec(steps=2, seed=53),
        dynamics=InteractionPipelineSpec(operators=[{"name": "ib.random_walk"}]),
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

    plugin = describe_plugin("ib.random_walk")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_ib_birth_plugin_is_native_and_matches_legacy():
    parameters = {"r_b": 0.24, "std": 0.02, "a_max": 1.0}
    spec = ModelSpec(
        description=Description(title="native equivalence ib.birth"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1, identity_based=True),
        time=TimeSpec(steps=2, seed=55),
        dynamics=InteractionPipelineSpec(operators=[{"name": "ib.birth", "parameters": parameters}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        ib=True,
        interaction="birth",
        bc="periodic",
        seed=55,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("ib.birth")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["r_b"], legacy.props["r_b"])


def test_ib_birthdeath_plugin_is_native_and_matches_legacy():
    parameters = {
        "r_b": 0.22,
        "r_d": 0.04,
        "std": 0.02,
        "a_max": 1.0,
        "track_inheritance": False,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence ib.birthdeath"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1, identity_based=True),
        time=TimeSpec(steps=2, seed=56),
        dynamics=InteractionPipelineSpec(operators=[{"name": "ib.birthdeath", "parameters": parameters}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        ib=True,
        interaction="birthdeath",
        bc="periodic",
        seed=56,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("ib.birthdeath")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["r_b"], legacy.props["r_b"])


def test_ib_birthdeath_discrete_plugin_is_native_and_matches_legacy():
    parameters = {
        "r_b": 0.23,
        "r_d": 0.04,
        "drb": 0.03,
        "a_max": 1.0,
        "pmut": 0.2,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence ib.birthdeath_discrete"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1, identity_based=True),
        time=TimeSpec(steps=2, seed=57),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "ib.birthdeath_discrete", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        ib=True,
        interaction="birthdeath_discrete",
        bc="periodic",
        seed=57,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("ib.birthdeath_discrete")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["r_b"], legacy.props["r_b"])


def test_ib_go_and_grow_mutations_plugin_is_native_and_matches_legacy():
    parameters = {
        "effect": "driver_mutation",
        "r_b": 0.45,
        "r_m": 0.55,
        "r_d": 0.03,
        "fitness_increase": 1.2,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence ib.go_and_grow_mutations"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.4, restchannels=1, identity_based=True),
        time=TimeSpec(steps=2, seed=61),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "ib.go_and_grow_mutations", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(
            observers=[NodeRecorder(), DensityRecorder(), FamilyPopulationRecorder()]
        ),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.4,
        restchannels=1,
        ib=True,
        interaction="go_and_grow_mutations",
        bc="periodic",
        seed=61,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("ib.go_and_grow_mutations")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_array_equal(result.lgca.props["family"], legacy.props["family"])
    np.testing.assert_array_equal(
        result.lgca.family_props["ancestor"],
        legacy.family_props["ancestor"],
    )
    assert result.lgca.family_props["descendants"] == legacy.family_props["descendants"]
    np.testing.assert_allclose(result.lgca.family_props["r_b"], legacy.family_props["r_b"])
    assert result.lgca.fam_pop_t.shape[0] == 3


def test_pipeline_timing_is_aggregated_in_constant_space():
    spec = _square_spec(
        operators=[{"name": "classical.random_walk"}], timesteps=100, seed=71
    )

    result = run_model(spec, showprogress=False)
    timings = result.metadata["runtime"]["operator_timings"]

    assert len(timings) == 2
    assert {entry["name"] for entry in timings} == {"classical.random_walk", "Propagation"}
    assert all(entry["count"] == 100 for entry in timings)


def test_pipeline_timing_trace_is_explicit_and_bounded():
    spec = _square_spec(
        operators=[{"name": "classical.random_walk"}], timesteps=10, seed=72
    )
    spec = replace(spec, time=replace(spec.time, timing_trace=3))

    result = run_model(spec, showprogress=False)
    trace = result.metadata["runtime"]["timing_trace"]

    assert len(trace) == 3
    assert [entry["name"] for entry in trace] == [
        "classical.random_walk",
        "Propagation",
        "classical.random_walk",
    ]
    assert [entry["step"] for entry in trace] == [1, 1, 2]


def test_ib_go_or_grow_plugin_is_native_and_matches_legacy():
    parameters = {
        "r_b": 0.31,
        "r_d": 0.04,
        "kappa": 4.6,
        "theta": 0.62,
        "kappa_std": 0.12,
        "theta_std": 0.03,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence ib.go_or_grow"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.45, restchannels=2, identity_based=True),
        time=TimeSpec(steps=2, seed=62),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "ib.go_or_grow", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.45,
        restchannels=2,
        ib=True,
        interaction="go_or_grow",
        bc="periodic",
        seed=62,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("ib.go_or_grow")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["kappa"], legacy.props["kappa"])
    np.testing.assert_allclose(result.lgca.props["theta"], legacy.props["theta"])


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        ("classical.birth", "birth", {"r_b": 0.17}),
        ("classical.birthdeath", "birthdeath", {"r_b": 0.19, "r_d": 0.07}),
    ],
)
def test_classical_birth_plugins_are_native_and_match_legacy(
    plugin_name, legacy_interaction, parameters
):
    spec = _square_spec(
        operators=[{"name": plugin_name, "parameters": parameters}],
        timesteps=2,
        seed=48,
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        interaction=legacy_interaction,
        bc="periodic",
        seed=48,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_go_or_rest_plugin_is_native_and_matches_legacy():
    parameters = {"kappa": 4.2, "theta": 0.55}
    spec = ModelSpec(
        description=Description(title="native equivalence classical.go_or_rest"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=2),
        time=TimeSpec(steps=2, seed=49),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.go_or_rest", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=2,
        interaction="go_or_rest",
        bc="periodic",
        seed=49,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("classical.go_or_rest")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_go_or_grow_plugin_is_native_and_matches_legacy():
    parameters = {"r_b": 0.13, "r_d": 0.04, "kappa": 4.1, "theta": 0.6}
    spec = ModelSpec(
        description=Description(title="native equivalence classical.go_or_grow"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=2),
        time=TimeSpec(steps=2, seed=50),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.go_or_grow", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=2,
        interaction="go_or_grow",
        bc="periodic",
        seed=50,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("classical.go_or_grow")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        ("classical.alignment", "alignment", {"beta": 1.2}),
        ("classical.persistent_walk", "persistent_motion", {"beta": 1.4}),
        ("classical.aggregation", "aggregation", {"beta": 1.6}),
    ],
)
def test_classical_reorientation_plugins_are_native_and_match_legacy(
    plugin_name, legacy_interaction, parameters
):
    spec = _square_spec(
        operators=[{"name": plugin_name, "parameters": parameters}],
        timesteps=2,
        seed=44,
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        interaction=legacy_interaction,
        bc="periodic",
        seed=44,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        ("classical.nematic", "nematic", {"beta": 1.1}),
        ("classical.contact_guidance", "contact_guidance", {"beta": 1.5}),
    ],
)
def test_classical_tensor_reorientation_plugins_are_native_and_match_legacy(
    plugin_name, legacy_interaction, parameters
):
    spec = _square_spec(
        operators=[{"name": plugin_name, "parameters": parameters}],
        timesteps=2,
        seed=45,
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        interaction=legacy_interaction,
        bc="periodic",
        seed=45,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_chemotaxis_plugin_is_native_and_matches_legacy():
    gradient = np.zeros((6, 7, 2), dtype=float)
    gradient[..., 0] = np.linspace(-1.0, 1.0, 6)[:, None]
    gradient[..., 1] = np.linspace(0.5, -0.5, 7)[None, :]
    parameters = {"beta": 1.25, "gradient": gradient}
    spec = _square_spec(
        operators=[{"name": "classical.chemotaxis", "parameters": parameters}],
        timesteps=2,
        seed=46,
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        interaction="chemotaxis",
        bc="periodic",
        seed=46,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("classical.chemotaxis")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_classical_wetting_plugin_is_native_and_matches_legacy():
    ecm = np.linspace(0.2, 1.0, 20, dtype=float).reshape(4, 5)
    parameters = {"beta": 1.1, "alpha": 0.3, "gamma": 1.4, "rho_0": 1.0}
    spec = ModelSpec(
        description=Description(title="native equivalence classical.wetting"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=2, fields={"ecm": ecm}),
        time=TimeSpec(steps=2, seed=47),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.wetting", "parameters": parameters}],
            propagation="default",
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=2,
        interaction="wetting",
        bc="periodic",
        seed=47,
        **parameters,
    )
    legacy.ecm = np.pad(ecm, [(legacy.r_int, legacy.r_int), (legacy.r_int, legacy.r_int)], mode="edge")
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("classical.wetting")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.ecm, legacy.ecm)


def test_build_model_exposes_compiled_pipeline_schedule():
    compiled = build_model(_square_spec(operators=[{"name": "classical.excitable_medium"}]))

    assert compiled.metadata["operator_names"] == ["classical.excitable_medium"]
    assert compiled.metadata["observer_names"] == ["NodeRecorder", "DensityRecorder"]
    assert compiled.metadata["reorientation_term_names"] == []
    schedule = compiled.pipeline.describe_schedule()
    assert "classical.excitable_medium" in schedule
    assert "birth_death" in schedule
    assert "lgca.interactions.excitable_medium" in schedule
    assert "status=native" in schedule
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
    if plugin_name == "phenotype_switch":
        return StateSpec(density=0.35, restchannels=1, n_species=2)
    if plugin_name == "classical.wetting":
        return StateSpec(
            density=0.35,
            restchannels=2,
            fields={"ecm": np.ones((4, 5), dtype=float)},
        )
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
    return StateSpec(density=0.35, restchannels=2)


@pytest.mark.parametrize(
    "plugin_name",
    [plugin.name for plugin in list_plugins(kind="interaction")],
)
def test_all_registered_interactions_compile_through_modelspec(plugin_name):
    spec = ModelSpec(
        description=Description(title=f"compile {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=_state_for_plugin(plugin_name),
        time=TimeSpec(steps=0, seed=19),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name}]),
    )

    compiled = build_model(spec)

    assert compiled.metadata["operator_names"] == [plugin_name]


@pytest.mark.parametrize(
    "plugin_name",
    [plugin.name for plugin in list_plugins(kind="interaction")],
)
def test_all_registered_interactions_run_one_step_through_modelspec(plugin_name):
    spec = ModelSpec(
        description=Description(title=f"run {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=_state_for_plugin(plugin_name),
        time=TimeSpec(steps=1, seed=21),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name}]),
    )

    result = run_model(spec, showprogress=False)

    assert result.metadata["operator_names"] == [plugin_name]
    assert result.lgca.cell_density.shape == result.lgca.nodes.shape[: len(result.lgca.dims)]


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,state",
    [
        (
            "ib.go_or_grow",
            "go_or_grow",
            StateSpec(density=0.35, restchannels=2, identity_based=True),
        ),
        (
            "nove_ib.evo_steric",
            "steric_evolution",
            StateSpec(density=0.35, restchannels=1, volume_exclusion=False, identity_based=True),
        ),
        (
            "multispecies.go_or_grow",
            "go_or_grow",
            StateSpec(density=0.35, restchannels=1, volume_exclusion=False, n_species=2),
        ),
    ],
)
def test_representative_legacy_wrappers_match_backend_families(plugin_name, legacy_interaction, state):
    spec = ModelSpec(
        description=Description(title=f"legacy equivalence {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=state,
        time=TimeSpec(steps=2, seed=31),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=state.restchannels,
        ib=state.identity_based,
        ve=state.volume_exclusion,
        n_species=state.n_species,
        interaction=legacy_interaction,
        bc="periodic",
        seed=31,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        ("nove.dd_alignment", "dd_alignment", {"beta": 1.7, "include_center": True}),
        ("nove.di_alignment", "di_alignment", {"beta": 1.3, "include_center": False}),
    ],
)
def test_nove_alignment_plugins_are_native_and_match_legacy(plugin_name, legacy_interaction, parameters):
    spec = ModelSpec(
        description=Description(title=f"native equivalence {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=0, volume_exclusion=False),
        time=TimeSpec(steps=2, seed=37),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name, "parameters": parameters}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=0,
        ve=False,
        interaction=legacy_interaction,
        bc="periodic",
        seed=37,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_nove_random_walk_plugin_is_native_and_matches_legacy():
    spec = ModelSpec(
        description=Description(title="native equivalence nove.random_walk"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.35, restchannels=1, volume_exclusion=False),
        time=TimeSpec(steps=2, seed=39),
        dynamics=InteractionPipelineSpec(operators=[{"name": "nove.random_walk"}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.35,
        restchannels=1,
        ve=False,
        interaction="random_walk",
        bc="periodic",
        seed=39,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove.random_walk")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_nove_ib_random_walk_plugin_is_native_and_matches_legacy():
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.random_walk"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=1.0,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
        ),
        time=TimeSpec(steps=2, seed=54),
        dynamics=InteractionPipelineSpec(operators=[{"name": "nove_ib.random_walk"}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=1.0,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="random_walk",
        bc="periodic",
        seed=54,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.random_walk")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        (
            "nove_ib.birth",
            "birth",
            {"capacity": 8, "r_b": 0.21, "std": 0.02, "a_max": 1.0, "gamma": 0.3},
        ),
        (
            "nove_ib.birthdeath",
            "birthdeath",
            {
                "capacity": 8,
                "r_b": 0.21,
                "r_d": 0.03,
                "std": 0.02,
                "a_max": 1.0,
                "gamma": 0.3,
            },
        ),
    ],
)
def test_nove_ib_birth_plugins_are_native_and_match_legacy(
    plugin_name, legacy_interaction, parameters
):
    spec = ModelSpec(
        description=Description(title=f"native equivalence {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=58),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name, "parameters": parameters}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction=legacy_interaction,
        bc="periodic",
        seed=58,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["r_b"], legacy.props["r_b"])


def test_nove_ib_birthdeath_cancerdfe_plugin_is_native_and_matches_legacy():
    parameters = {
        "capacity": 8,
        "r_b": 0.22,
        "r_d": 0.04,
        "p_d": 0.3,
        "p_p": 0.2,
        "s_d": 0.05,
        "s_p": 0.01,
        "a_max": 1.0,
        "gamma": 0.25,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.birthdeath_cancerdfe"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=59),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nove_ib.birthdeath_cancerdfe", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="birthdeath_cancerdfe",
        bc="periodic",
        seed=59,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.birthdeath_cancerdfe")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["r_b"], legacy.props["r_b"])


def test_nove_ib_go_or_grow_plugin_is_native_and_matches_legacy():
    parameters = {
        "capacity": 8,
        "r_b": 0.28,
        "r_d": 0.05,
        "kappa": 4.1,
        "theta": 0.46,
        "kappa_std": 0.11,
        "theta_std": 0.04,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.go_or_grow"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=63),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nove_ib.go_or_grow", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="go_or_grow",
        bc="periodic",
        seed=63,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.go_or_grow")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["kappa"], legacy.props["kappa"])
    np.testing.assert_allclose(result.lgca.props["theta"], legacy.props["theta"])


def test_nove_ib_go_or_grow_kappa_plugin_is_native_and_matches_legacy():
    parameters = {
        "capacity": 8,
        "r_b": 0.27,
        "r_d": 0.05,
        "kappa": 4.2,
        "theta": 0.48,
        "kappa_std": 0.1,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.go_or_grow_kappa"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=64),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nove_ib.go_or_grow_kappa", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="go_or_grow_kappa",
        bc="periodic",
        seed=64,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.go_or_grow_kappa")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["kappa"], legacy.props["kappa"])


def test_nove_ib_go_or_grow_kappa_chemo_plugin_is_native_and_matches_legacy():
    from lgca.nove_ib_interactions import go_or_grow_kappa_chemo

    parameters = {
        "capacity": 8,
        "r_b": 0.25,
        "r_d": 0.04,
        "kappa": 4.0,
        "theta": 0.47,
        "kappa_std": 0.08,
        "beta": 3.5,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.go_or_grow_kappa_chemo"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=65),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nove_ib.go_or_grow_kappa_chemo", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy_parameters = dict(parameters)
    beta = legacy_parameters.pop("beta")
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="go_or_grow_kappa",
        bc="periodic",
        seed=65,
        **legacy_parameters,
    )
    legacy.interaction = go_or_grow_kappa_chemo
    legacy.interaction_params["beta"] = beta
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.go_or_grow_kappa_chemo")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_allclose(result.lgca.props["kappa"], legacy.props["kappa"])


def test_nove_ib_go_or_grow_glioblastoma_plugin_is_native_and_matches_legacy():
    parameters = {
        "capacity": 8,
        "r_b": 0.3,
        "r_d": 0.04,
        "r_m": 0.45,
        "fitness_increase": 1.15,
        "theta": 0.49,
        "kappa": 4.3,
        "kappa_std": 0.09,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.go_or_grow_glioblastoma"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=66),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nove_ib.go_or_grow_glioblastoma", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="go_or_grow_glioblastoma",
        bc="periodic",
        seed=66,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.go_or_grow_glioblastoma")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_array_equal(result.lgca.props["family"], legacy.props["family"])
    np.testing.assert_allclose(result.lgca.family_props["r_b"], legacy.family_props["r_b"])
    np.testing.assert_allclose(result.lgca.family_props["kappa"], legacy.family_props["kappa"])
    np.testing.assert_array_equal(
        result.lgca.family_props["ancestor"],
        legacy.family_props["ancestor"],
    )
    assert result.lgca.family_props["descendants"] == legacy.family_props["descendants"]


def test_nove_ib_evo_steric_plugin_is_native_and_matches_legacy():
    parameters = {
        "capacity": 20,
        "r_b": 0.25,
        "r_m": 0.5,
        "r_d": 0.1,
        "alpha": 1.5,
        "gamma": 2.2,
        "fitness_increase": 1.12,
    }
    spec = ModelSpec(
        description=Description(title="native equivalence nove_ib.evo_steric"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=0.9,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": parameters["capacity"]},
        ),
        time=TimeSpec(steps=2, seed=67),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nove_ib.evo_steric", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.9,
        restchannels=1,
        ve=False,
        ib=True,
        interaction="steric_evolution",
        bc="periodic",
        seed=67,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("nove_ib.evo_steric")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
    np.testing.assert_array_equal(result.lgca.props["family"], legacy.props["family"])
    np.testing.assert_allclose(result.lgca.family_props["r_b"], legacy.family_props["r_b"])
    np.testing.assert_array_equal(
        result.lgca.family_props["ancestor"],
        legacy.family_props["ancestor"],
    )
    assert result.lgca.family_props["descendants"] == legacy.family_props["descendants"]


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        (
            "multispecies.birth",
            "birth",
            {
                "r_b": [0.18, 0.11],
                "gamma": 0.35,
                "mutation_matrix": [[0.9, 0.1], [0.2, 0.8]],
            },
        ),
        (
            "multispecies.birthdeath",
            "birthdeath",
            {
                "r_b": [0.16, 0.09],
                "r_d": [0.03, 0.05],
                "gamma": 0.25,
                "mutation_matrix": [[0.85, 0.15], [0.25, 0.75]],
            },
        ),
    ],
)
def test_multispecies_birth_plugins_are_native_and_match_legacy(
    plugin_name, legacy_interaction, parameters
):
    spec = ModelSpec(
        description=Description(title=f"native equivalence {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=1.1,
            restchannels=1,
            volume_exclusion=False,
            n_species=2,
            parameters={"capacity": 6},
        ),
        time=TimeSpec(steps=2, seed=52),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name, "parameters": parameters}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=1.1,
        restchannels=1,
        ve=False,
        n_species=2,
        capacity=6,
        interaction=legacy_interaction,
        bc="periodic",
        seed=52,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_multispecies_go_or_grow_plugin_is_native_and_matches_legacy():
    parameters = {
        "r_b": 0.18,
        "r_d": 0.04,
        "kappa": [4.2, 5.1],
        "theta": 0.48,
        "mutation_matrix": [[0.88, 0.12], [0.18, 0.82]],
    }
    spec = ModelSpec(
        description=Description(title="native equivalence multispecies.go_or_grow"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(
            density=1.1,
            restchannels=1,
            volume_exclusion=False,
            n_species=2,
            parameters={"capacity": 6},
        ),
        time=TimeSpec(steps=2, seed=69),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "multispecies.go_or_grow", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=1.1,
        restchannels=1,
        ve=False,
        n_species=2,
        capacity=6,
        interaction="go_or_grow",
        bc="periodic",
        seed=69,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("multispecies.go_or_grow")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


def test_multispecies_excitable_medium_plugin_is_native_and_matches_legacy():
    parameters = {"beta": 0.04, "alpha": 1.2, "N": 3}
    spec = ModelSpec(
        description=Description(title="native equivalence multispecies.excitable_medium_ms"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=0.6, restchannels=1, volume_exclusion=True, n_species=2),
        time=TimeSpec(steps=2, seed=70),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "multispecies.excitable_medium_ms", "parameters": parameters}]
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.6,
        restchannels=1,
        ve=True,
        n_species=2,
        interaction="excitable_medium_ms",
        bc="periodic",
        seed=70,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin("multispecies.excitable_medium_ms")
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)


@pytest.mark.parametrize(
    "plugin_name,legacy_interaction,parameters",
    [
        ("nove.go_or_rest", "go_or_rest", {"kappa": 3.8, "theta": 0.45}),
        (
            "nove.go_or_grow",
            "go_or_grow",
            {"r_b": 0.16, "r_d": 0.05, "kappa": 4.4, "theta": 0.5},
        ),
    ],
)
def test_nove_go_or_plugins_are_native_and_match_legacy(
    plugin_name, legacy_interaction, parameters
):
    spec = ModelSpec(
        description=Description(title=f"native equivalence {plugin_name}"),
        space=SpaceSpec(geometry="square", dims=(4, 5), boundary="periodic"),
        state=StateSpec(density=1.2, restchannels=1, volume_exclusion=False),
        time=TimeSpec(steps=2, seed=51),
        dynamics=InteractionPipelineSpec(operators=[{"name": plugin_name, "parameters": parameters}]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    legacy = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=1.2,
        restchannels=1,
        ve=False,
        interaction=legacy_interaction,
        bc="periodic",
        seed=51,
        **parameters,
    )
    legacy.timeevo(timesteps=2, record=True, recorddens=True, showprogress=False)

    plugin = describe_plugin(plugin_name)
    assert plugin.port_status == "native"
    assert plugin.test_status == "unit_tested"
    np.testing.assert_array_equal(result.lgca.nodes_t, legacy.nodes_t)
    np.testing.assert_allclose(result.lgca.dens_t, legacy.dens_t)
