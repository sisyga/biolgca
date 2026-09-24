"""Decorated rules register, run in every declared family and are checked by check_interaction."""

import numpy as np
import pytest

from lgca import interaction
from lgca.model import (Description, ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model,
                        model_spec_from_dict, model_spec_to_dict, run_model)
from lgca.pipeline import InteractionPipelineSpec
from lgca.plugins import default_registry, describe_plugin
from lgca.testing import InteractionCheckError, check_interaction


@pytest.fixture(autouse=True)
def _restore_registry():
    plugins, aliases = dict(default_registry._plugins), dict(default_registry._aliases)
    yield
    default_registry._plugins, default_registry._aliases = plugins, aliases


def _crowding_death():
    @interaction(kind="birth_death", families=("classical", "nove"), name="crowding_death")
    def crowding_death(state, r_d=0.1):
        """Each cell dies with probability r_d * density / capacity.

        Parameters
        ----------
        r_d : float
            Death probability of a cell in a
            full node.
        """
        state.remove_cells(np.minimum(r_d * state.density / state.capacity, 1))

    return crowding_death


def _constant_death():
    @interaction(kind="birth_death", families=("classical", "nove"), name="constant_death")
    def constant_death(state, p=0.2):
        """Each cell dies with probability p."""
        state.remove_cells(p)

    return constant_death


def _hpp():
    @interaction(kind="reorientation", families="classical", geometries="square", conserves="momentum",
                 name="hpp_collision")
    def hpp_collision(state):
        """Two cells meeting head-on leave at right angles (HPP rule)."""
        counts = state.counts.copy()
        velocity = counts[..., :4]
        horizontal = np.all(velocity == [1, 0, 1, 0], axis=-1)
        vertical = np.all(velocity == [0, 1, 0, 1], axis=-1)
        velocity[horizontal] = [0, 1, 0, 1]
        velocity[vertical] = [1, 0, 1, 0]
        state.counts = counts

    return hpp_collision


def _spec(operators, ve=True, n_species=1, geometry="square", density=2.0, steps=5):
    return ModelSpec(
        description=Description(title="decorated rule"),
        space=SpaceSpec(geometry=geometry, dims=(10, 8) if geometry != "lin" else 20),
        state=StateSpec(density=density, restchannels=1, volume_exclusion=ve, n_species=n_species),
        time=TimeSpec(steps=steps, seed=3),
        dynamics=InteractionPipelineSpec(operators=operators),
    )


def test_the_decorator_builds_metadata_from_signature_and_docstring():
    rule = _crowding_death()
    info = describe_plugin("crowding_death")

    assert info.operator_kind == "birth_death"
    assert info.description == "Each cell dies with probability r_d * density / capacity."
    assert info.parameter_specs["r_d"].default == 0.1
    assert info.parameter_specs["r_d"].description == "Death probability of a cell in a full node."
    assert "r_d (default 0.1)" in str(rule)


def test_calling_a_rule_gives_an_operator_entry_and_checks_parameter_names():
    rule = _crowding_death()

    assert rule(r_d=0.3) == {"name": "crowding_death", "parameters": {"r_d": 0.3}}
    with pytest.raises(ValueError, match="did you mean 'r_d'"):
        rule(r_dd=0.3)
    with pytest.raises(TypeError, match="by keyword"):
        rule(0.3)


def test_parameters_without_default_are_required():
    @interaction(kind="birth_death", families="classical", name="needs_rate")
    def needs_rate(state, rate):
        state.remove_cells(rate)

    with pytest.raises(ValueError, match="needs_rate.rate is required"):
        needs_rate()


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("n_species", [1, 2])
def test_a_rule_runs_with_and_without_volume_exclusion_and_several_species(ve, n_species):
    rule = _crowding_death()

    result = run_model(_spec([rule(r_d=1.0)], ve=ve, n_species=n_species), showprogress=False)

    assert result.lgca.nodes.sum() < build_model(_spec([rule(r_d=1.0)], ve=ve, n_species=n_species)).lgca.nodes.sum()


def test_defaults_are_applied_by_the_framework():
    seen = {}

    @interaction(kind="birth_death", families="classical", name="records_default")
    def records_default(state, rate=0.25):
        seen["rate"] = rate

    run_model(_spec([records_default()], steps=1), showprogress=False)

    assert seen == {"rate": 0.25}


def test_models_of_undeclared_families_and_geometries_are_rejected():
    rule = _hpp()

    with pytest.raises(ValueError, match="written for models with volume exclusion.*add 'nove'"):
        build_model(_spec([rule()], ve=False))
    with pytest.raises(ValueError, match="geometries square, not 'hex'"):
        build_model(_spec([rule()], geometry="hex"))


def test_identity_based_families_cannot_be_declared_yet():
    with pytest.raises(ValueError, match="identity-based"):
        interaction(kind="birth_death", families="ib")(lambda state: None)


def test_a_phenotype_switch_rule_needs_several_species():
    @interaction(kind="phenotype_switch", families=("classical", "nove"), name="swap_species")
    def swap_species(state, rate=0.1):
        state.switch_phenotype([[0, rate], [rate, 0]])

    with pytest.raises(ValueError, match="one species"):
        build_model(_spec([swap_species()]))
    check_interaction(swap_species, n_species=(2,))


def test_a_rule_that_breaks_its_kind_fails_when_it_runs():
    @interaction(kind="reorientation", families="classical", name="leaky_turn")
    def leaky_turn(state):
        state.remove_cells(0.5)

    with pytest.raises(ValueError, match="a reorientation must keep"):
        run_model(_spec([leaky_turn()]), showprogress=False)


def test_hpp_collisions_turn_head_on_pairs_and_keep_momentum():
    rule = _hpp()
    nodes = np.zeros((10, 8, 5), dtype=bool)
    nodes[2, 2, [0, 2]] = True
    nodes[5, 5, [1, 3]] = True
    nodes[7, 1, [0, 1]] = True
    spec = _spec([rule()], steps=1)
    spec = ModelSpec(description=spec.description, space=spec.space, time=spec.time,
                     state=StateSpec(nodes=nodes, restchannels=1),
                     dynamics=InteractionPipelineSpec(operators=[rule()], propagation=False))

    lgca = run_model(spec, showprogress=False).lgca
    interior = lgca.nodes[lgca.nonborder]

    assert interior[2, 2].tolist() == [False, True, False, True, False]
    assert interior[5, 5].tolist() == [True, False, True, False, False]
    assert interior[7, 1].tolist() == [True, True, False, False, False]
    check_interaction(rule)


def test_declared_momentum_conservation_is_enforced():
    @interaction(kind="reorientation", families="classical", conserves="momentum", name="bad_hpp")
    def bad_hpp(state):
        state.shuffle_cells()

    with pytest.raises(ValueError, match="conserves momentum"):
        run_model(_spec([bad_hpp()]), showprogress=False)


def test_rules_survive_a_round_trip_through_a_model_file():
    rule = _crowding_death()
    spec = _spec([rule(r_d=0.5)])

    loaded = model_spec_from_dict(model_spec_to_dict(spec))

    np.testing.assert_array_equal(run_model(loaded, showprogress=False).lgca.nodes,
                                  run_model(spec, showprogress=False).lgca.nodes)


def test_redefining_a_rule_in_the_same_module_replaces_it():
    first = _crowding_death()
    second = _crowding_death()

    assert describe_plugin("crowding_death") is second.info is not first.info


def test_a_rule_can_be_applied_to_a_state_directly():
    from lgca.lattice_state import LatticeState

    rule = _crowding_death()
    state = LatticeState(build_model(_spec([rule()])).lgca)
    before = state.density.sum()

    rule(state, r_d=1.0)

    assert state.density.sum() < before


def test_check_interaction_passes_a_correct_rule_and_measures_its_rate():
    assert check_interaction(_crowding_death(), parameters={"r_d": 0.5}).passed

    report = check_interaction(_constant_death(), parameters={"p": 0.3}, expected_growth=-0.3)

    assert report.passed
    assert len(report.rows) == 5 * 2 * 2 * 2  # geometries x families x species x boundaries
    assert "all passed" in str(report)


def test_check_interaction_reports_what_is_wrong():
    @interaction(kind="reorientation", families="classical", geometries="square", name="sloppy")
    def sloppy(state):
        state.remove_cells(np.random.random())

    with pytest.raises(InteractionCheckError) as error:
        check_interaction(sloppy)

    message = str(error.value)
    assert "square, classical, 1 species, periodic" in message
    assert "a reorientation must keep" in message


def test_check_interaction_detects_unseeded_randomness():
    @interaction(kind="birth_death", families="nove", geometries="lin", name="unseeded")
    def unseeded(state):
        state.remove_cells(np.random.random())

    report = check_interaction(unseeded, raise_on_failure=False)

    assert not report.passed
    assert "state.rng" in str(report)


def test_check_interaction_detects_a_wrong_rate():
    report = check_interaction(_constant_death(), parameters={"p": 0.3}, families=("classical",),
                               geometries=("square",), expected_growth=-0.1, raise_on_failure=False)

    assert "measured -0.30" in str(report)


def test_check_interaction_accepts_registered_names():
    assert check_interaction("classical.birth", parameters={"r_b": 0.2}).passed


def test_check_interaction_catches_ghost_nodes_that_leak_into_the_lattice():
    from lgca.plugins import BirthDeathOperator, PluginInfo, register_plugin

    class FillsGhosts(BirthDeathOperator):
        def apply(self, context, step):
            lgca = context.lgca
            interior = lgca.nodes[lgca.nonborder].copy()
            lgca.nodes[...] = True
            lgca.nodes[lgca.nonborder] = interior

    info = PluginInfo(name="fills_ghosts", operator_kind="birth_death", backend_families=("classical",))
    register_plugin(info, lambda parameters=None: FillsGhosts(info, parameters))

    report = check_interaction("fills_ghosts", raise_on_failure=False)

    failures = "\n".join(report.failures)
    assert "reflecting: step 1: changes to ghost nodes reach the lattice" in failures
    assert "periodic" not in failures
