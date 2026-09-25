"""Switching probabilities that respond to cues.

The tanh form p = max (1 + tanh(Σ kappa (cue - theta))) / 2, and the Boltzmann form with weights
w = rate exp(Σ beta cue) against staying.
"""

import numpy as np
import pytest

from lgca import switch_cue
from lgca.builtin_rules import tanh_switch
from lgca.lattice_state import LatticeState
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.switching import _values, parse_probability

DIMS = (60, 60)
DENSITY_CUE = {"max": 0.8, "cues": [{"name": "density", "kappa": 4.0, "theta": 0.5}]}


def _classical(ve, operators, nodes, fields=None, seed=1):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=1, n_species=nodes.shape[-2], volume_exclusion=ve,
                        capacity=None if ve else 10, fields=fields or {}),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False)))


def _assert_by_level(level, trials, successes, p):
    """Successes out of trials at every level, against the probability p(level)."""
    compared = 0
    for value in np.unique(level):
        at = level == value
        n = trials[at].sum()
        if n < 200:
            continue
        expected = p(value)
        assert abs(successes[at].sum() / n - expected) < 4 * np.sqrt(expected * (1 - expected) / n) + 1e-9, value
        compared += 1
    assert compared >= 3


@pytest.mark.parametrize("ve", [True, False])
def test_species_switch_responds_to_the_density(ve):
    # species 0 switches to species 1, which has no cells: every switch finds a free channel
    rng = np.random.default_rng(0)
    nodes = np.zeros(DIMS + (2, 5), dtype=bool if ve else np.int64)
    nodes[..., 0, :] = rng.random(DIMS + (5,)) < 0.5 if ve else rng.poisson(1.2, DIMS + (5,))
    model = _classical(ve, [{"name": "phenotype_switch", "parameters": {"rates": [[0, DENSITY_CUE], [0, 0]]}}],
                       nodes)
    capacity = 5 * 2 if ve else 10  # the capacity of the lattice state: n_species * K with volume exclusion
    cells = nodes[..., 0, :].sum(-1)
    model.step()
    switched = model.lgca.nodes[model.lgca.nonborder][..., 1, :].sum(-1)
    _assert_by_level(cells, cells, switched, lambda n: 0.8 * tanh_switch(n / capacity, 4.0, 0.5))


def test_species_switch_responds_to_a_field_of_the_species_it_senses():
    rng = np.random.default_rng(1)
    signal = np.add.outer(np.arange(DIMS[0]), np.zeros(DIMS[1])) / DIMS[0]
    nodes = np.zeros(DIMS + (2, 5), dtype=np.int64)
    nodes[..., 0, :] = rng.poisson(0.5, DIMS + (5,))
    rates = [[0, {"cues": [{"name": "field", "field": "signal", "kappa": 3.0, "theta": 0.5}]}], [0, 0]]
    model = _classical(False, [{"name": "phenotype_switch", "parameters": {"rates": rates}}], nodes,
                       fields={"signal": signal})
    cells = nodes[..., 0, :].sum(-1)
    model.step()
    switched = model.lgca.nodes[model.lgca.nonborder][..., 1, :].sum(-1)
    column = np.round(signal * 10).astype(int)  # levels of the field
    _assert_by_level(column, cells, switched, lambda level: tanh_switch(level / 10, 3.0, 0.5))


def test_cues_read_the_state():
    rng = np.random.default_rng(2)
    nodes = rng.poisson(0.7, DIMS + (2, 5))
    signal = rng.random(DIMS)
    model = _classical(False, [{"name": "random_walk"}], nodes, fields={"signal": signal})
    state = LatticeState(model.lgca, capacity=10)
    counts = nodes[..., :4]
    flux = counts.sum(-2) @ model.lgca.c.T

    def cue(name, **parameters):
        return _values(state, parse_probability({"cues": [{"name": name, **parameters}]}).cues[0])

    np.testing.assert_allclose(cue("density"), nodes.sum((-2, -1)) / 10)
    np.testing.assert_allclose(cue("density", sensed_species=1), nodes[..., 1, :].sum(-1) / 10)
    np.testing.assert_allclose(cue("field", field="signal"), signal)
    np.testing.assert_allclose(cue("gradient", field="signal"), np.linalg.norm(state.gradient("signal"), axis=-1))
    np.testing.assert_allclose(cue("flux"), np.linalg.norm(flux, axis=-1) / np.maximum(nodes.sum((-2, -1)), 1))
    np.testing.assert_allclose(cue("flux", normalize=False), np.linalg.norm(flux, axis=-1))


@pytest.mark.parametrize("ve", [True, False])
def test_cells_switch_a_trait_with_their_own_sensitivity_and_state(ve):
    # non-aligning cells (0) start to align with a density switch whose steepness is a trait of every
    # cell; aligning cells (2) are left alone by that event
    rng = np.random.default_rng(3)
    switch = {"when": {"alignment": 0}, "traits": {"alignment": {"value": 2.0, "operation": "set"}},
              "probability": {"cues": [{"name": "density", "kappa": "kappa", "theta": 0.3}]}}
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(density=2.5 if ve else 1.5, restchannels=1, identity_based=True, volume_exclusion=ve,
                        capacity=None if ve else 8, traits={"alignment": 0.0, "kappa": 0.0}),
        time=TimeSpec(steps=1, seed=4),
        dynamics=InteractionPipelineSpec(operators=[{"name": "trait_switch", "parameters": {"switch": switch}}],
                                         propagation=False)))
    lgca = model.lgca
    state = LatticeState(lgca)
    cells = state.cells
    kappa = rng.choice([-4.0, 4.0], len(cells))
    alignment = rng.choice([0.0, 2.0], len(cells))
    lgca.props["kappa"][cells.label] = kappa
    lgca.props["alignment"][cells.label] = alignment
    rho = (state.density / state.capacity).reshape(-1)[cells.index]
    model.step()
    after = np.asarray(lgca.props["alignment"])[cells.label]
    assert np.all(after[alignment == 2.0] == 2.0)
    for sign in (-4.0, 4.0):
        chosen = (alignment == 0) & (kappa == sign)
        _assert_by_level(rho[chosen], np.ones(chosen.sum()), after[chosen] == 2.0,
                         lambda level, sign=sign: tanh_switch(level, sign, 0.3))


def test_mutations_can_respond_to_cues():
    # daughters mutate only in sparse nodes
    mutation = {"probability": {"cues": [{"name": "density", "kappa": -50.0, "theta": 0.5}]},
                "traits": {"r_b": {"value": 0.0, "operation": "set"}}}
    nodes = np.zeros(DIMS + (5,), dtype=bool)
    nodes[:, : DIMS[1] // 2, 0] = True  # sparse half: one cell per node
    nodes[:, DIMS[1] // 2:, :4] = True  # crowded half: four cells per node
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, traits={"r_b": 1.0}),
        time=TimeSpec(steps=1, seed=5),
        dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death", "parameters": {
            "birth_rate": "r_b", "mutation": mutation}}], propagation=False)))
    first = int(model.lgca.maxlabel) + 1
    model.step()
    lgca = model.lgca
    labels = lgca.nodes[lgca.nonborder]
    daughters = labels >= first
    mutated = np.asarray(lgca.props["r_b"])[labels[daughters]] == 0.0
    side = np.nonzero(daughters)[1] < DIMS[1] // 2
    assert mutated[side].mean() > 0.99 and mutated[~side].mean() < 0.01


def test_a_cue_of_your_own():
    @switch_cue
    def left_half(state):
        """1 in the left half of the lattice."""
        values = np.zeros(state.dims)
        values[:, : state.dims[1] // 2] = 1.0
        return values

    nodes = np.zeros(DIMS + (2, 5), dtype=bool)
    nodes[..., 0, 0] = True
    rates = [[0, {"cues": [{"name": "left_half", "kappa": 50.0, "theta": 0.5}]}], [0, 0]]
    model = _classical(True, [{"name": "phenotype_switch", "parameters": {"rates": rates}}], nodes)
    model.step()
    switched = model.lgca.nodes[model.lgca.nonborder][..., 1, :].sum(-1)
    assert switched[:, : DIMS[1] // 2].all() and not switched[:, DIMS[1] // 2:].any()


@pytest.mark.parametrize("rates, message", [
    ([[0, {"cues": [{"name": "densty"}]}], [0, 0]], "unknown cue 'densty'"),
    ([[0, {"cues": [{"name": "density", "kappa": "kappa"}]}], [0, 0]], "reads cell traits"),
    ([[0, {"max": 0.7}], [0.2, 0]], None),
    ([[0, 0.7], [0.2, 0], [0, 0]], "2 x 2 matrix"),
    ([[0, {"cues": {"name": "density"}}], [0, 0]], "list of cues"),
])
def test_switching_probabilities_are_checked(rates, message):
    nodes = np.zeros(DIMS + (2, 5), dtype=bool)
    model = _classical(True, [{"name": "phenotype_switch", "parameters": {"rates": rates}}], nodes)
    if message is None:
        model.step()
        return
    with pytest.raises((ValueError, TypeError), match=message):
        model.step()


def _weight(rate, beta, cue):
    return rate * np.exp(beta * cue)


@pytest.mark.parametrize("ve", [True, False])
def test_boltzmann_rows_choose_between_several_switches(ve):
    # species 0 turns into 1 (more in crowded nodes) or into 2 (less in crowded nodes); both are empty,
    # so every switch finds a free channel
    rng = np.random.default_rng(6)
    nodes = np.zeros(DIMS + (3, 5), dtype=bool if ve else np.int64)
    nodes[..., 0, :] = rng.random(DIMS + (5,)) < rng.random(DIMS + (1,)) if ve else rng.poisson(1.5, DIMS + (5,))
    to_1 = {"rate": 0.3, "cues": [{"name": "density", "beta": 6.0}]}
    to_2 = {"rate": 1.5, "cues": [{"name": "density", "beta": -4.0}]}
    model = _classical(ve, [{"name": "phenotype_switch", "parameters": {"rates": [[0, to_1, to_2], [0, 0, 0],
                                                                                  [0, 0, 0]]}}], nodes)
    capacity = 5 * 3 if ve else 10
    cells = nodes[..., 0, :].sum(-1)
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]

    def probability(n, own, other):
        rho = n / capacity
        return _weight(*own, rho) / (1 + _weight(0.3, 6.0, rho) + _weight(1.5, -4.0, rho))

    _assert_by_level(cells, cells, after[..., 1, :].sum(-1), lambda n: probability(n, (0.3, 6.0), None))
    _assert_by_level(cells, cells, after[..., 2, :].sum(-1), lambda n: probability(n, (1.5, -4.0), None))


@pytest.mark.parametrize("ve", [True, False])
def test_a_boltzmann_event_happens_with_probability_w_over_1_plus_w(ve):
    # trait_switch with a weight whose sensitivity is a trait of every cell
    rng = np.random.default_rng(7)
    switch = {"when": {"alignment": 0}, "traits": {"alignment": {"value": 1.0, "operation": "set"}},
              "probability": {"rate": 0.4, "cues": [{"name": "density", "beta": "beta"}]}}
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(density=2.5 if ve else 1.5, restchannels=1, identity_based=True, volume_exclusion=ve,
                        capacity=None if ve else 8, traits={"alignment": 0.0, "beta": 0.0}),
        time=TimeSpec(steps=1, seed=8),
        dynamics=InteractionPipelineSpec(operators=[{"name": "trait_switch", "parameters": {"switch": switch}}],
                                         propagation=False)))
    lgca = model.lgca
    state = LatticeState(lgca)
    cells = state.cells
    beta = rng.choice([-3.0, 5.0], len(cells))
    lgca.props["beta"][cells.label] = beta
    rho = (state.density / state.capacity).reshape(-1)[cells.index]
    model.step()
    after = np.asarray(lgca.props["alignment"])[cells.label]
    for value in (-3.0, 5.0):
        chosen = beta == value
        _assert_by_level(rho[chosen], np.ones(chosen.sum()), after[chosen] == 1.0,
                         lambda level, value=value: _weight(0.4, value, level) / (1 + _weight(0.4, value, level)))


def test_the_boltzmann_form_of_two_states_is_the_tanh_form():
    rng = np.random.default_rng(9)
    nodes = rng.poisson(0.7, DIMS + (2, 5))
    model = _classical(False, [{"name": "random_walk"}], nodes)
    state = LatticeState(model.lgca, capacity=10)
    kappa, theta = 3.0, 0.4
    tanh = parse_probability({"cues": [{"name": "density", "kappa": kappa, "theta": theta}]})
    boltzmann = parse_probability({"rate": np.exp(-2 * kappa * theta),
                                   "cues": [{"name": "density", "beta": 2 * kappa}]})
    np.testing.assert_allclose(boltzmann.nodes(state), tanh.nodes(state), rtol=1e-12)
    np.testing.assert_allclose(tanh.nodes(state), tanh_switch(state.density / 10, kappa, theta), rtol=1e-12)


def test_boltzmann_weights_do_not_overflow():
    nodes = np.full(DIMS + (2, 5), 3)
    model = _classical(False, [{"name": "random_walk"}], nodes)
    state = LatticeState(model.lgca, capacity=10)
    huge = parse_probability({"rate": 1e300, "cues": [{"name": "density", "beta": 1e3}]})
    tiny = parse_probability({"rate": 0.0, "cues": [{"name": "density", "beta": 1e3}]})
    np.testing.assert_array_equal(huge.nodes(state), 1.0)
    np.testing.assert_array_equal(tiny.nodes(state), 0.0)
    assert parse_probability({"rate": 0.25}).nodes(state) == pytest.approx(0.2)


@pytest.mark.parametrize("rates, message", [
    ([[0, {"rate": 0.1}, 0.2], [0, 0, 0], [0, 0, 0]], r"mixes the Boltzmann form"),
    ([[0, {"rate": 0.1, "max": 0.5}, 0], [0, 0, 0], [0, 0, 0]], "'max' and 'rate'"),
    ([[0, {"rate": -1}, 0], [0, 0, 0], [0, 0, 0]], ">= 0"),
    ([[0, {"rate": 1, "cues": [{"name": "density", "kappa": 2}]}, 0], [0, 0, 0], [0, 0, 0]], "'beta'"),
    ([[0, {"max": 1, "cues": [{"name": "density", "beta": 2}]}, 0], [0, 0, 0], [0, 0, 0]], "Boltzmann"),
    ([[0, {"rate": 5.0}, {"rate": 9.0}], [0, 0, 0], [0, 0, 0]], None),  # weights need not sum to 1
    ([[0, {"rate": 5.0}, 0], [0, 0, 0], [0, 0, 0]], None),
])
def test_boltzmann_rates_are_checked(rates, message):
    nodes = np.zeros(DIMS + (3, 5), dtype=bool)
    model = _classical(True, [{"name": "phenotype_switch", "parameters": {"rates": rates}}], nodes)
    if message is None:
        model.step()
        return
    with pytest.raises((ValueError, TypeError), match=message):
        model.step()
