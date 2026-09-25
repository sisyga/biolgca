"""The resting term: go-or-rest as a Boltzmann reorientation.

A cell alone at its node rests with a switching probability p. Without volume exclusion every cell
rests with p; with it, the cells at rest follow Fisher's noncentral hypergeometric distribution: a rest
channel has the odds p / (1 - p) * v / r against a velocity channel.
"""

from math import comb

import numpy as np
import pytest

from lgca.builtin_rules import tanh_switch
from lgca.lattice_state import LatticeState
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec

SWITCH = {"cues": [{"name": "density", "kappa": 4.0, "theta": 0.5}]}
CHANNELS = {"lin": 2, "square": 4, "hex": 6}


def _model(geometry, dims, nodes, ve, operators, capacity=None, identity_based=False, traits=None, seed=1,
           restchannels=1):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims),
        state=StateSpec(nodes=nodes, restchannels=restchannels, volume_exclusion=ve, capacity=capacity,
                        identity_based=identity_based, traits=traits or {}),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False)))


def _dims(geometry):
    return (40000,) if geometry == "lin" else (200, 200)


def _fisher(n, v, r, odds):
    """Mean and variance of the cells at rest: n cells, v velocity and r rest channels, odds of a rest channel."""
    k = np.arange(max(0, n - v), min(n, r) + 1)
    weights = np.array([comb(r, int(i)) * comb(v, int(n - i)) for i in k]) * odds ** k
    weights = weights / weights.sum()
    mean = (k * weights).sum()
    return mean, (k ** 2 * weights).sum() - mean ** 2


def _odds(p, v=1, r=1):
    """The odds of a rest channel against a velocity channel for a lone cell that rests with p."""
    return p / (1 - p) * v / r


def _assert_level(values, expected_mean, expected_variance, label):
    error = np.sqrt(expected_variance / len(values))
    assert abs(values.mean() - expected_mean) <= 5 * error + 1e-9, (label, values.mean(), expected_mean)


@pytest.mark.parametrize("geometry", list(CHANNELS))
def test_without_volume_exclusion_every_cell_rests_with_the_probability(geometry):
    rng = np.random.default_rng(1)
    dims, v, r = _dims(geometry), CHANNELS[geometry], 1  # models without volume exclusion have one rest channel
    nodes = rng.poisson(rng.random(dims + (1,)) * 1.5, dims + (v + r,))
    model = _model(geometry, dims, nodes, False, [{"name": "resting", "parameters": {"probability": SWITCH}}],
                   capacity=8, restchannels=r)
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]
    cells = nodes.sum(-1)
    resting = after[..., v:].sum(-1)
    for n in range(1, 13):
        at = cells == n
        if at.sum() < 100:
            continue
        p = tanh_switch(n / 8, 4.0, 0.5)
        _assert_level(resting[at] / n, p, p * (1 - p) / n, n)
    # moving cells are spread uniformly over the velocity channels
    moving = after[..., :v].sum(axis=tuple(range(len(dims))))
    assert np.allclose(moving / moving.sum(), 1 / v, atol=5 * np.sqrt(1 / v / moving.sum()))


@pytest.mark.parametrize("geometry, restchannels", [("lin", 1), ("square", 1), ("square", 2), ("hex", 2)])
def test_with_volume_exclusion_the_cells_at_rest_follow_the_noncentral_hypergeometric_law(geometry, restchannels):
    rng = np.random.default_rng(2)
    dims, v, r = _dims(geometry), CHANNELS[geometry], restchannels
    K = v + r
    nodes = rng.random(dims + (K,)) < rng.random(dims + (1,))
    model = _model(geometry, dims, nodes, True, [{"name": "resting", "parameters": {"probability": SWITCH}}],
                   restchannels=r)
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]
    cells = nodes.sum(-1)
    np.testing.assert_array_equal(after.sum(-1), cells)
    resting = after[..., v:].sum(-1)
    for n in range(1, K):
        at = cells == n
        mean, variance = _fisher(n, v, r, _odds(tanh_switch(n / K, 4.0, 0.5), v, r))
        _assert_level(resting[at], mean, variance, n)


def test_the_boltzmann_form_and_beta():
    """A lone cell rests with w / (1 + w) for a weight w; beta = 0 is a random walk over all channels."""
    rng = np.random.default_rng(3)
    dims, v, r = (200, 200), 4, 1
    nodes = rng.poisson(rng.random(dims + (1,)) * 1.5, dims + (v + r,))
    cells = nodes.sum(-1)
    padded = np.pad(cells.astype(float), 1, mode="wrap")
    neighbourhood = (cells + padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]) / 5
    weight = 0.3 * np.exp(2.0 * neighbourhood / 8)
    probability = {"rate": 0.3, "cues": [{"name": "density", "beta": 2.0, "scope": "neighbourhood"}]}
    for beta, p in ((1.0, weight / (1 + weight)), (0.0, np.full(dims, r / (v + r)))):
        model = _model("square", dims, nodes, False, [{"name": "resting", "parameters": {
            "probability": probability, "beta": beta}}], capacity=8, seed=4)
        model.step()
        resting = model.lgca.nodes[model.lgca.nonborder][..., v:].sum(-1)
        level = np.digitize(neighbourhood, [1, 2, 3])
        for value in range(4):  # every cell rests with the probability of its node
            at = level == value
            expected = (cells[at] * p[at]).sum()
            error = np.sqrt((cells[at] * p[at] * (1 - p[at])).sum())
            assert abs(resting[at].sum() - expected) <= 5 * error, (beta, value)


@pytest.mark.parametrize("ve", [False, True])
def test_every_cell_rests_by_its_own_kappa_in_identity_based_models(ve):
    """kappa is a trait; nodes hold cells of one kappa, so each node follows the law of its kappa."""
    rng = np.random.default_rng(5)
    dims, v, r = (150, 150), 4, 1
    K = v + r
    probability = {"cues": [{"name": "density", "kappa": "kappa", "theta": 0.5}]}
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=dims),
        state=StateSpec(density=0.5 if ve else 0.4, restchannels=r, volume_exclusion=ve,
                        capacity=None if ve else 5, identity_based=True, traits={"kappa": 0.0}),
        time=TimeSpec(steps=1, seed=6),
        dynamics=InteractionPipelineSpec(operators=[{"name": "resting", "parameters": {
            "probability": probability}}], propagation=False)))
    lgca = model.lgca
    cells = LatticeState(lgca).cells
    node_kappa = rng.choice([-4.0, 4.0], dims)
    lgca.props["kappa"][cells.label] = node_kappa.reshape(-1)[cells.index]
    number = np.bincount(cells.index, minlength=node_kappa.size).reshape(dims)
    model.step()
    after = LatticeState(lgca).cells
    resting = np.bincount(after.index[after.in_channels("rest")], minlength=node_kappa.size).reshape(dims)
    np.testing.assert_array_equal(np.bincount(after.index, minlength=node_kappa.size).reshape(dims), number)
    for value in (-4.0, 4.0):
        for n in range(1, K if ve else 6):
            at = (number == n) & (node_kappa == value)
            if at.sum() < 100:
                continue
            p = tanh_switch(n / K, value, 0.5)
            if ve:
                mean, variance = _fisher(n, v, r, _odds(p, v, r))
            else:
                mean, variance = n * p, n * p * (1 - p)
            _assert_level(resting[at], mean, variance, (value, n))


def test_two_cells_with_their_own_odds_share_a_node_by_the_boltzmann_weights():
    """With volume exclusion, cells A and B (odds a, b per rest channel) at a node with one rest channel:
    P(A rests) = v a / (v (v - 1) + v a + v b)."""
    dims, v, K = (300, 300), 4, 5
    nodes = np.zeros(dims + (K,), dtype=np.int64)
    count = dims[0] * dims[1]
    nodes[..., 0] = np.arange(1, count + 1).reshape(dims)  # A: odd labels
    nodes[..., 1] = np.arange(count + 1, 2 * count + 1).reshape(dims)  # B
    probability = {"cues": [{"name": "density", "kappa": "kappa", "theta": 0.4}]}
    model = _model("square", dims, nodes, True, [{"name": "resting", "parameters": {"probability": probability}}],
                   identity_based=True, traits={"kappa": 0.0}, seed=7)
    lgca = model.lgca
    lgca.props["kappa"][1:count + 1] = 3.0
    lgca.props["kappa"][count + 1:2 * count + 1] = -3.0
    model.step()
    rest = lgca.nodes[lgca.nonborder][..., v]
    rho = 2 / K
    a = _odds(tanh_switch(rho, 3.0, 0.4)) * v
    b = _odds(tanh_switch(rho, -3.0, 0.4)) * v
    total = v * (v - 1) + v * a + v * b
    for resting_cells, weight in ((rest[(rest > 0) & (rest <= count)], a), (rest[rest > count], b)):
        p = v * weight / total
        _assert_level(np.r_[np.ones(len(resting_cells)), np.zeros(count - len(resting_cells))], p, p * (1 - p),
                      weight)


def test_resting_needs_rest_channels_and_traits_need_identity_based_models():
    nodes = np.ones((10, 10, 4), dtype=bool)
    with pytest.raises(ValueError, match="rest channel"):
        _model("square", (10, 10), nodes, True, [{"name": "resting"}], restchannels=0)
    nodes = np.ones((10, 10, 5), dtype=bool)
    probability = {"cues": [{"name": "density", "kappa": "kappa", "theta": 0.4}]}
    with pytest.raises(ValueError, match="identity-based"):
        _model("square", (10, 10), nodes, True, [{"name": "resting", "parameters": {"probability": probability}}])
