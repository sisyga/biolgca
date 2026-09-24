"""Reorientation terms scaled by a cell trait: a weight per cell in identity-based models.

With volume exclusion the labelled node states follow P(σ) ∝ exp(Σ_a s_a w_σ(a)) (Metropolis);
without it every cell draws its channel from its own softmax.
"""

import itertools

import numpy as np
import pytest

from lgca import reorientation_term
from lgca.model import (
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    model_spec_from_dict,
    model_spec_to_dict,
)
from lgca.pipeline import (
    _REORIENTATION_TERMS,
    _TERM_ALIASES,
    InteractionPipelineSpec,
    ReorientationSpec,
)

STRENGTHS = np.array([0.0, 1.0, 2.0, 4.0])
DIRECTION = np.array([1.0, 0.3])


@pytest.fixture(autouse=True)
def drift():
    terms, aliases = dict(_REORIENTATION_TERMS), dict(_TERM_ALIASES)

    @reorientation_term(coupling="flux")
    def drift(state, direction=(1.0, 0.0)):
        """Cells move in a fixed direction."""
        return np.asarray(direction, dtype=float)

    yield drift
    _REORIENTATION_TERMS.clear(), _REORIENTATION_TERMS.update(terms)
    _TERM_ALIASES.clear(), _TERM_ALIASES.update(aliases)


def _model(nodes, strength, terms, ve=True, sweeps=None, seed=5, geometry="hex"):
    dims = nodes.shape[:-1]
    parameters = {} if sweeps is None else {"sweeps": sweeps}
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=ve, identity_based=True,
                        traits={"strength": strength}, **({} if ve else {"capacity": 8})),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=terms, parameters=parameters)],
                                         propagation=False)))


def _channel_weights(model):
    angles = np.arange(6) * np.pi / 3
    return np.append(np.cos(angles) * DIRECTION[0] + np.sin(angles) * DIRECTION[1], 0.0)


def _exact(weights, strengths, K=7):
    """Mean channel weight and resting probability of every cell, by enumerating all arrangements."""
    maps = np.array(list(itertools.permutations(range(K), len(strengths))))
    p = np.exp((strengths * weights[maps]).sum(1))
    p /= p.sum()
    return (p[:, None] * weights[maps]).sum(0), (p[:, None] * (maps == K - 1)).sum(0)


def test_cells_with_their_own_strength_follow_the_joint_boltzmann_distribution(drift):
    # 60 x 60 hex nodes, each with 4 cells of strengths 0, 1, 2 and 4 in channels 0 to 3
    dims = (60, 60)
    n_nodes = np.prod(dims)
    nodes = np.zeros(dims + (7,), dtype=np.uint64)
    nodes[..., :4] = np.arange(1, 4 * n_nodes + 1).reshape(dims + (4,))
    model = _model(nodes, np.tile(STRENGTHS, n_nodes), [drift(beta=1.0, trait="strength", direction=DIRECTION)])
    weights = _channel_weights(model)
    model.step()
    labels = model.lgca.nodes[model.lgca.nonborder].reshape(n_nodes, 7)
    channel = np.argsort(labels, axis=1)[:, 3:]  # channels of the cells, in label order
    expected_weight, expected_rest = _exact(weights, STRENGTHS)
    measured_weight = weights[channel].mean(0)
    measured_rest = (channel == 6).mean(0)
    np.testing.assert_allclose(measured_weight, expected_weight, atol=4 * weights.std() / np.sqrt(n_nodes))
    np.testing.assert_allclose(measured_rest, expected_rest, atol=4 * np.sqrt(0.25 / n_nodes))
    assert measured_weight[3] > measured_weight[1] + 0.3  # the strongest cell follows the drift most


def test_without_volume_exclusion_every_cell_draws_from_its_own_weights(drift):
    dims = (60, 60)
    n_nodes = np.prod(dims)
    nodes = np.empty(dims + (7,), dtype=object)
    for index in np.ndindex(dims):
        first = 4 * np.ravel_multi_index(index, dims)
        nodes[index] = [list(range(first, first + 4))] + [[] for _ in range(6)]
    model = _model(nodes, np.tile(STRENGTHS, n_nodes), [drift(beta=1.0, trait="strength", direction=DIRECTION)],
                   ve=False)
    weights = _channel_weights(model)
    model.step()
    channel = np.empty(4 * n_nodes, dtype=int)
    for node in model.lgca.nodes[model.lgca.nonborder].reshape(n_nodes, 7):
        for k, content in enumerate(node):
            channel[content] = k
    for cell, strength in enumerate(STRENGTHS):
        p = np.exp(strength * weights)
        p /= p.sum()
        observed = np.bincount(channel[cell::4], minlength=7) / n_nodes
        np.testing.assert_allclose(observed, p, atol=4 * np.sqrt(0.25 / n_nodes))


def test_equal_strengths_give_the_classical_sampler(drift):
    dims = (60, 60)
    rng = np.random.default_rng(3)
    occupied = rng.random(dims + (7,)) < 0.45
    labels = occupied.astype(np.uint64)
    labels[occupied] = np.arange(1, occupied.sum() + 1)
    term = drift(beta=1.5, direction=DIRECTION)
    identity = _model(labels, np.full(occupied.sum(), 1.5), [drift(beta=1.0, trait="strength",
                                                                    direction=DIRECTION)])
    classical = build_model(ModelSpec(
        space=SpaceSpec(geometry="hex", dims=dims), state=StateSpec(nodes=occupied, restchannels=1),
        time=TimeSpec(steps=1, seed=5),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])], propagation=False)))
    identity.step()
    classical.step()
    flux = [model.lgca.calc_flux(model.lgca._channel_counts(model.lgca.nodes) if hasattr(
        model.lgca, "_channel_counts") else model.lgca.nodes)[model.lgca.nonborder] for model in (identity, classical)]
    density = occupied.sum(-1)
    for level in range(1, 7):
        at = density == level
        a, b = flux[0][at] @ DIRECTION, flux[1][at] @ DIRECTION
        error = np.sqrt((a.var() + b.var()) / at.sum())
        assert abs(a.mean() - b.mean()) < 4 * error, level


def test_trait_terms_are_checked_and_saved(drift):
    nodes = np.zeros((4, 4, 7), dtype=bool)
    nodes[0, 0, :2] = True
    term = drift(beta=1.0, trait="strength")
    with pytest.raises(ValueError, match="needs an identity-based model"):
        build_model(ModelSpec(space=SpaceSpec(geometry="hex", dims=(4, 4)),
                              state=StateSpec(nodes=nodes, restchannels=1),
                              dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])])))
    with pytest.raises(ValueError, match="no trait 'strength'"):
        build_model(ModelSpec(space=SpaceSpec(geometry="hex", dims=(4, 4)),
                              state=StateSpec(nodes=nodes, restchannels=1, identity_based=True),
                              dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])])))
    labels = nodes.astype(np.uint64)
    labels[nodes] = [1, 2]
    with pytest.raises(ValueError, match="sweeps must be a positive integer"):
        _model(labels, 1.0, [term], sweeps=0)
    with pytest.raises(ValueError, match="did you mean 'sweeps'"):
        build_model(ModelSpec(dynamics=InteractionPipelineSpec(operators=[
            ReorientationSpec(terms=[term], parameters={"sweps": 3})])))
    spec = ModelSpec(state=StateSpec(density=1, restchannels=1, identity_based=True, traits={"strength": 2.0}),
                     dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])]))
    assert model_spec_from_dict(model_spec_to_dict(spec)).dynamics.operators[0].terms[0].trait == "strength"


def test_more_sweeps_are_closer_to_the_joint_distribution(drift):
    dims = (40, 40)
    n_nodes = np.prod(dims)
    nodes = np.zeros(dims + (7,), dtype=np.uint64)
    nodes[..., :4] = np.arange(1, 4 * n_nodes + 1).reshape(dims + (4,))
    strength = np.tile(STRENGTHS * 2, n_nodes)
    errors = []
    for sweeps in (1, 10):
        model = _model(nodes, strength, [drift(beta=1.0, trait="strength", direction=DIRECTION)], sweeps=sweeps)
        weights = _channel_weights(model)
        model.step()
        labels = model.lgca.nodes[model.lgca.nonborder].reshape(n_nodes, 7)
        channel = np.argsort(labels, axis=1)[:, 3:]
        expected, _ = _exact(weights, STRENGTHS * 2)
        errors.append(np.abs(weights[channel].mean(0) - expected).max())
    assert errors[1] < errors[0] and errors[1] < 4 * weights.std() / np.sqrt(n_nodes)
