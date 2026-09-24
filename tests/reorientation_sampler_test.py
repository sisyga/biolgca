"""The Boltzmann reorientation without volume exclusion and in identity-based models.

Without volume exclusion every cell picks channel ``i`` independently with
``P(i) ∝ exp(w_i)``; identity-based models update their cell numbers like the
classical model and then place the node's cells on the occupied channels at
random.
"""

import numpy as np
import pytest
from scipy.special import softmax

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)

GEOMETRIES = [("lin", (20,)), ("square", (8, 8)), ("hex", (8, 8)), ("cubic", (4, 4, 4)), ("moore", (3, 3, 3))]
TERMS = [ReorientationTermSpec("polar_alignment", beta=0.5), ReorientationTermSpec("resting_bias", beta=0.3)]


def _model(geometry, dims, operators, *, nodes=None, density=3, restchannels=0, ve=False, ib=False,
           n_species=1, seed=3, propagation=True):
    extra = {} if ve else {"capacity": 8}
    state = StateSpec(nodes=nodes, density=None if nodes is not None else density, restchannels=restchannels, volume_exclusion=ve,
                      identity_based=ib, n_species=n_species, **extra)
    return build_model(ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims), state=state,
                                 time=TimeSpec(steps=1, seed=seed),
                                 dynamics=InteractionPipelineSpec(operators=operators, propagation=propagation)))


def _run(model, steps):
    for _ in range(steps):
        model.step()
    return model.lgca.nodes


@pytest.mark.parametrize("restchannels", [0, 1])
@pytest.mark.parametrize("geometry,dims", GEOMETRIES)
def test_without_terms_it_is_the_nove_random_walk(geometry, dims, restchannels):
    legacy = _model(geometry, dims, [{"name": "nove.random_walk"}], restchannels=restchannels)
    composed = _model(geometry, dims, [ReorientationSpec()], restchannels=restchannels)
    np.testing.assert_array_equal(_run(composed, 10), _run(legacy, 10))


@pytest.mark.parametrize("geometry,dims", [g for g in GEOMETRIES if g[0] != "hex"])
def test_polar_alignment_is_the_nove_density_dependent_alignment(geometry, dims):
    legacy = _model(geometry, dims, [{"name": "nove.dd_alignment", "parameters": {"beta": 0.7}}])
    composed = _model(geometry, dims, [ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment",
                                                                                       beta=0.7)])])
    np.testing.assert_array_equal(_run(composed, 10), _run(legacy, 10))


@pytest.mark.parametrize("geometry,dims", GEOMETRIES)
def test_polar_alignment_has_the_nove_alignment_probabilities(geometry, dims):
    # on hex, the neighbour sums differ from the legacy ones by rounding only,
    # so seeded runs drift apart; the probabilities agree
    model = _model(geometry, dims, [ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment",
                                                                                    beta=0.7)])])
    lgca, operator = model.lgca, model.pipeline.operators[0]
    for term in operator.terms:
        term.prepare(lgca)
    weights = operator._channel_weights(lgca)[..., 0, :]
    flux = lgca.nb_sum(lgca.calc_flux(lgca.nodes))
    expected = softmax(0.7 * np.einsum("...i,ij->...j", flux, lgca.c), axis=-1)[lgca.nonborder]
    np.testing.assert_allclose(softmax(weights, axis=-1), expected, rtol=1e-12, atol=1e-15)


def test_cells_choose_their_channels_independently():
    # two cells per node, rest weights (1, 1, 3): the pair is multinomial(2, (.2, .2, .6))
    nodes = np.zeros((20000, 3), dtype=int)
    nodes[:, 0] = 2
    model = _model("lin", (20000,), [ReorientationSpec(terms=[ReorientationTermSpec("resting_bias",
                                                                                     beta=np.log(3))])],
                   nodes=nodes, restchannels=1, propagation=False)
    counts = _run(model, 1)[model.lgca.nonborder]
    np.testing.assert_array_equal(counts.sum(-1), 2)
    p = np.array([.2, .2, .6])
    expected = {(0, 0, 2): p[2] ** 2, (1, 0, 1): 2 * p[0] * p[2], (1, 1, 0): 2 * p[0] * p[1],
                (2, 0, 0): p[0] ** 2}
    for state, probability in expected.items():
        observed = np.mean(np.all(counts == state, axis=-1))
        assert abs(observed - probability) < 5 * np.sqrt(probability * (1 - probability) / len(nodes))


def test_a_term_can_act_on_one_species_without_volume_exclusion():
    nodes = np.zeros((5000, 2, 3), dtype=int)
    nodes[..., 0] = 2
    terms = [ReorientationTermSpec("resting_bias", beta=20, species=1)]
    model = _model("lin", (5000,), [ReorientationSpec(terms=terms)], nodes=nodes, restchannels=1,
                   n_species=2, propagation=False)
    counts = _run(model, 1)[model.lgca.nonborder]
    np.testing.assert_array_equal(counts.sum(-1), 2)
    assert np.all(counts[:, 1, 2] == 2)
    resting = counts[:, 0, 2].mean() / 2
    assert abs(resting - 1 / 3) < 5 * np.sqrt(2 / 9 / 10000)


def _labels_per_node(lgca):
    inner = lgca.nodes[lgca.nonborder].reshape(-1, lgca.K)
    if inner.dtype == object:
        return [sorted(label for channel in node for label in channel) for node in inner]
    return [sorted(node[node > 0].tolist()) for node in inner]


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("geometry,dims", GEOMETRIES)
def test_identity_models_move_like_classical_ones_and_keep_their_cells(geometry, dims, ve):
    rng = np.random.default_rng(0)
    channels = _model(geometry, dims, [], ve=ve, restchannels=1).lgca.K
    if ve:
        classical_nodes = rng.random(dims + (channels,)) < (0.1 if geometry == "moore" else 0.4)
        identity_nodes = classical_nodes.astype(np.uint64)
        identity_nodes[classical_nodes] = np.arange(1, classical_nodes.sum() + 1)
    else:
        classical_nodes = identity_nodes = rng.poisson(0.5, dims + (channels,))
    operators = [ReorientationSpec(terms=TERMS)]
    classical = _model(geometry, dims, operators, nodes=classical_nodes, restchannels=1, ve=ve,
                       propagation=False, seed=7)
    identity = _model(geometry, dims, operators, nodes=identity_nodes, restchannels=1, ve=ve, ib=True,
                      propagation=False, seed=7)
    before = _labels_per_node(identity.lgca)
    classical.step()
    identity.step()  # the same random numbers for the cell numbers, then the labels
    lgca = identity.lgca
    counts = lgca._channel_counts(lgca.nodes[lgca.nonborder]).astype(int)
    np.testing.assert_array_equal(counts, classical.lgca.nodes[classical.lgca.nonborder].astype(int))
    assert _labels_per_node(lgca) == before


@pytest.mark.parametrize("ve", [True, False])
def test_identity_models_place_their_cells_at_random(ve):
    # every node holds cells 2k+1 and 2k+2 in channel 0; with rest weights (1, 1, 3),
    # each cell ends up at rest with probability 3/5 (VE: the pair takes channels
    # {0,1}, {0,2}, {1,2} with weights 1:3:3, and its cells are placed at random)
    n = 20000
    if ve:
        nodes = np.zeros((n, 3), dtype=np.uint64)
        nodes[:, 0] = np.arange(1, 2 * n, 2)
        nodes[:, 1] = np.arange(2, 2 * n + 1, 2)
        resting = 6 / 7 / 2
    else:
        nodes = np.empty((n, 3), dtype=object)
        for index in range(n):
            nodes[index] = [[2 * index + 1, 2 * index + 2], [], []]
        resting = 3 / 5
    terms = [ReorientationTermSpec("resting_bias", beta=np.log(3))]
    model = _model("lin", (n,), [ReorientationSpec(terms=terms)], nodes=nodes, restchannels=1, ve=ve,
                   ib=True, propagation=False)
    model.step()
    rest = model.lgca.nodes[model.lgca.nonborder][:, 2]
    at_rest = np.zeros(2 * n + 1, dtype=bool)
    at_rest[[label for channel in np.atleast_1d(rest) for label in np.atleast_1d(channel) if label]] = True
    odd, even = at_rest[1::2], at_rest[2::2]
    tolerance = 5 * np.sqrt(resting * (1 - resting) / n)
    assert abs(odd.mean() - resting) < tolerance
    assert abs(even.mean() - resting) < tolerance
    if not ve:  # independent cells
        assert abs(np.mean(odd & even) - resting ** 2) < 5 * np.sqrt(resting ** 2 / n)


@pytest.mark.parametrize("ve,growth", [(True, "ib.birthdeath"), (False, "nove_ib.birthdeath")])
def test_identity_growth_and_reorientation_run_together(ve, growth):
    model = _model("square", (10, 10), [{"name": growth}, ReorientationSpec(terms=TERMS)],
                   density=0.5 if ve else 2, restchannels=1, ve=ve, ib=True, seed=11)
    for _ in range(10):
        model.step()
    labels = [label for node in _labels_per_node(model.lgca) for label in node]
    assert len(labels) == len(set(labels)) > 0
