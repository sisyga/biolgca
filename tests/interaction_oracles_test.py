"""Interaction rules produce the transition probabilities of their model definition.

Each test prepares a lattice on which many nodes see the same neighbourhood,
applies one interaction step without propagation, and compares the measured
channel frequencies with probabilities computed here from the rule's
definition. Every implementation of a rule is checked against the same oracle:
the legacy interaction function, its translation to rules (the prefixed name) and, where one
exists, the composed reorientation term.
"""

import numpy as np
import pytest

from tests.legacy import legacy_lgca

# the prefixed names are the translations of the legacy names, which warn as deprecated
pytestmark = pytest.mark.filterwarnings("ignore:The interaction name:FutureWarning")
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec


E_X = np.array([1.0, 0.0])
SQUARE_C = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)  # channel velocities, independent of lgca.c


def _boltzmann(scores):
    weights = np.exp(np.asarray(scores, dtype=float))
    return weights / weights.sum()


def _legacy_states(geometry, nodes, interaction, repeats, ve=True, **params):
    lgca = legacy_lgca(geometry=geometry, nodes=nodes, ve=ve, interaction=interaction, seed=0, **params)
    start = lgca.nodes.copy()
    states = []
    for _ in range(repeats):
        lgca.nodes = start.copy()
        lgca.update_dynamic_fields()
        lgca.interaction(lgca)
        states.append(np.asarray(lgca.nodes[lgca.nonborder]))
    return np.stack(states)


def _model_spec_states(geometry, nodes, operator, repeats, ve=True, velocitychannels=4, fields=None,
                       capacity=None):
    compiled = build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=nodes.shape[:-1], boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=nodes.shape[-1] - velocitychannels,
                        volume_exclusion=ve, fields=fields or {}, capacity=capacity),
        time=TimeSpec(steps=1, seed=0),
        dynamics=InteractionPipelineSpec(operators=[operator], propagation=False),
    ))
    lgca = compiled.lgca
    start = lgca.nodes.copy()
    states = []
    for _ in range(repeats):
        lgca.nodes = start.copy()
        lgca.update_dynamic_fields()
        compiled.step()
        states.append(np.asarray(lgca.nodes[lgca.nonborder]))
    return np.stack(states)


def _states(entry, geometry, nodes, repeats, *, legacy, plugin, composed=None, ve=True, velocitychannels=4,
            fields=None, capacity=None):
    if entry == "legacy":
        interaction, params = legacy
        return _legacy_states(geometry, nodes, interaction, repeats, ve=ve, **params)
    operator = plugin if entry == "plugin" else composed
    return _model_spec_states(geometry, nodes, operator, repeats, ve=ve, velocitychannels=velocitychannels,
                              fields=fields, capacity=capacity)


def _assert_channel_frequencies(occupancy, expected, particles_per_node=1):
    """Compare the fraction of particles found in each channel with the expected probabilities."""
    samples = occupancy.reshape(-1, occupancy.shape[-1]).astype(float)
    trials = samples.shape[0] * particles_per_node
    observed = samples.sum(axis=0) / trials
    tolerance = 4 * np.sqrt(expected * (1 - expected) / trials) + 1e-12
    assert np.all(np.abs(observed - expected) <= tolerance), (observed, expected, tolerance)


def _uniform_right_movers(dims=(10, 10), restchannels=0):
    nodes = np.zeros(dims + (4 + restchannels,), dtype=bool)
    nodes[..., 0] = True
    return nodes


REORIENTATION_ENTRIES = ["legacy", "plugin", "composed"]


@pytest.mark.parametrize("entry", REORIENTATION_ENTRIES)
def test_polar_alignment_follows_the_boltzmann_weight_of_the_neighbour_flux(entry):
    beta = 0.3
    states = _states(entry, "square", _uniform_right_movers(), 30,
                     legacy=("alignment", dict(beta=beta)),
                     plugin={"name": "classical.alignment", "parameters": {"beta": beta}},
                     composed=ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=beta)]))

    # Four neighbours, each moving along +x: neighbour flux G = (4, 0).
    _assert_channel_frequencies(states, _boltzmann(beta * SQUARE_C @ (4 * E_X)))


@pytest.mark.parametrize("entry", REORIENTATION_ENTRIES)
def test_persistent_walk_follows_the_boltzmann_weight_of_the_own_flux(entry):
    beta = 1.0
    states = _states(entry, "square", _uniform_right_movers(), 30,
                     legacy=("persistent_motion", dict(beta=beta)),
                     plugin={"name": "classical.persistent_walk", "parameters": {"beta": beta}},
                     composed=ReorientationSpec(terms=[ReorientationTermSpec("persistent_walk", beta=beta)]))

    _assert_channel_frequencies(states, _boltzmann(beta * SQUARE_C @ E_X))


@pytest.mark.parametrize("entry", REORIENTATION_ENTRIES)
def test_nematic_alignment_favours_the_neighbour_axis_in_both_directions(entry):
    beta = 0.3
    states = _states(entry, "square", _uniform_right_movers(), 30,
                     legacy=("nematic", dict(beta=beta)),
                     plugin={"name": "classical.nematic", "parameters": {"beta": beta}},
                     composed=ReorientationSpec(terms=[ReorientationTermSpec("nematic_alignment", beta=beta)]))

    # Score 4 (c_k . e_x)^2 up to a constant: channels along x are favoured, both directions equally.
    _assert_channel_frequencies(states, _boltzmann(beta * 4 * (SQUARE_C @ E_X) ** 2))


@pytest.mark.parametrize("entry", REORIENTATION_ENTRIES)
def test_chemotaxis_follows_the_boltzmann_weight_of_the_signal_gradient(entry):
    beta = 1.0
    nodes = _uniform_right_movers()
    padded_gradient = np.zeros((12, 12, 2))
    padded_gradient[..., 0] = 1.0
    signal = np.repeat(np.arange(10, dtype=float)[:, None], 10, axis=1)  # slope 1 along x
    states = _states(entry, "square", nodes, 30,
                     legacy=("chemotaxis", dict(beta=beta, gradient=padded_gradient)),
                     plugin={"name": "classical.chemotaxis",
                             "parameters": {"beta": beta, "gradient": padded_gradient[1:-1, 1:-1]}},
                     composed=ReorientationSpec(terms=[ReorientationTermSpec("chemotaxis", beta=beta,
                                                                             parameters={"field": "signal"})]),
                     fields={"signal": signal})

    _assert_channel_frequencies(states, _boltzmann(beta * SQUARE_C @ E_X))


@pytest.mark.parametrize("entry", REORIENTATION_ENTRIES)
def test_aggregation_follows_the_boltzmann_weight_of_the_density_gradient(entry):
    beta = 1.0
    nodes = np.zeros((12, 6, 4), dtype=bool)
    nodes[1::3, :, 1] = True  # sampled column: one particle per node
    nodes[2::3, :, :2] = True  # its right neighbour holds two particles, its left neighbour none
    states = _states(entry, "square", nodes, 60,
                     legacy=("aggregation", dict(beta=beta)),
                     plugin={"name": "classical.aggregation", "parameters": {"beta": beta}},
                     composed=ReorientationSpec(terms=[ReorientationTermSpec("aggregation", beta=beta)]))

    # Density gradient: (n(x + 1) - n(x - 1)) / 2 = (2 - 0) / 2 along x.
    _assert_channel_frequencies(states[:, 1::3], _boltzmann(beta * SQUARE_C @ E_X))


@pytest.mark.parametrize("entry", REORIENTATION_ENTRIES)
def test_random_walk_places_a_particle_uniformly_in_all_channels(entry):
    states = _states(entry, "square", _uniform_right_movers(restchannels=1), 30,
                     legacy=("random_walk", {}),
                     plugin={"name": "classical.random_walk"},
                     composed=ReorientationSpec(terms=[ReorientationTermSpec("random_walk")]))

    _assert_channel_frequencies(states, np.full(5, 1 / 5))


@pytest.mark.parametrize("entry", ["legacy", "plugin"])
def test_birth_fills_empty_channels_in_proportion_to_local_density(entry):
    r_b = 0.5
    nodes = np.zeros((10, 10, 5), dtype=bool)
    nodes[::2, :, 0] = True  # occupied columns with one particle; the other columns stay empty
    states = _states(entry, "square", nodes, 30, legacy=("birth", dict(r_b=r_b)),
                     plugin={"name": "classical.birth", "parameters": {"r_b": r_b}})

    assert not states[:, 1::2].any(), "empty nodes must not give birth"
    births = states[:, ::2].sum(axis=-1) - 1
    # Each of the four empty channels is filled with probability r_b * n / K = r_b / 5.
    trials = 4 * births.size
    p = r_b / 5
    assert abs(births.sum() - trials * p) < 4 * np.sqrt(trials * p * (1 - p))


@pytest.mark.parametrize("entry", ["legacy", "plugin"])
@pytest.mark.parametrize("theta,r_d,expected_moving,expected_resting", [
    (0.0, 0.0, 0, 2),  # density above threshold: both cells switch to rest
    (1.0, 0.0, 2, 0),  # density below threshold: both cells keep moving
    (0.0, 1.0, 0, 0),  # certain death removes resting cells
    (1.0, 1.0, 0, 0),  # and moving cells
])
def test_go_or_grow_switches_by_density_threshold_and_removes_dead_cells(entry, theta, r_d, expected_moving,
                                                                         expected_resting):
    nodes = np.zeros((6, 6, 6), dtype=bool)
    nodes[..., :2] = True  # two moving cells, two free rest channels
    params = dict(r_b=0.0, r_d=r_d, kappa=50.0, theta=theta)
    states = _states(entry, "square", nodes, 3, legacy=("go_or_grow", params),
                     plugin={"name": "classical.go_or_grow", "parameters": params})

    assert np.all(states[..., :4].sum(axis=-1) == expected_moving)
    assert np.all(states[..., 4:].sum(axis=-1) == expected_resting)


@pytest.mark.parametrize("entry", ["legacy", "plugin"])
def test_nove_density_dependent_alignment_follows_the_neighbour_flux(entry):
    beta = 0.1
    nodes = np.zeros((10, 10, 4), dtype=np.uint)
    nodes[..., 0] = 3
    states = _states(entry, "square", nodes, 30, ve=False,
                     legacy=("dd_alignment", dict(beta=beta, restchannels=0)),
                     plugin={"name": "nove.dd_alignment", "parameters": {"beta": beta}})

    # Neighbour flux G = 4 nodes x 3 particles along +x = (12, 0).
    _assert_channel_frequencies(states, _boltzmann(beta * SQUARE_C @ (12 * E_X)), particles_per_node=3)


@pytest.mark.parametrize("entry", ["legacy", "plugin"])
def test_nove_random_walk_spreads_particles_over_all_channels(entry):
    nodes = np.zeros((10, 10, 5), dtype=np.uint)
    nodes[..., 0] = 5
    states = _states(entry, "square", nodes, 30, ve=False, legacy=("random_walk", dict(restchannels=1)),
                     plugin={"name": "nove.random_walk"})

    _assert_channel_frequencies(states, np.full(5, 1 / 5), particles_per_node=5)


@pytest.mark.parametrize("entry", ["legacy", "plugin"])
@pytest.mark.parametrize("cells,capacity", [(8, 8), (4, 8)])
def test_nove_go_or_grow_birth_is_logistic_in_node_density(entry, cells, capacity):
    r_b = 0.6
    nodes = np.zeros((10, 10, 5), dtype=np.uint)
    nodes[..., 4] = cells  # all cells resting; the steep switch keeps them resting
    params = dict(r_b=r_b, r_d=0.0, kappa=50.0, theta=0.0)
    states = _states(entry, "square", nodes, 20, ve=False, capacity=capacity,
                     legacy=("go_or_grow", dict(params, restchannels=1, capacity=capacity)),
                     plugin={"name": "nove.go_or_grow", "parameters": params})

    births = states.sum(axis=-1) - cells
    trials = cells * births.size
    p = r_b * (1 - cells / capacity)
    assert abs(births.sum() - trials * p) <= 4 * np.sqrt(trials * p * (1 - p))
