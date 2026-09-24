"""Reorientation restricted to a channel set, and the per-direction neighbour values behind steric repulsion."""

import numpy as np
import pytest

from lgca.lattice_state import LatticeState
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)

FAMILIES = [("classical", True, False), ("nove", False, False), ("ib", True, True), ("nove_ib", False, True)]
DIMS = (60, 60)


def _nodes(ve, identity, rng, K=5):
    if ve:
        occupied = rng.random(DIMS + (K,)) < 0.4
        return np.where(occupied, np.cumsum(occupied).reshape(occupied.shape), 0).astype(np.uint64) \
            if identity else occupied
    counts = rng.poisson(0.6, DIMS + (K,))
    if not identity:
        return counts
    nodes = np.empty(counts.shape, dtype=object)
    first = 0
    for index in np.ndindex(counts.shape):
        nodes[index] = list(range(first, first + counts[index]))
        first += counts[index]
    return nodes


def _counts(lgca, identity):
    interior = lgca.nodes[lgca.nonborder]
    if interior.dtype == object:
        return np.vectorize(len)(interior)
    return (interior > 0).astype(int) if identity else interior.astype(int)


@pytest.mark.parametrize("trait", [False, True])
@pytest.mark.parametrize("family, ve, identity", FAMILIES)
def test_only_cells_in_the_channel_set_move_and_only_among_it(family, ve, identity, trait):
    if trait and not identity:
        pytest.skip("traits need identity-based models")
    rng = np.random.default_rng(3)
    nodes = _nodes(ve, identity, rng)
    term = ReorientationTermSpec("persistent_walk", beta=2.0, trait="strength" if trait else None)
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=ve, identity_based=identity,
                        capacity=None if ve else 8, **({"traits": {"strength": 1.0}} if identity else {})),
        time=TimeSpec(steps=1, seed=2),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term],
                                                                      parameters={"channels": "velocity"})],
                                         propagation=False)))
    before = _counts(model.lgca, identity)
    if identity:
        resting = model.lgca.nodes[model.lgca.nonborder][..., 4].copy()
    model.step()
    after = _counts(model.lgca, identity)
    np.testing.assert_array_equal(after[..., 4], before[..., 4])  # resting cells stay
    np.testing.assert_array_equal(after[..., :4].sum(-1), before[..., :4].sum(-1))
    assert np.any(after[..., :4] != before[..., :4])
    if identity:  # and they are the same cells
        np.testing.assert_array_equal(model.lgca.nodes[model.lgca.nonborder][..., 4], resting)


def test_a_restricted_sampler_is_the_sampler_of_the_channel_subset():
    # classical, volume exclusion: velocity states follow exp(w . s) among states with as many cells
    nodes = np.zeros(DIMS + (5,), dtype=bool)
    nodes[..., 0] = nodes[..., 4] = True  # one moving cell (east) and one resting cell per node
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS), state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=6),
        dynamics=InteractionPipelineSpec(operators=[{"name": "persistent_walk", "parameters": {
            "beta": 1.0, "channels": "velocity"}}], propagation=False)))
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder].reshape(-1, 5)
    # the flux (1, 0) scores c_i . (1, 0): 1 east, 0 north and south, -1 west
    p = np.exp([1.0, 0.0, -1.0, 0.0])
    p /= p.sum()
    observed = after[:, :4].mean(0)
    np.testing.assert_allclose(observed, p, atol=4 * np.sqrt(0.25 / len(after)))
    assert after[:, 4].all()


@pytest.mark.parametrize("bc", ["periodic", "reflecting"])
@pytest.mark.parametrize("geometry, dims", [("lin", (9,)), ("square", (6, 7)), ("hex", (8, 8)),
                                            ("cubic", (5, 4, 6)), ("moore", (5, 4, 6))])
def test_neighbour_values_point_along_the_velocities(geometry, dims, bc):
    model = build_model(ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims, boundary=bc),
                                  state=StateSpec(density=0.5, restchannels=1)))
    lgca = model.lgca
    state = LatticeState(lgca)
    values = np.random.default_rng(0).random(dims)
    neighbours = state.neighbor_values(values)
    np.testing.assert_allclose(neighbours.sum(-1), state.neighbor_sum(values))
    # positions of the nodes, and the neighbour each velocity points to
    if geometry == "hex":
        position = np.stack([lgca.xcoords, lgca.ycoords], -1)
    else:
        position = np.stack(np.meshgrid(*[np.arange(size) for size in dims], indexing="ij"), -1).astype(float)
    flat = position.reshape(-1, position.shape[-1])
    for node in [tuple(size // 2 for size in dims), tuple(1 for _ in dims)]:
        for k in range(lgca.velocitychannels):
            target = position[node] + lgca.c[:, k]
            distance = np.linalg.norm(flat - target, axis=1)
            if distance.min() < 1e-6:
                expected = values.ravel()[distance.argmin()]
                assert neighbours[node + (k,)] == pytest.approx(expected), (node, k)


def test_steric_repulsion_scores_minus_the_cells_ahead():
    nodes = np.zeros((5, 5, 5), dtype=bool)
    nodes[3, 2, :3] = True  # three cells east of the centre
    model = build_model(ModelSpec(space=SpaceSpec(geometry="square", dims=(5, 5)),
                                  state=StateSpec(nodes=nodes, restchannels=1),
                                  dynamics=InteractionPipelineSpec(operators=[{"name": "steric_repulsion"}])))
    term = model.pipeline.operators[0].terms[0]
    term.prepare(model.lgca)
    np.testing.assert_allclose(term.weights[2, 2], [-3, 0, 0, 0, 0])


@pytest.mark.parametrize("identity", [False, True])
def test_go_or_rest_can_sense_the_neighbourhood(identity):
    # without volume exclusion every cell rests with (1 + tanh(kappa (rho - theta))) / 2, rho the mean
    # density of the node and its neighbours over the capacity
    from lgca.builtin_rules import tanh_switch

    rng = np.random.default_rng(1)
    nodes = _nodes(False, identity, rng)
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=False, identity_based=identity, capacity=4),
        time=TimeSpec(steps=1, seed=2),
        dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_rest", "parameters": {
            "kappa": 6.0, "theta": 0.6, "density": "neighbourhood"}}], propagation=False)))
    lgca = model.lgca
    state = LatticeState(lgca)
    density = state.density
    rho = (density + state.neighbor_sum(density)) / 5 / 4
    p = tanh_switch(rho, 6.0, 0.6)
    model.step()
    resting = _counts(lgca, identity)[..., 4]
    expected, variance = (density * p).sum(), (density * p * (1 - p)).sum()
    assert abs(resting.sum() - expected) < 4 * np.sqrt(variance)
    alone = (density * tanh_switch(density / 4, 6.0, 0.6)).sum()  # with the node density alone
    assert abs(resting.sum() - alone) > 4 * np.sqrt(variance)


def test_go_or_rest_switches_every_species_with_its_own_kappa():
    # without volume exclusion every cell of species s rests with tanh_switch(rho, kappa_s, theta_s)
    from lgca.builtin_rules import tanh_switch

    counts = np.random.default_rng(2).poisson(0.8, DIMS + (2, 5))
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=counts, restchannels=1, volume_exclusion=False, n_species=2, capacity=8),
        time=TimeSpec(steps=1, seed=4),
        dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_rest", "parameters": {
            "kappa": [6.0, -6.0], "theta": [0.4, 0.6]}}], propagation=False)))
    lgca = model.lgca
    cells = counts.sum(-1)
    rho = cells.sum(-1) / 8
    model.step()
    resting = lgca.nodes[lgca.nonborder][..., 4]
    for species, (kappa, theta) in enumerate(((6.0, 0.4), (-6.0, 0.6))):
        p = tanh_switch(rho, kappa, theta)
        expected, variance = (cells[..., species] * p).sum(), (cells[..., species] * p * (1 - p)).sum()
        assert abs(resting[..., species].sum() - expected) < 4 * np.sqrt(variance)
    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder].sum(-1), cells)  # species keep their cells


def test_directed_motion_follows_a_given_vector_field():
    nodes = np.zeros(DIMS + (4,), dtype=bool)
    nodes[..., 1] = True  # one cell per node, moving north
    field = np.zeros(DIMS + (2,))
    field[..., 0] = 1.0  # a flow to the east
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=0, fields={"flow": field}),
        time=TimeSpec(steps=1, seed=5),
        dynamics=InteractionPipelineSpec(operators=[{"name": "directed_motion", "parameters": {
            "beta": 1.5, "field": "flow"}}], propagation=False)))
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder].reshape(-1, 4)
    p = np.exp(1.5 * np.array([1.0, 0.0, -1.0, 0.0]))
    p /= p.sum()
    np.testing.assert_allclose(after.mean(0), p, atol=4 * np.sqrt(0.25 / len(after)))


def _two_species(ve, operators, seed=3):
    rng = np.random.default_rng(seed)
    nodes = rng.random(DIMS + (2, 5)) < 0.3 if ve else rng.poisson(0.5, DIMS + (2, 5))
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=1, n_species=2, volume_exclusion=ve, capacity=None if ve else 8,
                        fields={"signal": np.add.outer(np.arange(DIMS[0]), np.zeros(DIMS[1])) / 10}),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False)))


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("operator", [
    {"name": "chemotaxis", "parameters": {"beta": 2.0, "field": "signal", "species": 0}},
    {"name": "random_walk", "parameters": {"species": 0}},
    ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=2.0)], parameters={"species": [0]}),
    {"name": "go_or_rest", "parameters": {"kappa": 5.0, "theta": 0.2, "species": 0}},
    {"name": "birth_death", "parameters": {"birth_rate": 0.5, "death_rate": 0.2, "species": 0}},
], ids=["chemotaxis", "random_walk", "spec", "go_or_rest", "birth_death"])
def test_rules_for_one_species_leave_the_other_species_as_they_are(ve, operator):
    model = _two_species(ve, [operator])
    before = model.lgca.nodes[model.lgca.nonborder].copy()
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]
    np.testing.assert_array_equal(after[..., 1, :], before[..., 1, :])
    assert np.any(after[..., 0, :] != before[..., 0, :])


@pytest.mark.parametrize("ve", [True, False])
def test_a_cue_senses_the_chosen_species(ve):
    # polar alignment of species 0 with the cells of species 1: the neighbours' flux of species 1 only
    model = _two_species(ve, [{"name": "polar_alignment", "parameters": {"species": 0, "sensed_species": 1}}])
    lgca = model.lgca
    term = model.pipeline.operators[0].terms[0]
    term.prepare(lgca)
    state = LatticeState(lgca)
    flux = state.counts[..., 1, :4] @ lgca.c.T
    np.testing.assert_allclose(term.weights[..., :4], state.neighbor_sum(flux) @ lgca.c)
    all_species = _two_species(ve, [{"name": "polar_alignment"}]).pipeline.operators[0].terms[0]
    all_species.prepare(lgca)
    assert not np.allclose(all_species.weights, term.weights)


def test_species_that_do_not_exist_are_refused():
    for operator in ({"name": "polar_alignment", "parameters": {"species": 2}},
                     {"name": "polar_alignment", "parameters": {"sensed_species": [0, 2]}},
                     {"name": "birth_death", "parameters": {"birth_rate": 0.1, "species": 2}}):
        with pytest.raises(ValueError, match="species"):
            _two_species(True, [operator]).step()
