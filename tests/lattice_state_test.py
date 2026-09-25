"""LatticeState operations mean the same per cell in every classical model family."""

import warnings

import numpy as np
import pytest

from lgca import get_lgca
from lgca.lattice_state import LatticeState

DIMS = {"lin": 12, "square": (6, 4), "hex": (6, 4), "cubic": (4, 3, 3), "moore": (3, 3, 3)}


def _model(geometry="square", ve=True, n_species=1, density=0.6, seed=1, bc="periodic", **kwargs):
    arguments = dict(geometry=geometry, dims=DIMS[geometry], ve=ve, density=density, seed=seed,
                     restchannels=1, bc=bc, **kwargs)
    if n_species > 1:
        arguments["n_species"] = n_species
    if not ve:
        arguments["interaction"] = "only_propagation"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_lgca(**arguments)


def _nodes(ve, n_species, dims, channels, value):
    """A lattice filled with ``value`` cells per channel and species."""
    shape = tuple(dims) + ((n_species,) if n_species > 1 else ()) + (channels,)
    return np.full(shape, value, dtype=bool if ve else int)


MATRIX = [(geometry, ve, n_species) for geometry in DIMS for ve in (True, False) for n_species in (1, 2)]


def _every_operation(state):
    state.remove_cells(0.3)
    state.divide_cells(0.5, channels="rest")
    state.add_cells(1, channels="velocity")
    state.shuffle_cells("velocity")
    if state.n_species > 1:
        state.switch_phenotype(np.full((2, 2), 0.4))
        state.switch_phenotype(np.full((2, 2), 0.4), channels="rest")


@pytest.mark.parametrize("geometry, ve, n_species", MATRIX)
def test_operations_keep_shapes_types_and_ghost_nodes(geometry, ve, n_species):
    lgca = _model(geometry, ve, n_species)
    shape, dtype = lgca.nodes.shape, lgca.nodes.dtype
    ghosts = np.ones(lgca.nodes.shape[:len(lgca.dims)], dtype=bool)
    ghosts[lgca.nonborder] = False
    before = lgca.nodes[ghosts].copy()

    state = LatticeState(lgca)
    assert state.counts.shape == tuple(lgca.dims) + (n_species, lgca.K)
    _every_operation(state)
    state.commit()

    assert lgca.nodes.shape == shape and lgca.nodes.dtype == dtype
    np.testing.assert_array_equal(lgca.nodes[ghosts], before)
    assert state.counts.min() >= 0
    if ve:
        assert state.counts.max() <= 1
    np.testing.assert_array_equal(lgca.cell_density[lgca.nonborder], state.density)


@pytest.mark.parametrize("geometry, ve, n_species", MATRIX)
def test_the_same_seed_gives_the_same_state(geometry, ve, n_species):
    results = []
    for _ in range(2):
        state = LatticeState(_model(geometry, ve, n_species, seed=5))
        _every_operation(state)
        results.append(state.counts.copy())

    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("geometry, ve, n_species", MATRIX)
def test_reorientations_and_switches_keep_what_their_kind_conserves(geometry, ve, n_species):
    lgca = _model(geometry, ve, n_species)
    state = LatticeState(lgca, kind="reorientation")
    species_density = state.species_density

    state.shuffle_cells()
    state.shuffle_cells("velocity", species=0)
    np.testing.assert_array_equal(state.species_density, species_density)
    state.commit()

    if n_species > 1:
        state = LatticeState(lgca, kind="phenotype_switch")
        density = state.density
        state.switch_phenotype([[0, 0.5], [0.5, 0]])
        state.switch_phenotype([[0, 0.5], [0.5, 0]], channels="all")
        np.testing.assert_array_equal(state.density, density)
        state.commit()


@pytest.mark.parametrize("kind, operation", [
    ("reorientation", lambda state: state.remove_cells(1.0)),
    ("phenotype_switch", lambda state: state.add_cells(1)),
])
def test_commit_rejects_a_state_that_breaks_its_conservation_law(kind, operation):
    state = LatticeState(_model(density=2), kind=kind)
    operation(state)

    with pytest.raises(ValueError, match=f"a {kind} must keep"):
        state.commit()


@pytest.mark.parametrize("geometry", ["lin", "square", "hex", "cubic"])
@pytest.mark.parametrize("bc", ["periodic", "reflecting", "absorbing"])
def test_neighbor_sum_matches_the_model_after_its_boundary_conditions(geometry, bc):
    lgca = _model(geometry, bc=bc)
    lgca.apply_boundaries()
    lgca.update_dynamic_fields()
    state = LatticeState(lgca)

    expected = lgca.nb_sum(lgca.cell_density)[lgca.nonborder]
    np.testing.assert_array_equal(state.neighbor_sum(state.density), expected)
    assert state.neighbor_sum(state.flux).shape == state.flux.shape


def test_neighbor_sum_wraps_only_across_periodic_boundaries():
    values = np.zeros(DIMS["square"])
    values[0, 0] = 1

    periodic = LatticeState(_model(bc="periodic")).neighbor_sum(values)
    reflecting = LatticeState(_model(bc="reflecting")).neighbor_sum(values)

    assert periodic[-1, 0] == periodic[0, -1] == 1
    assert reflecting[-1, 0] == reflecting[0, -1] == 0
    assert reflecting[1, 0] == reflecting[0, 1] == 1


def test_cells_without_volume_exclusion_die_one_by_one():
    # 5 cells per channel dying with probability 0.5: variance 5 * 0.25, not 25 * 0.25
    lgca = _model(ve=False, nodes=_nodes(False, 1, (60, 50), 5, 5))
    state = LatticeState(lgca)

    state.remove_cells(0.5)
    removed = 5 - state.counts

    assert abs(removed.mean() - 2.5) < 0.05
    assert abs(removed.var() - 1.25) < 0.1


@pytest.mark.parametrize("ve", [True, False])
def test_divisions_happen_at_the_given_rate_into_the_given_channels(ve):
    nodes = _nodes(ve, 1, (60, 50), 5, 0)
    nodes[..., :4] = 1  # velocity channels full, rest channel empty
    state = LatticeState(_model(ve=ve, nodes=nodes))

    added = state.divide_cells(0.1, channels="rest")

    if ve:  # at most one daughter fits into the single rest channel
        assert abs(added.mean() - (1 - 0.9 ** 4)) < 0.02
    else:
        assert abs(added.mean() - 0.4) < 0.03
    np.testing.assert_array_equal(state.counts[..., 0, :4], 1)
    np.testing.assert_array_equal(state.counts[..., 0, 4], added[..., 0])


def test_daughters_join_their_mothers_channel_only_without_volume_exclusion():
    state = LatticeState(_model(ve=False, nodes=_nodes(False, 1, (6, 4), 5, 2)))
    before = state.counts.copy()

    state.divide_cells(1.0, channels="same")

    np.testing.assert_array_equal(state.counts, 2 * before)
    with pytest.raises(ValueError, match="same"):
        LatticeState(_model()).divide_cells(0.5, channels="same")


def test_capacity_is_a_crowding_scale_and_not_a_limit():
    state = LatticeState(_model(nodes=_nodes(True, 1, (6, 4), 5, 0)), capacity=2)

    state.add_cells(3)

    assert state.capacity == 2
    np.testing.assert_array_equal(state.density, 3)


def test_cells_are_only_added_to_free_channels_of_their_species():
    nodes = _nodes(True, 2, (6, 4), 5, 0)
    nodes[..., 0, :] = True
    state = LatticeState(_model(n_species=2, nodes=nodes))

    added = state.add_cells(np.array([2, 7]))

    np.testing.assert_array_equal(added[..., 0], 0)
    np.testing.assert_array_equal(added[..., 1], 5)


@pytest.mark.parametrize("ve", [True, False])
def test_cells_switch_species_at_the_given_rates(ve):
    nodes = _nodes(ve, 3, (60, 50), 5, 0)
    nodes[..., 0, :] = 1
    state = LatticeState(_model(ve=ve, n_species=3, nodes=nodes), kind="phenotype_switch")

    switched = state.switch_phenotype([[0, 0.2, 0.3], [0, 0, 0], [0, 0, 0]])

    total = nodes[..., 0, :].sum()
    assert abs(switched[..., 0, 1].sum() / total - 0.2) < 0.01
    assert abs(switched[..., 0, 2].sum() / total - 0.3) < 0.01
    # channels="same": every cell kept its channel
    np.testing.assert_array_equal(state.counts.sum(axis=-2), nodes.sum(axis=-2))
    state.commit()


def test_a_switch_into_an_occupied_channel_is_rejected_with_volume_exclusion():
    nodes = _nodes(True, 2, (6, 4), 5, 1)
    state = LatticeState(_model(n_species=2, nodes=nodes))

    switched = state.switch_phenotype([[0, 1], [1, 0]])

    assert switched.sum() == 0
    np.testing.assert_array_equal(state.counts, nodes)


SWITCH_DIMS = {"lin": (30000,), "square": (180, 180), "hex": (180, 180), "cubic": (32, 32, 32),
               "moore": (40, 40, 40)}


@pytest.mark.parametrize("geometry", list(SWITCH_DIMS))
def test_a_switch_succeeds_if_its_random_target_channel_is_free(geometry):
    """One switching cell into a species with n cells in C channels succeeds with probability 1 - n/C."""
    rng = np.random.default_rng(3)
    dims = SWITCH_DIMS[geometry]
    channels = LatticeState(_model(geometry, n_species=2)).K
    nodes = np.zeros(dims + (2, channels), dtype=bool)
    nodes[..., 0, 0] = True
    level = np.arange(np.prod(dims)).reshape(dims) % (channels + 1)
    keys = rng.random(dims + (channels,))
    nodes[..., 1, :] = np.argsort(np.argsort(keys, axis=-1), axis=-1) < level[..., None]
    state = LatticeState(_model(geometry, n_species=2, nodes=nodes, seed=4))

    switched = state.switch_phenotype([[0, 1], [0, 0]], channels="all")[..., 0, 1]

    for n in range(channels + 1):
        success = 1 - n / channels
        trials = (level == n).sum()
        error = np.sqrt(success * (1 - success) / trials)
        assert abs(switched[level == n].mean() - success) <= 5 * error + 1e-12, (n, channels)
    assert state.counts.max() <= 1
    np.testing.assert_array_equal(state.density, nodes.sum(axis=(-2, -1)))
    # the cells that switched sit on channels that were free for species 1
    np.testing.assert_array_equal(state.counts[..., 1, :] & nodes[..., 1, :], nodes[..., 1, :])


@pytest.mark.parametrize("channels, targets", [("all", 5), ("velocity", 4), ("rest", 1)])
def test_more_switchers_than_channels_aim_at_every_channel_once(channels, targets):
    """With at least as many switchers as channels, exactly the free channels of the set are filled."""
    nodes = _nodes(True, 2, (40, 40), 5, 0)
    nodes[..., 0, :] = True
    nodes[:20, :, 1, 0] = True  # half of the nodes: one velocity channel of species 1 is occupied
    state = LatticeState(_model(n_species=2, nodes=nodes, seed=2))

    switched = state.switch_phenotype([[0, 1], [0, 0]], channels=channels)[..., 0, 1]

    blocked = 1 if channels in ("all", "velocity") else 0
    np.testing.assert_array_equal(switched[:20], targets - blocked)
    np.testing.assert_array_equal(switched[20:], targets)
    assert state.counts.max() <= 1
    np.testing.assert_array_equal(state.density, nodes.sum(axis=(-2, -1)))
    if targets < 5:  # which switchers succeed is random
        assert len(np.unique(state.counts[20:, :, 0, :].reshape(-1, 5), axis=0)) > 1


@pytest.mark.parametrize("channels", ["same", "all"])
def test_a_channel_vacated_by_a_switch_is_not_free_in_the_same_step(channels):
    nodes = _nodes(True, 2, (20, 20), 5, 1)
    state = LatticeState(_model(n_species=2, nodes=nodes))

    switched = state.switch_phenotype([[0, 1], [1, 0]], channels=channels)

    assert switched.sum() == 0
    np.testing.assert_array_equal(state.counts, nodes)


def test_competing_switches_fill_each_free_channel_once():
    nodes = _nodes(True, 3, (40, 40), 5, 0)
    nodes[..., :2, :] = True  # species 0 and 1 both want the free channels of species 2
    state = LatticeState(_model(n_species=3, nodes=nodes))

    for channels in ("same", "rest"):
        before = state.counts.copy()
        state.switch_phenotype([[0, 0, 1], [0, 0, 1], [0, 0, 0]], channels=channels)
        assert state.counts.max() <= 1
        np.testing.assert_array_equal(state.density, before.sum(axis=(-2, -1)))


def test_shuffling_moves_only_the_selected_cells_within_their_set():
    nodes = _nodes(True, 2, (20, 20), 5, 0)
    nodes[..., 0, 0] = nodes[..., 1, 4] = True
    state = LatticeState(_model(n_species=2, nodes=nodes), kind="reorientation")

    state.shuffle_cells("velocity", species=0)

    np.testing.assert_array_equal(state.counts[..., 1, :], nodes[..., 1, :])
    np.testing.assert_array_equal(state.counts[..., 0, 4], 0)
    moved = state.counts[..., 0, :4].argmax(-1)
    assert set(np.unique(moved)) == {0, 1, 2, 3}


@pytest.mark.parametrize("value, message", [
    (np.full((6, 4, 1, 5), 2), "at most one cell"),
    (np.full((6, 4, 1, 5), -1), "non-negative"),
    (np.full((6, 4, 1, 5), 0.5), "integers"),
    (np.zeros((6, 4, 5, 1)), "shape"),
])
def test_replacing_the_counts_is_checked(value, message):
    state = LatticeState(_model())

    with pytest.raises(ValueError, match=message):
        state.counts = value


def test_counts_are_read_only_but_can_be_replaced():
    state = LatticeState(_model())
    with pytest.raises(ValueError):
        state.counts[0, 0, 0, 0] = 1

    new = np.zeros(state.counts.shape[:-2] + (state.K,), dtype=bool)
    new[..., -1] = True
    state.counts = new

    np.testing.assert_array_equal(state.density, 1)


@pytest.mark.parametrize("p", [0.1, np.full(DIMS["square"], 0.1), [0.1, 0.2],
                               np.full(DIMS["square"] + (2,), 0.1), np.full(DIMS["square"] + (2, 5), 0.1)])
def test_probabilities_broadcast_per_node_species_or_channel(p):
    LatticeState(_model(n_species=2)).remove_cells(p)


@pytest.mark.parametrize("p, message", [([0.1, 0.2, 0.3], "shape"), (1.5, "between 0 and 1")])
def test_invalid_probabilities_are_explained(p, message):
    with pytest.raises(ValueError, match=message):
        LatticeState(_model(n_species=2)).remove_cells(p)


def _identity_model(ve, nodes, seed=1):
    from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
    from lgca.pipeline import InteractionPipelineSpec

    dims = nodes.shape[:-1]
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="lin" if len(dims) == 1 else "square", dims=dims),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=ve, identity_based=True),
        time=TimeSpec(steps=1, seed=seed), dynamics=InteractionPipelineSpec(operators=[]))).lgca


@pytest.mark.parametrize("ve", [True, False])
def test_identity_based_counts_can_be_read_but_not_assigned(ve):
    lgca = get_lgca(geometry="square", dims=(4, 4), ib=True, ve=ve, density=2 if not ve else 0.5,
                    restchannels=1, seed=1)
    state = LatticeState(lgca)
    counts = lgca._channel_counts(lgca.nodes[lgca.nonborder])
    np.testing.assert_array_equal(state.counts[..., 0, :], counts)
    assert state.volume_exclusion == ve
    np.testing.assert_allclose(state.flux, lgca.calc_flux(lgca.nodes)[lgca.nonborder])
    with pytest.raises(TypeError, match="can be read but not assigned"):
        state.counts = state.counts
    with pytest.raises(TypeError, match="new cells need traits"):
        state.add_cells(1)


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("geometry", list(DIMS))
def test_identity_shuffle_moves_labels_and_keeps_other_channels(geometry, ve):
    classical = _model(geometry, ve, 1, density=0.4 if ve else 2, seed=4)
    counts = classical.nodes[classical.nonborder]
    if ve:
        labels = counts.astype(np.uint64)
        labels[counts] = np.arange(1, counts.sum() + 1)
    else:
        labels = counts.astype(np.int64)
    lgca = get_lgca(geometry=geometry, dims=classical.dims, nodes=labels, ib=True, ve=ve,
                    restchannels=classical.restchannels, seed=9)
    classical.rng = np.random.default_rng(9)
    before = lgca.nodes[lgca.nonborder].copy()
    state = LatticeState(lgca, kind="reorientation")
    state.shuffle_cells("velocity")
    state.commit()
    expected = LatticeState(classical, kind="reorientation")
    expected.shuffle_cells("velocity")  # the same random numbers for the cell numbers
    after = lgca.nodes[lgca.nonborder]
    np.testing.assert_array_equal(state.counts, expected.counts)
    velocity = lgca.velocitychannels
    rest_before, rest_after = before[..., velocity:], after[..., velocity:]
    assert rest_before.tolist() == rest_after.tolist()  # resting cells keep label and channel
    for old, new in zip(before.reshape(-1, lgca.K), after.reshape(-1, lgca.K)):
        if ve:
            assert sorted(old[old > 0]) == sorted(new[new > 0])
        else:
            assert sorted(x for c in old for x in c) == sorted(x for c in new for x in c)


@pytest.mark.parametrize("ve", [True, False])
def test_identity_shuffle_places_labels_at_random(ve):
    # every node: cells 2k+1 and 2k+2 in velocity channel 0, cell 3k in the rest channel (lin)
    n = 10000
    if ve:
        nodes = np.zeros((n, 3), dtype=np.uint64)
        nodes[:, 0], nodes[:, 1] = np.arange(1, 2 * n, 2), np.arange(2, 2 * n + 1, 2)
    else:
        nodes = np.empty((n, 3), dtype=object)
        for index in range(n):
            nodes[index] = [[2 * index + 1, 2 * index + 2], [], []]
    lgca = _identity_model(ve, nodes)
    state = LatticeState(lgca, kind="reorientation")
    state.shuffle_cells("velocity")
    state.commit()
    first = lgca.nodes[lgca.nonborder][:, 0]
    in_first = np.zeros(2 * n + 1, dtype=bool)
    in_first[[label for channel in first for label in np.atleast_1d(channel) if label]] = True
    odd, even = in_first[1::2], in_first[2::2]
    tolerance = 5 * np.sqrt(0.25 / n)
    # with volume exclusion the cell numbers cannot change here; only the labels move
    assert abs(odd.mean() - 0.5) < tolerance and abs(even.mean() - 0.5) < tolerance
    if ve:
        assert np.all(odd != even)
    else:  # independent cells
        assert abs(np.mean(odd & even) - 0.25) < 5 * np.sqrt(0.25 * 0.75 / n)


def _field_model(boundary):
    from lgca.model import (
        Description,
        ModelSpec,
        SpaceSpec,
        StateSpec,
        TimeSpec,
        build_model,
    )

    signal = np.add.outer(np.arange(6.0), np.zeros(4))  # rises by 1 per node along x
    return build_model(ModelSpec(
        description=Description(title="field access"),
        space=SpaceSpec(geometry="square", dims=(6, 4), boundary=boundary),
        state=StateSpec(density=0.5, fields={"signal": signal}),
        time=TimeSpec(steps=1, seed=1),
    )).lgca, signal


def test_fields_of_a_model_spec_are_available_without_ghost_nodes():
    lgca, signal = _field_model("reflecting")
    state = LatticeState(lgca)

    np.testing.assert_array_equal(state.field("signal"), signal)
    with pytest.raises(KeyError, match="StateSpec.fields"):
        state.field("missing")


def test_gradients_are_centred_and_take_boundary_values_from_ghost_nodes():
    lgca, signal = _field_model("reflecting")
    state = LatticeState(lgca)

    # a named field uses its stored ghost nodes (edge values): half the slope at the walls
    np.testing.assert_allclose(state.gradient("signal")[..., 0], [[0.5] * 4] + [[1.0] * 4] * 4 + [[0.5] * 4])
    np.testing.assert_allclose(state.gradient("signal")[..., 1], 0)
    # an array gets the ghost values of the cells, zero beyond a reflecting wall
    np.testing.assert_allclose(state.gradient(signal)[0, :, 0], 0.5)
    np.testing.assert_allclose(state.gradient(signal)[-1, :, 0], -2.0)


@pytest.mark.parametrize("geometry", ["lin", "square", "hex", "cubic"])
@pytest.mark.parametrize("bc", ["periodic", "reflecting"])
def test_gradient_of_the_density_matches_the_model_after_its_boundary_conditions(geometry, bc):
    lgca = _model(geometry, bc=bc, density=1.5)
    lgca.apply_boundaries()
    lgca.update_dynamic_fields()
    state = LatticeState(lgca)

    expected = lgca.gradient(lgca.cell_density)[lgca.nonborder]
    np.testing.assert_allclose(state.gradient(state.density), expected)


@pytest.mark.parametrize("channels, cells", [(5, 2), (7, 3), (27, 3), (27, 13)])
def test_random_occupancy_places_the_cells_uniformly(channels, cells):
    # 5 and 7 channels: one table of all states; 27 channels with 3 cells: the table of that
    # number of cells; with 13 cells too many states to enumerate: ranked random keys
    from math import comb

    from lgca.lattice_state import _enumerable, _state_table, random_occupancy

    assert (_state_table(channels) is not None) == (channels < 27)
    assert _enumerable(channels, cells) == (cells != 13)
    rng = np.random.default_rng(0)
    draws = 40000
    number = np.full(draws, cells)
    number[::7] = 1  # entries with another number of cells in the same call
    placed = random_occupancy(rng, number, channels)
    np.testing.assert_array_equal(placed.sum(-1), number)
    chosen = placed[number == cells]
    frequency = chosen.mean(0)  # every channel is occupied with probability cells / channels
    p = cells / channels
    np.testing.assert_allclose(frequency, p, atol=4 * np.sqrt(p * (1 - p) / len(chosen)))
    if comb(channels, cells) <= 35:  # every state equally often
        states, counts = np.unique(chosen, axis=0, return_counts=True)
        assert len(states) == comb(channels, cells)
        expected = len(chosen) / len(states)
        assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected))
