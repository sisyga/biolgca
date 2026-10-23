"""Compact integration tests for the core LGCA invariants."""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.simulation import (
    DensityRecorder,
    NodeRecorder,
    PopulationRecorder,
    ScalarTimeSeriesRecorder,
    SimulationRunner,
)
from tests.combinations import pairwise


FAMILIES = {
    "classical_ve": {"ve": True, "ib": False},
    "identity_ve": {"ve": True, "ib": True},
    "classical_nove": {"ve": False, "ib": False},
    "identity_nove": {"ve": False, "ib": True},
    "multispecies_ve": {"ve": True, "n_species": 2},
    "multispecies_nove": {"ve": False, "n_species": 2},
}

GEOMETRIES = {
    "lin": (3,),
    "square": (3, 3),
    "hex": (3, 4),
    "cubic": (3, 3, 3),
    "moore": (3, 3, 3),
}


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_identity_boolean_initialization_assigns_unique_physical_ids(geometry):
    template = _make_lgca("identity_ve", geometry)
    nodes = np.ones(template.dims + (template.K,), dtype=bool)
    lgca = get_lgca(geometry=geometry, nodes=nodes, ib=True,
                    interaction="only_propagation")
    labels = lgca.nodes[lgca.nonborder].ravel()
    np.testing.assert_array_equal(np.sort(labels), np.arange(1, nodes.size + 1))
    assert lgca.maxlabel == nodes.size
    lgca.timestep()
    np.testing.assert_array_equal(np.sort(lgca.nodes[lgca.nonborder].ravel()),
                                  np.arange(1, nodes.size + 1))
    with pytest.raises(ValueError, match="unique"):
        get_lgca(geometry=geometry, nodes=np.ones_like(nodes, dtype=int),
                 ib=True, interaction="only_propagation")


@pytest.mark.parametrize("nodes", [np.array([[1.5, 2.]]), np.array([[-1, 2]]),
                                  np.array([[np.nan, 2.]]), np.array([[np.inf, 2.]])])
def test_identity_initialization_rejects_invalid_labels(nodes):
    with pytest.raises(ValueError, match="non-negative integer"):
        get_lgca(geometry="lin", nodes=nodes, ib=True, interaction="only_propagation")


def _make_lgca(family, geometry, *, bc="pbc", density=0):
    kwargs = dict(FAMILIES[family])
    # Identity-based NoVE models have exactly one rest channel by definition.
    restchannels = 1 if family == "identity_nove" else 0
    return get_lgca(
        geometry=geometry,
        dims=GEOMETRIES[geometry],
        density=density,
        restchannels=restchannels,
        interaction="only_propagation",
        bc=bc,
        seed=1729,
        **kwargs,
    )


def _channel_in_direction(lgca, direction):
    velocities = np.asarray(lgca.c).T
    matches = np.flatnonzero(np.all(np.isclose(velocities, direction), axis=1))
    assert matches.size == 1
    return int(matches[0])


def _channel_index(family, coordinate, channel):
    if family.startswith("multispecies"):
        return coordinate + (1, channel)
    return coordinate + (channel,)


def _put_particle(lgca, family, coordinate, channel):
    index = _channel_index(family, coordinate, channel)
    lgca.nodes[index] = [1] if family == "identity_nove" else 1
    lgca.update_dynamic_fields()


def _occupancy(lgca, family, coordinate, channel):
    value = lgca.nodes[_channel_index(family, coordinate, channel)]
    return len(value) if family == "identity_nove" else int(value != 0)


def _particle_count(lgca):
    return int(np.asarray(lgca.cell_density[lgca.nonborder]).sum())


@pytest.mark.parametrize("family, geometry, bc", pairwise(list(FAMILIES), list(GEOMETRIES), ["pbc", "rbc", "abc"]))
def test_plus_x_boundary_crossing_obeys_core_invariants(family, geometry, bc):
    """A single particle must wrap, reflect, or be absorbed exactly once."""
    lgca = _make_lgca(family, geometry, bc=bc)
    ndim = len(lgca.dims)
    plus_x = _channel_in_direction(lgca, (1,) + (0,) * (ndim - 1))
    minus_x = _channel_in_direction(lgca, (-1,) + (0,) * (ndim - 1))
    right = (lgca.r_int + lgca.dims[0] - 1,) + (lgca.r_int,) * (ndim - 1)

    _put_particle(lgca, family, right, plus_x)
    lgca.timestep()

    if bc == "pbc":
        wrapped = (lgca.r_int,) + (lgca.r_int,) * (ndim - 1)
        assert _particle_count(lgca) == 1
        assert _occupancy(lgca, family, wrapped, plus_x) == 1
    elif bc == "rbc":
        assert _particle_count(lgca) == 1
        assert _occupancy(lgca, family, right, minus_x) == 1
    else:
        assert _particle_count(lgca) == 0


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_seeded_initialization_is_reproducible(family, geometry):
    first = _make_lgca(family, geometry, density=0.25)
    second = _make_lgca(family, geometry, density=0.25)

    assert np.array_equal(first.nodes, second.nodes)


@pytest.mark.parametrize("family", FAMILIES)
def test_dense_recorders_have_one_row_per_simulation_step(family, tmp_path):
    lgca = _make_lgca(family, "lin", density=0.5)
    scalar = ScalarTimeSeriesRecorder(output_path=tmp_path / "population.csv")
    observers = [NodeRecorder(), DensityRecorder(), PopulationRecorder(), scalar]

    total = _particle_count(lgca)
    assert total > 0

    SimulationRunner(
        lgca, timesteps=2, observers=observers, showprogress=False
    ).run()

    # the initial state and one row per step; propagation on a periodic lattice keeps the cells
    assert lgca.nodes_t.shape[0] == 3
    np.testing.assert_array_equal(lgca.nodes_t[-1], lgca.nodes[lgca.nonborder])
    np.testing.assert_array_equal(lgca.dens_t, lgca._channel_counts(lgca.nodes_t).sum(-1))  # per species
    np.testing.assert_array_equal(lgca.n_t, [total] * 3)
    assert [record["step"] for record in scalar.records] == [0, 1, 2]
    assert [record["population"] for record in scalar.records] == [total] * 3


@pytest.mark.parametrize("ve", [True, False])
def test_seeded_identity_birthdeath_is_reproducible(ve):
    def run():
        lgca = get_lgca(geometry="square", dims=6, ib=True, ve=ve, restchannels=1, density=1,
                        interaction="birthdeath", seed=11)
        lgca.timeevo(timesteps=5, showprogress=False)
        return lgca

    first, second = run(), run()

    assert list(first.props["r_b"]) == list(second.props["r_b"])
    assert [list(np.atleast_1d(x)) for x in first.nodes.flat] == [list(np.atleast_1d(x)) for x in second.nodes.flat]
    assert len(first.props["r_b"]) == int(first.maxlabel) + 1
