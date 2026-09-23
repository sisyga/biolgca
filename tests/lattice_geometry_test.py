"""Transport, neighbour sums and gradients agree with the channel velocity vectors ``c``.

For a linear field ``f(r) = a . r`` every lattice satisfies
``sum_i f(r + c_i) = b f(r)`` because the velocities sum to zero, and the
lattice gradient is the physical gradient ``a`` in lattice units.
"""

import numpy as np
import pytest

from lgca import get_lgca


GEOMETRIES = {"lin": (9,), "square": (9, 10), "hex": (9, 10), "cubic": (7, 8, 6), "moore": (7, 8, 6)}


def _empty(geometry):
    return get_lgca(geometry=geometry, dims=GEOMETRIES[geometry], density=0, interaction="only_propagation")


def _coordinates(lgca):
    return [getattr(lgca, name) for name in ("xcoords", "ycoords", "zcoords")[:len(lgca.dims)]]


@pytest.mark.parametrize("row_offset", [0, 1])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_each_velocity_channel_moves_a_particle_by_its_velocity_vector(geometry, row_offset):
    template = _empty(geometry)
    origin = tuple(size // 2 + (row_offset if axis == 1 else 0) for axis, size in enumerate(template.dims))
    for channel in range(template.velocitychannels):
        nodes = np.zeros(template.dims + (template.K,), dtype=bool)
        nodes[origin + (channel,)] = True
        lgca = get_lgca(geometry=geometry, nodes=nodes, interaction="only_propagation")

        lgca.timestep()

        (target,) = np.argwhere(lgca.nodes[lgca.nonborder][..., channel])
        displacement = [coordinate[tuple(target)] - coordinate[origin] for coordinate in _coordinates(lgca)]
        np.testing.assert_allclose(displacement, lgca.c[:, channel], atol=1e-12)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_neighbour_sum_and_gradient_of_a_linear_field(geometry):
    lgca = _empty(geometry)
    slope = np.array([0.5, -1.25, 2.0])[:len(lgca.dims)]
    field = np.zeros(lgca.nodes.shape[:-1])
    field[lgca.nonborder] = sum(a * coordinate for a, coordinate in zip(slope, _coordinates(lgca)))
    interior = (slice(1, -1),) * len(lgca.dims)  # nodes whose neighbours all lie inside the lattice

    neighbour_sum = lgca.nb_sum(field)[lgca.nonborder][interior]
    gradient = lgca.gradient(field)[lgca.nonborder][interior]

    np.testing.assert_allclose(neighbour_sum, lgca.velocitychannels * field[lgca.nonborder][interior])
    np.testing.assert_allclose(gradient, np.broadcast_to(slope, gradient.shape), atol=1e-12)
