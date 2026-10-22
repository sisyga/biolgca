"""Distances from the centre of the lattice and the core and rim of a tumour, shared by the zoo models."""

from __future__ import annotations

import numpy as np


def centre_distance(lgca) -> np.ndarray:
    """Distance of every node from the central node, in lattice units (``lgca.xcoords``, ``lgca.ycoords``)."""
    centre = tuple(size // 2 for size in lgca.dims)
    if len(lgca.dims) == 1:
        return np.abs(np.arange(lgca.dims[0]) - centre[0]).astype(float)
    x, y = lgca.xcoords, lgca.ycoords
    return np.hypot(x - x[centre], y - y[centre])


def core_and_rim_cells(lgca, rim: float):
    """The cells of ``lgca`` and two masks over them: the core (the inner half of the cells by distance from
    the centre) and the rim (the outermost fraction ``rim`` of the cells)."""
    from lgca.lattice_state import LatticeState

    cells = LatticeState(lgca).cells
    distance = centre_distance(lgca)[cells.node]
    if not len(cells):
        return cells, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    inner, outer = np.quantile(distance, [0.5, 1 - rim])
    return cells, distance <= inner, distance >= outer
