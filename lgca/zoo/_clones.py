"""Clones in identity-based models: drivers per cell, clonal diversity, the dominant clone of each node.

A clone is a family (``new_family=True`` in ``birth_death``): the cells that descend from the same
mutated daughter without a further mutation. The initial cells form family 1, below the root 0
that only holds the tree together.
"""

from __future__ import annotations

import numpy as np


def family_depth(lgca) -> np.ndarray:
    """Depth of every family in the family tree: 1 for the initial clone, 2 for its mutants, and so on."""
    if not hasattr(lgca, "family_props"):
        return np.array([0, 1])
    ancestor = np.asarray(lgca.family_props["ancestor"], dtype=np.int64)
    depth = np.zeros(len(ancestor), dtype=np.int64)
    for family in range(1, len(ancestor)):  # parents have lower numbers than their children
        depth[family] = depth[ancestor[family]] + 1
    return depth


def cell_families(lgca) -> np.ndarray:
    """The family of every living cell (1 for all cells before the first mutation)."""
    from lgca.lattice_state import LatticeState

    cells = LatticeState(lgca).cells
    if "family" not in lgca.props:
        return np.ones(len(cells), dtype=np.int64)
    return np.asarray(cells["family"], dtype=np.int64)


def clonal_indices(lgca) -> dict[str, float]:
    """The two indices of Noble et al. (2022) and the number of clones.

    ``"n"`` is the mean number of drivers per cell, counting the one that founded the initial
    clone (so n = 1 before the first mutation); ``"D"`` the clonal diversity, the inverse Simpson
    index 1 / Σ p_i² of the clone frequencies p_i (D = k for k clones of equal size); ``"clones"``
    the number of clones alive.
    """
    families = cell_families(lgca)
    if len(families) == 0:
        return {"n": np.nan, "D": np.nan, "clones": 0}
    counts = np.bincount(families)
    p = counts[counts > 0] / len(families)
    return {"n": float(family_depth(lgca)[families].mean()), "D": float(1 / np.sum(p ** 2)),
            "clones": int(np.count_nonzero(counts))}


def dominant_clone(lgca) -> np.ndarray:
    """The most frequent family of every node, -1 where the node is empty."""
    from lgca.lattice_state import LatticeState

    cells = LatticeState(lgca).cells
    families = cell_families(lgca)
    shape = lgca.dims
    dominant = np.full(int(np.prod(shape)), -1, dtype=np.int64)
    if len(families):
        node = np.ravel_multi_index(cells.node, shape)
        order = np.lexsort((families, node))  # cells sorted by node, then family
        node, families = node[order], families[order]
        # runs of equal (node, family): their length is the family's count at the node
        start = np.flatnonzero(np.r_[True, (np.diff(node) != 0) | (np.diff(families) != 0)])
        length = np.diff(np.r_[start, len(node)])
        run_node, run_family = node[start], families[start]
        best = np.lexsort((-length, run_node))  # per node, the longest run first
        first = np.r_[True, np.diff(run_node[best]) != 0]
        dominant[run_node[best][first]] = run_family[best][first]
    return dominant.reshape(shape)


def diversity_bounds(n) -> dict[str, np.ndarray]:
    """Curves of the (n, D) plane of Noble et al. (2022), Fig 3b.

    ``"upper"``: D ≤ 1 / (2 − n)² for n < 2, the most diverse trees with n drivers per cell
    (infinite beyond); ``"sweeps"``: 1 / ((1 − {n})² + {n}²), a population that evolves by one
    complete selective sweep after the other ({n} the fractional part of n).
    """
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore"):
        upper = np.where(n < 2, 1 / (2 - n) ** 2, np.inf)
    fraction = n - np.floor(n)
    return {"upper": upper, "sweeps": 1 / ((1 - fraction) ** 2 + fraction ** 2)}


def largest_clones(dominant: np.ndarray, number: int = 9) -> np.ndarray:
    """Map of the ``number`` clones that dominate the most nodes, for plotting with a qualitative colormap.

    Nodes dominated by the k-th largest of them get k (1 for the largest), nodes dominated by
    another clone 0, empty nodes NaN.
    """
    occupied = dominant >= 0
    families, nodes = np.unique(dominant[occupied], return_counts=True)
    ranked = families[np.argsort(-nodes, kind="stable")][:number]
    rank = {family: k + 1 for k, family in enumerate(ranked.tolist())}
    categories = np.full(dominant.shape, np.nan)
    categories[occupied] = [rank.get(family, 0) for family in dominant[occupied].tolist()]
    return categories


def family_trait(lgca, name: str) -> list[float]:
    """The value of the trait ``name`` in every family, for colouring Muller plots.

    For traits that change only in mutated daughters, which found a family, every cell of a family
    has its family's value; the first cell of each family (dead ones included) gives it. The root
    family 0, which holds no cells, gets the value of the initial clone.
    """
    from lgca.cells import trait_array
    from lgca.nove_ib_base import NoVE_IBLGCA_base

    # Label 0 is an empty channel only with volume exclusion; otherwise it is a real cell.
    start = 0 if isinstance(lgca, NoVE_IBLGCA_base) else 1
    families = np.asarray(trait_array(lgca, "family").values, dtype=np.int64)[start:]
    values = np.asarray(trait_array(lgca, name).values, dtype=float)[start:]
    result = np.full(int(lgca.maxfamily) + 1, np.nan)
    present, first = np.unique(families, return_index=True)
    result[present] = values[first]
    if len(result) > 1:
        result[0] = result[1]
    return result.tolist()
