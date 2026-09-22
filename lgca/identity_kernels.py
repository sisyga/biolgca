"""Shared numerical kernels for identity-based volume-exclusion interactions."""

import numpy as np


def apply_identity_birth(lgca, *, a_max, std):
    """Apply VE identity birth, mutation and final channel shuffling in order."""
    from .ib_interactions import trunc_gauss

    relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
        lgca.cell_density[lgca.nonborder] < lgca.K
    )
    coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        r_bs = np.array([lgca.props["r_b"][label] for label in node])
        proliferating = lgca.rng.random(lgca.K) < r_bs
        for label in node[proliferating]:
            ind = lgca.rng.choice(lgca.K)
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props["r_b"][label]
                lgca.props["r_b"].append(
                    float(trunc_gauss(0, a_max, r_b, sigma=std, rng=lgca.rng))
                )
        lgca.nodes[coord] = node
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)
