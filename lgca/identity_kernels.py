"""Shared numerical kernels for identity-based volume-exclusion interactions."""

import numpy as np


def inherit_missing_properties(lgca, parent):
    """Complete the newest daughter's property row by copying its parent.

    The growth operator first appends any explicitly mutated traits. All other
    cell properties, including family membership, inherit unchanged. This
    helper consumes no randomness and preserves existing mutation draw order.
    """
    label = int(lgca.maxlabel)
    for name, values in lgca.props.items():
        if len(values) == label:
            value = values[int(parent)]
            if isinstance(values, np.ndarray):
                lgca.props[name] = np.append(values, value)
            else:
                values.append(value)
        elif len(values) != label + 1:
            raise ValueError(f"Property {name!r} has no contiguous row for daughter {label}")


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
                inherit_missing_properties(lgca, label)
        lgca.nodes[coord] = node
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)
