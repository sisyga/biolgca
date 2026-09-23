"""Shared numerical kernels for identity-based interactions."""

import numpy as np
from scipy.stats import truncnorm


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


def sample_truncated_normal(lower, upper, mean, std, rng):
    """Draw one value per entry of ``mean`` from a normal distribution truncated to ``[lower, upper]``.

    All values are drawn in a single vectorized call from ``rng``. With
    ``std == 0`` the means are returned unchanged.

    Parameters
    ----------
    lower, upper : float
        Truncation bounds.
    mean : array_like
        Mean of the untruncated distribution for each sample.
    std : float
        Standard deviation of the untruncated distribution.
    rng : numpy.random.Generator
        Random number generator, normally ``lgca.rng``.

    Returns
    -------
    numpy.ndarray
        Samples with the shape of ``mean``.
    """
    mean = np.asarray(mean, dtype=float)
    if std == 0 or mean.size == 0:
        return mean.copy()
    return truncnorm.rvs((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std,
                         size=mean.shape, random_state=rng)


def append_daughter_properties(lgca, parents, **traits):
    """Append property rows for the daughters labelled ``maxlabel - len(parents) + 1`` to ``maxlabel``.

    Daughter ``i`` descends from ``parents[i]``. Traits passed as keyword
    arguments are appended as given; every other property is copied from the
    parent.
    """
    parents = [int(parent) for parent in parents]
    if not parents:
        return
    unknown = set(traits) - set(lgca.props)
    if unknown:
        raise ValueError(f"Unknown cell properties: {sorted(unknown)}")
    first = int(lgca.maxlabel) - len(parents) + 1
    for name, values in lgca.props.items():
        if len(values) != first:
            raise ValueError(f"Property {name!r} has no contiguous rows for daughters {first}..{lgca.maxlabel}")
        if name in traits:
            new_values = np.asarray(traits[name]).tolist()
        else:
            new_values = [values[parent] for parent in parents]
        if isinstance(values, np.ndarray):
            lgca.props[name] = np.concatenate([values, np.asarray(new_values, dtype=values.dtype)])
        else:
            values.extend(new_values)


def append_mutated_daughters(lgca, parents, *, a_max, std):
    """Append daughters whose birth rate ``r_b`` mutates around the parent's rate within ``[0, a_max]``."""
    if not parents:
        return
    parent_rates = [lgca.props["r_b"][parent] for parent in parents]
    rates = sample_truncated_normal(0.0, a_max, parent_rates, std, lgca.rng)
    append_daughter_properties(lgca, parents, r_b=rates)


def apply_identity_birth(lgca, *, a_max, std):
    """Apply volume-exclusion identity birth with mutating birth rates, then shuffle channels.

    Each cell divides with its own probability ``r_b`` into a randomly chosen
    channel of its node if that channel is empty. Daughter birth rates are drawn
    after all divisions of the step.
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
        lgca.cell_density[lgca.nonborder] < lgca.K
    )
    coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
    r_b = np.asarray(lgca.props["r_b"])
    parents = []
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        proliferating = lgca.rng.random(lgca.K) < r_b[node]
        for label in node[proliferating]:
            ind = lgca.rng.choice(lgca.K)
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                parents.append(label)
        lgca.nodes[coord] = node
    append_mutated_daughters(lgca, parents, a_max=a_max, std=std)
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


def apply_identity_birthdeath(lgca, *, r_d, a_max, std):
    """Apply volume-exclusion identity birth and death with mutating birth rates, then shuffle channels.

    Cells chosen to die with probability ``r_d`` may still divide in the same
    step. Each dividing cell targets a distinct random channel of its node and
    places a daughter there if the channel is empty.
    """
    dying = (lgca.rng.random(size=lgca.nodes.shape) < r_d) & lgca.occupied

    relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
        lgca.cell_density[lgca.nonborder] < lgca.K
    )
    coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
    r_b = np.asarray(lgca.props["r_b"])
    parents = []
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        occ = lgca.occupied[coord]
        proliferating = (lgca.rng.random(lgca.K) * occ) < r_b[node]
        n_p = proliferating.sum()
        if n_p == 0:
            continue
        targetchannels = lgca.rng.choice(lgca.K, size=n_p, replace=False)
        for ind, label in zip(targetchannels, node[proliferating]):
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                parents.append(label)
        lgca.nodes[coord] = node
    append_mutated_daughters(lgca, parents, a_max=a_max, std=std)

    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


def apply_nove_identity_birth(lgca, *, capacity, a_max, std, channel_weights, r_d=None):
    """Apply logistic identity birth (and death if ``r_d`` is given) without volume exclusion.

    Each cell divides with probability ``r_b * (1 - n / capacity)``, where ``n``
    is the node population, and dies with probability ``r_d``. Surviving cells
    and daughters are redistributed over the channels with ``channel_weights``.
    """
    from .nove_ib_interactions import _cells_from_node, _split_cells_into_channels

    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
    r_b = lgca.props["r_b"]
    parents = []
    for coord in zip(*coords):
        rho = lgca.cell_density[coord] / capacity
        cells = _cells_from_node(lgca.nodes[coord])
        newcells = cells.copy()
        for cell in cells:
            if r_d is not None and lgca.rng.random() < r_d:
                newcells.remove(cell)
            if lgca.rng.random() < r_b[cell] * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                parents.append(cell)

        channeldist = lgca.rng.multinomial(len(newcells), channel_weights).cumsum()
        lgca.rng.shuffle(newcells)
        lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)
    append_mutated_daughters(lgca, parents, a_max=a_max, std=std)
