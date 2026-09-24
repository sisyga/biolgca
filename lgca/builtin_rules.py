"""Built-in interactions written with the :func:`lgca.interaction` decorator.

They work with and without volume exclusion and for any number of species
unless stated otherwise, and serve as worked examples of the decorator.

Go-or-grow
----------
Cells in velocity channels migrate, cells in rest channels rest and divide.
One time step of the classical go-or-grow model is the pipeline ::

    operators=[
        {"name": "go_or_rest", "parameters": {"kappa": 4.0, "theta": 0.75}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01}},
        {"name": "random_walk", "parameters": {"channels": "velocity"}},
    ]

in this order, followed by propagation: cells switch between moving and
resting (a reorientation between velocity and rest channels), die, and
resting cells divide into rest channels; then the moving cells pick new
directions. This reproduces ``classical.go_or_grow`` and ``nove.go_or_grow``
in distribution.

In the two-species variant, migrating cells are species 0 (in velocity
channels) and resting cells species 1 (in rest channels), and the switch is
a phenotype switch, ``go_or_grow.switch``; ``go_or_grow.growth`` and the
random walk (with ``"species": 0``) work unchanged. It gives both phenotypes
their own counts in recordings. The rules keep each species in its channels,
so the initial state must place migrating cells in velocity channels and
resting cells in rest channels.
"""

from __future__ import annotations

import numpy as np

from .interactions import tanh_switch
from .rules import interaction, register_single_cue, reorientation_term

__all__ = ["birth_death", "go_or_grow_growth", "go_or_grow_switch", "go_or_rest", "random_walk"]


# Terms of the Boltzmann reorientation (ReorientationSpec). J(s') is the flux of
# the candidate state; see lgca.rules.reorientation_term for the couplings.

@reorientation_term(coupling="rest", name="random_walk", aliases="uniform")
def uniform(state):
    """Score 0: all channel states are equally likely."""
    return 0.0


@reorientation_term(coupling="rest", name="resting_bias")
def resting_bias(state):
    """Number of cells in rest channels: cells prefer to rest."""
    return 1.0


@reorientation_term(coupling="flux", name="persistent_walk", aliases="persistent_motion")
def persistent_walk(state):
    """J(s) · J(s'): cells keep the direction they had at this node."""
    return state.flux


@reorientation_term(coupling="flux", name="polar_alignment")
def polar_alignment(state, include_center=False, normalize=False):
    """J_nb · J(s'), with J_nb the flux of the neighbouring nodes: cells move with their neighbours.

    Parameters
    ----------
    include_center : bool
        Add the flux of the node's own cells to J_nb.
    normalize : bool
        Divide J_nb by the number of cells it sums over (at least one), so the
        cue measures the mean direction rather than the number of neighbours
        (density-independent alignment).
    """
    flux = state.neighbor_sum(state.flux)
    cells = state.neighbor_sum(state.density)
    if include_center:
        flux, cells = flux + state.flux, cells + state.density
    if normalize:
        flux = flux / np.maximum(cells, 1)[..., None]
    return flux


@reorientation_term(coupling="channels", name="nematic_alignment", aliases=("nematic", "alignment"))
def nematic_alignment(state):
    """Cells share an axis with neighbouring cells; opposite directions count the same.

    A cell in channel i gains Σ_k n_k [(c_k · c_i)² - |c_k|² |c_i|² / d] from the n_k
    neighbouring cells in channel k (d the spatial dimension): the product of the
    traceless tensors c cᵀ - |c|² I / d of the two directions. Moving along the
    neighbours' axis scores above resting (0), moving across it below.
    """
    neighbours = state.neighbor_sum(state.counts[..., :state.velocitychannels].sum(axis=-2))
    c = state.c
    lengths = (c ** 2).sum(0)
    return neighbours @ ((c.T @ c) ** 2 - np.multiply.outer(lengths, lengths) / c.shape[0])


@reorientation_term(coupling="flux", name="aggregation")
def aggregation(state):
    """∇ρ · J(s'): cells move up the gradient of the cell density."""
    return state.gradient(state.density)


@reorientation_term(coupling="flux", name="chemotaxis")
def chemotaxis(state, field):
    """∇f · J(s'): cells move up the gradient of a field.

    Parameters
    ----------
    field : str
        Name of a scalar field in StateSpec.fields. Its gradient is centred
        everywhere; beyond the lattice edge the field keeps the ghost values
        the model stores for it (the edge values unless changed).
    """
    values = state.field(field)
    if values.shape != state.dims:
        raise ValueError(f"state.fields.{field} must have shape {state.dims}, got {values.shape}")
    return state.gradient(field)


@reorientation_term(coupling="channels", name="contact_guidance")
def contact_guidance(state, field="director"):
    """Cells move along the axis of a director field.

    A cell in channel i scores (n · c_i)² - |c_i|² / d, with n the unit director and d
    the spatial dimension: above resting (0) along the axis, below it across. Where
    the director is zero, all channels score 0.

    Parameters
    ----------
    field : str
        Name of a vector field in StateSpec.fields; only its direction matters.
    """
    director = np.asarray(state.field(field), dtype=float)
    if director.shape != state.dims + (state.c.shape[0],):
        raise ValueError(f"state.fields.{field} must have shape {state.dims + (state.c.shape[0],)}, "
                         f"got {director.shape}")
    norm = np.linalg.norm(director, axis=-1, keepdims=True)
    director = np.divide(director, norm, out=np.zeros_like(director), where=norm > 0)
    c = state.c
    return (director @ c) ** 2 - (director ** 2).sum(-1)[..., None] * (c ** 2).sum(0) / c.shape[0]

# Every built-in cue also works alone as an operator: {"name": "chemotaxis", "parameters": {"beta": 2,
# "field": "signal"}} is a ReorientationSpec with this one term.
for _cue, _aliases in ((polar_alignment, ()), (nematic_alignment, ("nematic",)),
                       (persistent_walk, ("persistent_motion",)), (aggregation, ()), (chemotaxis, ()),
                       (contact_guidance, ()), (resting_bias, ())):
    register_single_cue(_cue, aliases=_aliases)

@interaction(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), name="birth_death")
def birth_death(state, birth_rate=0.0, death_rate=0.0, crowding=True, mutation_matrix=None, mutation=None,
                new_family=False):
    """Cells die and divide at the same time; crowded nodes have less room for daughters.

    In a time step, every cell dies with probability ``death_rate`` and,
    independently, tries to divide with probability ``birth_rate``; both are
    decided on the state at the start of the step, so a dying cell may still
    divide, and daughters do not die in the step they are born. The daughter
    needs room:

    - with volume exclusion, it goes to a random channel of its species at
      the node (dividing cells pick different channels) and survives only
      if that channel was empty, which happens with probability
      ``1 - n_s / K`` for ``n_s`` cells of its species;
    - a capacity (``StateSpec.capacity``; always set without volume
      exclusion) scales the division probability by ``1 - n / capacity``,
      with ``n`` all cells at the node. With volume exclusion it is an
      optional soft limit in addition to the channels, e.g. to make species
      compete for space.

    The expected change of a node with one species is
    ``birth_rate * n * (1 - n / capacity) - death_rate * n``, with
    ``capacity = K`` under volume exclusion. For death before division,
    list two operators: one with only ``death_rate``, then one with only
    ``birth_rate``.

    Parameters
    ----------
    birth_rate : float, sequence or str
        Probability per time step that a cell tries to divide: one number,
        one per species, or in identity-based models the name of a cell trait
        (a value per cell).
    death_rate : float, sequence or str
        Probability per time step that a cell dies, given like ``birth_rate``.
    crowding : bool
        False: cells divide with ``birth_rate`` regardless of crowding;
        daughters take free channels, and ``state.capacity`` is a hard limit
        on the cells per node. Divisions beyond either fail, chosen at random.
    mutation_matrix : array_like or None
        Classical models with several species: entry ``[a][b]`` is the
        probability that a daughter of species ``a`` belongs to species ``b``
        (rows sum to 1). Default: daughters have their mother's species.
    mutation : dict or None
        Identity-based models: how the daughters' traits change, by trait
        name. A number is the standard deviation of a normal change,
        ``{"std": s, "bounds": [low, high]}`` a normal change truncated to
        the bounds, ``{"step": d, "probability": p}`` a change by ``+d`` or
        ``-d`` with probability ``p`` (clipped to ``"bounds"`` if given).
    new_family : bool or float
        Identity-based models: every daughter founds a new family (True), or
        each one with this probability, for lineage analyses such as Muller
        plots.

    Examples
    --------
    Logistic growth, ``{"name": "birth_death", "parameters": {"birth_rate":
    0.2, "death_rate": 0.02}}``; in an identity-based model with a mutating
    birth rate per cell, ``{"birth_rate": "r_b", "death_rate": 0.02,
    "mutation": {"r_b": {"std": 0.01, "bounds": [0, 0.5]}}}``.
    """
    if not isinstance(crowding, (bool, np.bool_)):
        raise TypeError(f"crowding must be True or False, got {crowding!r}")
    if state.identity_based:
        if mutation_matrix is not None and not np.array_equal(np.asarray(mutation_matrix), [[1]]):
            raise ValueError("mutation_matrix needs a classical model with several species; identity-based "
                             "models have one species and change traits with mutation=")
        _birth_death_cells(state, birth_rate, death_rate, crowding, mutation or {}, new_family)
        return
    if mutation or new_family:
        raise ValueError("mutation and new_family change cell traits and families, which only "
                         "identity-based models have")
    n_species, rng = state.n_species, state.rng
    death = _per_species(death_rate, "death_rate", n_species)
    birth = _per_species(birth_rate, "birth_rate", n_species)
    matrix = _mutation_matrix(mutation_matrix, n_species)
    channels = state.counts
    deaths = rng.binomial(channels, death[:, None])  # decided on the state at the start of the step
    counts, density = channels.sum(axis=-1), state.density
    if crowding and state.has_capacity:
        birth = birth * np.clip(1 - density / state.capacity, 0, 1)[..., None]
    births = rng.binomial(counts, birth)
    if matrix is not None:
        births = rng.multinomial(births, matrix).sum(axis=-2)
    if crowding and state.volume_exclusion:  # daughters pick distinct random channels; the empty ones hold them
        births = _into_empty(rng, births, counts, state.K)
    elif not crowding:
        births = _at_most(rng, births, np.maximum(state.capacity - density, 0))
    state.add_cells(births)  # into channels that were free at the start of the step
    state.counts = state.counts - deaths


def _birth_death_cells(state, birth_rate, death_rate, crowding, mutation, new_family):
    """birth_death for identity-based models: death and division per cell, decided at once."""
    cells, rng = state.cells, state.rng
    dying = rng.random(len(cells)) < _per_cell(cells, death_rate, "death_rate")
    birth = _per_cell(cells, birth_rate, "birth_rate")
    density = state.density
    if crowding and state.has_capacity:
        birth = birth * np.clip(1 - density / state.capacity, 0, 1)[cells.node]
    dividing = rng.random(len(cells)) < birth
    if crowding and state.volume_exclusion:
        attempts = np.bincount(cells.index[dividing], minlength=density.size).reshape(density.shape)
        dividing = cells.pick(dividing, _into_empty(rng, attempts, density, state.K))
    elif not crowding:
        dividing = cells.pick(dividing, np.maximum(state.capacity - density, 0))
    founders = _founders(rng, cells, dividing, new_family)
    daughters = cells.divide(dividing, new_family=founders)
    _mutate(state, daughters, mutation)
    cells.kill(np.concatenate([dying, np.zeros(len(daughters), dtype=bool)]))


def _into_empty(rng, attempts, occupied, channels):
    """How many of ``attempts`` daughters, sent to distinct random channels of ``channels``, find them empty."""
    attempts = np.minimum(attempts, channels)
    return rng.hypergeometric(channels - occupied, occupied, attempts)


def _per_species(value, name, n_species):
    rates = np.asarray(_number(value, name), dtype=float)
    if rates.ndim == 0:
        rates = np.full(n_species, float(rates))
    if rates.shape != (n_species,):
        raise ValueError(f"{name} must be one number or one per species ({n_species}), got {value!r}")
    if np.any((rates < 0) | (rates > 1)):
        raise ValueError(f"{name} must be probabilities, got {value!r}")
    return rates


def _mutation_matrix(value, n_species):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (n_species, n_species):
        raise ValueError(f"mutation_matrix must have shape ({n_species}, {n_species}), got {matrix.shape}")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("mutation_matrix rows must be probabilities that sum to 1")
    return None if np.array_equal(matrix, np.eye(n_species)) else matrix / matrix.sum(axis=1, keepdims=True)


def _at_most(rng, births, room):
    """Keep at most ``room`` of the births at each node, chosen uniformly among all species' births."""
    total = births.sum(axis=-1)
    kept = np.empty_like(births)
    left, wanted = np.minimum(total, room), total
    for species in range(births.shape[-1]):
        kept[..., species] = rng.hypergeometric(births[..., species], wanted - births[..., species], left)
        left, wanted = left - kept[..., species], wanted - births[..., species]
    return kept


def _founders(rng, cells, dividing, new_family):
    """Mask of the dividing cells whose daughters found a family: all, none, or each with a probability."""
    if isinstance(new_family, (bool, np.bool_)):
        return bool(new_family)
    probability = float(new_family)
    if not 0 <= probability <= 1:
        raise ValueError(f"new_family must be True, False or a probability, got {new_family!r}")
    return dividing & (rng.random(len(cells)) < probability)


def _mutate(state, daughters, mutation):
    """Change the daughters' traits as ``mutation`` says (see birth_death)."""
    cells, rng = state.cells, state.rng
    for name, spec in mutation.items():
        if not isinstance(spec, dict):
            spec = {"std": spec}
        unknown = set(spec) - {"std", "bounds", "step", "probability"}
        if unknown or ("std" in spec) == ("step" in spec):
            raise ValueError(f"mutation[{name!r}] must be a standard deviation, {{'std': s, 'bounds': "
                             f"[low, high]}} or {{'step': d, 'probability': p}}, got {spec!r}")
        low, high = spec.get("bounds") or (-np.inf, np.inf)
        low, high = -np.inf if low is None else float(low), np.inf if high is None else float(high)
        if not low <= high:
            raise ValueError(f"mutation[{name!r}]['bounds'] must be [low, high] with low <= high")
        values = np.asarray(cells[name][daughters], dtype=float)
        if "std" in spec:
            std = float(spec["std"])
            if not np.isfinite(std) or std < 0:
                raise ValueError(f"mutation[{name!r}] needs a non-negative standard deviation")
            values = _truncated_normal(rng, values, std, low, high)
        else:
            step, probability = float(spec["step"]), float(spec.get("probability", 1.0))
            if not 0 <= probability <= 1:
                raise ValueError(f"mutation[{name!r}]['probability'] must be a probability")
            mutates = rng.random(len(values)) < probability
            signs = np.where(rng.random(len(values)) < 0.5, -1.0, 1.0)
            values = np.clip(values + mutates * signs * step, low, high)
        cells.set_trait(daughters, name, values)


def _truncated_normal(rng, mean, std, low, high):
    """Normal values around ``mean`` with standard deviation ``std``, truncated to [low, high]."""
    if std == 0 or len(mean) == 0:
        return np.clip(mean, low, high)
    if np.isinf(low) and np.isinf(high):
        return mean + rng.normal(0.0, std, len(mean))
    from scipy.stats import truncnorm

    return truncnorm.rvs((low - mean) / std, (high - mean) / std, loc=mean, scale=std, size=len(mean),
                         random_state=rng)


_MIGRATING, _RESTING = 0, 1
_WHEN_FULL = ("legacy", "reject")


@interaction(kind="reorientation", families=("classical", "nove", "ib", "nove_ib"), name="random_walk")
def random_walk(state, channels="all", species=None):
    """Cells move to uniformly random channels of their node, within a set of channels.

    In identity-based models the cells keep their labels; cells outside the
    set keep their channels.

    Parameters
    ----------
    channels : str or list of int
        Channels the moving cells are spread over: ``"all"``, ``"velocity"``,
        ``"rest"`` or channel indices. Cells in other channels stay put.
    species : int, list of int or None
        Species that move; the others stay in their channels. Default: all.
    """
    state.shuffle_cells(channels, species=species)


@interaction(kind="reorientation", families=("classical", "nove", "ib", "nove_ib"), n_species=1,
             name="go_or_rest")
def go_or_rest(state, kappa=5.0, theta=0.75, when_full="legacy"):
    """Moving cells start resting on crowded nodes, resting cells start moving on sparse ones.

    Parameters
    ----------
    kappa : float or str
        Steepness of the switch. With kappa > 0 crowded cells rest, with
        kappa < 0 they move. In identity-based models, the name of a cell
        trait gives every cell its own value.
    theta : float or str
        Relative density at which half of the cells rest, or the name of a
        cell trait. The relative density is the number of cells at the node
        over its capacity: the number of channels K with volume exclusion,
        ``StateSpec.capacity`` (the crowding scale) without.
    when_full : {"legacy", "reject"}
        With volume exclusion, what happens when a switching cell finds no free
        channel. "legacy" reproduces the original rule: the number of switching
        cells is drawn from the cells that fit into free channels. "reject":
        every cell tries to switch, and switches into full channels fail.
    """
    _check_mode(when_full)
    if state.restchannels < 1:
        raise ValueError("go_or_rest needs at least one rest channel for resting cells")
    if state.identity_based:
        _go_or_rest_cells(state, kappa, theta, when_full)
        return
    rest = tanh_switch(state.density / _node_capacity(state), _number(kappa, "kappa"), _number(theta, "theta"))
    velocity, rng = state.velocitychannels, state.rng
    cells = state.counts[..., 0, :]
    moving, resting = cells[..., :velocity], cells[..., velocity:]
    if state.volume_exclusion:
        n_m, n_r = moving.sum(-1), resting.sum(-1)
        free_rest, free_velocity = state.restchannels - n_r, velocity - n_m
        if when_full == "legacy":
            to_rest = rng.binomial(np.minimum(n_m, free_rest), rest)
            to_move = rng.binomial(np.minimum(n_r, free_velocity), 1 - rest)
        else:
            to_rest = np.minimum(rng.binomial(n_m, rest), free_rest)
            to_move = np.minimum(rng.binomial(n_r, 1 - rest), free_velocity)
        new_moving = moving - _pick(rng, moving == 1, to_rest) + _pick(rng, moving == 0, to_move)
        new_resting = resting - _pick(rng, resting == 1, to_move) + _pick(rng, resting == 0, to_rest)
    else:
        leaving_moving = rng.binomial(moving, rest[..., None])
        leaving_resting = rng.binomial(resting, (1 - rest)[..., None])
        new_moving = moving - leaving_moving + _spread(rng, leaving_resting.sum(-1), velocity)
        new_resting = resting - leaving_resting + _spread(rng, leaving_moving.sum(-1), state.restchannels)
    state.counts = np.concatenate((new_moving, new_resting), axis=-1)[..., None, :]


@interaction(kind="phenotype_switch", families=("classical", "nove"), n_species=2, name="go_or_grow.switch")
def go_or_grow_switch(state, kappa=5.0, theta=0.75, when_full="legacy"):
    """Migrating cells start resting in crowded nodes, resting cells start migrating in sparse ones.

    The two-species form of go_or_rest: migrating cells are species 0 (in
    velocity channels), resting cells species 1 (in rest channels).

    Parameters
    ----------
    kappa : float
        Steepness of the switch. With kappa > 0 crowded cells rest, with
        kappa < 0 they migrate.
    theta : float
        Relative density at which half of the cells rest: cells at the node
        over K with volume exclusion, over ``StateSpec.capacity`` without.
    when_full : {"legacy", "reject"}
        With volume exclusion, what happens when a switching cell finds no free
        channel. "legacy" reproduces the original rule: the number of switching
        cells is drawn from the cells that fit into free channels. "reject":
        every cell tries to switch, and switches into full channels fail.
    """
    _check_layout(state)
    _check_mode(when_full)
    rest = tanh_switch(state.density / _node_capacity(state), kappa, theta)
    if state.volume_exclusion and when_full == "legacy":
        counts = state.counts.copy()
        velocity = state.velocitychannels
        n_m = counts[..., _MIGRATING, :].sum(-1)
        n_r = counts[..., _RESTING, :].sum(-1)
        to_rest = state.rng.binomial(np.minimum(n_m, state.restchannels - n_r), rest)
        to_move = state.rng.binomial(np.minimum(n_r, velocity - n_m), 1 - rest)
        rng = state.rng
        counts[..., _MIGRATING, :] -= _pick(rng, counts[..., _MIGRATING, :] == 1, to_rest)
        counts[..., _RESTING, :] -= _pick(rng, counts[..., _RESTING, :] == 1, to_move)
        free_rest = state.counts[..., _RESTING, :] == 0
        free_rest[..., :velocity] = False
        free_velocity = state.counts[..., _MIGRATING, :] == 0
        free_velocity[..., velocity:] = False
        counts[..., _RESTING, :] += _pick(rng, free_rest, to_rest)
        counts[..., _MIGRATING, :] += _pick(rng, free_velocity, to_move)
        state.counts = counts
        return
    rates = np.zeros(state.dims + (2, 2))
    rates[..., _MIGRATING, _RESTING] = rest
    rates[..., _RESTING, _MIGRATING] = 1 - rest
    state.switch_phenotype(rates, channels={_MIGRATING: "velocity", _RESTING: "rest"})


@interaction(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), name="go_or_grow.growth")
def go_or_grow_growth(state, r_b=0.2, r_d=0.01, r_d_resting=None, when_full="legacy", mutation=None,
                      new_family=False):
    """Cells die, and resting cells divide into free rest channels.

    Resting cells are the cells in rest channels, or species 1 in the
    two-species form of go-or-grow.

    Parameters
    ----------
    r_b : float or str
        Division probability of a resting cell. Without volume exclusion it is
        scaled by 1 - density / capacity. In identity-based models, the name
        of a cell trait gives every cell its own value.
    r_d : float or str
        Death probability of a moving cell, and of a resting cell unless
        r_d_resting is given; or the name of a cell trait.
    r_d_resting : float, str or None
        Death probability of a resting cell. Default: r_d.
    when_full : {"legacy", "reject"}
        With volume exclusion, how divisions meet full rest channels. "legacy"
        reproduces the original rule: the number of divisions is drawn from as
        many resting cells as there are free rest channels. "reject": every
        resting cell tries to divide, and divisions into full channels fail.
    mutation : dict or None
        Identity-based models: traits of the daughters that mutate, e.g.
        ``{"kappa": 0.2}`` for a normal change with standard deviation 0.2;
        see :func:`birth_death` for bounded and discrete changes.
    new_family : bool or float
        Identity-based models: every daughter founds a new family (True), or
        each one with this probability, for lineage analyses such as Muller
        plots.
    """
    _check_mode(when_full)
    if state.identity_based:
        _growth_cells(state, r_b, r_d, r_d_resting, when_full, mutation or {}, new_family)
        return
    r_b, r_d = _number(r_b, "r_b"), _number(r_d, "r_d")
    r_d_resting = None if r_d_resting is None else _number(r_d_resting, "r_d_resting")
    if state.n_species == 1:
        if state.restchannels < 1:
            raise ValueError("go-or-grow needs at least one rest channel for resting cells")
        resting = np.arange(state.K) >= state.velocitychannels
        resting_species = 0
    elif state.n_species == 2:
        _check_layout(state)
        resting = np.array([np.zeros(state.K, dtype=bool), np.ones(state.K, dtype=bool)])
        resting_species = _RESTING
    else:
        raise ValueError(f"go_or_grow.growth works with one or two species, not {state.n_species}")
    resting = np.broadcast_to(resting, (state.n_species, state.K))
    crowding = state.density / _node_capacity(state)
    death = np.where(resting, r_d if r_d_resting is None else r_d_resting, r_d)
    state.remove_cells(np.broadcast_to(death, state.dims + death.shape))
    if state.volume_exclusion and when_full == "legacy":
        n_r = (state.counts * resting).sum(axis=(-2, -1))
        births = np.zeros(state.dims + (state.n_species,), dtype=np.int64)
        births[..., resting_species] = state.rng.binomial(np.minimum(n_r, state.restchannels - n_r), r_b)
        state.add_cells(births, channels={resting_species: "rest"})
        return
    rate = r_b if state.volume_exclusion else np.clip(r_b * (1 - crowding), 0, 1)
    division = np.where(resting, np.asarray(rate, dtype=float)[..., None, None], 0.0)
    state.divide_cells(np.broadcast_to(division, state.dims + resting.shape),
                       channels={resting_species: "rest"})


def _go_or_rest_cells(state, kappa, theta, when_full):
    """go_or_rest for identity-based models: every cell switches with its own probability."""
    cells, rng = state.cells, state.rng
    rho = state.density[cells.node] / _node_capacity(state)
    rest = tanh_switch(rho, _per_cell(cells, kappa, "kappa"), _per_cell(cells, theta, "theta"))
    resting = cells.in_channels("rest")
    to_rest = ~resting & (rng.random(len(cells)) < rest)
    to_move = resting & (rng.random(len(cells)) < 1 - rest)
    if state.volume_exclusion:
        # only as many cells as there were free channels before the switch
        n_r = state.counts[..., 0, state.velocitychannels:].sum(-1)
        n_m = state.counts[..., 0, :state.velocitychannels].sum(-1)
        free_rest, free_velocity = state.restchannels - n_r, state.velocitychannels - n_m
        if when_full == "legacy":  # the cells that fit try to switch
            to_rest = cells.pick(~resting, free_rest) & to_rest
            to_move = cells.pick(resting, free_velocity) & to_move
        else:  # every cell tries; the successful ones are chosen at random
            to_rest = cells.pick(to_rest, free_rest)
            to_move = cells.pick(to_move, free_velocity)
    cells.move(to_rest, "rest")
    cells.move(to_move, "velocity")


def _growth_cells(state, r_b, r_d, r_d_resting, when_full, mutation, new_family):
    """go_or_grow.growth for identity-based models: death and division per cell."""
    cells, rng = state.cells, state.rng
    crowding = state.density / _node_capacity(state)
    resting = cells.in_channels("rest")
    death = _per_cell(cells, r_d, "r_d")
    if r_d_resting is not None:
        death = np.where(resting, _per_cell(cells, r_d_resting, "r_d_resting"), death)
    alive = rng.random(len(cells)) >= death
    cells.kill(~alive)
    resting = resting[alive]
    birth = _per_cell(cells, r_b, "r_b")
    if state.volume_exclusion:
        free_rest = state.restchannels - state.counts[..., 0, state.velocitychannels:].sum(-1)
        if when_full == "legacy":  # as many resting cells as there are free rest channels try
            dividing = cells.pick(resting, free_rest) & (rng.random(len(cells)) < birth)
        else:
            dividing = resting & (rng.random(len(cells)) < birth)
    else:
        rate = np.clip(birth * (1 - crowding[cells.node]), 0, 1)
        dividing = resting & (rng.random(len(cells)) < rate)
    founders = _founders(rng, cells, dividing, new_family)
    daughters = cells.divide(dividing, channels="rest", new_family=founders)
    _mutate(state, daughters, mutation)


def _per_cell(cells, value, name):
    """A parameter per cell: the named trait, or one number for all cells."""
    if isinstance(value, str):
        return cells[value]
    return np.full(len(cells), _number(value, name))


def _number(value, name):
    if isinstance(value, str):
        # a model mismatch rather than a wrong type: the same value is valid in identity-based models
        raise ValueError(f"{name}={value!r} names a cell trait, which only identity-based models have")  # noqa: TRY004
    return value


def go_or_grow_layout(state):
    """Move migrating cells to velocity channels and resting cells to rest channels.

    For random initial states of the two-species form, e.g. in
    :func:`lgca.testing.check_interaction`: cells in the wrong channels are
    dropped. States with one species are left as they are.
    """
    if state.n_species == 1:
        return
    counts = state.counts.copy()
    velocity = state.velocitychannels
    counts[..., _MIGRATING, velocity:] = 0
    counts[..., _RESTING, :velocity] = 0
    state.counts = counts


def _node_capacity(state):
    """Cells a node can hold: its channels with volume exclusion, the model capacity without."""
    return state.K if state.volume_exclusion else state.capacity


def _check_mode(when_full):
    if when_full not in _WHEN_FULL:
        raise ValueError(f"when_full must be one of {_WHEN_FULL}, got {when_full!r}")


def _spread(rng, number, channels):
    """Spread ``number`` cells per node uniformly over ``channels`` channels."""
    return rng.multinomial(number, np.full(channels, 1 / channels))


def _check_layout(state):
    velocity = state.velocitychannels
    if state.restchannels < 1:
        raise ValueError("go-or-grow needs at least one rest channel for resting cells")
    if np.any(state.counts[..., _MIGRATING, velocity:]) or np.any(state.counts[..., _RESTING, :velocity]):
        raise ValueError("go-or-grow keeps migrating cells (species 0) in velocity channels and resting "
                         "cells (species 1) in rest channels; the state has cells elsewhere")


def _pick(rng, candidates, number):
    """Indicator of ``number`` uniformly chosen channels among ``candidates`` at every node."""
    scores = rng.random(candidates.shape)
    scores[~candidates] = np.inf
    ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1)
    return (ranks < number[..., None]).astype(np.int64)
