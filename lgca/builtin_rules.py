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
        {"name": "channel_random_walk", "parameters": {"channels": "velocity"}},
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
from .rules import interaction, reorientation_term

__all__ = ["channel_random_walk", "go_or_grow_growth", "go_or_grow_switch", "go_or_rest"]


# Terms of the Boltzmann reorientation (ReorientationSpec). J(s') is the flux of
# the candidate state; see lgca.rules.reorientation_term for the couplings.

@reorientation_term(coupling="rest", name="random_walk", aliases="uniform")
def random_walk(state):
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
def polar_alignment(state):
    """J_nb · J(s'), with J_nb the flux of the neighbouring nodes: cells move with their neighbours."""
    return state.neighbor_sum(state.flux)


@reorientation_term(coupling="channels", name="nematic_alignment", aliases=("nematic", "alignment"))
def nematic_alignment(state):
    """Cells share an axis with neighbouring cells; opposite directions count the same."""
    neighbours = state.neighbor_sum(state.counts[..., :state.velocitychannels].sum(axis=-2))
    return neighbours @ (state.c.T @ state.c) ** 2


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
    """Σ (d · c_i)² over occupied velocity channels: cells move along the axis of a director field.

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
    return (director @ state.c) ** 2

_MIGRATING, _RESTING = 0, 1
_CAPACITY_MODES = ("legacy", "reject")


@interaction(kind="reorientation", families=("classical", "nove", "ib", "nove_ib"), name="channel_random_walk")
def channel_random_walk(state, channels="all", species=None):
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
def go_or_rest(state, kappa=5.0, theta=0.75, capacity="legacy"):
    """Moving cells start resting on crowded nodes, resting cells start moving on sparse ones.

    Parameters
    ----------
    kappa : float or str
        Steepness of the switch. With kappa > 0 crowded cells rest, with
        kappa < 0 they move. In identity-based models, the name of a cell
        trait gives every cell its own value.
    theta : float or str
        Density (cells per node over capacity) at which half of the cells
        rest, or the name of a cell trait.
    capacity : {"legacy", "reject"}
        With volume exclusion, what happens when a switching cell finds no free
        channel. "legacy" reproduces the original rule: the number of switching
        cells is drawn from the cells that fit into free channels. "reject":
        every cell tries to switch, and switches into full channels fail.
    """
    _check_mode(capacity)
    if state.restchannels < 1:
        raise ValueError("go_or_rest needs at least one rest channel for resting cells")
    if state.identity_based:
        _go_or_rest_cells(state, kappa, theta, capacity)
        return
    rest = tanh_switch(state.density / _node_capacity(state), _number(kappa, "kappa"), _number(theta, "theta"))
    velocity, rng = state.velocitychannels, state.rng
    cells = state.counts[..., 0, :]
    moving, resting = cells[..., :velocity], cells[..., velocity:]
    if state.volume_exclusion:
        n_m, n_r = moving.sum(-1), resting.sum(-1)
        free_rest, free_velocity = state.restchannels - n_r, velocity - n_m
        if capacity == "legacy":
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
def go_or_grow_switch(state, kappa=5.0, theta=0.75, capacity="legacy"):
    """Migrating cells start resting in crowded nodes, resting cells start migrating in sparse ones.

    The two-species form of go_or_rest: migrating cells are species 0 (in
    velocity channels), resting cells species 1 (in rest channels).

    Parameters
    ----------
    kappa : float
        Steepness of the switch. With kappa > 0 crowded cells rest, with
        kappa < 0 they migrate.
    theta : float
        Density (cells per node over capacity) at which half of the cells rest.
    capacity : {"legacy", "reject"}
        With volume exclusion, what happens when a switching cell finds no free
        channel. "legacy" reproduces the original rule: the number of switching
        cells is drawn from the cells that fit into free channels. "reject":
        every cell tries to switch, and switches into full channels fail.
    """
    _check_layout(state)
    _check_mode(capacity)
    rest = tanh_switch(state.density / _node_capacity(state), kappa, theta)
    if state.volume_exclusion and capacity == "legacy":
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
def go_or_grow_growth(state, r_b=0.2, r_d=0.01, r_d_resting=None, capacity="legacy", mutation=None,
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
    capacity : {"legacy", "reject"}
        With volume exclusion, how divisions meet full rest channels. "legacy"
        reproduces the original rule: the number of divisions is drawn from as
        many resting cells as there are free rest channels. "reject": every
        resting cell tries to divide, and divisions into full channels fail.
    mutation : dict or None
        Identity-based models: traits of the daughters that mutate, with the
        standard deviation of a normal change, e.g. ``{"kappa": 0.2}``.
    new_family : bool
        Identity-based models: every daughter founds a new family, for
        lineage analyses such as Muller plots.
    """
    _check_mode(capacity)
    if state.identity_based:
        _growth_cells(state, r_b, r_d, r_d_resting, capacity, mutation or {}, new_family)
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
    if state.volume_exclusion and capacity == "legacy":
        n_r = (state.counts * resting).sum(axis=(-2, -1))
        births = np.zeros(state.dims + (state.n_species,), dtype=np.int64)
        births[..., resting_species] = state.rng.binomial(np.minimum(n_r, state.restchannels - n_r), r_b)
        state.add_cells(births, channels={resting_species: "rest"})
        return
    rate = r_b if state.volume_exclusion else np.clip(r_b * (1 - crowding), 0, 1)
    division = np.where(resting, np.asarray(rate, dtype=float)[..., None, None], 0.0)
    state.divide_cells(np.broadcast_to(division, state.dims + resting.shape),
                       channels={resting_species: "rest"})


def _go_or_rest_cells(state, kappa, theta, capacity):
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
        if capacity == "legacy":  # the cells that fit try to switch
            to_rest = cells.pick(~resting, free_rest) & to_rest
            to_move = cells.pick(resting, free_velocity) & to_move
        else:  # every cell tries; the successful ones are chosen at random
            to_rest = cells.pick(to_rest, free_rest)
            to_move = cells.pick(to_move, free_velocity)
    cells.move(to_rest, "rest")
    cells.move(to_move, "velocity")


def _growth_cells(state, r_b, r_d, r_d_resting, capacity, mutation, new_family):
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
        if capacity == "legacy":  # as many resting cells as there are free rest channels try
            dividing = cells.pick(resting, free_rest) & (rng.random(len(cells)) < birth)
        else:
            dividing = resting & (rng.random(len(cells)) < birth)
    else:
        rate = np.clip(birth * (1 - crowding[cells.node]), 0, 1)
        dividing = resting & (rng.random(len(cells)) < rate)
    daughters = cells.divide(dividing, channels="rest", new_family=new_family)
    for name, std in mutation.items():
        if not np.isfinite(std) or std < 0:
            raise ValueError(f"mutation[{name!r}] must be a non-negative standard deviation")
        values = cells[name][daughters]
        cells.set_trait(daughters, name, values + rng.normal(0.0, std, len(daughters)))


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


def _check_mode(capacity):
    if capacity not in _CAPACITY_MODES:
        raise ValueError(f"capacity must be one of {_CAPACITY_MODES}, got {capacity!r}")


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
