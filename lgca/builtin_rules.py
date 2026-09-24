"""Built-in interactions written with the :func:`lgca.interaction` decorator.

They work with and without volume exclusion and for any number of species
unless stated otherwise, and serve as worked examples of the decorator.

Go-or-grow as a two-species model
---------------------------------
Migrating cells are species 0 and live in velocity channels; resting cells are
species 1 and live in rest channels. One time step of the classical
go-or-grow model is the pipeline ::

    operators=[
        {"name": "go_or_grow.switch", "parameters": {"kappa": 4.0, "theta": 0.75}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01}},
        {"name": "species_random_walk", "parameters": {"species": 0, "channels": "velocity"}},
    ]

in this order, followed by propagation. The rules keep each species in its
channels, so the initial state must place migrating cells in velocity
channels and resting cells in rest channels.
"""

from __future__ import annotations

import numpy as np

from .interactions import tanh_switch
from .rules import interaction, reorientation_term

__all__ = ["go_or_grow_growth", "go_or_grow_switch", "species_random_walk"]


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


@interaction(kind="reorientation", families=("classical", "nove"), name="species_random_walk")
def species_random_walk(state, species=None, channels="all"):
    """Cells move to uniformly random channels of their node.

    Parameters
    ----------
    species : int, list of int or None
        Species that move; the others stay in their channels. Default: all.
    channels : str or list of int
        Channels the moving cells are spread over: ``"all"``, ``"velocity"``,
        ``"rest"`` or channel indices. Cells in other channels stay put.
    """
    state.shuffle_cells(channels, species=species)


@interaction(kind="phenotype_switch", families=("classical", "nove"), n_species=2, name="go_or_grow.switch")
def go_or_grow_switch(state, kappa=5.0, theta=0.75, capacity="legacy"):
    """Migrating cells start resting in crowded nodes, resting cells start migrating in sparse ones.

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
    if capacity not in _CAPACITY_MODES:
        raise ValueError(f"capacity must be one of {_CAPACITY_MODES}, got {capacity!r}")
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


@interaction(kind="birth_death", families=("classical", "nove"), n_species=2, name="go_or_grow.growth")
def go_or_grow_growth(state, r_b=0.2, r_d=0.01, r_d_resting=None, capacity="legacy"):
    """Cells die, and resting cells divide into free rest channels.

    Parameters
    ----------
    r_b : float
        Division probability of a resting cell. Without volume exclusion it is
        scaled by 1 - density / capacity.
    r_d : float
        Death probability of a migrating cell, and of a resting cell unless
        r_d_resting is given.
    r_d_resting : float or None
        Death probability of a resting cell. Default: r_d.
    capacity : {"legacy", "reject"}
        With volume exclusion, how divisions meet full rest channels. "legacy"
        reproduces the original rule: the number of divisions is drawn from as
        many resting cells as there are free rest channels. "reject": every
        resting cell tries to divide, and divisions into full channels fail.
    """
    _check_layout(state)
    if capacity not in _CAPACITY_MODES:
        raise ValueError(f"capacity must be one of {_CAPACITY_MODES}, got {capacity!r}")
    crowding = state.density / _node_capacity(state)
    state.remove_cells([r_d, r_d if r_d_resting is None else r_d_resting])
    if state.volume_exclusion and capacity == "legacy":
        n_r = state.species_density[..., _RESTING]
        births = np.zeros(state.dims + (2,), dtype=np.int64)
        births[..., _RESTING] = state.rng.binomial(np.minimum(n_r, state.restchannels - n_r), r_b)
        state.add_cells(births, channels={_RESTING: "rest"})
        return
    division = np.zeros(state.dims + (2,))
    division[..., _RESTING] = r_b if state.volume_exclusion else np.clip(r_b * (1 - crowding), 0, 1)
    state.divide_cells(division, channels={_RESTING: "rest"})


def go_or_grow_layout(state):
    """Move migrating cells to velocity channels and resting cells to rest channels.

    For random initial states, e.g. in :func:`lgca.testing.check_interaction`:
    cells in the wrong channels are dropped.
    """
    counts = state.counts.copy()
    velocity = state.velocitychannels
    counts[..., _MIGRATING, velocity:] = 0
    counts[..., _RESTING, :velocity] = 0
    state.counts = counts


def _node_capacity(state):
    """Cells a node can hold: its channels with volume exclusion, the model capacity without."""
    return state.K if state.volume_exclusion else state.capacity


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
