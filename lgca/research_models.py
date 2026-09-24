"""Research models of earlier biolgca versions, rebuilt from the generic rules.

Each model step is a :func:`~lgca.rules.stack` of the built-in rules, so the
functions below double as worked examples of how to put a published model
together: a phenotype switch, growth with mutations and movement, in the
order the model applies them. They are registered under the names of the
legacy interactions without the family prefix, e.g.
``{"name": "go_or_grow_glioblastoma", "parameters": {"r_m": 0.01}}``, and work
in identity-based models with and without volume exclusion.

Parameters named in ``traits=`` also give the initial value of that cell
trait, unless ``StateSpec.traits`` sets it. Capacities come from
``StateSpec.capacity``.

Differences from the legacy code, all small:

- traits that the legacy models stored per family (``family_props["r_b"]``
  and ``["kappa"]`` in ``go_or_grow_glioblastoma``, ``evo_steric`` and
  ``go_and_grow_mutations``) are cell traits here. They only change when a
  mutated daughter founds a family, so every cell of a family has its
  family's value, and the dynamics are the same;
- cues computed from the cell density (``aggregation``, ``steric_repulsion``)
  see the density after the growth of the step, where the legacy code used
  the density at its start;
- ``go_or_grow_kappa_chemo`` averages the neighbourhood density over the node
  and its neighbours, as ``go_or_grow_kappa`` does; the legacy code divided
  the sum over ``velocitychannels + 1`` nodes by ``velocitychannels``.
"""

from __future__ import annotations

import numpy as np

from .lattice_state import random_occupancy
from .pipeline import ReorientationSpec, ReorientationTermSpec
from .rules import interaction, stack

__all__ = ["birthdeath_cancerdfe", "birthdeath_discrete", "evo_steric", "excitable_medium",
           "go_and_grow_mutations", "go_or_grow_glioblastoma", "go_or_grow_kappa", "go_or_grow_kappa_chemo"]

_IDENTITY = ("ib", "nove_ib")


@stack(kind="birth_death", families=_IDENTITY, traits="r_b", name="birthdeath_cancerdfe")
def birthdeath_cancerdfe(state, r_b=0.2, r_d=0.02, p_d=1.4e-5, p_p=0.1, s_d=None, s_p=None, a_max=1.0,
                         gamma=0.0):
    """Logistic growth in which daughters acquire driver and passenger mutations of their birth rate.

    Parameters
    ----------
    r_b : float
        Initial birth rate of the cells (the trait ``r_b``).
    r_d : float
        Death rate.
    p_d : float
        Probability that a daughter acquires a driver mutation, which raises
        its birth rate.
    p_p : float
        Probability that a daughter acquires a passenger mutation, which
        lowers its birth rate; a daughter can acquire both.
    s_d : float
        Mean effect of a driver mutation, exponentially distributed.
        Default: ``0.1 * r_b``.
    s_p : float
        Mean effect of a passenger mutation, exponentially distributed.
        Default: ``0.001 * r_b``.
    a_max : float
        Largest birth rate.
    gamma : float
        Preference for rest channels when cells pick new channels.
    """
    s_d = 0.1 * r_b if s_d is None else s_d
    s_p = 0.001 * r_b if s_p is None else s_p
    passenger = {"distribution": "exponential", "scale": s_p, "operation": "subtract", "bounds": [None, a_max]}
    driver = {"distribution": "exponential", "scale": s_d, "bounds": [None, a_max]}
    return [
        {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": r_d, "mutation": [
            {"probability": p_p, "traits": {"r_b": passenger}},
            {"probability": p_d, "traits": {"r_b": driver}},
        ]}},
        {"name": "resting_bias", "parameters": {"beta": gamma}},
    ]


@stack(kind="birth_death", families=_IDENTITY, traits="r_b", name="birthdeath_discrete")
def birthdeath_discrete(state, r_b=0.2, r_d=0.02, drb=0.01, a_max=1.0, pmut=0.1):
    """Cells die, then divide; a daughter's birth rate mutates up or down by a fixed step.

    Parameters
    ----------
    r_b : float
        Initial birth rate of the cells (the trait ``r_b``).
    r_d : float
        Death rate.
    drb : float
        Step of a mutation of the birth rate, up or down with equal chance.
    a_max : float
        Largest birth rate.
    pmut : float
        Probability that a daughter mutates.
    """
    step = {"distribution": "choice", "a": [-drb, drb], "bounds": [None, a_max]}
    return [
        {"name": "birth_death", "parameters": {"death_rate": r_d}},
        {"name": "birth_death", "parameters": {"birth_rate": "r_b", "mutation": {
            "probability": pmut, "traits": {"r_b": step}}}},
        {"name": "random_walk"},
    ]


@stack(kind="birth_death", families=_IDENTITY, traits="r_b", name="go_and_grow_mutations")
def go_and_grow_mutations(state, r_b=0.5, r_d=0.02, r_m=1e-3, effect="passenger_mutation",
                          fitness_increase=1.1):
    """Cells die, then divide; mutated daughters found new families, with a higher birth rate for drivers.

    Parameters
    ----------
    r_b : float
        Initial birth rate of the cells (the trait ``r_b``).
    r_d : float
        Death rate.
    r_m : float
        Probability that a daughter mutates and founds a new family.
    effect : {"passenger_mutation", "driver_mutation"}
        Passenger mutations leave the birth rate as it is; driver mutations
        multiply it by ``fitness_increase``.
    fitness_increase : float
        Factor of a driver mutation.
    """
    if effect not in ("passenger_mutation", "driver_mutation"):
        raise ValueError(f"effect must be 'passenger_mutation' or 'driver_mutation', got {effect!r}")
    traits = {"r_b": {"value": fitness_increase, "operation": "multiply"}} if effect == "driver_mutation" else {}
    return [
        {"name": "birth_death", "parameters": {"death_rate": r_d}},
        {"name": "birth_death", "parameters": {"birth_rate": "r_b", "new_family": True,
                                               "mutation": {"probability": r_m, "traits": traits}}},
        {"name": "random_walk"},
    ]


@stack(kind="birth_death", families=_IDENTITY, traits="r_b", name="evo_steric")
def evo_steric(state, r_b=0.1, r_d=None, r_m=1e-3, fitness_increase=1.1, alpha=2.0, gamma=3.0):
    """Logistic growth with driver mutations; cells avoid crowded neighbours and prefer to rest.

    Parameters
    ----------
    r_b : float
        Initial birth rate of the cells (the trait ``r_b``).
    r_d : float
        Death rate. Default: ``0.98 * r_b``.
    r_m : float
        Probability that a daughter acquires a driver mutation, which founds
        a new family and multiplies its birth rate by ``fitness_increase``.
    fitness_increase : float
        Factor of a driver mutation.
    alpha : float
        Strength of the steric repulsion from crowded neighbouring nodes.
    gamma : float
        Preference for rest channels.
    """
    r_d = 0.98 * r_b if r_d is None else r_d
    return [
        {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": r_d, "new_family": True,
                                               "mutation": {"probability": r_m, "traits": {"r_b": {
                                                   "value": fitness_increase, "operation": "multiply"}}}}},
        ReorientationSpec(terms=[ReorientationTermSpec("steric_repulsion", beta=alpha),
                                 ReorientationTermSpec("resting_bias", beta=gamma)]),
    ]


@stack(kind="birth_death", families=_IDENTITY, traits="kappa", name="go_or_grow_kappa")
def go_or_grow_kappa(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.5, kappa_std=0.2):
    """Go-or-grow with a switch steepness kappa per cell that mutates, sensing the neighbourhood density.

    Cells rest with probability ``(1 + tanh(kappa * (rho - theta))) / 2``,
    with ``rho`` the mean density of the node and its neighbours over the
    capacity; then they die; resting cells divide, and their daughters rest
    and inherit kappa with a normal change; moving cells pick new velocity
    channels.

    Parameters
    ----------
    r_b : float
        Birth rate of resting cells.
    r_d : float
        Death rate.
    kappa : float
        Initial switch steepness of the cells (the trait ``kappa``).
    theta : float
        Relative density at which half of the cells rest.
    kappa_std : float
        Standard deviation of the change of kappa in daughters.
    """
    return [
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": theta, "density": "neighbourhood"}},
        {"name": "birth_death", "parameters": {"death_rate": r_d}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": r_b, "r_d": 0.0, "mutation": {"kappa": kappa_std}}},
        {"name": "random_walk", "parameters": {"channels": "velocity"}},
    ]


@stack(kind="birth_death", families=_IDENTITY, traits="kappa", name="go_or_grow_kappa_chemo")
def go_or_grow_kappa_chemo(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.5, kappa_std=0.2, beta=5.0):
    """go_or_grow_kappa in which moving cells move up the gradient of the relative cell density.

    Parameters
    ----------
    r_b : float
        Birth rate of resting cells.
    r_d : float
        Death rate.
    kappa : float
        Initial switch steepness of the cells (the trait ``kappa``).
    theta : float
        Relative density at which half of the cells rest.
    kappa_std : float
        Standard deviation of the change of kappa in daughters.
    beta : float
        Sensitivity to the gradient of the density over the capacity.
    """
    return [
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": theta, "density": "neighbourhood"}},
        {"name": "birth_death", "parameters": {"death_rate": r_d}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": r_b, "r_d": 0.0, "mutation": {"kappa": kappa_std}}},
        # up the gradient of the density over the capacity, among the velocity channels
        {"name": "aggregation", "parameters": {"beta": beta / state.capacity, "channels": "velocity"}},
    ]


@stack(kind="birth_death", families=_IDENTITY, traits=("r_b", "kappa"), name="go_or_grow_glioblastoma")
def go_or_grow_glioblastoma(state, r_b=0.2, r_d=0.01, r_m=1e-3, fitness_increase=1.1, theta=0.5, kappa=5.0,
                            kappa_std=0.2):
    """Go-or-grow in which driver mutations found clones with a higher birth rate and a new kappa.

    As :func:`go_or_grow_kappa`, but a daughter mutates with probability
    ``r_m``: it founds a new family, its birth rate is multiplied by
    ``fitness_increase`` and its kappa changes by a normal step.

    Parameters
    ----------
    r_b : float
        Initial birth rate of resting cells (the trait ``r_b``).
    r_d : float
        Death rate.
    r_m : float
        Probability that a daughter acquires a driver mutation.
    fitness_increase : float
        Factor of the birth rate of a mutated daughter.
    theta : float
        Relative density at which half of the cells rest.
    kappa : float
        Initial switch steepness of the cells (the trait ``kappa``).
    kappa_std : float
        Standard deviation of the change of kappa in mutated daughters.
    """
    driver = {"probability": r_m, "traits": {
        "r_b": {"value": fitness_increase, "operation": "multiply"},
        "kappa": {"distribution": "normal", "scale": kappa_std},
    }}
    return [
        # cells rest or move depending on the density around them, each with its own kappa
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": theta, "density": "neighbourhood"}},
        {"name": "birth_death", "parameters": {"death_rate": r_d}},
        # resting cells divide; mutated daughters found clones
        {"name": "go_or_grow.growth", "parameters": {"r_b": "r_b", "r_d": 0.0, "mutation": driver,
                                                     "new_family": True}},
        {"name": "random_walk", "parameters": {"channels": "velocity"}},
    ]


@interaction(kind="birth_death", families="classical", name="excitable_medium")
def excitable_medium(state, alpha=1.0, beta=0.05, N=50):
    """Excitable medium after Barkley: activator cells move, inhibitor cells rest.

    With one species, activators occupy velocity channels and inhibitors
    rest channels; with two, species 1 are activators in velocity channels
    and species 0 inhibitors in rest channels. Per time step the
    inhibitors change by one reaction and the activators by ``N`` fast ones;
    their densities are the occupied fractions ``x`` and ``y`` of these
    channels. Activators are added with probability
    ``x² (1 + (y + beta) / alpha)`` and removed with ``x³ + x (y + beta) /
    alpha``; inhibitors are added with probability ``x`` and removed with
    ``y``.

    Parameters
    ----------
    alpha : float
        Excitability of the medium.
    beta : float
        Threshold of excitation.
    N : int
        Fast activator reactions per time step.
    """
    if state.n_species not in (1, 2):
        raise ValueError(f"excitable_medium works with one or two species, not {state.n_species}")
    if state.restchannels < 1:
        raise ValueError("excitable_medium needs rest channels for the inhibitor")
    velocity, rest, rng = state.velocitychannels, state.restchannels, state.rng
    activator, inhibitor = (0, 0) if state.n_species == 1 else (1, 0)
    counts = state.counts
    n_x = counts[..., activator, :velocity] @ np.ones(velocity, dtype=np.int64)
    n_y = counts[..., inhibitor, velocity:] @ np.ones(rest, dtype=np.int64)
    rho_y = n_y / rest
    inhibition = (rho_y + beta) / alpha
    n_y = np.clip(n_y + _net_change(rng, n_x / velocity, rho_y), 0, rest)
    # nodes without activators keep none (both rates vanish), so the fast reactions run on the others
    active = np.flatnonzero(n_x)
    x, inhibition = n_x.ravel()[active], inhibition.ravel()[active]
    rho = np.arange(velocity + 1) / velocity
    rho2 = rho * rho
    for _ in range(int(N)):
        square = rho2[x]
        x = np.clip(x + _net_change(rng, square * (1 + inhibition), rho[x] * (square + inhibition)), 0, velocity)
    n_x = np.zeros_like(n_x)
    n_x.ravel()[active] = x
    # activators in random velocity channels; the rest channels are all alike
    new = np.zeros_like(counts)
    new[..., activator, :velocity] = random_occupancy(rng, n_x, velocity)
    new[..., inhibitor, velocity:] = np.arange(rest) < n_y[..., None]
    state.counts = new


def _net_change(rng, gain, loss):
    """+1 with probability ``gain``, -1 with ``loss``, independently: their sum, from one random number."""
    gain, loss = np.clip(gain, 0, 1), np.clip(loss, 0, 1)
    draws = rng.random(np.shape(gain))
    return (draws < gain * (1 - loss)).astype(np.int64) - (draws >= 1 - loss * (1 - gain))
