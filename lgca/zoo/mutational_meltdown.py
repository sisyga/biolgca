"""Drivers, passengers and mutational meltdown: a tumour in a tug-of-war with its own mutations.

Syga S (2023). Evolutionary cellular automata for tumor growth and invasion. PhD thesis, TU Dresden,
chapter "The interplay of invasion and mutational meltdown". Code: Zenodo,
https://doi.org/10.5281/zenodo.10014813

Cells divide with their own proliferation rate ``α`` times ``1 - n/K`` and die with probability
``δ``. A daughter acquires a *passenger* mutation with probability ``p_p``, which lowers its ``α``
a little, and, rarely, a *driver* with probability ``p_d``, which raises it a lot; the effects are
exponentially distributed with means ``α₀ s_p`` and ``α₀ s_d``. The parameters are those of
McFarland et al. (2014) for tumours. Passengers are so weak that selection barely sees them, and
they accumulate by drift; drivers must be found and fixed. In a population of about ``N`` cells,
drivers win if ``N`` exceeds the critical size ``N* = T_p s_p / (T_d s_d²)``; below it the mean
fitness declines and the population melts down (Muller's ratchet). Invasion, i.e. spreading into
free space, can carry a small tumour past ``N*``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
)
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import PopulationRecorder, Schedule
from lgca.zoo._card import Parameter, ZooEntry

CARD = ZooEntry(
    name="mutational_meltdown",
    title="Drivers, passengers and mutational meltdown",
    question="Rare strong drivers against frequent weak passengers: when does a tumour's own mutational load "
             "drive it extinct, and can invasion rescue it?",
    authors="Syga S",
    paper="Evolutionary cellular automata for tumor growth and invasion (PhD thesis), chapter 'The interplay "
          "of invasion and mutational meltdown'",
    year=2023,
    venue="TU Dresden",
    doi="10.5281/zenodo.10014813",
    reproduces="the tug-of-war in isolated nodes (N₀ = N*/2, N*, 2N*: extinction, persistence, growth), invasion "
               "at three rest-channel weights, and a coarse version of the probability-of-cancer diagram",
    mechanisms=("driver and passenger mutations", "Muller's ratchet", "logistic division", "death",
                "random walk with resting"),
    lattice="1D, one rest channel, identity-based without volume exclusion",
    fidelity="same rules; the scan on a coarse grid with fewer runs and a step limit",
)

#: Parameters of McFarland et al. (2014) as used in the thesis (Table 5.1): mutation probability per
#: locus and division, driver and passenger loci, relative effects of drivers and passengers
P_LOCUS, T_D, T_P, S_D, S_P = 1e-8, 700, 5e6, 0.2, 1e-3


def critical_size(t_d: float = T_D, t_p: float = T_P, s_d: float = S_D, s_p: float = S_P) -> float:
    """The critical population size ``N* = T_p s_p / (T_d s_d²)`` of McFarland et al. (2014), about 179."""
    return t_p * s_p / (t_d * s_d ** 2)


PARAMETERS = {
    "alpha_0": Parameter("α₀", "initial proliferation rate of every cell", 0.5, "state.traits.r_b"),
    "delta": Parameter("δ", "death probability (0.25 in isolated nodes)", 0.375,
                       "dynamics.operators[birth_death].death_rate"),
    "p_p": Parameter("p_p = T_p p_l", "probability of a passenger mutation per daughter", T_P * P_LOCUS,
                     "dynamics.operators[birth_death].mutation[0].probability"),
    "p_d": Parameter("p_d = T_d p_l", "probability of a driver mutation per daughter", T_D * P_LOCUS,
                     "dynamics.operators[birth_death].mutation[1].probability"),
    "s_p": Parameter("s_p", "mean relative effect of a passenger (absolute: α₀ s_p)", S_P),
    "s_d": Parameter("s_d", "mean relative effect of a driver (absolute: α₀ s_d)", S_D),
    "gamma": Parameter("γ", "rest-channel weight: a cell moves with probability 2 / (2 + e^γ)", 8.0,
                       "dynamics.operators[1].terms[resting_bias].beta"),
    "N0": Parameter("N₀ / N*", "initial cells over the critical size; the capacity is K = N₀ / (1 − δ/α₀)", 0.25),
}


def _mutations(alpha_0, p_d, p_p, s_d, s_p):
    """Passengers subtract, drivers add an exponentially distributed change of α (kept in [0, 1])."""
    return [
        {"probability": p_p, "traits": {"r_b": {"distribution": "exponential", "scale": alpha_0 * s_p,
                                                "operation": "subtract", "bounds": [0, 1]}}},
        {"probability": p_d, "traits": {"r_b": {"distribution": "exponential", "scale": alpha_0 * s_d,
                                                "bounds": [0, 1]}}},
    ]


def build_spec(full: bool = False, *, n0: float = 0.25, gamma: float = 8.0, alpha_0: float = 0.5,
               delta: float = 0.375, p_d: float = T_D * P_LOCUS, p_p: float = T_P * P_LOCUS, s_d: float = S_D,
               s_p: float = S_P, length: int | None = None, steps: int | None = None, record_every: int = 20,
               seed: int | None = 1) -> ModelSpec:
    """A tumour of ``N₀ = n0 N*`` cells at the left end of a 1D tissue, which it can invade.

    As in the thesis code: the capacity is ``K = N₀ / (1 − δ/α₀)``, so that ``N₀`` is the
    equilibrium of one node, and the tissue has ``⌈4 N* / N₀⌉`` nodes, enough to exceed ``N*``
    by far. :func:`outcome` runs it until it dies out or reaches ``max(2 N₀, N*)`` cells.

    Parameters
    ----------
    full : bool, default=False
        Allow 100 000 steps instead of 20 000 (the time limit of :func:`outcome` and the run).
    n0 : float, default=0.25
        Initial cells over the critical size N*.
    gamma : float, default=8
        Rest-channel weight γ; a cell moves with probability 2 / (2 + e^γ).
    alpha_0, delta : float
        Initial proliferation rate α₀ and death probability δ (0.375 as in the thesis code of the
        scan; with 0.25, as in its text, a founder cell survives more often and invasion is easier).
    p_d, p_p, s_d, s_p : float
        Probabilities of a driver and a passenger per daughter and their mean relative effects.
    length : int, optional
        Override the number of nodes.
    steps : int, optional
        Override the number of steps.
    record_every : int, default=20
        Record the population every so many steps.
    seed : int or None, default=1
        Seed of the run.
    """
    n_star = critical_size()
    capacity = max(1, round(n0 * n_star / (1 - delta / alpha_0)))
    cells = max(1, round(capacity * (1 - delta / alpha_0)))
    length = length or int(np.ceil(4 * n_star / cells))
    nodes = np.zeros((length, 3), dtype=np.int64)
    nodes[0, -1] = cells
    return ModelSpec(
        description=Description(title=f"{CARD.title}: N₀ = {cells}, γ = {gamma}", details=CARD.citation),
        space=SpaceSpec(geometry="lin", dims=length, boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits={"r_b": alpha_0}),
        time=TimeSpec(steps=steps if steps is not None else 100_000 if full else 20_000, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # death with δ, division with α (1 − n/K); daughters may carry a passenger and a driver
            {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": delta,
                                                   "mutation": _mutations(alpha_0, p_d, p_p, s_d, s_p)}},
            # every cell takes a channel: the two velocity channels with weight 1, the rest channel e^γ
            ReorientationSpec(terms=[ReorientationTermSpec("resting_bias", beta=gamma)]),
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder(Schedule(every=record_every))]),
    )


def isolated_nodes(n0: float = 1.0, nodes: int = 10, *, alpha_0: float = 0.5, delta: float = 0.25,
                   p_d: float = T_D * P_LOCUS, p_p: float = T_P * P_LOCUS, s_d: float = S_D, s_p: float = S_P,
                   steps: int = 40_000, seed: int | None = 1) -> ModelSpec:
    """``nodes`` isolated populations of ``N₀ = n0 N*`` cells each, as in the thesis (after McFarland).

    The capacity is ``K = N₀ / (1 − δ/α₀)`` (``K = 2 N₀`` with δ = 0.25). The cells stay in the rest
    channel of their node: only they die and divide, and daughters go to the rest channel too, so
    the nodes do not exchange cells and are independent repetitions.
    """
    n_star = critical_size()
    capacity = round(n0 * n_star / (1 - delta / alpha_0))
    cells = round(capacity * (1 - delta / alpha_0))
    state = np.zeros((nodes, 3), dtype=np.int64)
    state[:, -1] = cells
    return ModelSpec(
        description=Description(title=f"{CARD.title}: isolated nodes, N₀ = {cells}", details=CARD.citation),
        space=SpaceSpec(geometry="lin", dims=nodes, boundary="reflecting"),
        state=StateSpec(nodes=state, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits={"r_b": alpha_0}),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": delta, "channels": "rest",
                                                   "mutation": _mutations(alpha_0, p_d, p_p, s_d, s_p)}},
        ]),
    )


def node_histories(spec: ModelSpec, every: int = 100) -> dict[str, np.ndarray]:
    """Run isolated nodes and record the cells and the mean α of every node every ``every`` steps.

    Returns ``"steps"``, ``"cells"`` (steps × nodes) and ``"alpha"`` (steps × nodes, NaN once a
    node has died out).
    """
    from lgca.lattice_state import LatticeState
    from lgca.model import build_model

    model = build_model(spec)
    lattice = model.lgca
    size = lattice.dims[0]
    out = {"steps": [], "cells": [], "alpha": []}
    for step in range(spec.time.steps + 1):
        if step % every == 0:
            cells = LatticeState(lattice).cells
            node = cells.node[0]
            count = np.bincount(node, minlength=size)
            with np.errstate(invalid="ignore", divide="ignore"):
                alpha = np.bincount(node, weights=np.asarray(cells["r_b"], dtype=float), minlength=size) / count
            out["steps"].append(step)
            out["cells"].append(count)
            out["alpha"].append(alpha)
        if step < spec.time.steps:
            model.step()
    return {key: np.array(values) for key, values in out.items()}


def outcome(spec: ModelSpec, every: int = 1, history: bool = False):
    """Run the tumour until it dies out, reaches ``max(2 N₀, N*)`` cells or runs out of steps.

    Returns ``"cancer"``, ``"extinct"`` or ``"undecided"`` and the number of steps; with
    ``history``, also a dict of ``"steps"``, ``"cells"``, ``"extent"`` (occupied nodes) and
    ``"alpha"`` (mean proliferation rate), recorded every ``every`` steps.
    """
    from lgca.lattice_state import LatticeState
    from lgca.model import build_model

    model = build_model(replace(spec, analysis=AnalysisSpec()))
    lattice = model.lgca
    target = max(2 * int(spec.state.nodes.sum()), critical_size())
    record = {"steps": [], "cells": [], "extent": [], "alpha": []}
    result = "undecided"
    for step in range(spec.time.steps + 1):
        density = lattice.cell_density[lattice.nonborder]
        total = int(density.sum())
        if history and (step % every == 0 or total == 0 or total >= target):
            record["steps"].append(step)
            record["cells"].append(total)
            record["extent"].append(int(np.count_nonzero(density)))
            record["alpha"].append(float(np.mean(LatticeState(lattice).cells["r_b"])) if total else np.nan)
        if total == 0 or total >= target:
            result = "extinct" if total == 0 else "cancer"
            break
        if step < spec.time.steps:
            model.step()
    if history:
        return result, step, {key: np.array(values) for key, values in record.items()}
    return result, step


def _scan_one(arguments):
    n0, gamma, seed, options = arguments
    return outcome(build_spec(n0=n0, gamma=gamma, seed=seed, **options))[0]


def scan(n0s, gammas, seeds=range(10), n_jobs: int = 1, **options):
    """The outcome for every initial size, rest-channel weight and seed (the thesis' probability of cancer).

    Returns a :class:`pandas.DataFrame` with the columns ``n0``, ``gamma``, ``seed`` and
    ``outcome``. ``options`` go to :func:`build_spec` (e.g. ``p_p=0`` for a tumour without
    mutations); ``n_jobs`` runs that many processes.
    """
    import pandas as pd

    n0s, gammas, seeds = list(n0s), list(gammas), list(seeds)
    bad = [seed for seed in seeds
           if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0]
    if bad:  # int() would run 1.9 and True as seed 1
        raise ValueError(f"seeds must be non-negative integers, got {bad[0]!r}")
    tasks = [(float(n0), float(gamma), int(seed), options) for n0 in n0s for gamma in gammas for seed in seeds]
    if n_jobs == 1:
        results = [_scan_one(task) for task in tasks]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context(method)) as pool:
            results = list(pool.map(_scan_one, tasks))  # one run per task: their lengths differ a lot
    table = pd.DataFrame(tasks, columns=["n0", "gamma", "seed", "options"]).drop(columns="options")
    table["outcome"] = results
    return table
