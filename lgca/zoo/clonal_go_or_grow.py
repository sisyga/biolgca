"""Clonal evolution of the go-or-grow switch: driver mutations change how fast cells divide and when they move.

A new model (the research model ``go_or_grow_glioblastoma`` of earlier biolgca versions), after the
go-or-grow hypothesis for glioblastoma: a cell either migrates or rests and divides. It rests with
probability ``r_κ(ρ_N) = (1 + tanh(κ (ρ_N - θ))) / 2``, where ``ρ_N`` is the mean density of its
node and the neighbouring nodes over the capacity. Every cell carries two traits, its birth rate
``r_b`` and its switch parameter κ. A daughter acquires a driver mutation with probability ``r_m``:
it founds a new clone, its birth rate is multiplied by a fitness factor and its κ takes a normal
step. Mutations are rare, so a tumour consists of clones that differ in both traits.

The tumour grows from one node of cells that switch independently of their neighbours (κ = 0).
The question is which clones win, where, and whether the switch evolves along with the birth rate;
comparing with κ fixed (``evolve_kappa=False``) separates the two traits.
"""

from __future__ import annotations

import numpy as np

from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import FamilyPopulationRecorder, PopulationRecorder, Schedule
from lgca.zoo._card import Parameter, ZooEntry
from lgca.zoo._clones import clonal_indices, dominant_clone

CARD = ZooEntry(
    name="clonal_go_or_grow",
    title="Clonal evolution of the go-or-grow switch",
    question="When driver mutations change both how fast cells divide and when they switch between migrating "
             "and dividing, which clones win in a growing tumour, and where?",
    authors="Syga S, Deutsch A",
    paper="the research model go_or_grow_glioblastoma of earlier biolgca versions",
    year=None,
    venue="biolgca model zoo",
    doi=None,
    reproduces="no paper: the clones, the switch parameter κ and the birth rate across a growing tumour, with "
               "and without evolution of κ",
    mechanisms=("go-or-grow switch", "driver mutations", "clones", "evolving traits", "death", "logistic division",
                "random walk"),
    lattice="hexagonal, one rest channel, identity-based without volume exclusion",
    fidelity="new model",
)

PARAMETERS = {
    "r_b": Parameter("r_b", "initial birth rate of resting cells on an empty node", 0.2, "state.traits.r_b"),
    "r_d": Parameter("r_d", "death probability of every cell", 0.05, "dynamics.operators[birth_death].death_rate"),
    "r_m": Parameter("r_m", "probability that a daughter acquires a driver mutation", 0.01,
                     "dynamics.operators[go_or_grow.growth].mutation.probability"),
    "fitness": Parameter("f", "factor of the birth rate of a mutated daughter", 1.1,
                         "dynamics.operators[go_or_grow.growth].mutation.traits.r_b.value"),
    "theta": Parameter("θ", "neighbourhood density at which both phenotypes are equally likely", 0.5,
                       "dynamics.operators[go_or_rest].theta"),
    "kappa": Parameter("κ₀", "initial switch parameter of all cells (independent switching)", 0.0,
                       "state.traits.kappa"),
    "kappa_std": Parameter("Δκ", "standard deviation of the change of κ in a mutated daughter (research model: 0.2)", 1.0,
                           "dynamics.operators[go_or_grow.growth].mutation.traits.kappa.scale"),
    "capacity": Parameter("K", "carrying capacity of a node; the tumour starts from K resting cells", 50,
                          "state.capacity"),
}


def build_spec(full: bool = False, *, evolve_kappa: bool = True, r_b: float = 0.2, r_d: float = 0.05,
               r_m: float = 0.01, fitness: float = 1.1, theta: float = 0.5, kappa: float = 0.0,
               kappa_std: float = 1.0, capacity: int = 50, size: int | None = None, steps: int | None = None,
               record_every: int = 10, seed: int | None = 1) -> ModelSpec:
    """A tumour growing from ``capacity`` resting cells at the centre of a hexagonal lattice.

    Parameters
    ----------
    full : bool, default=False
        250 × 250 nodes and 700 steps; by default 150 × 150 and 400 steps. The tumour grows by about
        0.15 nodes per step and stays clear of the edge.
    evolve_kappa : bool, default=True
        Whether a driver mutation also changes κ. With False, κ stays at ``kappa`` in every cell
        and only the birth rate evolves.
    r_b, r_d : float
        Initial birth rate of resting cells (it falls as r_b (1 − n/K)) and death probability.
    r_m, fitness : float
        Probability of a driver mutation per daughter and its factor on the birth rate.
    theta, kappa, kappa_std : float
        Switch threshold θ, initial switch parameter κ₀ of all cells and the standard deviation Δκ
        of its change in a mutated daughter. The research model's Δκ = 0.2 changes κ so slowly that
        little happens within 400 steps.
    capacity : int, default=50
        Carrying capacity K of a node.
    size, steps : int, optional
        Override the side of the lattice and the number of steps.
    record_every : int, default=10
        Record the population and the cells of every clone every so many steps (for Muller plots).
    seed : int or None, default=1
        Seed of the run.
    """
    size = size or (250 if full else 150)
    nodes = np.zeros((size, size, 7), dtype=np.int64)
    nodes[size // 2, size // 2, -1] = capacity  # K resting cells at the centre
    driver = {"probability": r_m, "traits": {"r_b": {"value": fitness, "operation": "multiply"}}}
    if evolve_kappa:
        driver["traits"]["kappa"] = {"distribution": "normal", "scale": kappa_std}
    every = Schedule(every=record_every)
    return ModelSpec(
        description=Description(title=f"{CARD.title}: κ {'evolves' if evolve_kappa else 'fixed'}",
                                details=CARD.citation),
        space=SpaceSpec(geometry="hex", dims=(size, size), boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits={"r_b": r_b, "kappa": kappa}),
        time=TimeSpec(steps=steps if steps is not None else 700 if full else 400, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # rest with (1 + tanh(κ (ρ_N − θ))) / 2, κ of each cell, ρ_N of the node and its neighbours
            {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": theta, "density": "neighbourhood"}},
            {"name": "birth_death", "parameters": {"death_rate": r_d}},
            # resting cells divide with r_b (1 − n/K); a mutated daughter founds a clone
            {"name": "go_or_grow.growth", "parameters": {"r_b": "r_b", "r_d": 0.0, "mutation": driver,
                                                         "new_family": True}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}},
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder(every), FamilyPopulationRecorder(every)]),
    )


def node_maps(lgca) -> dict[str, np.ndarray]:
    """Per node: the cells, the mean κ and the mean birth rate (NaN where empty) and the dominant clone."""
    from lgca.plot_data import mean_trait

    return {"cells": lgca.cell_density[lgca.nonborder].copy(), "kappa": mean_trait(lgca, "kappa"),
            "r_b": mean_trait(lgca, "r_b"), "clone": dominant_clone(lgca)}


def core_and_rim(lgca, rim: float = 0.1) -> dict[str, float]:
    """Mean κ and birth rate of the cells in the core (the inner half by distance from the centre), at
    the rim (the outermost fraction ``rim``) and of all cells, with the cells and clones of the tumour.

    An extinct population has zero cells and clones and NaN trait means.
    """
    from lgca.lattice_state import LatticeState
    from lgca.zoo.phenotypic_plasticity import _distance

    cells = LatticeState(lgca).cells
    if not len(cells):
        return {"cells": 0, "clones": 0,
                **{trait: {region: float("nan") for region in ("core", "rim", "all")}
                   for trait in ("kappa", "r_b")}}
    distance = _distance(lgca)[cells.node]
    inner, outer = np.quantile(distance, [0.5, 1 - rim])
    summary = {"cells": len(cells), "clones": clonal_indices(lgca)["clones"]}
    for trait in ("kappa", "r_b"):
        values = np.asarray(cells[trait], dtype=float)
        summary[trait] = {"core": float(values[distance <= inner].mean()),
                          "rim": float(values[distance >= outer].mean()), "all": float(values.mean())}
    return summary
