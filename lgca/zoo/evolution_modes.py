"""Contact inhibition and the mode of tumour evolution: whether cells keep moving in a full tissue.

A new model (the research model ``evo_steric`` of earlier biolgca versions), after Noble et al.
(2022), who showed that the spatial structure of a tumour and the way its cells disperse decide how
it evolves: by selective sweeps, by branching or almost neutrally.

Every node is a deme, a gland, that holds at most ``K`` cells (a hard capacity). Cells divide with
their own birth rate ``r_b`` as long as the node has room and die with probability ``r_d``, so a
full node turns over like a Moran process. A daughter acquires a driver mutation with probability
``r_m``: it founds a clone and its birth rate is multiplied by a fitness factor. After birth and
death, cells pick channels: a velocity channel pointing to a node with relative density ``ρ`` has
the weight ``exp(-α ρ)`` (contact inhibition: cells avoid moving into crowded nodes), the rest
channel ``exp(γ)`` (cells prefer to rest). In a full tissue every direction is crowded, so the
larger ``α``, the fewer cells move; at the edge of a colony cells still move out into free space.
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
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.rules import reorientation_term
from lgca.simulation import FamilyPopulationRecorder, PopulationRecorder, Schedule
from lgca.zoo._card import Parameter, ZooEntry
from lgca.zoo._clones import clonal_indices

CARD = ZooEntry(
    name="evolution_modes",
    title="Contact inhibition and the mode of tumour evolution",
    question="Does it matter for how a tumour evolves whether its cells keep moving through the crowded tissue "
             "or stop where it is full?",
    authors="Syga S, Deutsch A",
    paper="the research model evo_steric of earlier biolgca versions, after Noble R et al. (2022), Spatial "
          "structure governs the mode of tumour evolution, Nat Ecol Evol 6: 207-217, "
          "doi:10.1038/s41559-021-01615-9",
    year=None,
    venue="biolgca model zoo",
    doi=None,
    reproduces="no paper: clonal diversity D and drivers per cell n (the indices of Noble et al.) as the "
               "motility of cells in crowded tissue decreases",
    mechanisms=("driver mutations", "clones", "hard carrying capacity", "contact inhibition of locomotion",
                "resting bias"),
    lattice="hexagonal, one rest channel, identity-based without volume exclusion",
    fidelity="new model",
)

B = 6  # velocity channels of the hexagonal lattice


@reorientation_term(coupling="channels", name="evolution_modes.contact_inhibition")
def contact_inhibition(state):
    """Channel i scores minus the relative density n/K of the node it points to (0 beyond the edge)."""
    return -state.neighbor_values(state.density / state.capacity)


PARAMETERS = {
    "r_b": Parameter("r_b", "initial birth rate while the node has room", 0.2, "state.traits.r_b"),
    "r_d": Parameter("r_d", "death probability", 0.1, "dynamics.operators[birth_death].death_rate"),
    "r_m": Parameter("r_m", "probability that a daughter acquires a driver mutation", 1e-4,
                     "dynamics.operators[birth_death].mutation.probability"),
    "fitness": Parameter("f", "factor of the birth rate of a mutated daughter", 1.1,
                         "dynamics.operators[birth_death].mutation.traits.r_b.value"),
    "alpha": Parameter("α", "contact inhibition: weight exp(-α ρ) of a move into a node of relative density ρ",
                       0.0, "dynamics.operators[1].terms[evolution_modes.contact_inhibition].beta"),
    "gamma": Parameter("γ", "resting bias: weight exp(γ) of the rest channel", 0.0,
                       "dynamics.operators[1].terms[resting_bias].beta"),
    "capacity": Parameter("K", "cells a node (a gland) holds at most", 64, "state.capacity"),
}


def build_spec(full: bool = False, *, alpha: float = 0.0, gamma: float = 0.0, r_b: float = 0.2, r_d: float = 0.1,
               r_m: float = 1e-4, fitness: float = 1.1, capacity: int | None = None, start: str = "centre",
               size: int | None = None, steps: int | None = None, record_every: int = 20,
               seed: int | None = 1) -> ModelSpec:
    """A tissue of glands with driver mutations on a hexagonal lattice.

    Parameters
    ----------
    full : bool, default=False
        60 × 60 glands of K = 512 cells (the gland size of Noble et al.) for 2000 steps; by default
        40 × 40 glands of 64 cells for 1000 steps.
    alpha, gamma : float
        Contact inhibition α and resting bias γ of the movement.
    r_b, r_d : float
        Initial birth rate and death probability.
    r_m, fitness : float
        Probability of a driver mutation per daughter and its factor on the birth rate.
    capacity : int, optional
        Cells a node holds at most; default 64, 512 with ``full``.
    start : {"centre", "full"}, default="centre"
        One full node at the centre, from which the tumour grows, or every node full of resting
        cells of one clone (an established tissue).
    size, steps : int, optional
        Override the side of the lattice and the number of steps.
    record_every : int, default=20
        Record the population and the cells of every clone every so many steps.
    seed : int or None, default=1
        Seed of the run.
    """
    if start not in ("full", "centre"):
        raise ValueError(f"start must be 'full' or 'centre', got {start!r}")
    size = size or (60 if full else 40)
    capacity = capacity or (512 if full else 64)
    nodes = np.zeros((size, size, B + 1), dtype=np.int64)
    if start == "full":
        nodes[..., -1] = capacity
    else:
        nodes[size // 2, size // 2, -1] = capacity
    driver = {"probability": r_m, "traits": {"r_b": {"value": fitness, "operation": "multiply"}}}
    every = Schedule(every=record_every)
    return ModelSpec(
        description=Description(title=f"{CARD.title}: α = {alpha}, γ = {gamma}", details=CARD.paper),
        space=SpaceSpec(geometry="hex", dims=(size, size), boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits={"r_b": r_b}),
        time=TimeSpec(steps=steps if steps is not None else 2000 if full else 1000, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # death with r_d; division with r_b while the node has room (hard capacity); drivers found clones
            {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": r_d, "crowding": False,
                                                   "mutation": driver, "new_family": True}},
            # velocity channels weighted exp(-α ρ_neighbour), the rest channel exp(γ)
            ReorientationSpec(terms=[ReorientationTermSpec("evolution_modes.contact_inhibition", beta=alpha),
                                     ReorientationTermSpec("resting_bias", beta=gamma)]),
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder(every), FamilyPopulationRecorder(every)]),
    )


def moving_fraction(alpha: float, gamma: float = 0.0, density: float = 1.0) -> float:
    """Fraction of cells that move when all neighbours have the relative density ``density``.

    Each of the b = 6 velocity channels has the weight exp(-α ρ), the rest channel exp(γ).
    """
    moving = B * np.exp(-alpha * density)
    return float(moving / (moving + np.exp(gamma)))


def record(spec: ModelSpec, every: int = 20):
    """Run the model and record, every ``every`` steps, the indices of Noble et al. and more.

    Returns ``(run, record)``: the result of :func:`lgca.model.run_model` (with the family
    populations for Muller plots) and a dict with ``"steps"``, ``"cells"``, ``"n"`` (drivers per
    cell), ``"D"`` (clonal diversity), ``"clones"`` and ``"r_b"`` (the mean birth rate).
    """
    from dataclasses import replace

    from lgca.lattice_state import LatticeState
    from lgca.model import run_model
    from lgca.simulation import CallbackObserver

    out = {key: [] for key in ("steps", "cells", "n", "D", "clones", "r_b")}

    def measure(lattice, step):
        cells = LatticeState(lattice).cells
        indices = clonal_indices(lattice)
        out["steps"].append(step)
        out["cells"].append(len(cells))
        for key in ("n", "D", "clones"):
            out[key].append(indices[key])
        out["r_b"].append(float(np.mean(cells["r_b"])) if len(cells) else np.nan)

    observers = [*spec.analysis.observers, CallbackObserver(measure, Schedule(every=every))]
    run = run_model(replace(spec, analysis=replace(spec.analysis, observers=observers)), showprogress=False)
    return run, {key: np.array(values) for key, values in out.items()}
