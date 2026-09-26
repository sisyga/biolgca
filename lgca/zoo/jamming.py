"""Jamming transitions in invasion: cell–cell adhesion and matrix confinement set the invasion mode.

Ilina O, Gritsenko PG, Syga S, Lippoldt J, La Porta CAM, Chepizhko O, Grosser S, Vullings M,
Bakker GJ, Starruß J, Bult P, Käs JA, Zapperi S, Deutsch A, Friedl P (2020). Cell–cell adhesion and
3D matrix confinement determine jamming transitions in breast cancer invasion. Nat Cell Biol 22:
1103–1115. https://doi.org/10.1038/s41556-020-0552-6

The model of the paper's Fig 5, from its supplement (github.com/sisyga/jamminglgca,
``definition.pdf``). Cells on a hexagonal lattice with volume exclusion, 6 velocity and 3 rest
channels, reorient in one Boltzmann decision, ``P(s') ∝ exp(E(s'))``, with

- steric repulsion from nodes above the homeostatic density ρ₀ (strength β_steric),
- confinement by the extracellular matrix (ECM), a field ρ_ECM that makes cells rest,
- adhesion: cells keep an optimal number of neighbours and align with them (strength β).

Moving cells open up the matrix: ρ_ECM ← ρ_ECM (1 − α n/K) at every step. New cells flow in from
the source (the lower edge of the lattice in the paper, or a spheroid) at rate r_b per free
channel. The invasion mode is read from the cumulative number of single cells and the velocity
correlation between neighbouring nodes.

The terms and rules below are written with the public decorators, as an example of adding the
rules of a paper to the library.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from lgca.lattice_state import LatticeState
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
from lgca.rules import interaction, reorientation_term
from lgca.simulation import ScalarTimeSeriesRecorder, Schedule
from lgca.zoo._card import Parameter, ZooEntry

CARD = ZooEntry(
    name="jamming",
    title="Jamming transitions in invasion",
    question="How do cell–cell adhesion and confinement by the matrix decide whether cancer cells invade as a "
             "jammed sheet, a fluid sheet or as single cells?",
    authors="Ilina O, Gritsenko PG, Syga S, Lippoldt J, La Porta CAM, Chepizhko O, Grosser S, Vullings M, "
            "Bakker GJ, Starruß J, Bult P, Käs JA, Zapperi S, Deutsch A, Friedl P",
    paper="Cell–cell adhesion and 3D matrix confinement determine jamming transitions in breast cancer invasion",
    year=2020,
    venue="Nat Cell Biol 22: 1103–1115",
    doi="10.1038/s41556-020-0552-6",
    reproduces="Fig 5d (the invasion modes for low and high adhesion and matrix density) and a coarse version "
               "of Fig 5e (single cells and velocity correlation over both); beyond the paper, invasion "
               "from a spheroid",
    mechanisms=("adhesion", "alignment", "steric repulsion", "matrix confinement", "matrix degradation",
                "influx"),
    lattice="hexagonal, 6 velocity and 3 rest channels, classical with volume exclusion, reflecting",
    fidelity="same rules, from the supplement's model definition; a coarser phase diagram by default",
)

K = 9  # channels per node: 6 velocity and 3 rest channels
RHO_0 = 3.0  # homeostatic density ρ₀, cells per node

PARAMETERS = {
    "beta": Parameter("β", "adhesion strength (aggregation and alignment)", 1.0,
                      "dynamics.operators[1].terms[jamming.adhesion].beta"),
    "ecm": Parameter("ρ̄_ECM", "initial density of the matrix", 1.0, "state.fields.ecm"),
    "beta_steric": Parameter("β_steric", "strength of the steric repulsion", 5.0,
                             "dynamics.operators[1].terms[jamming.pressure].beta"),
    "alpha": Parameter("α", "rate at which cells open up the matrix", 1.0,
                       "dynamics.operators[jamming.degradation].alpha"),
    "rho_0": Parameter("ρ₀", "homeostatic density, cells per node", RHO_0),
    "r_b": Parameter("r_b", "influx: probability that a free channel of the source is filled", 0.05,
                     "dynamics.operators[jamming.influx].rate"),
    "K": Parameter("K", "channels per node, 3 of them rest channels", K),
}


# ------------------------------------------------------------------ the rules of the paper

def _gradient(state, values):
    """Σ_i c_i values(r + c_i): the discrete gradient of the paper (zero beyond the walls)."""
    return state.neighbor_values(values) @ state.c.T


@reorientation_term(coupling="channels", name="jamming.adhesion")
def adhesion(state, rho_0=RHO_0, scale=1.0):
    """E_aggregation + E_alignment / β of the paper: weights per channel, to be scaled by β.

    Aggregation: j(s') · ∇u with u = scale · n_nb (1 − n_nb/n_crit)⁺ / (2 n_crit), n_nb the cells on
    the neighbouring nodes and n_crit = (b + 1) ρ₀. Alignment: j(s') · J_nb / (2b) with J_nb the flux
    of the neighbours, and n_rest(s') n_rest,nb / (b ρ₀) with n_rest,nb the resting cells of the
    neighbours.

    Parameters
    ----------
    rho_0 : float
        Homeostatic density ρ₀, cells per node.
    scale : float
        Factor of the adhesion potential u: 1 is the model definition of the supplement, 4 the
        original code.
    """
    b = state.velocitychannels
    n = state.density.astype(float)
    neighbours = state.neighbor_sum(n)
    n_crit = (b + 1) * rho_0
    u = scale * neighbours * np.clip(1 - neighbours / n_crit, 0, None) / (2 * n_crit)
    moving = _gradient(state, u) + state.neighbor_sum(state.flux) / (2 * b)
    resting = state.neighbor_sum(state.counts[..., b:].sum(axis=(-2, -1)).astype(float)) / (b * rho_0)
    weights = np.empty(state.dims + (state.K,))
    weights[..., :b] = moving @ state.c
    weights[..., b:] = resting[..., None]
    return weights


@reorientation_term(coupling="flux", name="jamming.pressure")
def pressure(state, rho_0=RHO_0):
    """Steric repulsion: j(s') · (−Σ c_i ρ̃(r + c_i)), ρ̃ = (n − ρ₀)/(K − ρ₀) above ρ₀ and 0 below.

    Parameters
    ----------
    rho_0 : float
        Homeostatic density ρ₀, cells per node.
    """
    excess = np.clip(state.density - rho_0, 0, None) / (state.K - rho_0)
    return -_gradient(state, excess)


@reorientation_term(coupling="rest", name="jamming.confinement")
def confinement(state, field="ecm"):
    """Confinement by the matrix: each resting cell scores ρ_ECM, so cells rest where it is dense.

    Parameters
    ----------
    field : str
        Name of the matrix field.
    """
    return state.field(field)


@interaction(kind="field", families="classical", name="jamming.degradation")
def degradation(state, field="ecm", alpha=1.0):
    """Cells open up the matrix: ρ_ECM ← ρ_ECM (1 − α n/K).

    Parameters
    ----------
    field : str
        Name of the matrix field.
    alpha : float
        Rate α at which cells open up the matrix; with α = 1 a full node removes it in one step.
    """
    state.set_field(field, state.field(field) * (1 - alpha * state.density / state.K))


@interaction(kind="birth_death", families="classical", name="jamming.influx")
def influx(state, source="source", rate=0.05):
    """Cells flow in: every free channel of the source nodes is filled with probability ``rate``.

    Parameters
    ----------
    source : str
        Name of the field that marks the source: nodes where it is positive.
    rate : float
        Probability r_b that a free channel of the source is filled in a step.
    """
    counts = state.counts
    at_source = (state.field(source) > 0)[..., None, None]
    new = (counts == 0) & at_source & (state.rng.random(counts.shape) < rate)
    state.counts = counts + new


# ------------------------------------------------------------------ observables

def single_cells(lgca) -> int:
    """Cells without another cell on their node or the neighbouring nodes."""
    state = LatticeState(lgca)
    n = state.density
    return int(np.count_nonzero((n == 1) & (state.neighbor_sum(n) == 0)))


def local_correlation(lgca) -> np.ndarray:
    """Velocity correlation of every node with its neighbours: the cosine of the angle between the node's flux j
    and the neighbours' flux J_nb; NaN where either vanishes."""
    state = LatticeState(lgca)
    flux = state.flux
    neighbours = state.neighbor_sum(flux)
    norms = np.linalg.norm(flux, axis=-1) * np.linalg.norm(neighbours, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 1e-12, np.sum(flux * neighbours, axis=-1) / norms, np.nan)


def velocity_correlation(lgca) -> float:
    """Mean next-neighbour velocity correlation over the nodes where it is defined."""
    values = local_correlation(lgca)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def invasion_mode(result) -> dict[str, float]:
    """The observables of Fig 5e: single cells summed over the observed steps, and the mean correlation."""
    return {"single_cells": float(np.sum(result.data["single_cells"])),
            "correlation": float(np.nanmean(result.data["correlation"]))}


def total_single_cells(result) -> float:
    """Single cells summed over the observed steps (for :func:`lgca.study.sweep`)."""
    return invasion_mode(result)["single_cells"]


def mean_correlation(result) -> float:
    """Mean next-neighbour velocity correlation over the observed steps (for :func:`lgca.study.sweep`)."""
    return invasion_mode(result)["correlation"]


# ------------------------------------------------------------------ the model

def build_spec(full: bool = False, *, beta: float = 1.0, ecm: float = 1.0, setup: str = "sheet",
               beta_steric: float = 5.0, alpha: float = 1.0, rate: float = 0.05, rho_0: float = RHO_0,
               adhesion_scale: float = 1.0,
               size: int | None = None, radius: int = 6, transient: int = 50, steps: int = 200,
               seed: int | None = 1) -> ModelSpec:
    """The model of Ilina et al. (2020), Fig 5.

    Parameters
    ----------
    full : bool, default=False
        Has no effect on a single run, which has the paper's size (50 × 50, 50 + 200 steps); the
        paper's full phase diagram is :data:`FULL_GRID`.
    beta : float, default=1.0
        Adhesion strength β (β_align = β_ag = β in the paper).
    ecm : float, default=1.0
        Initial, uniform matrix density ρ̄_ECM.
    setup : {"sheet", "spheroid"}, default="sheet"
        ``"sheet"``: the paper's interface assay, two rows of cells at the lower edge, which is also
        the source. ``"spheroid"``: a disc of radius ``radius`` in the centre of the lattice, which
        is the source; the cells invade in all directions (beyond the paper).
    beta_steric, alpha, rate, rho_0 : float
        Steric strength β_steric, matrix degradation α, influx r_b and homeostatic density ρ₀.
    adhesion_scale : float, default=1.0
        Factor of the adhesion potential u: 1 is the model definition of the supplement, 4 the
        original code.
    size : int, optional
        Side of the lattice; default 50 for the sheet, 80 for the spheroid.
    radius : int, default=6
        Radius of the spheroid.
    transient, steps : int
        Steps before the observables are recorded, and steps recorded.
    seed : int or None, default=1
        Seed of the run and of the initial cells.
    """
    if setup not in ("sheet", "spheroid"):
        raise ValueError(f"setup must be 'sheet' (the paper's) or 'spheroid', got {setup!r}")
    size = size or (50 if setup == "sheet" else 80)
    dims = (size, size)
    rng = np.random.default_rng(seed)
    if setup == "sheet":
        source = np.zeros(dims)
        source[:, :2] = 1.0  # the rows y = 0 and 1
    else:
        from lgca import get_lgca

        coordinates = get_lgca(geometry="hex", dims=dims, density=0.0, restchannels=3)
        x, y = coordinates.xcoords, coordinates.ycoords
        centre = (size // 2, size // 2)
        source = (np.hypot(x - x[centre], y - y[centre]) <= radius).astype(float)
    # ρ₀ cells per source node, in random channels
    nodes = np.zeros(dims + (K,), dtype=bool)
    for index in np.argwhere(source > 0):
        nodes[tuple(index)][rng.choice(K, size=int(rho_0), replace=False)] = True
    series = ScalarTimeSeriesRecorder(
        metrics={"single_cells": single_cells, "correlation": velocity_correlation},
        schedule=Schedule(steps=range(transient + 1, transient + steps + 1)),
        output_path=Path(tempfile.gettempdir()) / "lgca_zoo_jamming.csv",
    )
    return ModelSpec(
        description=Description(title=f"{CARD.title}: β = {beta}, ρ̄_ECM = {ecm}, {setup}", details=CARD.citation),
        space=SpaceSpec(geometry="hex", dims=dims, boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=K - 6,
                        fields={"ecm": float(ecm), "source": source}),
        time=TimeSpec(steps=transient + steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # the order of the original code: influx, reorientation, matrix degradation
            {"name": "jamming.influx", "parameters": {"source": "source", "rate": rate}},
            ReorientationSpec(terms=[
                ReorientationTermSpec("jamming.pressure", beta=beta_steric, parameters={"rho_0": rho_0}),
                ReorientationTermSpec("jamming.confinement", beta=1.0, parameters={"field": "ecm"}),
                ReorientationTermSpec("jamming.adhesion", beta=beta,
                                      parameters={"rho_0": rho_0, "scale": adhesion_scale}),
            ]),
            {"name": "jamming.degradation", "parameters": {"field": "ecm", "alpha": alpha}},
        ]),
        analysis=AnalysisSpec(observers=[series]),
    )


#: β and ρ̄_ECM of the paper's phase diagram (Fig 5e): 0 to 10 in steps of 0.2, 5 seeds each
FULL_GRID = {"beta": np.round(np.arange(0, 10.01, 0.2), 1), "ecm": np.round(np.arange(0, 10.01, 0.2), 1)}
