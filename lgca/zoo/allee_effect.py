"""An emerging Allee effect: go-or-grow makes small tumour cell populations die out.

Böttger K, Hatzikirou H, Voss-Böhme A, Cavalcanti-Adam EA, Herrero MA, Deutsch A (2015).
An emerging Allee effect is critical for tumor initiation and persistence.
PLoS Comput Biol 11(9): e1004366. https://doi.org/10.1371/journal.pcbi.1004366

Cells either move (velocity channels) or rest and divide (rest channels). A moving cell starts
resting with probability ``r_s(ϱ) = (1 + tanh(κ (ϱ - θ))) / 2``, a resting cell starts moving with
``1 - r_s(ϱ)``, where ``ϱ = n / K`` is the density of its node. With ``κ > 0`` cells in sparse
regions keep moving and rarely divide, so a population below a critical density shrinks and dies
out, although every cell could divide: an Allee effect that emerges from the switch.

One time step, as in the paper: death (R1), division of resting cells into free rest channels
(R2), the switch (R3), a random walk of the moving cells (R4), and propagation.
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
from lgca.simulation import DensityRecorder, PopulationRecorder, Schedule
from lgca.zoo._card import Parameter, ZooEntry

CARD = ZooEntry(
    name="allee_effect",
    title="An emerging Allee effect",
    question="Why do small tumour cell populations die out although every cell can divide?",
    authors="Böttger K, Hatzikirou H, Voss-Böhme A, Cavalcanti-Adam EA, Herrero MA, Deutsch A",
    paper="An emerging Allee effect is critical for tumor initiation and persistence",
    year=2015,
    venue="PLoS Comput Biol 11(9): e1004366",
    doi="10.1371/journal.pcbi.1004366",
    reproduces="Fig 3: extinction frequency against the initial density, and bimodal outcomes near "
               "the threshold; Fig 5: the per-capita growth rate of the mean-field model",
    mechanisms=("go-or-grow switch", "death", "division of resting cells", "random walk"),
    lattice="square, 4 velocity and 4 rest channels, classical with volume exclusion, periodic",
    fidelity="same rules; smaller lattice and shorter runs by default",
)

K = 8  # channels per node: 4 velocity and 4 rest channels

PARAMETERS = {
    "r_b": Parameter("r_b", "division probability of a resting cell", 0.2,
                     "dynamics.operators[go_or_grow.growth].r_b"),
    "r_d": Parameter("r_d", "death probability of every cell", 0.01,
                     "dynamics.operators[go_or_grow.growth].r_d"),
    "kappa": Parameter("κ", "switch intensity; κ > 0: crowded cells rest (attraction)", 4.4,
                       "dynamics.operators[go_or_rest].kappa"),
    "theta": Parameter("θ", "density at which moving and resting are equally likely", 0.75,
                       "dynamics.operators[go_or_rest].theta"),
    "K": Parameter("K", "channels per node (the carrying capacity)", K),
    # state.density counts cells per node, K ϱ₀; build_spec(density=...) takes ϱ₀
    "density": Parameter("ϱ₀", "initial density, occupied fraction of all channels", 0.26),
}


def build_spec(full: bool = False, *, density: float = 0.26, kappa: float = 4.4, theta: float = 0.75,
               r_b: float = 0.2, r_d: float = 0.01, steps: int | None = None, seed: int | None = 1,
               record_every: int = 10) -> ModelSpec:
    """The model of Böttger et al. (2015), starting from a uniform random density ``density``.

    Parameters
    ----------
    full : bool, default=False
        The paper's 100 × 100 lattice and 5000 steps; by default 50 × 50 and 1000 steps, in which
        the outcome is decided.
    density : float, default=0.26
        Initial density ϱ₀: every channel is occupied with this probability, so a node holds
        ``K ϱ₀`` cells on average (``state.density`` counts cells per node).
    kappa, theta : float
        Switch intensity κ and switch density θ; the defaults are those of Fig 3.
    r_b, r_d : float
        Division probability of resting cells and death probability of all cells.
    steps : int, optional
        Number of steps; default 5000 with ``full`` and 1000 without.
    seed : int or None, default=1
        Seed of the run.
    record_every : int, default=10
        Steps between two recorded densities (the population is recorded every step).
    """
    if not 0 <= density <= 1:
        raise ValueError(f"density is the occupied fraction of the channels, between 0 and 1; got {density!r}")
    size = 100 if full else 50
    return ModelSpec(
        description=Description(title=f"{CARD.title}: κ = {kappa}, θ = {theta}, ϱ₀ = {density}",
                                details=CARD.citation),
        space=SpaceSpec(geometry="square", dims=(size, size), boundary="periodic"),
        state=StateSpec(density=density * K, restchannels=K - 4),
        time=TimeSpec(steps=steps if steps is not None else 5000 if full else 1000, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # R1 and R2: every cell dies with r_d, then resting cells divide into free rest channels
            {"name": "go_or_grow.growth", "parameters": {"r_b": r_b, "r_d": r_d}},
            # R3: moving cells rest with r_s(n / K), resting cells move with 1 - r_s
            {"name": "go_or_rest", "parameters": {"kappa": kappa, "theta": theta}},
            # R4: moving cells pick a random velocity channel
            {"name": "random_walk", "parameters": {"channels": "velocity"}},
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder(),
                                         DensityRecorder(schedule=Schedule(every=record_every))]),
    )


def switch(rho, kappa: float = 4.4, theta: float = 0.75):
    """The switch probability r_s(ϱ) = (1 + tanh(κ (ϱ − θ))) / 2 of Eq (1)."""
    return 0.5 * (1 + np.tanh(kappa * (np.asarray(rho, dtype=float) - theta)))


def per_capita_growth(rho, kappa: float = 4.4, theta: float = 0.75, r_b: float = 0.2, r_d: float = 0.01):
    """Per-capita growth rate of the mean-field model, r_s(ϱ) r_b − r_d (Eq 2 and Fig 5).

    It assumes that switching is fast compared with division, so that a fraction r_s(ϱ) of the
    cells rests, and that the rest channels are not full. Negative at small ϱ means an Allee
    effect.
    """
    return switch(rho, kappa, theta) * r_b - r_d


def per_capita_growth_nodes(rho, kappa: float = 4.4, theta: float = 0.75, r_b: float = 0.2, r_d: float = 0.01):
    """As :func:`per_capita_growth`, averaged over the cells of nodes with binomial occupancy.

    In the lattice a node with ``n`` cells switches with ``r_s(n / K)``, not with the mean
    density. With every channel occupied with probability ϱ, ``n`` is binomial, and a cell sits
    on a node with ``n`` cells with probability ``n P(n) / (K ϱ)``. Because ``r_s`` is convex at
    small densities, crowded nodes grow although the mean density lies below the threshold of
    :func:`per_capita_growth`.
    """
    from scipy.stats import binom

    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    n = np.arange(K + 1)
    weights = n * binom.pmf(n[None, :], K, rho[:, None])  # cells on nodes with n cells
    rates = switch(n / K, kappa, theta) * r_b - r_d
    with np.errstate(invalid="ignore", divide="ignore"):
        growth = (weights * rates).sum(-1) / weights.sum(-1)
    growth = np.where(rho > 0, growth, rates[1])  # a lone cell in an empty lattice
    return growth if np.ndim(growth) and growth.size > 1 else float(growth[0])


def final_density(result) -> float:
    """Density at the end of a run: cells over all channels of the lattice."""
    lgca = result.lgca
    return float(lgca.cell_density[lgca.nonborder].sum() / (lgca.cell_density[lgca.nonborder].size * K))
