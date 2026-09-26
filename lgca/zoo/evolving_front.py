"""Evolution at an invasion front: the fastest-dividing cells gather at the front and speed it up.

Syga S, Nava-Sedeño JM, Deutsch A (2026). A novel cellular automaton approach for modeling genotypic
and phenotypic heterogeneity in cell systems. Eur Phys J Spec Top.
https://doi.org/10.1140/epjs/s11734-026-02186-1

An evolutionary LGCA (identity-based, without volume exclusion): every cell carries its own
proliferation rate α. In every step each cell dies with probability δ and divides with probability
α (1 − n/K); the daughter's α mutates with probability p_μ to a value drawn from a normal
distribution around the mother's. Then every cell takes a channel at random, a velocity channel
with weight 1 and the rest channel with weight e^γ, which sets the diffusion coefficient D.

The population invades empty space from the left edge of a long strip as a Fisher–KPP wave. Its
speed v = 2 √(D (ᾱ − δ)) depends on the proliferation rate ᾱ, which evolves: selection enriches the
front with fast-dividing cells, and the front accelerates.
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
from lgca.simulation import PopulationRecorder
from lgca.zoo._card import Parameter, ZooEntry

CARD = ZooEntry(
    name="evolving_front",
    title="Evolution at an invasion front",
    question="How does a population whose proliferation rate evolves invade empty space, and where do the "
             "fastest-dividing cells end up?",
    authors="Syga S, Nava-Sedeño JM, Deutsch A",
    paper="A novel cellular automaton approach for modeling genotypic and phenotypic heterogeneity in cell systems",
    year=2026,
    venue="Eur Phys J Spec Top",
    doi="10.1140/epjs/s11734-026-02186-1",
    reproduces="Fig 3: kymographs of the density and of the mean proliferation rate, the accelerating front "
               "against the Fisher-KPP predictions, and the mean proliferation rate over time",
    mechanisms=("evolving trait", "mutation", "logistic division", "death", "random walk with resting"),
    lattice="square strip, 4 velocity channels and 1 rest channel, identity-based without volume exclusion",
    fidelity="same rules; K and the run length are not given in the main text and are chosen here",
)

B = 4  # velocity channels of the square lattice


def gamma_for(D: float) -> float:
    """Rest-channel weight γ for the diffusion coefficient D.

    A cell moves with probability p = b / (b + e^γ) and then one node in one of the b = 4
    directions, so its displacement along an axis has variance p / 2 per step and D = p / 4.
    """
    moving = 4 * D
    if not 0 < moving < 1:
        raise ValueError(f"D must lie between 0 and 1/4 on the square lattice, got {D!r}")
    return float(np.log(B / moving - B))


PARAMETERS = {
    "alpha_0": Parameter("α₀", "initial proliferation rate of every cell", 0.2, "state.traits.r_b"),
    "delta": Parameter("δ", "death probability", 0.01, "dynamics.operators[birth_death].death_rate"),
    "p_mu": Parameter("p_μ", "probability that a daughter's α mutates", 1.0,
                      "dynamics.operators[birth_death].mutation.probability"),
    "sigma": Parameter("σ", "standard deviation of a mutation of α (σ² = 10⁻⁴)", 0.01,
                       "dynamics.operators[birth_death].mutation.traits.r_b.scale"),
    "D": Parameter("D", "diffusion coefficient, set by the rest-channel weight γ", 0.14),
    "gamma": Parameter("γ", "rest-channel weight, ln(b / 4D − b)", gamma_for(0.14),
                       "dynamics.operators[1].terms[resting_bias].beta"),
    "capacity": Parameter("K", "carrying capacity of a node (chosen here)", 100, "state.capacity"),
}


def build_spec(full: bool = False, *, alpha_0: float = 0.2, delta: float = 0.01, p_mu: float = 1.0,
               sigma: float = 0.01, D: float = 0.14, capacity: int = 100, length: int = 500, width: int = 10,
               start: int = 2, steps: int | None = None, seed: int | None = 1) -> ModelSpec:
    """The model of Syga et al. (2026), invading a strip from its left edge.

    Parameters
    ----------
    full : bool, default=False
        1400 steps, until the front approaches the right edge; by default 1000.
    alpha_0, delta, p_mu, sigma : float
        Initial proliferation rate α₀, death probability δ, mutation probability p_μ and standard
        deviation σ of a mutation of α (normal, kept within [0, 1]).
    D : float, default=0.14
        Diffusion coefficient; sets the rest-channel weight γ (:func:`gamma_for`).
    capacity : int, default=100
        Carrying capacity K of a node.
    length, width : int
        The strip, 500 × 10 nodes in the paper, reflecting.
    start : int, default=2
        Columns at the left edge that start at the equilibrium density Ψ₀ = 1 − δ/α₀.
    steps : int, optional
        Override the number of steps.
    seed : int or None, default=1
        Seed of the run and of the initial cells.
    """
    gamma = gamma_for(D)
    psi_0 = 1 - delta / alpha_0
    rng = np.random.default_rng(seed)
    nodes = np.zeros((length, width, B + 1), dtype=np.int64)
    weights = np.r_[np.ones(B), np.exp(gamma)]
    nodes[:start] = rng.multinomial(round(psi_0 * capacity), weights / weights.sum(), size=(start, width))
    mutation = {"probability": p_mu, "traits": {"r_b": {"distribution": "normal", "scale": sigma, "bounds": [0, 1]}}}
    return ModelSpec(
        description=Description(title=f"{CARD.title}: α₀ = {alpha_0}, δ = {delta}", details=CARD.citation),
        space=SpaceSpec(geometry="square", dims=(length, width), boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits={"r_b": alpha_0}),
        time=TimeSpec(steps=steps if steps is not None else 1400 if full else 1000, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # death with δ and division with α (1 − n/K) at the same time; every daughter's α mutates
            {"name": "birth_death", "parameters": {"death_rate": delta, "birth_rate": "r_b", "mutation": mutation}},
            # every cell takes a channel: velocity channels with weight 1, the rest channel with e^γ
            ReorientationSpec(terms=[ReorientationTermSpec("resting_bias", beta=gamma)]),
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder()]),
    )


def record(spec: ModelSpec, every: int = 10) -> dict[str, np.ndarray]:
    """Run the model and record, every ``every`` steps, what Fig 3 shows.

    Returns ``"steps"``; ``"density"`` and ``"alpha"``, the cells per node and their mean α at each
    x, averaged over y (steps × length; α is NaN where there are no cells); ``"front"``, the first x
    at which the density falls below 0.1 Ψ₀ K; and the mean, standard deviation and the mean of the
    fastest 10 % of α over all cells (``"alpha_mean"``, ``"alpha_std"``, ``"alpha_top"``).
    """
    from lgca.lattice_state import LatticeState
    from lgca.model import build_model

    model = build_model(spec)
    lattice = model.lgca
    capacity = spec.state.capacity
    delta = spec.dynamics.operators[0]["parameters"]["death_rate"]
    alpha_0 = float(spec.state.traits["r_b"])
    threshold = 0.1 * (1 - delta / alpha_0) * capacity
    out = {key: [] for key in ("steps", "density", "alpha", "front", "alpha_mean", "alpha_std", "alpha_top")}
    for step in range(spec.time.steps + 1):
        if step % every == 0:
            cells = LatticeState(lattice).cells
            alpha = np.asarray(cells["r_b"], dtype=float)
            x = cells.node[0]
            count = np.bincount(x, minlength=lattice.dims[0])
            density = count / lattice.dims[1]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_alpha = np.bincount(x, weights=alpha, minlength=lattice.dims[0]) / count
            out["steps"].append(step)
            out["density"].append(density)
            out["alpha"].append(mean_alpha)
            out["front"].append(front_position(density, threshold))
            out["alpha_mean"].append(alpha.mean())
            out["alpha_std"].append(alpha.std())
            out["alpha_top"].append(np.mean(np.sort(alpha)[-max(1, len(alpha) // 10):]))
        if step < spec.time.steps:
            model.step()
    return {key: np.array(values) for key, values in out.items()}


def front_position(density, threshold: float) -> int:
    """The first x at which the density falls below ``threshold``."""
    below = np.flatnonzero(np.asarray(density) < threshold)
    return int(below[0]) if len(below) else len(density)


def wave_speed(alpha, D: float = 0.14, delta: float = 0.01):
    """Fisher-KPP speed 2 √(D (α − δ)) (Eq 6); zero where α ≤ δ."""
    return 2 * np.sqrt(D * np.clip(np.asarray(alpha, dtype=float) - delta, 0, None))


def predicted_front(steps, alpha, x_0: float, D: float = 0.14, delta: float = 0.01) -> np.ndarray:
    """Front position if it moved at the Fisher-KPP speed of the proliferation rate ``alpha(t)``.

    ``alpha`` is a number or one value per entry of ``steps``; the speed is integrated with the
    trapezoidal rule from ``x_0`` at ``steps[0]``.
    """
    steps = np.asarray(steps, dtype=float)
    speed = wave_speed(np.broadcast_to(alpha, steps.shape), D, delta)
    increments = 0.5 * (speed[1:] + speed[:-1]) * np.diff(steps)
    return x_0 + np.concatenate([[0.0], np.cumsum(increments)])
