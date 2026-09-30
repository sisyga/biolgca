"""Evolution of phenotypic plasticity: the go-or-grow switch evolves in a growing tumour.

Syga S, Jain HP, Krellner M, Hatzikirou H, Deutsch A (2024). Evolution of phenotypic plasticity
leads to tumor heterogeneity with implications for therapy. PLoS Comput Biol 20(8): e1012003.
https://doi.org/10.1371/journal.pcbi.1012003

Every cell carries its own switch parameter κ, its genotype. A cell rests and divides (the
proliferative phenotype) with probability ``r_κ(ρ_N) = (1 + tanh(κ (ρ_N - θ))) / 2`` and migrates
otherwise, where ``ρ_N`` is the mean density of its node and the neighbouring nodes over the
capacity. κ > 0 is the attractive strategy (rest where it is crowded), κ < 0 the repulsive one
(migrate away from crowding), κ ≈ 0 the independent one. Daughters inherit κ with a normal change.

One time step, as in the paper: death (δ), the switch, division of resting cells with
``α (1 - n / K)`` (daughters rest, κ_d ~ N(κ_mother, Δκ²)), a random walk of migrating cells, and
propagation.
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
from lgca.simulation import PopulationRecorder
from lgca.zoo._card import Parameter, ZooEntry

CARD = ZooEntry(
    name="phenotypic_plasticity",
    title="Evolution of phenotypic plasticity",
    question="How does the switch between migrating and dividing evolve in a growing tumour, and where do the "
             "different strategies end up?",
    authors="Syga S, Jain HP, Krellner M, Hatzikirou H, Deutsch A",
    paper="Evolution of phenotypic plasticity leads to tumor heterogeneity with implications for therapy",
    year=2024,
    venue="PLoS Comput Biol 20(8): e1012003",
    doi="10.1371/journal.pcbi.1012003",
    reproduces="Fig 3 in two dimensions (S1-S3 Figs): density, phenotypes and the switch parameter κ in the "
               "three evolutionary regimes; Fig 3 A-F in one dimension",
    mechanisms=("go-or-grow switch", "evolving trait", "mutation", "death", "logistic division", "random walk"),
    lattice="hexagonal (or 1D), one rest channel, identity-based without volume exclusion",
    fidelity="same rules; the 2D runs (S1-S3 Figs) on a smaller lattice, the 1D runs (Fig 3) at the paper's size",
)

#: (θ, δ) of the three regimes of Fig 3 and S1-S3 Figs
REGIMES = {
    1: {"theta": 0.5, "delta": 0.0},  # negligible death: independent cells, many migrate
    2: {"theta": 0.2, "delta": 0.2},  # death, low threshold: attractive core, repulsive rim
    3: {"theta": 0.9, "delta": 0.2},  # death, high threshold: repulsive throughout
}

PARAMETERS = {
    "alpha": Parameter("α", "division probability of a resting cell on an empty node", 1.0,
                       "dynamics.operators[go_or_grow.growth].r_b"),
    "delta": Parameter("δ", "death probability of every cell", 0.2, "dynamics.operators[birth_death].death_rate"),
    "theta": Parameter("θ", "neighbourhood density at which both phenotypes are equally likely", 0.2,
                       "dynamics.operators[go_or_rest].theta"),
    "kappa_std": Parameter("Δκ", "standard deviation of the change of κ in a daughter", 0.2,
                           "dynamics.operators[go_or_grow.growth].mutation.kappa"),
    "capacity": Parameter("K", "carrying capacity of a node (100 in 1D)", 50, "state.capacity"),
    "cells": Parameter("N₀", "initial cells, resting at the central node (K)", 50),
    "kappa_0": Parameter("κ₀", "initial κ of the cells, uniform on this interval", (-4.0, 4.0)),
}


def build_spec(full: bool = False, *, geometry: str = "hex", theta: float = 0.2, delta: float = 0.2,
               alpha: float = 1.0, capacity: int | None = None, kappa_std: float = 0.2, cells: int | None = None,
               kappa_0=(-4.0, 4.0), steps: int | None = None, size: int | None = None,
               seed: int | None = 1) -> ModelSpec:
    """The model of Syga et al. (2024), growing from ``cells`` resting cells at the centre.

    Parameters
    ----------
    full : bool, default=False
        On the hexagonal lattice the paper's 2D runs, 250 × 250 nodes and 300 steps; by default
        120 × 120 and 200 steps. In 1D the paper's size is the default.
    geometry : {"hex", "lin"}, default="hex"
        The 2D runs of S1-S3 Figs, or the 1D runs of Fig 3 (1001 nodes, K = 100, 1000 steps).
    theta, delta : float
        Switch threshold θ and death probability δ; the defaults are regime 2, see :data:`REGIMES`.
    alpha : float, default=1.0
        Division probability α of a resting cell on an empty node; it falls as α (1 − n/K).
    capacity : int, optional
        Carrying capacity K of a node; default 50 on the hexagonal lattice, 100 in 1D.
    kappa_std : float, default=0.2
        Standard deviation Δκ of the change of κ from mother to daughter.
    cells : int, optional
        Initial cells, in the rest channel of the central node; default K.
    kappa_0 : (float, float), default=(-4, 4)
        Interval of the uniform distribution of the initial κ.
    steps, size : int, optional
        Override the number of steps and the side of the lattice.
    seed : int or None, default=1
        Seed of the run; it also draws the initial κ.
    """
    if geometry == "hex":
        size, capacity = size or (250 if full else 120), capacity or 50
        default_steps, dims = (300 if full else 200), None
    elif geometry == "lin":
        size, capacity = size or 1001, capacity or 100
        default_steps, dims = 1000, None
    else:
        raise ValueError(f"geometry must be 'hex' (2D, S1-S3 Figs) or 'lin' (1D, Fig 3), got {geometry!r}")
    cells = capacity if cells is None else cells
    dims = (size, size) if geometry == "hex" else (size,)
    channels = 7 if geometry == "hex" else 3  # the last channel is the rest channel
    nodes = np.zeros(dims + (channels,), dtype=np.int64)
    nodes[tuple(n // 2 for n in dims) + (channels - 1,)] = cells
    kappa = np.random.default_rng(seed).uniform(*kappa_0, cells)
    return ModelSpec(
        description=Description(title=f"{CARD.title}: θ = {theta}, δ = {delta}", details=CARD.citation),
        space=SpaceSpec(geometry=geometry, dims=dims if geometry == "hex" else size, boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits={"kappa": kappa}),
        time=TimeSpec(steps=steps if steps is not None else default_steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"death_rate": delta}},
            # rest with (1 + tanh(κ (ρ_N − θ))) / 2, κ of each cell, ρ_N of the node and its neighbours
            {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": theta, "density": "neighbourhood"}},
            # resting cells divide with α (1 − n/K); daughters rest, κ_d ~ N(κ_mother, Δκ²)
            {"name": "go_or_grow.growth", "parameters": {"r_b": alpha, "r_d": 0.0, "mutation": {"kappa": kappa_std}}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}},
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder()]),
    )


def kymographs(spec: ModelSpec, every: int = 10) -> dict[str, np.ndarray]:
    """Run a 1D model and record the cells and the mean κ per node every ``every`` steps (Fig 3 A-F).

    Returns ``"steps"``, ``"cells"`` (steps × nodes) and ``"kappa"`` (steps × nodes, NaN where empty).
    """
    from lgca.model import build_model
    from lgca.plot_data import mean_trait

    model = build_model(spec)
    lgca = model.lgca
    steps, cells, kappa = [], [], []
    for step in range(spec.time.steps + 1):
        if step % every == 0:
            steps.append(step)
            cells.append(lgca.cell_density[lgca.nonborder].copy())
            kappa.append(mean_trait(lgca, "kappa"))
        if step < spec.time.steps:
            model.step()
    return {"steps": np.array(steps), "cells": np.array(cells), "kappa": np.array(kappa), "lgca": lgca}


def regime(number: int, **options) -> ModelSpec:
    """The model in regime 1, 2 or 3 of the paper, e.g. ``regime(3, full=True)``."""
    if number not in REGIMES:
        raise KeyError(f"the paper has regimes 1, 2 and 3, not {number!r}")
    return build_spec(**REGIMES[number], **options)


def node_maps(lgca) -> dict[str, np.ndarray]:
    """Per node: all cells, migrating cells, resting cells, and the mean κ (NaN where empty)."""
    from lgca.plot_data import mean_trait

    counts = lgca._channel_counts(lgca.nodes[lgca.nonborder])
    migrating = counts[..., :lgca.velocitychannels].sum(-1)
    resting = counts[..., lgca.velocitychannels:].sum(-1)
    return {"cells": migrating + resting, "migrating": migrating, "resting": resting,
            "kappa": mean_trait(lgca, "kappa")}


def radial_profiles(lgca, bins: int = 12) -> dict[str, np.ndarray]:
    """Profiles against the distance from the centre, out to the front.

    The front is the distance within which 95 % of the occupied nodes lie. Returns the bin
    centres in units of the front radius (``"r"``), the cells per node, the fractions of migrating
    and resting cells, and the mean and standard deviation of κ over the cells of each ring.
    After extinction the densities are zero and the front, radii and trait measures are NaN.
    Radii are also NaN while the population occupies only the central node (front radius zero).
    """
    from lgca.lattice_state import LatticeState

    r = _distance(lgca)
    maps = node_maps(lgca)
    occupied = maps["cells"] > 0
    if not occupied.any():
        return {"front": np.nan, **{name: np.full(bins, np.nan) for name in
                                    ("r", "migrating_fraction", "kappa_mean", "kappa_std")},
                **{name: np.zeros(bins) for name in ("cells", "migrating", "resting")}}
    front = np.percentile(r[occupied], 95)
    edges = np.linspace(0, 1.1 * (front or 1.0), bins + 1)
    ring = np.digitize(r, edges) - 1  # ring of every node; bins and beyond
    cells = LatticeState(lgca).cells
    kappa = np.asarray(cells["kappa"], dtype=float)
    cell_ring = ring[cells.node]
    profile = {"r": 0.5 * (edges[1:] + edges[:-1]) / front if front else np.full(bins, np.nan), "front": front}
    for name in ("cells", "migrating", "resting"):
        total = np.bincount(ring.ravel(), weights=maps[name].ravel(), minlength=bins + 1)[:bins]
        nodes = np.bincount(ring.ravel(), minlength=bins + 1)[:bins]
        profile[name] = total / np.maximum(nodes, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile["migrating_fraction"] = profile["migrating"] / profile["cells"]
        count = np.bincount(cell_ring, minlength=bins + 1)[:bins]
        mean = np.bincount(cell_ring, weights=kappa, minlength=bins + 1)[:bins] / count
        square = np.bincount(cell_ring, weights=kappa ** 2, minlength=bins + 1)[:bins] / count
    profile["kappa_mean"], profile["kappa_std"] = mean, np.sqrt(np.maximum(square - mean ** 2, 0))
    return profile


def core_and_rim(lgca, rim: float = 0.1) -> dict[str, float]:
    """Mean κ of the cells in the core (the inner half of the cells by distance from the centre) and at
    the rim (the outermost fraction ``rim`` of the cells), of all cells, and the fraction migrating.

    All measures are NaN when the population is extinct.
    """
    from lgca.lattice_state import LatticeState

    cells = LatticeState(lgca).cells
    if not len(cells):
        return {key: float("nan") for key in ("core", "rim", "all", "migrating")}
    distance = _distance(lgca)[cells.node]
    kappa = np.asarray(cells["kappa"], dtype=float)
    inner, outer = np.quantile(distance, [0.5, 1 - rim])
    return {"core": float(kappa[distance <= inner].mean()), "rim": float(kappa[distance >= outer].mean()),
            "all": float(kappa.mean()), "migrating": float(cells.in_channels("velocity").mean())}


def _distance(lgca) -> np.ndarray:
    """Distance of every node from the central node, in lattice units."""
    centre = tuple(size // 2 for size in lgca.dims)
    if len(lgca.dims) == 1:
        return np.abs(np.arange(lgca.dims[0]) - centre[0]).astype(float)
    x, y = lgca.xcoords, lgca.ycoords
    return np.hypot(x - x[centre], y - y[centre])
