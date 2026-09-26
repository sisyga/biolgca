"""Discrete excitable media: spiral waves of two interacting species, and their mean field, Barkley's model.

Syga S, Nava-Sedeño JM, Brusch L, Deutsch A (2019). A lattice-gas cellular automaton model for
discrete excitable media. In: Tsuji K, Müller SC (eds) Spirals and Vortices. The Frontiers
Collection, pp 253–264. Springer, Cham. https://doi.org/10.1007/978-3-030-05798-5_15

Two species on a hexagonal lattice: the excited species X moves (velocity channels), the refractory
species Y rests (the a = K − 6 rest channels). In every time step Y reacts once, with birth
probability ρ_X and death probability ρ_Y, and X reacts N times, with birth probability
ρ_X² (1 + (ρ_Y + B)/A) and death probability ρ_X (ρ_Y + B)/A + ρ_X³; then the X cells are mixed
over the velocity channels and move. ρ_X and ρ_Y are the occupied fractions of the velocity and of
the rest channels. The mean field is Barkley's model,

    ∂ρ_X/∂t = D Δρ_X + N f,   ∂ρ_Y/∂t = g,
    f = ρ_X (1 − ρ_X) (ρ_X − (ρ_Y + B)/A),   g = ρ_X − ρ_Y,

with D = 1/4 in lattice units. The rule is the built-in ``excitable_medium`` (A = ``alpha``,
B = ``beta``).
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
from lgca.simulation import NodeRecorder, Schedule
from lgca.zoo._card import Parameter, ZooEntry

CARD = ZooEntry(
    name="excitable_media",
    title="Discrete excitable media",
    question="Do spiral waves survive when an excitable medium consists of a small number of discrete "
             "individuals, and how do they differ from those of the continuum model?",
    authors="Syga S, Nava-Sedeño JM, Brusch L, Deutsch A",
    paper="A lattice-gas cellular automaton model for discrete excitable media",
    year=2019,
    venue="In: Tsuji K, Müller SC (eds) Spirals and Vortices. The Frontiers Collection, pp 253–264. Springer",
    doi="10.1007/978-3-030-05798-5_15",
    reproduces="Fig 2: spiral waves in the LGCA and in Barkley's model from the same start, the nullclines, "
               "and the distribution of the LGCA's node states around the orbit of the mean field",
    mechanisms=("excitable birth and death", "two species", "random walk"),
    lattice="hexagonal, 6 velocity channels (X) and K − 6 rest channels (Y), classical with volume exclusion, "
            "absorbing",
    fidelity="same rules; the lattice size is not given in the chapter and is chosen here",
)

VELOCITY = 6

PARAMETERS = {
    "A": Parameter("A", "excitability", 0.75, "dynamics.operators[excitable_medium].alpha"),
    "B": Parameter("B", "excitation threshold", 0.02, "dynamics.operators[excitable_medium].beta"),
    "N": Parameter("N", "fast reactions of X per time step", 50, "dynamics.operators[excitable_medium].N"),
    "K": Parameter("K", "channels per node: 6 velocity channels and K − 6 rest channels", 23),
}


def quadrants(size: int, K: int = 23) -> np.ndarray:
    """The initial state of Fig 2: quadrants with (ρ_X, ρ_Y) = (0, 0), (1, 0), (0, 1) and (1, 1)."""
    nodes = np.zeros((size, size, K), dtype=bool)
    half = size // 2
    nodes[half:, :half, :VELOCITY] = True  # excited
    nodes[:half, half:, VELOCITY:] = True  # refractory
    nodes[half:, half:, :] = True  # both
    return nodes


def build_spec(full: bool = False, *, A: float = 0.75, B: float = 0.02, N: int = 50, K: int = 23,
               size: int | None = None, steps: int | None = None, record_every: int = 10,
               seed: int | None = 1) -> ModelSpec:
    """The LGCA of Syga et al. (2019), starting from four quadrants.

    Parameters
    ----------
    full : bool, default=False
        A 200 × 200 lattice and 2000 steps; by default 100 × 100 and 600 steps.
    A, B : float
        Excitability A and threshold B.
    N : int, default=50
        Fast reactions of the excited species per time step.
    K : int, default=23
        Channels per node; the chapter varies it from 12 to 100 (small K means large fluctuations).
    size, steps : int, optional
        Override the side of the lattice and the number of steps.
    record_every : int, default=10
        Steps between two recorded states.
    seed : int or None, default=1
        Seed of the run.
    """
    if K <= VELOCITY:
        raise ValueError(f"K must exceed the 6 velocity channels, got {K!r}")
    size = size or (200 if full else 100)
    return ModelSpec(
        description=Description(title=f"{CARD.title}: A = {A}, B = {B}, N = {N}, K = {K}", details=CARD.citation),
        space=SpaceSpec(geometry="hex", dims=(size, size), boundary="absorbing"),
        state=StateSpec(nodes=quadrants(size, K), restchannels=K - VELOCITY),
        time=TimeSpec(steps=steps if steps is not None else 2000 if full else 600, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # Y reacts once, X N times, then X is mixed over the velocity channels
            {"name": "excitable_medium", "parameters": {"alpha": A, "beta": B, "N": N}},
        ]),
        analysis=AnalysisSpec(observers=[NodeRecorder(schedule=Schedule(every=record_every))]),
    )


def fractions(nodes) -> tuple[np.ndarray, np.ndarray]:
    """ρ_X and ρ_Y, the occupied fractions of the velocity and of the rest channels of every node."""
    nodes = np.asarray(nodes)
    return nodes[..., :VELOCITY].mean(-1), nodes[..., VELOCITY:].mean(-1)


def colour(rho_x, rho_y) -> np.ndarray:
    """The colours of Fig 2: red ∝ ρ_X, green ∝ ρ_Y, yellow where both are high."""
    return np.stack([rho_x, rho_y, np.zeros_like(rho_x)], axis=-1).clip(0, 1)


def reaction(rho_x, rho_y, A: float = 0.75, B: float = 0.02):
    """Barkley's reaction terms f and g (Eq 10)."""
    return rho_x * (1 - rho_x) * (rho_x - (rho_y + B) / A), rho_x - rho_y


def barkley(lgca, rho_x, rho_y, steps: int, *, A: float = 0.75, B: float = 0.02, N: int = 50, K: int = 23,
            substeps: int = 20, record_every: int = 10, probe=None) -> dict[str, np.ndarray]:
    """Integrate the mean field of the LGCA, Barkley's model, on the lattice of ``lgca``.

    In the LGCA's units, one time step and one node,

        ∂ρ_X/∂k = D Δρ_X + (N / b) f,   ∂ρ_Y/∂k = g / a,

    since each of the N reactions of X changes n_X = b ρ_X by f on average and the reaction of Y
    changes n_Y = a ρ_Y by g; D = 1/4, as every X cell moves one node per step in one of the b = 6
    directions. Rescaling time and space gives Barkley's form (Eq 8 of the chapter). Explicit Euler
    with ``substeps`` steps per LGCA step and the Laplacian of the lattice
    (:func:`lgca.fields.laplacian`), with ρ_X = 0 beyond the edge, as cells leave through absorbing
    boundaries. Returns the recorded ``"steps"``, ``"x"`` and ``"y"``; with ``probe``, a node such as
    ``(50, 50)``, also ``"probe"``: its ρ_X and ρ_Y at every step, shape ``(steps + 1, 2)``.
    """
    from lgca.fields import laplacian

    matrix, _ = laplacian(lgca, {"value": 0.0})
    shape = tuple(lgca.dims)
    x, y = np.asarray(rho_x, dtype=float).ravel().copy(), np.asarray(rho_y, dtype=float).ravel().copy()
    dt = 1.0 / substeps
    out = {"steps": [], "x": [], "y": [], "probe": []}
    at = None if probe is None else int(np.ravel_multi_index(tuple(probe), shape))
    for step in range(steps + 1):
        if at is not None:
            out["probe"].append((x[at], y[at]))
        if step % record_every == 0:
            out["steps"].append(step)
            out["x"].append(x.reshape(shape).copy())
            out["y"].append(y.reshape(shape).copy())
        if step == steps:
            break
        for _ in range(substeps):
            f, g = reaction(x, y, A, B)
            x, y = x + dt * (0.25 * (matrix @ x) + N / VELOCITY * f), y + dt * g / (K - VELOCITY)
    if at is None:
        del out["probe"]
    return {key: np.array(value) for key, value in out.items()}


def mean_return_time(nodes_t, start: int = 0) -> float:
    """Mean number of recorded frames until a node is again in the state (n_X, n_Y) it had at ``start``.

    The observable of Fig 3a: close to 1 without spirals, the rotation period with them. Nodes that
    never return are left out.
    """
    counts = np.stack([np.asarray(nodes_t)[..., :VELOCITY].sum(-1), np.asarray(nodes_t)[..., VELOCITY:].sum(-1)], -1)
    reference = counts[start]
    same = np.all(counts[start + 1:] == reference, axis=-1)  # frames × nodes
    returned = same.any(axis=0)
    first = np.argmax(same, axis=0) + 1
    return float(first[returned].mean()) if returned.any() else float("nan")
