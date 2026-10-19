"""Glioblastoma with oxygen: the go-or-grow switch evolves in response to oxygen from blood vessels.

The model extends the evolutionary go-or-grow model of Syga et al. (2024), :mod:`lgca.zoo.phenotypic_plasticity`,
by explicit oxygen (Whiteside, Syga & Deutsch, in preparation). Every cell carries two traits, its genotype: the
oxygen sensitivity κ and the critical oxygen θ. A cell rests and may divide (the proliferative phenotype) with
probability

    r(O) = (1 + tanh(κ (O - θ))) / 2,

where ``O`` is the oxygen at its node relative to the oxygen of the blood, and migrates otherwise. κ > 0 is the
attractive strategy (proliferate where oxygen is high, migrate where it is low), κ < 0 the repulsive one, κ ≈ 0
the independent one. Daughters inherit κ and θ with small normal changes, so the population evolves.

Oxygen obeys the reaction–diffusion equation of Martínez-González et al. (2012),

    ∂O/∂t = D ΔO - α n O / (K_m + O),

with ``n`` the cells at a node, and is supplied by blood vessels, which hold it at the oxygen of the blood
(Dirichlet point sources, one per (200 µm)^d of tissue). Tumour cells clog vessels: an open vessel closes with
a probability that rises with the cells at its node, and stays closed. Oxygen diffuses across a node in
milliseconds, so it is at equilibrium with the cells at every step (``solver="steady"``).

Oxygen also sets division and death: resting cells divide with probability ``α_max O² / (O_h² + O²)`` times
``1 - n / K``, which falls in hypoxia (``O < O_h``), and every cell dies with probability
``δ + δ_a O_a⁴ / (O_a⁴ + O⁴)``, which rises in anoxia (``O < O_a``).

One time step: oxygen (steady), clogging, death, the switch, division of resting cells (daughters mutate),
a random walk of migrating cells, and propagation. :func:`build_spec` builds the model in 1D (cells start at
the left edge and invade to the right, as a radius) or on the hexagonal lattice (cells start at the centre);
``clogging=0`` gives vessels that never close (the "endless oxygen" case), a positive value the case of a
necrotic core. :func:`resection` removes the bulk of the tumour and its vessels at a given step.

The physical parameters and their sources are those of :class:`Parameters`; several are assumptions to be
checked against the original implementation (marked "to check").

References
----------
Syga S, Jain HP, Krellner M, Hatzikirou H, Deutsch A (2024) Evolution of phenotypic plasticity leads to tumor
heterogeneity with implications for therapy. PLoS Comput Biol 20(8): e1012003.

Martínez-González A, Calvo GF, Pérez Romasanta LA, Pérez-García VM (2012) Hypoxic cell waves around necrotic
cores in glioblastoma: a biomathematical model and its therapeutic implications. Bull Math Biol 74:2875-2896.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lgca.examples._helpers import main, run_spec
from lgca.examples._types import ExampleInfo
from lgca.fields import PDESpec, reaction
from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.rules import interaction
from lgca.simulation import FieldRecorder, PopulationRecorder, Schedule

INFO = ExampleInfo(
    name="glioblastoma_oxygen",
    title="Glioblastoma with oxygen example",
    category="evolution",
    question="How does the oxygen supplied by blood vessels shape the evolution of the go-or-grow switch in "
             "glioblastoma?",
    concepts=("identity-based LGCA", "fields", "go-or-grow", "evolution", "phenotype switch"),
    source_path="lgca/examples/glioblastoma_oxygen.py",
    source="Syga et al. (2024) with oxygen after Martínez-González et al. (2012); Whiteside, Syga & Deutsch "
           "(in preparation)",
)

#: initial interval of κ and θ, and the bounds of θ
KAPPA_0 = (-4.0, 4.0)
THETA_0 = (0.0, 1.0)
MMHG = 1.316e-3  # µmol O2 per cm³ and mmHg in tissue at 37 °C (after Martínez-González et al. 2012)


@dataclass(frozen=True)
class Parameters:
    """Physical parameters and their lattice values.

    Lengths in µm, times in seconds, oxygen as a partial pressure in mmHg. Oxygen in the model is relative to
    ``vessel_oxygen``.

    Attributes
    ----------
    node_size : float, default=50
        Width of a node (0.005 cm, so that 1000 nodes span 5 cm).
    capacity : int, default=100
        Cells per node at which division stops (Syga et al. 2024, 1D).
    vessel_spacing : float, default=200
        Mean distance between blood vessels: one vessel per (200 µm)³ of brain tissue (Michalowski 2022);
        a node holds a vessel with probability ``(node_size / vessel_spacing)^d``.
    oxygen_diffusion : float, default=2000
        Diffusion coefficient of oxygen in µm²/s (2e-5 cm²/s; to check).
    consumption : float, default=1e-11
        Maximal oxygen uptake of a cell in µmol/s (after Martínez-González et al. 2012).
    michaelis_constant : float, default=2.5
        Oxygen at which uptake is half its maximum (Martínez-González et al. 2012).
    vessel_oxygen : float, default=40
        Oxygen at a blood vessel (to check; brain tissue near capillaries 35 to 40 mmHg).
    hypoxia : float, default=0.1
        Relative oxygen below which division slows (hypoxia).
    anoxia : float, default=0.01
        Relative oxygen below which cells die (anoxia).
    """

    node_size: float = 50.0
    capacity: int = 100
    vessel_spacing: float = 200.0
    oxygen_diffusion: float = 2000.0
    consumption: float = 1e-11
    michaelis_constant: float = 2.5
    vessel_oxygen: float = 40.0
    hypoxia: float = 0.1
    anoxia: float = 0.01

    def vessel_probability(self, dimensions: int) -> float:
        """The probability that a node holds a blood vessel."""
        return min(1.0, (self.node_size / self.vessel_spacing) ** dimensions)

    def lattice(self) -> dict[str, float]:
        """Diffusion and uptake in lattice units per second, and K_m relative to ``vessel_oxygen``.

        A node is a voxel of tissue, ``node_size³``. The steady field depends only on the ratio of diffusion
        and uptake, so the unit of time does not matter.
        """
        volume = (self.node_size * 1e-4) ** 3  # cm³
        blood = self.vessel_oxygen * MMHG  # µmol/cm³
        return {
            "diffusion": self.oxygen_diffusion / self.node_size**2,  # nodes² per s
            "uptake": self.consumption / (volume * blood),  # per cell and s, relative to the blood
            "michaelis_constant": self.michaelis_constant / self.vessel_oxygen,
        }


DEFAULT = Parameters()


@reaction(name="glioblastoma.vessels")
def vessel_supply(state, c, rate=1e6, field="vessels"):
    """Oxygen relaxes to the blood's, 1, at ``rate`` where ``field`` marks an open vessel: a Dirichlet source."""
    open_vessels = rate * (state.field(field) > 0)
    return open_vessels, open_vessels


@interaction(kind="field", families=("ib", "nove_ib", "classical", "nove"), name="glioblastoma.clogging")
def clogging(state, rate=0.0, field="vessels"):
    """Tumour cells clog blood vessels: an open vessel closes with probability ``rate * n / capacity`` per step.

    ``n`` is the number of cells at its node. Closed vessels stay closed (no angiogenesis).

    Parameters
    ----------
    rate : float
        Probability per step that an open vessel at a node with ``capacity`` cells closes; 0 keeps every
        vessel open.
    field : str
        Name of the field that marks open vessels (1) and closed ones or none (0).
    """
    if rate <= 0:
        return
    vessels = state.field(field)
    p = np.clip(rate * state.density / state.capacity, 0.0, 1.0)
    closing = (vessels > 0) & (state.rng.random(state.dims) < p)
    if closing.any():
        state.set_field(field, np.where(closing, 0.0, vessels))


@interaction(kind="birth_death", families=("ib", "nove_ib", "classical", "nove"), name="glioblastoma.resection")
def resection(state, at_step=-1, fraction=0.999, threshold=0.1, field="vessels"):
    """Surgery in step ``at_step`` (the step that leads to time ``at_step``): where the tumour is dense, a
    ``fraction`` of the cells and all vessels go.

    The resected region is every node with at least ``threshold * capacity`` cells. Removed vessels stay
    removed; the remaining cells regrow the tumour.

    Parameters
    ----------
    at_step : int
        The step of the surgery; -1 (the default) never.
    fraction : float
        Probability that a cell in the resected region is removed.
    threshold : float
        Cells per node over the capacity from which a node belongs to the resected region.
    field : str
        Name of the field that marks open vessels.
    """
    if state.step != at_step:
        return
    tumour = state.density >= threshold * state.capacity
    if not tumour.any():
        return
    p = np.where(tumour, fraction, 0.0)
    state.remove_cells(p[..., None, None] * np.ones((state.n_species, state.velocitychannels + state.restchannels)))
    state.set_field(field, np.where(tumour, 0.0, state.field(field)))


def _oxygen(K, n):
    """The Hill response to oxygen, rising (n > 0) or falling (n < 0) at K."""
    return {"name": "field", "field": "oxygen", "K": K, "n": n}


def build_vessels(dims, parameters: Parameters = DEFAULT, seed: int | None = 0) -> np.ndarray:
    """A map of blood vessels: 1 at a node with a vessel, else 0, drawn with ``seed``."""
    p = parameters.vessel_probability(len(dims))
    return (np.random.default_rng(seed).random(dims) < p).astype(float)


def build_spec(geometry: str = "lin", size: int | None = None, steps: int | None = None, *,
               clogging: float = 0.0, alpha: float = 1.0, delta: float = 0.0, delta_anoxic: float = 0.5,
               kappa_std: float = 0.2, theta_std: float = 0.05, cells: int | None = None,
               kappa_0=KAPPA_0, theta_0=THETA_0, resect_at: int | None = None,
               parameters: Parameters = DEFAULT, seed: int | None = 1, record_every: int = 10) -> ModelSpec:
    """The glioblastoma model with oxygen.

    Parameters
    ----------
    geometry : {"lin", "hex"}
        A line of nodes from the origin of the tumour (1D, 5 cm by default), or the hexagonal lattice.
    size : int, optional
        Nodes of the line (default 1000) or of the side of the hexagonal lattice (default 200).
    steps : int, optional
        Steps to run; default 1000 in 1D, 300 on the hexagonal lattice.
    clogging : float, default=0
        Probability per step that an open vessel at a full node closes; 0 keeps every vessel open.
    alpha : float, default=1
        Maximal division probability of a resting cell on an empty node with plenty of oxygen.
    delta : float, default=0
        Death probability of every cell, regardless of oxygen.
    delta_anoxic : float, default=0.5
        Maximal additional death probability in anoxia.
    kappa_std, theta_std : float
        Standard deviations of the changes of κ and θ from mother to daughter; θ stays in [0, 1].
    cells : int, optional
        Initial cells, resting at the first node (1D) or the centre; default the capacity.
    kappa_0, theta_0 : (float, float)
        Intervals of the uniform distributions of the initial κ and θ.
    resect_at : int, optional
        Step at which :func:`resection` removes the tumour; default never.
    parameters : Parameters
        The physical parameters.
    seed : int or None
        Seed of the run; it also draws the vessels and the initial traits.
    record_every : int
        Steps between recordings of the oxygen and the vessels.
    """
    if geometry == "lin":
        size = size or 1000
        dims, channels, origin = (size,), 3, (0,)
        default_steps = 1000
    elif geometry == "hex":
        size = size or 200
        dims, channels, origin = (size, size), 7, (size // 2, size // 2)
        default_steps = 300
    else:
        raise ValueError(f"geometry must be 'lin' or 'hex', got {geometry!r}")
    capacity = parameters.capacity
    cells = capacity if cells is None else cells
    rng = np.random.default_rng(seed)
    nodes = np.zeros(dims + (channels,), dtype=np.int64)
    nodes[origin + (channels - 1,)] = cells  # the last channel is the rest channel
    traits = {"kappa": rng.uniform(*kappa_0, cells), "theta": rng.uniform(*theta_0, cells)}
    vessels = build_vessels(dims, parameters, seed=rng.integers(2**32))
    lattice = parameters.lattice()
    mutation = {"traits": {
        "kappa": {"distribution": "normal", "scale": kappa_std},
        "theta": {"distribution": "normal", "scale": theta_std, "bounds": [0.0, 1.0], "at_bounds": "redraw"},
    }}
    operators = [
        # oxygen at equilibrium with the cells; open vessels hold it at the blood's
        PDESpec(field="oxygen", diffusion=lattice["diffusion"], solver="steady",
                reactions=[{"name": "glioblastoma.vessels"}],
                cells=[{"uptake": lattice["uptake"], "saturation": lattice["michaelis_constant"]}]),
        {"name": "glioblastoma.clogging", "parameters": {"rate": clogging}},
    ]
    if resect_at is not None:
        operators.append({"name": "glioblastoma.resection", "parameters": {"at_step": resect_at}})
    if delta > 0:
        operators.append({"name": "birth_death", "parameters": {"death_rate": delta}})
    operators += [
        # death in anoxia
        {"name": "birth_death", "parameters": {
            "death_rate": {"max": delta_anoxic, "hill": [_oxygen(parameters.anoxia, -4)]}}},
        # rest with (1 + tanh(κ (O − θ))) / 2, κ and θ of each cell
        {"name": "go_or_rest", "parameters": {"probability": {"max": 1.0, "cues": [
            {"name": "field", "field": "oxygen", "kappa": "kappa", "theta": "theta"}]}}},
        # resting cells divide, slower in hypoxia; daughters rest and inherit κ and θ with a change
        {"name": "birth_death", "parameters": {
            "birth_rate": {"max": alpha, "hill": [_oxygen(parameters.hypoxia, 2)]},
            "channels": "rest", "mutation": mutation}},
        {"name": "random_walk", "parameters": {"channels": "velocity"}},
    ]
    regime = "clogging vessels" if clogging > 0 else "fixed vessels"
    return ModelSpec(
        description=Description(title=f"{INFO.title}: {regime}", details=INFO.source,
                                tags=("example", "evolution", "oxygen", "glioblastoma", geometry)),
        space=SpaceSpec(geometry=geometry, dims=size if geometry == "lin" else dims, boundary="reflecting"),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, volume_exclusion=False,
                        capacity=capacity, traits=traits, fields={"oxygen": 1.0, "vessels": vessels}),
        time=TimeSpec(steps=steps if steps is not None else default_steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators),
        analysis=AnalysisSpec(observers=[
            PopulationRecorder(),
            FieldRecorder(["oxygen", "vessels"], schedule=Schedule(every=record_every)),
        ]),
    )


def kymographs(spec: ModelSpec, every: int = 10) -> dict[str, np.ndarray]:
    """Run a 1D model and record, every ``every`` steps, the cells, migrating cells, oxygen, open vessels and
    the mean κ and θ per node (NaN where empty).

    Returns arrays of shape (samples, nodes) and ``"steps"``.
    """
    from lgca.lattice_state import LatticeState
    from lgca.model import build_model
    from lgca.plot_data import mean_trait

    model = build_model(spec)
    lgca = model.lgca
    record = {key: [] for key in ("steps", "cells", "migrating", "oxygen", "vessels", "kappa", "theta")}
    for step in range(spec.time.steps + 1):
        if step % every == 0:
            state = LatticeState(lgca)
            record["steps"].append(step)
            record["cells"].append(lgca.cell_density[lgca.nonborder].copy())
            counts = lgca._channel_counts(lgca.nodes[lgca.nonborder])
            record["migrating"].append(counts[..., :lgca.velocitychannels].sum(-1))
            record["oxygen"].append(state.field("oxygen").copy())
            record["vessels"].append(state.field("vessels").copy())
            record["kappa"].append(mean_trait(lgca, "kappa"))
            record["theta"].append(mean_trait(lgca, "theta"))
        if step < spec.time.steps:
            model.step()
    result = {key: np.array(values) for key, values in record.items()}
    result["lgca"] = lgca
    return result


def node_size_cm(parameters: Parameters = DEFAULT) -> float:
    """The width of a node in cm, for axes in physical units."""
    return parameters.node_size * 1e-4


def doubling_steps(alpha: float = 1.0) -> float:
    """Steps in which a resting population with plenty of oxygen and room doubles."""
    return math.log(2) / math.log1p(alpha)


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
