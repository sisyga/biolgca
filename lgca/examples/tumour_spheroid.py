"""Tumour spheroid: proliferating, quiescent and necrotic cells in an oxygen gradient.

A multicellular tumour spheroid grows from a small ball of cells in culture.
Oxygen diffuses in from the medium and the cells consume it, so it falls
towards the centre. Where it runs low, cells stop dividing (quiescent); where
it runs out, they die (necrotic). The spheroid grows into layers: a necrotic
core, a ring of quiescent cells and a rim of proliferating cells, whose
thickness the supply of oxygen sets.

The model follows the hybrid cellular automaton of Dormann & Deutsch (2002)
in outline: cells of several classes on a lattice, a nutrient field that
they consume, and necrotic material that stays in place. It differs where
later measurements allow a closer model: the nutrient is oxygen, with its
measured diffusion, consumption and thresholds; quiescent cells are a class
of their own, which consumes less oxygen; and there is no chemotaxis up a
signal from necrotic cells, which Dormann & Deutsch needed for their growth
plateau and later work disputed (McElwain & Pettet 1993). The physical
parameters, with their sources, are those of :class:`Parameters`.

The three classes are species 0 (proliferating), 1 (quiescent) and 2
(necrotic) of a model without volume exclusion: a node is a voxel of tissue
that holds up to ``capacity`` cells. One step is one hour. Oxygen is at
equilibrium with the cells at every step (it diffuses across the spheroid in
minutes). The medium is stirred: it keeps its oxygen except in an unstirred
layer around the spheroid, through which oxygen diffuses to the cells, as
Dormann & Deutsch refilled the nutrient outside the tumour (the rule
``tumour_spheroid.stir`` marks the stirred medium at every step, the
reaction ``tumour_spheroid.medium`` holds its oxygen). Cells switch class
with rates that respond to the local oxygen, proliferating cells divide
where they have room, and cells of full nodes move to neighbouring nodes,
which is how the spheroid expands. On a hexagonal lattice the model is a
cross-section (a cylinder rather than a ball of cells); on the 3D Moore
lattice it is the spheroid. A model file of the example needs this module
imported first, e.g. ``biolgca run model.json --plugins
lgca.examples.tumour_spheroid``.

:func:`mean_uptake` gives every node the mean oxygen uptake of its cells, a
property of their classes (1 proliferating, 0.4 quiescent, 0 necrotic), and
:func:`layers` the radii of the layers. The notebook *Tumour spheroid* in the
documentation runs both lattices and compares them with measurements.

References
----------
Dormann S, Deutsch A (2002) Modeling of self-organized avascular tumor growth
with a hybrid cellular automaton. In Silico Biology 2:393-406.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

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
from lgca.plot_data import mean_species_property
from lgca.rules import interaction
from lgca.simulation import DensityRecorder, FieldRecorder, Schedule

INFO = ExampleInfo(
    name="tumour_spheroid",
    title="Tumour spheroid example",
    category="tumor growth",
    question="How does the supply of oxygen layer a growing tumour spheroid into proliferating, quiescent and "
             "necrotic cells?",
    concepts=("multispecies", "fields", "phenotype switch", "hexagonal lattice", "3D lattice"),
    source_path="lgca/examples/tumour_spheroid.py",
    source="Dormann & Deutsch (2002), with measured parameters of oxygen and spheroid growth",
)

PROLIFERATING, QUIESCENT, NECROTIC = 0, 1, 2
CLASSES = ("proliferating", "quiescent", "necrotic")
GEOMETRIES = {"hex": (80, 80), "moore": (40, 40, 40)}


@dataclass(frozen=True)
class Parameters:
    """Physical parameters of the spheroid and their sources.

    Times in hours, lengths in µm, oxygen as a partial pressure in mmHg
    (1.3 µM per mmHg in water at 37 °C).

    Attributes
    ----------
    node_size : float, default=30
        Distance between neighbouring nodes. A node is a voxel of tissue
        with ``capacity`` cells, about 12 for this size (below).
    cell_volume : float, default=1.4
        Volume of a cell in pL: DLD1 cells of radius 7 µm (Grimes et al.
        2014); HCT116 1.2 pL (Mao et al. 2018).
    packing : float, default=0.65
        Volume fraction of the cells in the tissue, about 4.6e8 cells per
        cm³.
    doubling_time : float, default=20
        Doubling time of proliferating cells: 20 to 24 h for EMT6/Ro
        spheroids (Freyer & Sutherland 1986), 22 h for HCT116 (Mao et al.
        2018).
    oxygen_diffusion : float, default=2000
        Diffusion coefficient of oxygen in µm²/s (2e-9 m²/s; Grimes et al.
        2014; 1.75e-9 m²/s in tumour tissue, Grote et al. 1977), also in the
        unstirred medium.
    medium_oxygen : float, default=140
        Oxygen of the medium: in air with 5 % CO2 at 37 °C, 141 mmHg (Wenger
        et al. 2015).
    unstirred_layer : float, default=90
        Thickness in µm of the medium around the spheroid that is not
        stirred, through which oxygen diffuses to it: a diffusion-depleted
        zone of up to about 100 µm (Mueller-Klieser & Sutherland 1982;
        Grimes et al. 2014, who measured 100 mmHg next to DLD1 spheroids).
        Farther out the medium keeps ``medium_oxygen``.
    uptake : float, default=22
        Oxygen consumption of tissue of proliferating cells, in mmHg/s
        (about 0.03 mol per m³ and s): 22 mmHg/s for DLD1 spheroids (Grimes
        et al. 2014); 11 to 28 mmHg/s for four other lines (Grimes et al.
        2016).
    michaelis_constant : float, default=1
        Oxygen at which the consumption is half its maximum (Michaelis–Menten;
        at most about 1 mmHg, Grimes et al. 2014).
    quiescent_uptake : float, default=0.4
        Consumption of quiescent relative to proliferating cells: 2.5 times
        less in large EMT6/Ro spheroids (Freyer & Sutherland 1985), 5 times
        less for V79 cells (Freyer et al. 1984).
    quiescence_oxygen : float, default=5
        Oxygen below which cells stop dividing and become quiescent: 0.5 to
        5 mmHg, depending on glucose (Grimes et al. 2016, after Höckel &
        Vaupel 2001).
    quiescence_time : float, default=4
        Mean time until a cell below ``quiescence_oxygen`` stops in the cycle
        (an assumption: cells arrest in late G1 within hours).
    recovery_time : float, default=10
        Mean time until a quiescent cell above ``quiescence_oxygen`` divides
        again (an assumption; DNA synthesis resumes hours after
        reoxygenation, Amellem & Pettersen 1991).
    necrosis_oxygen : float, default=0.5
        Oxygen below which cells die: severe hypoxia at 0.8 mmHg and below
        (Grimes et al. 2014); HCT116 cells die after a day below 0.1 mmHg
        (Mao et al. 2018).
    necrosis_time : float, default=48
        Mean time until a cell below ``necrosis_oxygen`` dies: a day below
        the threshold and death a day later (Mao et al. 2018).
    lysis_time : float, default=2000
        Mean time until necrotic material dissolves (Dormann & Deutsch 2002;
        not measured).
    crowding : float, default=0.9
        Fraction of the capacity above which most cells of a node move to
        neighbouring nodes: the relaxation of a tissue that division
        compresses (an assumption of the model). Below it, cells rest: they
        move about 10 µm²/h or less, as single tumour cells in tissue do
        (Klank et al. 2018).
    """

    node_size: float = 30.0
    cell_volume: float = 1.4
    packing: float = 0.65
    doubling_time: float = 20.0
    oxygen_diffusion: float = 2000.0
    medium_oxygen: float = 140.0
    unstirred_layer: float = 90.0
    uptake: float = 22.0
    michaelis_constant: float = 1.0
    quiescent_uptake: float = 0.4
    quiescence_oxygen: float = 5.0
    quiescence_time: float = 4.0
    recovery_time: float = 10.0
    necrosis_oxygen: float = 0.5
    necrosis_time: float = 48.0
    lysis_time: float = 2000.0
    crowding: float = 0.9

    def node_volume(self, geometry: str) -> float:
        """Volume of a node in pL: a cube, or on the hexagonal lattice a prism of a hexagon (a cross-section
        one node thick)."""
        area = math.sqrt(3) / 2 * self.node_size**2 if geometry == "hex" else self.node_size**2
        return area * self.node_size * 1e-3

    def capacity(self, geometry: str) -> int:
        """Cells per node in tissue: about 12 on the Moore and 11 on the hexagonal lattice."""
        return max(1, round(self.packing * self.node_volume(geometry) / self.cell_volume))


DEFAULT = Parameters()


def probability(time: float, step: float = 1.0) -> float:
    """The probability per step of an event with mean waiting time ``time``."""
    return -math.expm1(-step / time)


def lattice_parameters(parameters: Parameters = DEFAULT, geometry: str = "hex") -> dict[str, float]:
    """The parameters of the model in lattice units: lengths in nodes, times in steps of one hour, oxygen
    relative to the medium."""
    capacity = parameters.capacity(geometry)
    oxygen = parameters.medium_oxygen
    return {
        "capacity": capacity,
        # nodes² per step
        "diffusion": parameters.oxygen_diffusion * 3600 / parameters.node_size**2,
        # tissue at the capacity consumes `uptake`: per cell, in units of the medium's oxygen per step
        "uptake": parameters.uptake * 3600 / capacity / oxygen,
        "michaelis_constant": parameters.michaelis_constant / oxygen,
        "quiescence_oxygen": parameters.quiescence_oxygen / oxygen,
        "necrosis_oxygen": parameters.necrosis_oxygen / oxygen,
        "division": probability(parameters.doubling_time / math.log(2)),
        "quiescence": probability(parameters.quiescence_time),
        "recovery": probability(parameters.recovery_time),
        "necrosis": probability(parameters.necrosis_time),
        "lysis": probability(parameters.lysis_time),
    }


@interaction(kind="field", families=("classical", "nove"), name="tumour_spheroid.stir")
def stir(state, field="medium", layer=3.0):
    """Mark the stirred medium in ``field``: 1 at the nodes farther than ``layer`` node spacings from any
    cell, 0 elsewhere.

    The nodes in between are an unstirred layer around the spheroid, through which oxygen diffuses (in
    culture up to about 100 µm thick; Grimes et al. 2014), and gaps within the tissue are not medium.

    Parameters
    ----------
    field : str
        Name of the field that marks the stirred medium, read by the reaction ``tumour_spheroid.medium``.
    layer : float
        Thickness of the unstirred layer in node spacings.
    """
    state.set_field(field, _stirred(state.geometry, state.dims, state.density > 0, layer))


@reaction(name="tumour_spheroid.medium")
def medium(state, c, rate=1e6, field="medium"):
    """Oxygen relaxes to the medium's value, 1, at ``rate`` per step where ``field`` is 1 (see :func:`stir`)."""
    stirred = rate * state.field(field)
    return stirred, stirred


def _stirred(geometry, dims, occupied, layer):
    """1 at the nodes farther than ``layer`` from every occupied node, else 0."""
    occupied = np.asarray(occupied, dtype=bool).ravel()
    stirred = np.ones(occupied.shape)
    if occupied.any():
        positions = _flat_positions(geometry, tuple(dims))
        # the bound is exclusive: nodes at exactly ``layer`` are in the unstirred layer
        bound = np.nextafter(layer, np.inf)
        distance, _ = cKDTree(positions[occupied]).query(positions[~occupied], distance_upper_bound=bound)
        stirred[occupied] = 0.0
        stirred[~occupied] = np.isinf(distance)
    return stirred.reshape(tuple(dims))


@functools.lru_cache(maxsize=4)
def _flat_positions(geometry, dims):
    """The positions of the nodes, one row per node in C order."""
    return np.stack([axis.ravel() for axis in _positions(geometry, dims)], axis=-1)


def _oxygen(K, n):
    """The Hill response to oxygen, rising (n > 0) or falling (n < 0) at K."""
    return {"name": "field", "field": "oxygen", "K": K, "n": n}


def build_spec(geometry: str = "hex", dims=None, steps: int = 240, radius: float = 100.0,
               parameters: Parameters = DEFAULT, seed: int = 2002, record_every: int = 24) -> ModelSpec:
    """Build the model specification for this example.

    Parameters
    ----------
    geometry : {"hex", "moore"}
        A cross-section on the hexagonal lattice, or the spheroid on the 3D
        Moore lattice.
    dims : tuple of int, optional
        Nodes per axis; default 80 x 80 (2.4 mm) or 40 x 40 x 40 (1.2 mm,
        which the spheroid fills in about ten days).
    steps : int
        Hours to run.
    radius : float
        Radius of the initial spheroid in µm, filled with proliferating
        cells.
    parameters : Parameters
        The physical parameters.
    seed : int
        Random seed.
    record_every : int
        Steps between recordings of the cells (``"density"``, per class) and
        the oxygen (``"oxygen"``).
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {', '.join(GEOMETRIES)}, got {geometry!r}")
    dims = tuple(dims or GEOMETRIES[geometry])
    lattice = lattice_parameters(parameters, geometry)
    capacity, K_m = lattice["capacity"], lattice["michaelis_constant"]
    hypoxic, anoxic = lattice["quiescence_oxygen"], lattice["necrosis_oxygen"]
    nodes = _ball(geometry, dims, radius / parameters.node_size, capacity)
    layer = parameters.unstirred_layer / parameters.node_size
    steep = 4  # thresholds in the Hill form: from 10 % to 90 % of the response within a factor 3 of oxygen
    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Proliferating, quiescent and necrotic cells of a spheroid in the oxygen that the cells "
                    "consume, with measured parameters; after Dormann & Deutsch (2002).",
            tags=("example", "multispecies", "tumour", "oxygen", geometry),
        ),
        space=SpaceSpec(geometry=geometry, dims=dims, boundary="reflecting"),
        state=StateSpec(
            nodes=nodes,
            restchannels=1,
            n_species=3,
            volume_exclusion=False,
            capacity=capacity,
            fields={"oxygen": 1.0, "medium": _stirred(geometry, dims, nodes.sum(axis=(-2, -1)) > 0, layer)},
        ),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            # oxygen at equilibrium with the cells, in a stirred medium
            {"name": "tumour_spheroid.stir", "parameters": {"layer": layer}},
            # (conjugate gradients: the moving edge of the medium changes the matrix at every step, and
            # rebuilding a multigrid hierarchy for it costs more than the iterations it saves)
            PDESpec(field="oxygen", diffusion=lattice["diffusion"], boundary={"value": 1.0}, solver="steady",
                    solver_options={"backend": "cg"}, reactions=[{"name": "tumour_spheroid.medium"}],
                    cells=[{"uptake": lattice["uptake"], "saturation": K_m, "species": PROLIFERATING},
                           {"uptake": lattice["uptake"] * parameters.quiescent_uptake, "saturation": K_m,
                            "species": QUIESCENT}]),
            # cells stop dividing where oxygen is low, divide again where it returns, and die where it runs out
            # (into the rest channel: necrotic material does not move)
            {"name": "phenotype_switch", "parameters": {"channels": "rest", "rates": [
                [0, {"max": lattice["quiescence"], "hill": [_oxygen(hypoxic, -steep)]},
                 {"max": lattice["necrosis"], "hill": [_oxygen(anoxic, -steep)]}],
                [{"max": lattice["recovery"], "hill": [_oxygen(hypoxic, steep)]}, 0,
                 {"max": lattice["necrosis"], "hill": [_oxygen(anoxic, -steep)]}],
                [0, 0, 0],
            ]}},
            # proliferating cells divide where there is oxygen, up to the capacity of their node; necrotic
            # material dissolves slowly
            {"name": "birth_death", "parameters": {"crowding": False,
                "birth_rate": [{"max": lattice["division"], "hill": [_oxygen(hypoxic, steep)]}, 0.0, 0.0],
                "death_rate": [0.0, 0.0, lattice["lysis"]],
            }},
            # cells of full nodes move to neighbouring nodes, the others stay
            {"name": "go_or_rest", "parameters": {"kappa": -20.0, "theta": parameters.crowding,
                                                  "species": [PROLIFERATING, QUIESCENT]}},
            {"name": "random_walk", "parameters": {"channels": "velocity", "species": [PROLIFERATING, QUIESCENT]}},
        ]),
        analysis=AnalysisSpec(observers=[
            DensityRecorder(schedule=Schedule(every=record_every)),
            FieldRecorder(["oxygen"], schedule=Schedule(every=record_every)),
        ]),
    )


def _ball(geometry, dims, radius, capacity):
    """Nodes within ``radius`` (in nodes) of the centre, filled to the capacity with proliferating cells."""
    channels = {"hex": 7, "moore": 27}[geometry]
    nodes = np.zeros(tuple(dims) + (3, channels), dtype=np.int64)
    inside = _distance(geometry, dims) <= radius
    nodes[inside, PROLIFERATING, -1] = capacity  # in the rest channel
    return nodes


def _positions(geometry, dims):
    """The positions of the nodes, one array per axis, in node spacings: hexagonal rows are offset by half a
    node and sqrt(3)/2 apart."""
    index = np.indices(dims, dtype=float)
    if geometry == "hex":
        return [index[0] + 0.5 * (index[1] % 2), index[1] * math.sqrt(3) / 2]
    return list(index)


def _distance(geometry, dims, centre=None):
    """Distance of every node from ``centre`` (default the middle of the lattice), in node spacings."""
    position = _positions(geometry, dims)
    if centre is None:
        centre = [(axis.min() + axis.max()) / 2 for axis in position]
    return np.sqrt(sum((axis - middle) ** 2 for axis, middle in zip(position, centre)))


def mean_uptake(lgca, density=None, parameters: Parameters = DEFAULT):
    """The mean oxygen uptake of the cells at every node, relative to proliferating cells; NaN where no cell is.

    A property of the classes: 1 for proliferating cells, ``quiescent_uptake`` for quiescent ones, 0 for
    necrotic material. ``density`` as in :func:`lgca.plot_data.mean_species_property`.
    """
    return mean_species_property(lgca, (1.0, parameters.quiescent_uptake, 0.0), density)


def layers(density, geometry, parameters: Parameters = DEFAULT) -> dict[str, np.ndarray]:
    """Radii of the spheroid and its layers in µm, from the cells of every class.

    The cells of the necrotic core, of the core and the quiescent ring, and of the whole spheroid are taken
    as concentric balls (discs on the hexagonal lattice) of tissue at the capacity; the radius of each holds
    their volume. ``density`` is one state or a recording (species last).
    """
    density = np.asarray(density, dtype=float)
    spatial = 3 if geometry == "moore" else 2
    counts = density.sum(axis=tuple(range(density.ndim - 1 - spatial, density.ndim - 1)))  # per class
    cumulative = {"necrotic": counts[..., NECROTIC], "quiescent": counts[..., NECROTIC] + counts[..., QUIESCENT],
                  "spheroid": counts.sum(-1)}
    volume = parameters.node_volume(geometry) / parameters.capacity(geometry)  # pL of tissue per cell
    radii = {}
    for name, cells in cumulative.items():
        if geometry == "moore":
            radii[name] = (3 * cells * volume * 1e3 / (4 * math.pi)) ** (1 / 3)
        else:  # a disc of tissue one node thick
            radii[name] = np.sqrt(cells * volume * 1e3 / parameters.node_size / math.pi)
    return radii


def radial_profile(values, geometry, density=None, width=1.0):
    """The mean of ``values`` (one per node) in rings around the centre of the cells.

    ``density``, the cells per node and class of the same state, sets the centre (default: the middle of
    the lattice); rings are ``width`` nodes wide, and NaN values are left out. Returns the distances of the
    rings' middles, in nodes, and the means.
    """
    values = np.asarray(values, dtype=float)
    dims = values.shape
    centre = None
    if density is not None:
        cells = np.asarray(density, dtype=float).sum(-1)
        centre = [float((axis * cells).sum() / cells.sum()) for axis in _positions(geometry, dims)]
    ring = (_distance(geometry, dims, centre) / width).astype(int).ravel()
    valid = np.isfinite(values.ravel())
    total = np.bincount(ring[valid], weights=values.ravel()[valid], minlength=ring.max() + 1)
    count = np.bincount(ring[valid], minlength=ring.max() + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.arange(len(count)) + 0.5) * width, total / count


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
