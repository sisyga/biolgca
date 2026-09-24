"""Go-or-grow example: an emerging Allee effect.

Cells either migrate (species 0, in velocity channels) or rest and divide
(species 1, in rest channels), and switch between the two depending on how
crowded their node is: a migrating cell starts resting with probability
``(1 + tanh(kappa * (rho - theta))) / 2``, where ``rho`` is the fraction of
occupied channels. One time step is a phenotype switch
(``go_or_grow.switch``), death and division (``go_or_grow.growth``) and a
random walk of the migrating cells over the velocity channels, followed by
propagation. This reproduces the original single-species go-or-grow rule
(``classical.go_or_grow``) in distribution.

With the default ``kappa=4, theta=0.75``, cells in sparse regions keep
migrating and rarely divide. A colony that starts from one fully occupied node
spreads out, stops dividing and shrinks (from 12 to a handful of cells in 100
steps):
growth needs a minimum population, an Allee effect that emerges from the
switching rule. Colonies of 25 x 25 fully occupied nodes, in contrast, grow.

Compare with ``build_spec(kappa=-4.0)``: crowded cells now migrate and sparse
cells rest and divide, and the same single node grows to more than a thousand
cells. For example::

    from lgca.examples.go_or_grow import build_spec
    from lgca.model import run_model

    allee = run_model(build_spec(), showprogress=False)
    invasion = run_model(build_spec(kappa=-4.0), showprogress=False)
    print(allee.lgca.n_t[-1], invasion.lgca.n_t[-1])
"""

from __future__ import annotations

import numpy as np

from lgca.examples._helpers import main, run_spec
from lgca.examples._types import ExampleInfo
from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import DensityRecorder, NodeRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="go_or_grow",
    title="Go-or-grow example",
    category="tumor growth",
    question="How does switching between motion and birth change expansion?",
    concepts=("go-or-grow", "phenotype switching", "rest channels"),
    source_path="lgca/examples/go_or_grow.py",
    source="BioLGCA.ipynb go or grow example",
)


def build_initial_nodes() -> np.ndarray:
    """Seed one fully occupied node in the centre: 6 migrating and 6 resting cells."""

    nodes = np.zeros((50, 50, 2, 12), dtype=bool)
    nodes[25, 25, 0, :6] = True  # migrating cells in the velocity channels
    nodes[25, 25, 1, 6:] = True  # resting cells in the rest channels
    return nodes


def build_spec(kappa: float = 4.0) -> ModelSpec:
    """Build the model specification for this example.

    Parameters
    ----------
    kappa : float, default=4.0
        Steepness of the switch between moving and resting. Positive values
        make crowded cells rest (Allee effect), negative values make them move.
    """

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="A small go-or-grow colony shrinks: an Allee effect emerges from density-dependent switching.",
            tags=("example", "go-or-grow", "phenotype-switching"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=6, n_species=2),
        time=TimeSpec(steps=100, seed=111),
        dynamics=InteractionPipelineSpec(
            operators=[
                {"name": "go_or_grow.switch", "parameters": {"kappa": kappa, "theta": 0.75}},
                {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01}},
                {"name": "species_random_walk", "parameters": {"species": 0, "channels": "velocity"}},
            ],
        ),
        analysis=AnalysisSpec(
            observers=[NodeRecorder(), DensityRecorder(), PopulationRecorder()],
        ),
    )


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
