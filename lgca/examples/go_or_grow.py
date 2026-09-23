"""Go-or-grow example.

Cells either migrate (velocity channels) or rest and divide (rest channels),
and switch between the two depending on how crowded their node is. With
``kappa < 0``, crowded cells start to migrate, so the colony spreads while its
core keeps growing: one fully occupied node grows to several thousand cells in
100 steps. With ``kappa > 0`` (e.g. ``kappa=4, theta=0.75``), isolated cells
keep migrating and rarely divide, and a small colony shrinks. Compare both
signs of ``kappa``.
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
    """Seed one fully occupied node in the centre."""

    nodes = np.zeros((50, 50, 12), dtype=bool)
    nodes[25, 25, :] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Crowded cells migrate, resting cells divide: a colony spreads from one node.",
            tags=("example", "go-or-grow", "phenotype-switching"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=6),
        time=TimeSpec(steps=100, seed=111),
        dynamics=InteractionPipelineSpec(
            operators=[
                {
                    "name": "classical.go_or_grow",
                    "parameters": {"r_b": 0.2, "r_d": 0.01, "kappa": -4.0, "theta": 0.5},
                }
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
