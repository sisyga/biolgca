"""Go-or-grow example.

This follows the ``BioLGCA.ipynb`` go-or-grow section with ``kappa=4`` and a
fully occupied central node.
"""

from __future__ import annotations

try:
    from ._helpers import ensure_project_root_on_path, main
except ImportError:
    from _helpers import ensure_project_root_on_path, main

ensure_project_root_on_path(__file__)

import numpy as np

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
    """Seed a fully occupied central node, as in the notebook."""

    nodes = np.zeros((50, 50, 12), dtype=bool)
    nodes[25, 25, :] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Cells switch between moving and proliferating states.",
            tags=("example", "go-or-grow", "phenotype-switching"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=6),
        time=TimeSpec(steps=15, seed=111),
        dynamics=InteractionPipelineSpec(
            operators=[
                {
                    "name": "classical.go_or_grow",
                    "parameters": {"r_b": 0.2, "r_d": 0.01, "kappa": 4.0, "theta": 0.75},
                }
            ],
        ),
        analysis=AnalysisSpec(
            observers=[NodeRecorder(), DensityRecorder(), PopulationRecorder()],
        ),
    )


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    from dataclasses import replace

    from lgca.model import run_model

    spec = build_spec()
    if steps is not None:
        spec = replace(spec, time=replace(spec.time, steps=int(steps)))
    return run_model(spec, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
