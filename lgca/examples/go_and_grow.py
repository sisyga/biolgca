"""Go-and-grow example.

This follows the ``BioLGCA.ipynb`` go-and-grow section: one resting cell is
placed in the center and the birth rule expands the population.
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
    name="go_and_grow",
    title="Go-and-grow example",
    category="tumor growth",
    question="How does local birth expand a seeded population?",
    concepts=("birth", "rest channels", "tumor growth"),
    source_path="lgca/examples/go_and_grow.py",
    source="BioLGCA.ipynb go and grow example",
)


def build_initial_nodes() -> np.ndarray:
    """Seed one resting cell in the center of a hexagonal LGCA."""

    nodes = np.zeros((50, 50, 12), dtype=bool)
    nodes[25, 25, -1] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="A central resting cell grows through the classical birth rule.",
            tags=("example", "go-and-grow", "birth"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=6),
        time=TimeSpec(steps=100, seed=110),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.birth", "parameters": {"r_b": 0.2}}],
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
