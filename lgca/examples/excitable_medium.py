"""Excitable medium example.

This follows the ``BioLGCA.ipynb`` excitable-medium setup: many rest channels,
``N=20``, and a structured initial condition separating moving and resting
particles.
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
    name="excitable_medium",
    title="Excitable medium example",
    category="pattern formation",
    question="How can local excitation create propagating waves?",
    concepts=("excitable medium", "rest channels", "wave propagation"),
    source_path="lgca/examples/excitable_medium.py",
    source="BioLGCA.ipynb excitable medium example",
)


def build_initial_nodes() -> np.ndarray:
    """Create the notebook's left-moving and lower-resting initial condition."""

    nodes = np.zeros((50, 50, 26), dtype=bool)
    nodes[:25, :, :6] = True
    nodes[:, :25, 6:] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Structured excitation with 20 rest channels and N=20.",
            tags=("example", "excitable-medium", "pattern-formation"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="reflecting"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=20),
        time=TimeSpec(steps=100, seed=113),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.excitable_medium", "parameters": {"N": 20}}],
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
