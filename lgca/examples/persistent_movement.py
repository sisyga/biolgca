"""Persistent movement example.

This follows ``BioLGCA.ipynb`` by placing one cell on a 12 by 12 reflecting
lattice and using a strong persistence parameter.
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
    name="persistent_movement",
    title="Persistent movement example",
    category="movement",
    question="How does directional memory change a single-cell trajectory?",
    concepts=("persistent motion", "single-cell initial state", "reflecting boundary"),
    source_path="lgca/examples/persistent_movement.py",
    source="BioLGCA.ipynb persistent movement example",
)


def build_initial_nodes() -> np.ndarray:
    """Place one cell in a single velocity channel, as in the notebook."""

    nodes = np.zeros((12, 12, 4), dtype=bool)
    nodes[5, 5, 0] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="A single cell moves with strong directional persistence.",
            tags=("example", "persistent-motion", "single-cell"),
        ),
        space=SpaceSpec(geometry="square", dims=(12, 12), boundary="reflecting"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=0),
        time=TimeSpec(steps=50, seed=108),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.persistent_walk", "parameters": {"beta": 8.0}}],
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
