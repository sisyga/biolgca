"""Aggregation example.

This follows the ``BioLGCA.ipynb`` aggregation section: a square lattice,
moderate density, and three rest channels so cells can collect in dense regions.
"""

from __future__ import annotations

try:
    from ._helpers import ensure_project_root_on_path, main
except ImportError:
    from _helpers import ensure_project_root_on_path, main

ensure_project_root_on_path(__file__)

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
    name="aggregation",
    title="Aggregation example",
    category="collective motion",
    question="How does density-biased movement create clusters?",
    concepts=("aggregation", "density", "rest channels"),
    source_path="lgca/examples/aggregation.py",
    source="BioLGCA.ipynb aggregation example",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Square-lattice aggregation with notebook density and rest channels.",
            tags=("example", "aggregation", "density"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="periodic"),
        state=StateSpec(density=0.3, restchannels=3),
        time=TimeSpec(steps=100, seed=106),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.aggregation", "parameters": {"beta": 2.0}}],
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
