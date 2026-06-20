"""Nematic interaction example.

This mirrors the ``BioLGCA.ipynb`` nematic example, where neighboring cells
prefer aligned axes without distinguishing head from tail.
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
from lgca.simulation import DensityRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="nematic_interaction",
    title="Nematic interaction example",
    category="collective motion",
    question="How does axis alignment differ from polar alignment?",
    concepts=("nematic alignment", "orientation", "flux"),
    source_path="lgca/examples/nematic_interaction.py",
    source="BioLGCA.ipynb nematic interaction example",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Notebook-style nematic interaction on a hexagonal lattice.",
            tags=("example", "nematic", "collective-motion"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(density=0.1, restchannels=0),
        time=TimeSpec(steps=100, seed=107),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.nematic", "parameters": {"beta": 2.0}}],
        ),
        analysis=AnalysisSpec(
            observers=[DensityRecorder(), PopulationRecorder()],
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
