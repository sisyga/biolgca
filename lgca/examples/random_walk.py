"""Random walk example.

This mirrors the introductory simulation loop in ``BioLGCA.ipynb``: choose a
geometry, initialize a population, run unbiased random-walk dynamics, and record
density and population output.
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
    name="random_walk",
    title="Random walk example",
    category="movement",
    question="How does unbiased cell movement spread a population?",
    concepts=("movement", "diffusion", "density"),
    source_path="lgca/examples/random_walk.py",
    source="BioLGCA.ipynb introduction and simulation sections",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Unbiased random movement on the default notebook-scale square lattice.",
            tags=("example", "random-walk", "movement"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="periodic"),
        state=StateSpec(density=0.1, restchannels=0),
        time=TimeSpec(steps=100, seed=101),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.random_walk"}],
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
