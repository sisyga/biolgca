"""Random walk example.

This is the smallest curated model: choose an LGCA geometry, initialize a
population density, run unbiased random-walk dynamics, and record density and
population output.

What to inspect after running:
- ``result.lgca.dens_t`` shows density snapshots.
- ``result.lgca.n_t`` shows total population over time.
"""

from __future__ import annotations

from dataclasses import replace

from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    run_model,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import DensityRecorder, PopulationRecorder

from ._types import ExampleInfo


INFO = ExampleInfo(
    name="random_walk",
    title="Random walk example",
    category="movement",
    question="How does unbiased cell movement spread a population?",
    concepts=("movement", "diffusion", "density"),
    source_path="lgca/examples/random_walk.py",
    source="BioLGCA.ipynb class initialization and simulation sections",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Unbiased random movement on a small square lattice.",
            tags=("example", "random-walk", "movement"),
        ),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=2, seed=101),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.random_walk"}],
        ),
        analysis=AnalysisSpec(
            observers=[DensityRecorder(), PopulationRecorder()],
        ),
    )


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    spec = build_spec()
    if steps is not None:
        spec = replace(spec, time=replace(spec.time, steps=int(steps)))
    return run_model(spec, showprogress=showprogress)


if __name__ == "__main__":
    result = run(steps=10)
    print(result.metadata)
