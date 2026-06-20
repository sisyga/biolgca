"""Alignment example.

This model changes one idea from the random walk: cells still move on an LGCA
lattice, but the reorientation step now favors alignment with local flux.

What to inspect after running:
- ``result.lgca.dens_t`` shows where cells accumulate.
- ``result.lgca.plot_flux()`` can be used interactively to inspect direction.
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
from lgca.simulation import DensityRecorder

from ._types import ExampleInfo


INFO = ExampleInfo(
    name="alignment",
    title="Alignment example",
    category="collective motion",
    question="How do local alignment rules create coherent streams?",
    concepts=("collective motion", "flux", "reorientation"),
    source_path="lgca/examples/alignment.py",
    source="BioLGCA.ipynb alignment example",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details=(
                "Alignment on a small hexagonal lattice with beta controlling "
                "coupling strength."
            ),
            tags=("example", "alignment", "collective-motion"),
        ),
        space=SpaceSpec(geometry="hex", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=0),
        time=TimeSpec(steps=2, seed=102),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.alignment", "parameters": {"beta": 1.0}}],
        ),
        analysis=AnalysisSpec(observers=[DensityRecorder()]),
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
