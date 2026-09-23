"""Alignment example.

Cells on a hexagonal lattice with reflecting walls reorient towards the mean
direction of their neighbours. Starting from random headings, they form
coherent streams within 100 steps: the global polarization (length of the mean
velocity) rises from about 0.03 to about 0.4. Lower ``beta`` or ``density``
to see where order breaks down.
"""

from __future__ import annotations

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
            details="Hexagonal alignment with reflecting boundaries; random headings order into streams.",
            tags=("example", "alignment", "collective-motion"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="reflecting"),
        state=StateSpec(density=0.5, restchannels=0),
        time=TimeSpec(steps=100, seed=102),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.alignment", "parameters": {"beta": 3.0}}],
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
