"""Nematic interaction example.

Neighbouring cells prefer to share an axis of movement but not a direction:
moving left and moving right count the same. Within 100 steps the cells form
lanes of opposite traffic. The nematic order (alignment of axes) rises from
about 0.02 to about 0.3 while the polarization (alignment of directions) stays
near zero, unlike in the polar alignment example.
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
            details="Nematic alignment on a hexagonal lattice: cells share axes, not directions.",
            tags=("example", "nematic", "collective-motion"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(density=0.5, restchannels=0),
        time=TimeSpec(steps=100, seed=103),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "nematic_alignment", "parameters": {"beta": 3.0}}],
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
