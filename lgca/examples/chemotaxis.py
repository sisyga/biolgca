"""Chemotaxis example.

A signal increases from 0 at the left wall to 10 at the right wall. The
chemotaxis term makes cells prefer to move up its gradient, so within 100
steps more than half of the cells gather in the right fifth of the lattice. The signal is an explicit named
field, so you can see where the directional cue enters the model; try another
shape, e.g. a peak in the middle, or a smaller ``beta``.
"""

from __future__ import annotations

import numpy as np

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
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
from lgca.simulation import DensityRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="chemotaxis",
    title="Chemotaxis example",
    category="guidance",
    question="How does a signal field bias cell movement?",
    concepts=("signal field", "gradient sensing", "reorientation"),
    source_path="lgca/examples/chemotaxis.py",
    source="BioLGCA.ipynb chemotaxis example",
)


def build_signal_field() -> np.ndarray:
    """Create a signal rising from 0 at the left to 10 at the right of the 50 by 50 lattice."""

    return np.linspace(0.0, 10.0, 50)[:, None] + np.zeros((50, 50))


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Cells climb a linear signal gradient and gather near the right wall.",
            tags=("example", "chemotaxis", "signal-field"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="reflecting"),
        state=StateSpec(
            density=0.1,
            restchannels=0,
            fields={"signal": build_signal_field()},
        ),
        time=TimeSpec(steps=100, seed=103),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=2.0,
                            parameters={"field": "signal"},
                        )
                    ],
                )
            ],
        ),
        analysis=AnalysisSpec(
            observers=[DensityRecorder(), PopulationRecorder()],
        ),
    )


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
