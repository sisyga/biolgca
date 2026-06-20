"""Chemotaxis example.

This model adds a named signal field to the state and then points a
reorientation term at that field. The field is deliberately small and explicit
so students can change the gradient by editing one array expression.

What to inspect after running:
- ``result.context.fields["signal"]`` is the static guidance field.
- ``result.lgca.dens_t`` shows the density response.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    run_model,
)
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
from lgca.simulation import DensityRecorder

from ._types import ExampleInfo


INFO = ExampleInfo(
    name="chemotaxis",
    title="Chemotaxis example",
    category="guidance",
    question="How does a signal field bias cell movement?",
    concepts=("signal field", "gradient sensing", "reorientation"),
    source_path="lgca/examples/chemotaxis.py",
    source="BioLGCA.ipynb chemotaxis; Morpheus multiscale chemotaxis examples",
)


def build_signal_field() -> np.ndarray:
    """Create a left-to-right signal gradient for the 4 by 4 lattice."""

    return np.linspace(0.0, 1.0, 4)[:, None] + np.zeros((4, 4))


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Chemotaxis uses a named state field and a reorientation term that reads it.",
            tags=("example", "chemotaxis", "signal-field"),
        ),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(
            density=0.25,
            restchannels=1,
            fields={"signal": build_signal_field()},
        ),
        time=TimeSpec(steps=2, seed=103),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=1.0,
                            parameters={"field": "signal"},
                        )
                    ],
                )
            ],
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
