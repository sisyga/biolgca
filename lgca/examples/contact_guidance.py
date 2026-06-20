"""Nematic contact guidance example.

This adapts the ``BioLGCA.ipynb`` contact-guidance section into an explicit
director-field model, so the guiding structure is visible in the example file.
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
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
from lgca.simulation import DensityRecorder, NodeRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="contact_guidance",
    title="Nematic contact guidance example",
    category="guidance",
    question="How does an oriented scaffold bias movement?",
    concepts=("contact guidance", "director field", "single-cell initial state"),
    source_path="lgca/examples/contact_guidance.py",
    source="BioLGCA.ipynb nematic contact guidance example",
)


def build_initial_nodes() -> np.ndarray:
    """Place one cell near the notebook's starting location."""

    nodes = np.zeros((50, 50, 4), dtype=bool)
    nodes[10, 10, 1] = True
    return nodes


def build_director_field() -> np.ndarray:
    """Create a horizontal director field for square-lattice contact guidance."""

    director = np.zeros((50, 50, 2), dtype=float)
    director[..., 0] = 1.0
    return director


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="A single cell follows an explicit horizontal director field.",
            tags=("example", "contact-guidance", "director-field"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="periodic"),
        state=StateSpec(
            nodes=build_initial_nodes(),
            restchannels=0,
            fields={"director": build_director_field()},
        ),
        time=TimeSpec(steps=50, seed=109),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="contact_guidance",
                            beta=2.0,
                            parameters={"field": "director"},
                        )
                    ],
                )
            ],
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
