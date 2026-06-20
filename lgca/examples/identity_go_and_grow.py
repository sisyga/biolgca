"""Identity-based go-and-grow example.

This follows the one-dimensional identity-based go-and-grow section from
``BioLGCA.ipynb`` and records the evolving population over 200 steps.
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
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import DensityRecorder, NodeRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="identity_go_and_grow",
    title="Identity-based go-and-grow example",
    category="evolution",
    question="How does an identity-based lineage grow from one seed?",
    concepts=("identity-based LGCA", "birth-death", "lineage properties"),
    source_path="lgca/examples/identity_go_and_grow.py",
    source="BioLGCA.ipynb identity-based go-and-grow example",
)


def build_initial_nodes() -> np.ndarray:
    """Seed the center node of a one-dimensional identity-based LGCA."""

    nodes = np.zeros((100, 8), dtype=bool)
    nodes[50, :] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="One-dimensional identity-based birth-death growth.",
            tags=("example", "identity-based", "go-and-grow"),
        ),
        space=SpaceSpec(geometry="lin", dims=(100,), boundary="reflecting"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=6, identity_based=True),
        time=TimeSpec(steps=200, seed=112),
        dynamics=InteractionPipelineSpec(
            operators=[
                {
                    "name": "ib.birthdeath",
                    "parameters": {"r_b": 0.2, "r_d": 0.02, "std": 0.01},
                }
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
