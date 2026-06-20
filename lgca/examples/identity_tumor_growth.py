"""Identity-based tumor growth example.

This is a compact two-dimensional identity-based go-or-grow model inspired by
the BioLGCA go-or-grow notebook section.
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
    name="identity_tumor_growth",
    title="Identity tumor growth example",
    category="tumor growth",
    question="How can individual cell properties drive go-or-grow tumor expansion?",
    concepts=("identity-based LGCA", "go-or-grow", "tumor growth"),
    source_path="lgca/examples/identity_tumor_growth.py",
    source="BioLGCA.ipynb go-or-grow example adapted to an identity-based 2D model",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Identity-based tumor growth with a go-or-grow interaction.",
            tags=("example", "identity-based", "tumor-growth"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="periodic"),
        state=StateSpec(
            density=0.2,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": 8},
        ),
        time=TimeSpec(steps=50, seed=105),
        dynamics=InteractionPipelineSpec(
            operators=[
                {
                    "name": "nove_ib.go_or_grow",
                    "parameters": {
                        "capacity": 8,
                        "r_b": 0.2,
                        "r_d": 0.01,
                        "kappa": 5.0,
                        "theta": 0.5,
                    },
                }
            ],
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
