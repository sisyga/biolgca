"""Identity-based tumor growth example.

This model uses a no-volume-exclusion identity-based LGCA. That means cells can
share a node and can carry individual properties while a go-or-grow rule changes
motile and proliferative behavior.

What to inspect after running:
- ``result.lgca.n_t`` records population size.
- The ``state`` block selects the identity-based, no-volume-exclusion backend.
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
    name="identity_tumor_growth",
    title="Identity tumor growth example",
    category="tumor growth",
    question="How can individual cell properties drive go-or-grow tumor expansion?",
    concepts=("identity-based LGCA", "go-or-grow", "tumor growth"),
    source_path="lgca/examples/identity_tumor_growth.py",
    source="BioLGCA.ipynb go-and-grow/go-or-grow examples",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Identity-based tumor growth with a go-or-grow interaction.",
            tags=("example", "identity-based", "tumor-growth"),
        ),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(
            density=0.8,
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            parameters={"capacity": 8},
        ),
        time=TimeSpec(steps=2, seed=105),
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

    spec = build_spec()
    if steps is not None:
        spec = replace(spec, time=replace(spec.time, steps=int(steps)))
    return run_model(spec, showprogress=showprogress)


if __name__ == "__main__":
    result = run(steps=10)
    print(result.metadata)
