"""Multispecies birth-death example.

This model shows how a single BioLGCA model can track two species with
different birth rates. The operator is backend-aware, so the same model spec
states the biological rates while BioLGCA chooses the correct implementation.

What to inspect after running:
- ``result.lgca.n_t`` records total population through time.
- The species-specific rates are in ``BirthDeathSpec.parameters``.
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
from lgca.pipeline import BirthDeathSpec, InteractionPipelineSpec
from lgca.simulation import DensityRecorder, PopulationRecorder

from ._types import ExampleInfo


INFO = ExampleInfo(
    name="multispecies_birth_death",
    title="Multispecies birth-death example",
    category="population dynamics",
    question="How do different birth and death rates change competing populations?",
    concepts=("multispecies", "birth", "death"),
    source_path="lgca/examples/multispecies_birth_death.py",
    source="Morpheus ODE and multiscale population examples",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Two species share a lattice but use different birth rates.",
            tags=("example", "multispecies", "birth-death"),
        ),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.5, restchannels=1, n_species=2),
        time=TimeSpec(steps=2, seed=104),
        dynamics=InteractionPipelineSpec(
            operators=[
                BirthDeathSpec(
                    name="birth_death",
                    parameters={"birth_rate": [0.05, 0.02], "death_rate": [0.01, 0.01]},
                )
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
