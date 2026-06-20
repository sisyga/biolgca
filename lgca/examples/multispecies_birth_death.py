"""Multispecies birth-death example.

This extends the notebook population-dynamics theme to two species with
different birth rates. It is retained as a compact bridge from the canonical
single-species notebook models toward multispecies BioLGCA use.
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
from lgca.pipeline import BirthDeathSpec, InteractionPipelineSpec
from lgca.simulation import DensityRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="multispecies_birth_death",
    title="Multispecies birth-death example",
    category="population dynamics",
    question="How do different birth and death rates change competing populations?",
    concepts=("multispecies", "birth", "death"),
    source_path="lgca/examples/multispecies_birth_death.py",
    source="BioLGCA.ipynb population examples extended to multispecies LGCA",
)


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Two species share a lattice but use different birth rates.",
            tags=("example", "multispecies", "birth-death"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="periodic"),
        state=StateSpec(density=0.2, restchannels=1, n_species=2),
        time=TimeSpec(steps=100, seed=104),
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

    from dataclasses import replace

    from lgca.model import run_model

    spec = build_spec()
    if steps is not None:
        spec = replace(spec, time=replace(spec.time, steps=int(steps)))
    return run_model(spec, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
