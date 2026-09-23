"""Multispecies birth-death example.

Two species share the lattice and its free channels. Species 0 divides faster
(birth rate 0.05) than species 1 (0.02); both die at rate 0.01. Both grow
while free channels remain, and the faster species ends up with most of the
cells (about 10400 of 10900 after 100 steps).
Change the rates to explore coexistence and competitive exclusion.
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

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
