"""Identity-based tumor growth example.

A small tumour of 64 cells grows without volume exclusion (up to 8 cells per
node). Every cell carries its own switching steepness ``kappa``; daughters
inherit it with a small random change (``kappa_std``). With ``kappa < 0``,
crowded cells migrate and resting cells divide, so the tumour invades its
surroundings and grows to about 3000 cells in 100 steps. Inspect the inherited
trait with ``result.lgca.props["kappa"]``.
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
            details="A seeded tumour invades by go-or-grow; each cell inherits a mutating switch steepness.",
            tags=("example", "identity-based", "tumor-growth"),
        ),
        space=SpaceSpec(geometry="square", dims=(50, 50), boundary="periodic"),
        state=StateSpec(
            restchannels=1,
            volume_exclusion=False,
            identity_based=True,
            capacity=8,
            # a 3 x 3 block of fully occupied nodes in the centre
            initializer={"name": "region", "parameters": {"extent": 3, "density": 8}},
        ),
        time=TimeSpec(steps=100, seed=105),
        dynamics=InteractionPipelineSpec(
            operators=[
                {
                    "name": "nove_ib.go_or_grow",
                    "parameters": {
                        "r_b": 0.2,
                        "r_d": 0.01,
                        "kappa": -5.0,
                        "theta": 0.5,
                        "kappa_std": 0.2,
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

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
