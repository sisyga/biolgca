"""Evolutionary go-and-grow example.

This follows ``Evolutionary LGCA.ipynb``: one identity-based family starts at
the left boundary and birth/death dynamics mutate the inherited birth rate.
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
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import DensityRecorder, NodeRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="evolutionary_go_and_grow",
    title="Evolutionary go-and-grow example",
    category="evolution",
    question="How does heritable birth-rate variation change growth?",
    concepts=("identity-based LGCA", "birth-death", "evolution"),
    source_path="lgca/examples/evolutionary_go_and_grow.py",
    source="Evolutionary LGCA.ipynb proof-of-principle go-and-grow example",
)


def build_initial_nodes() -> np.ndarray:
    """Seed the first node exactly as in the evolutionary notebook."""

    nodes = np.zeros((100, 4), dtype=bool)
    nodes[0] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Identity-based birth-death growth with heritable rate variation.",
            tags=("example", "evolution", "birth-death"),
        ),
        space=SpaceSpec(geometry="lin", dims=(100,), boundary="reflecting"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=2, identity_based=True,
                        traits={"r_b": 0.2}),
        time=TimeSpec(steps=200, seed=114),
        dynamics=InteractionPipelineSpec(
            operators=[
                # every cell has its own birth rate; daughters inherit it with a normal change
                {"name": "birth_death", "parameters": {
                    "birth_rate": "r_b", "death_rate": 0.01, "mutation": {"r_b": {
                        "distribution": "normal", "scale": 0.05, "bounds": [0, 1], "at_bounds": "redraw"}}}},
                {"name": "random_walk"},
            ],
        ),
        analysis=AnalysisSpec(
            observers=[NodeRecorder(), DensityRecorder(), PopulationRecorder()],
        ),
    )


def run(steps: int | None = None, showprogress: bool = False):
    """Run this example and return a :class:`lgca.model.ModelRunResult`."""

    return run_spec(build_spec, steps=steps, showprogress=showprogress)


if __name__ == "__main__":
    main(run)
