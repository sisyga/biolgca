"""Evolutionary go-or-grow example.

This follows ``Evolutionary LGCA.ipynb``: one identity-based family starts at
the left boundary and evolves with a go-or-grow switch on a short one-
dimensional lattice.
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
    name="evolutionary_go_or_grow",
    title="Evolutionary go-or-grow example",
    category="evolution",
    question="How does switching affect evolutionary expansion?",
    concepts=("identity-based LGCA", "go-or-grow", "evolution"),
    source_path="lgca/examples/evolutionary_go_or_grow.py",
    source="Evolutionary LGCA.ipynb go-or-grow example",
)


def build_initial_nodes() -> np.ndarray:
    """Seed the first node exactly as in the evolutionary notebook."""

    nodes = np.zeros((25, 4), dtype=bool)
    nodes[0] = True
    return nodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Identity-based go-or-grow dynamics on a short line.",
            tags=("example", "evolution", "go-or-grow"),
        ),
        space=SpaceSpec(geometry="lin", dims=(25,), boundary="reflecting"),
        state=StateSpec(nodes=build_initial_nodes(), restchannels=2, identity_based=True,
                        traits={"kappa": 0.0, "theta": 0.75}),
        time=TimeSpec(steps=100, seed=115),
        dynamics=InteractionPipelineSpec(
            operators=[
                # every cell switches with its own kappa and theta; daughters inherit them with a change
                {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": "theta"}},
                {"name": "go_or_grow.growth", "parameters": {
                    "r_b": 0.2, "r_d": 0.01, "mutation": {"kappa": 0.2, "theta": 0.05}}},
                {"name": "random_walk", "parameters": {"channels": "velocity"}},
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
