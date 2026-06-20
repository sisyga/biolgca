"""Custom rest-or-align interaction example.

This ports the custom interaction rule from ``BioLGCA.ipynb`` into a
``ModelSpec`` example. It combines alignment with a local preference for rest
channels, so students can see where custom Python dynamics still fit.
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
from lgca.plugins import ConservationLaw, LegacyInteractionOperator, PluginInfo
from lgca.simulation import DensityRecorder, NodeRecorder, PopulationRecorder


INFO = ExampleInfo(
    name="custom_rest_or_align",
    title="Custom rest-or-align example",
    category="custom dynamics",
    question="How can a notebook interaction rule become a reusable model spec?",
    concepts=("custom interaction", "alignment", "rest channels"),
    source_path="lgca/examples/custom_rest_or_align.py",
    source="BioLGCA.ipynb custom interaction rule",
)


def rest_or_align(lgca) -> None:
    """Reorient particles using alignment and local resting-channel support."""

    newnodes = np.zeros_like(lgca.nodes)
    resting = lgca.nodes[..., lgca.velocitychannels :].sum(-1)
    resting = lgca.nb_sum(resting)
    flux = lgca.nb_sum(lgca.calc_flux(lgca.nodes))
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
        lgca.cell_density[lgca.nonborder] < lgca.K
    )
    coords = [axis[relevant] for axis in lgca.nonborder]
    beta = float(lgca.interaction_params.get("beta", 2.0))
    alpha = float(lgca.interaction_params.get("alpha", 2.0))

    for coord in zip(*coords):
        n_particles = int(lgca.cell_density[coord])
        try:
            permutations = lgca.get_permutations(n_particles)
        except (AttributeError, KeyError, TypeError):
            permutations = lgca.permutations[n_particles]
        try:
            permutation_flux = lgca.get_flux_permutations(n_particles)
        except (AttributeError, KeyError, TypeError):
            permutation_flux = lgca.j[n_particles]
        n_rest = permutations[:, lgca.velocitychannels :].sum(-1)
        weights = np.exp(
            beta * np.einsum("i,ij", flux[coord], permutation_flux)
            + alpha * n_rest * resting[coord]
        ).cumsum()
        index = np.searchsorted(weights, lgca.rng.random() * weights[-1])
        newnodes[coord] = permutations[index]

    lgca.nodes = newnodes


def build_spec() -> ModelSpec:
    """Build the model specification for this example."""

    return ModelSpec(
        description=Description(
            title=INFO.title,
            details="Notebook custom dynamics wrapped as a ModelSpec interaction operator.",
            tags=("example", "custom-interaction", "alignment"),
        ),
        space=SpaceSpec(geometry="hex", dims=(50, 50), boundary="periodic"),
        state=StateSpec(density=0.1, restchannels=1),
        time=TimeSpec(steps=100, seed=112),
        dynamics=InteractionPipelineSpec(
            operators=[
                LegacyInteractionOperator(
                    info=PluginInfo(
                        name="custom.rest_or_align",
                        operator_kind="reorientation",
                        backend_families=("classical",),
                        legacy_source="BioLGCA.ipynb custom interaction rule",
                        conservation_law=ConservationLaw(True, True, False),
                        port_status="example",
                        test_status="smoke",
                        description="Alignment with an added resting-channel preference.",
                    ),
                    legacy_interaction="alignment",
                    parameters={"beta": 2.0, "alpha": 2.0},
                    function_module=__name__,
                    function_name="rest_or_align",
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
