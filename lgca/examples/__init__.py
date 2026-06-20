"""Curated executable ModelSpec examples."""

from __future__ import annotations

import difflib

import numpy as np

from lgca.model import AnalysisSpec, Description, ModelSpec, SpaceSpec, StateSpec, TimeSpec
from lgca.pipeline import (
    BirthDeathSpec,
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import DensityRecorder, PopulationRecorder

__all__ = [
    "alignment_spec",
    "all_example_specs",
    "chemotaxis_spec",
    "example_names",
    "get_example_spec",
    "identity_tumor_growth_spec",
    "multispecies_birth_death_spec",
    "random_walk_spec",
]


def random_walk_spec() -> ModelSpec:
    """Return a small classical random-walk example."""

    return ModelSpec(
        description=Description(title="Random walk example"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=2, seed=101),
        dynamics=InteractionPipelineSpec(operators=[{"name": "classical.random_walk"}]),
        analysis=AnalysisSpec(observers=[DensityRecorder(), PopulationRecorder()]),
    )


def alignment_spec() -> ModelSpec:
    """Return a small alignment-interaction example."""

    return ModelSpec(
        description=Description(title="Alignment example"),
        space=SpaceSpec(geometry="hex", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=0),
        time=TimeSpec(steps=2, seed=102),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.alignment", "parameters": {"beta": 1.0}}],
        ),
        analysis=AnalysisSpec(observers=[DensityRecorder()]),
    )


def chemotaxis_spec() -> ModelSpec:
    """Return a small chemotaxis example with a static signal field."""

    signal = np.linspace(0.0, 1.0, 4)[:, None] + np.zeros((4, 4))
    return ModelSpec(
        description=Description(title="Chemotaxis example"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1, fields={"signal": signal}),
        time=TimeSpec(steps=2, seed=103),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=1.0,
                            parameters={"field": "signal"},
                        )
                    ],
                )
            ],
        ),
        analysis=AnalysisSpec(observers=[DensityRecorder()]),
    )


def multispecies_birth_death_spec() -> ModelSpec:
    """Return a small multispecies birth-death example."""

    return ModelSpec(
        description=Description(title="Multispecies birth-death example"),
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
        analysis=AnalysisSpec(observers=[DensityRecorder(), PopulationRecorder()]),
    )


def identity_tumor_growth_spec() -> ModelSpec:
    """Return a small identity-based tumor-growth example."""

    return ModelSpec(
        description=Description(title="Identity tumor growth example"),
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
        analysis=AnalysisSpec(observers=[DensityRecorder(), PopulationRecorder()]),
    )


def all_example_specs() -> dict[str, ModelSpec]:
    """Return all curated examples keyed by stable example name."""

    return {
        "alignment": alignment_spec(),
        "chemotaxis": chemotaxis_spec(),
        "identity_tumor_growth": identity_tumor_growth_spec(),
        "multispecies_birth_death": multispecies_birth_death_spec(),
        "random_walk": random_walk_spec(),
    }


def example_names() -> tuple[str, ...]:
    """Return the stable names of the curated examples."""

    return tuple(sorted(_EXAMPLE_FACTORIES))


def get_example_spec(name: str) -> ModelSpec:
    """Return one curated example by name."""

    try:
        return _EXAMPLE_FACTORIES[name]()
    except KeyError as exc:
        suggestion = difflib.get_close_matches(name, _EXAMPLE_FACTORIES, n=1)
        valid = ", ".join(example_names())
        if suggestion:
            raise ValueError(
                f"Unknown example {name!r}. Did you mean {suggestion[0]!r}? "
                f"Available examples: {valid}."
            ) from exc
        raise ValueError(f"Unknown example {name!r}. Available examples: {valid}.") from exc


_EXAMPLE_FACTORIES = {
    "alignment": alignment_spec,
    "chemotaxis": chemotaxis_spec,
    "identity_tumor_growth": identity_tumor_growth_spec,
    "multispecies_birth_death": multispecies_birth_death_spec,
    "random_walk": random_walk_spec,
}
