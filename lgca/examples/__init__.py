"""Curated executable ModelSpec examples."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, replace

import numpy as np

from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    run_model,
    save_model_spec,
)
from lgca.pipeline import (
    BirthDeathSpec,
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import DensityRecorder, PopulationRecorder

__all__ = [
    "ExampleInfo",
    "alignment_spec",
    "all_example_specs",
    "chemotaxis_spec",
    "describe_example",
    "example_gallery",
    "example_names",
    "get_example_spec",
    "identity_tumor_growth_spec",
    "multispecies_birth_death_spec",
    "random_walk_spec",
    "run_example",
    "save_example_spec",
]


@dataclass(frozen=True)
class ExampleInfo:
    """Beginner-facing metadata for a curated example."""

    name: str
    title: str
    category: str
    question: str
    concepts: tuple[str, ...]
    source: str = ""


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

    return {name: _EXAMPLE_FACTORIES[name]() for name in example_names()}


def example_names() -> tuple[str, ...]:
    """Return the stable names of the curated examples."""

    return tuple(sorted(_EXAMPLE_FACTORIES))


def example_gallery(category: str | None = None) -> tuple[ExampleInfo, ...]:
    """Return beginner-facing example cards, optionally filtered by category."""

    if category is None:
        return tuple(_EXAMPLE_INFOS[name] for name in _GALLERY_ORDER)
    categories = {info.category for info in _EXAMPLE_INFOS.values()}
    if category not in categories:
        valid = ", ".join(sorted(categories))
        raise ValueError(f"Unknown example category {category!r}. Available categories: {valid}.")
    return tuple(
        _EXAMPLE_INFOS[name]
        for name in _GALLERY_ORDER
        if _EXAMPLE_INFOS[name].category == category
    )


def describe_example(name: str) -> ExampleInfo:
    """Return beginner-facing metadata for one curated example."""

    try:
        return _EXAMPLE_INFOS[name]
    except KeyError as exc:
        raise _unknown_example_error(name, exc)


def get_example_spec(name: str) -> ModelSpec:
    """Return one curated example by name."""

    try:
        return _EXAMPLE_FACTORIES[name]()
    except KeyError as exc:
        raise _unknown_example_error(name, exc)


def run_example(name: str, steps: int | None = None, showprogress: bool = False):
    """Run a curated example by stable name and return its model result."""

    spec = get_example_spec(name)
    if steps is not None:
        spec = replace(spec, time=replace(spec.time, steps=int(steps)))
    return run_model(spec, showprogress=showprogress)


def save_example_spec(name: str, path, file_format: str | None = None):
    """Save a curated example as a JSON or YAML model specification."""

    return save_model_spec(get_example_spec(name), path, file_format=file_format)


def _unknown_example_error(name: str, exc: KeyError) -> ValueError:
    suggestion = difflib.get_close_matches(name, _EXAMPLE_FACTORIES, n=1)
    valid = ", ".join(example_names())
    if suggestion:
        return ValueError(
            f"Unknown example {name!r}. Did you mean {suggestion[0]!r}? "
            f"Available examples: {valid}."
        )
    return ValueError(f"Unknown example {name!r}. Available examples: {valid}.")


_GALLERY_ORDER = (
    "random_walk",
    "alignment",
    "chemotaxis",
    "multispecies_birth_death",
    "identity_tumor_growth",
)


_EXAMPLE_FACTORIES = {
    "alignment": alignment_spec,
    "chemotaxis": chemotaxis_spec,
    "identity_tumor_growth": identity_tumor_growth_spec,
    "multispecies_birth_death": multispecies_birth_death_spec,
    "random_walk": random_walk_spec,
}

_EXAMPLE_INFOS = {
    "random_walk": ExampleInfo(
        name="random_walk",
        title="Random walk",
        category="movement",
        question="How does unbiased cell movement spread a population?",
        concepts=("movement", "diffusion", "density"),
        source="BioLGCA.ipynb class initialization and simulation sections",
    ),
    "alignment": ExampleInfo(
        name="alignment",
        title="Alignment",
        category="collective motion",
        question="How do local alignment rules create coherent streams?",
        concepts=("collective motion", "flux", "reorientation"),
        source="BioLGCA.ipynb alignment example",
    ),
    "chemotaxis": ExampleInfo(
        name="chemotaxis",
        title="Chemotaxis",
        category="guidance",
        question="How does a signal field bias cell movement?",
        concepts=("signal field", "gradient sensing", "reorientation"),
        source="BioLGCA.ipynb chemotaxis; Morpheus multiscale chemotaxis examples",
    ),
    "multispecies_birth_death": ExampleInfo(
        name="multispecies_birth_death",
        title="Multispecies birth-death",
        category="population dynamics",
        question="How do different birth and death rates change competing populations?",
        concepts=("multispecies", "birth", "death"),
        source="Morpheus ODE and multiscale population examples",
    ),
    "identity_tumor_growth": ExampleInfo(
        name="identity_tumor_growth",
        title="Identity-based tumor growth",
        category="tumor growth",
        question="How can individual cell properties drive go-or-grow tumor expansion?",
        concepts=("identity-based LGCA", "go-or-grow", "tumor growth"),
        source="BioLGCA.ipynb go-and-grow/go-or-grow examples",
    ),
}
