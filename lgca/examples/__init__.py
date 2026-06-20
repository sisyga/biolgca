"""Curated executable ModelSpec examples.

Each curated example lives in its own module under :mod:`lgca.examples`. The
package-level helpers are only an index; students should read the individual
files to see how a model is assembled and run.
"""

from __future__ import annotations

import difflib
import importlib
from dataclasses import replace
from types import ModuleType

from lgca.model import ModelSpec, run_model, save_model_spec

from ._types import ExampleInfo

__all__ = [
    "ExampleInfo",
    "alignment",
    "alignment_spec",
    "all_example_specs",
    "chemotaxis",
    "chemotaxis_spec",
    "describe_example",
    "example_gallery",
    "example_names",
    "get_example_spec",
    "identity_tumor_growth",
    "identity_tumor_growth_spec",
    "multispecies_birth_death",
    "multispecies_birth_death_spec",
    "random_walk",
    "random_walk_spec",
    "run_example",
    "save_example_spec",
]


def random_walk_spec() -> ModelSpec:
    """Return the random-walk example specification."""

    return _example_module("random_walk").build_spec()


def alignment_spec() -> ModelSpec:
    """Return the alignment example specification."""

    return _example_module("alignment").build_spec()


def chemotaxis_spec() -> ModelSpec:
    """Return the chemotaxis example specification."""

    return _example_module("chemotaxis").build_spec()


def multispecies_birth_death_spec() -> ModelSpec:
    """Return the multispecies birth-death example specification."""

    return _example_module("multispecies_birth_death").build_spec()


def identity_tumor_growth_spec() -> ModelSpec:
    """Return the identity-based tumor-growth example specification."""

    return _example_module("identity_tumor_growth").build_spec()


def all_example_specs() -> dict[str, ModelSpec]:
    """Return all curated examples keyed by stable example name."""

    return {name: _example_module(name).build_spec() for name in example_names()}


def example_names() -> tuple[str, ...]:
    """Return the stable names of the curated examples."""

    return tuple(sorted(_EXAMPLE_MODULE_NAMES))


def example_gallery(category: str | None = None) -> tuple[ExampleInfo, ...]:
    """Return beginner-facing example cards, optionally filtered by category."""

    if category is None:
        return tuple(_example_module(name).INFO for name in _GALLERY_ORDER)
    categories = {_example_module(name).INFO.category for name in _EXAMPLE_MODULE_NAMES}
    if category not in categories:
        valid = ", ".join(sorted(categories))
        raise ValueError(f"Unknown example category {category!r}. Available categories: {valid}.")
    return tuple(
        _example_module(name).INFO
        for name in _GALLERY_ORDER
        if _example_module(name).INFO.category == category
    )


def describe_example(name: str) -> ExampleInfo:
    """Return beginner-facing metadata for one curated example."""

    try:
        return _example_module(name).INFO
    except KeyError as exc:
        raise _unknown_example_error(name, exc)


def get_example_spec(name: str) -> ModelSpec:
    """Return one curated example by name."""

    try:
        return _example_module(name).build_spec()
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


def __getattr__(name: str):
    if name in _EXAMPLE_MODULE_NAMES:
        return _example_module(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _unknown_example_error(name: str, exc: KeyError) -> ValueError:
    suggestion = difflib.get_close_matches(name, _EXAMPLE_MODULE_NAMES, n=1)
    valid = ", ".join(example_names())
    if suggestion:
        return ValueError(
            f"Unknown example {name!r}. Did you mean {suggestion[0]!r}? "
            f"Available examples: {valid}."
        )
    return ValueError(f"Unknown example {name!r}. Available examples: {valid}.")


def _example_module(name: str) -> ModuleType:
    if name not in _EXAMPLE_MODULE_NAMES:
        raise KeyError(name)
    return importlib.import_module(f"{__name__}.{name}")


_GALLERY_ORDER = (
    "random_walk",
    "alignment",
    "chemotaxis",
    "multispecies_birth_death",
    "identity_tumor_growth",
)

_EXAMPLE_MODULE_NAMES = frozenset(_GALLERY_ORDER)
