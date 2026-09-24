"""Plugin registry of interaction operators.

Every interaction a model file can name is a registered plugin: its metadata
(kind, model families, parameters, description) and a factory for its
operator. Rules written with the decorators of :mod:`lgca.rules` register
themselves; the legacy names of earlier versions are registered by
:mod:`lgca.legacy_names` and warn when a model file uses them.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from ._warnings import warn_user

__all__ = [
    "BirthDeathOperator",
    "ConservationLaw",
    "InteractionOperator",
    "ParameterSpec",
    "PhenotypeSwitchOperator",
    "PluginInfo",
    "PluginRegistry",
    "ReorientationOperator",
    "ReorientationTerm",
    "create_plugin",
    "default_registry",
    "describe_plugin",
    "interaction_coverage_table",
    "list_plugins",
    "register_plugin",
    "validate_plugin_parameters",
]


from .operator_base import (
    BirthDeathOperator,
    ConservationLaw,
    InteractionOperator,
    ParameterSpec,
    PhenotypeSwitchOperator,
    PluginInfo,
    ReorientationOperator,
    ReorientationTerm,
)

PluginFactory = Callable[[Mapping[str, Any] | None], InteractionOperator]


from .operator_registry import PluginRegistry

default_registry = PluginRegistry()


def register_plugin(info: PluginInfo, factory: PluginFactory, replace: bool = False) -> None:
    """Register a plugin in the default registry.

    Registering the same name again from the same module replaces the earlier
    entry, so a notebook cell that defines and registers a plugin can be run
    again. Pass ``replace=True`` to replace a plugin registered by another
    module, such as a built-in interaction.
    """

    default_registry.register(info, factory, replace=replace)


def list_plugins(kind: str | None = None, deprecated: bool = False) -> list[PluginInfo]:
    """List plugins from the default registry; ``deprecated=True`` includes deprecated names."""

    return [info for info in default_registry.list(kind=kind) if deprecated or not info.deprecated]


def describe_plugin(name: str) -> PluginInfo:
    """Return metadata for one registered plugin."""

    return default_registry.describe(name)


def validate_plugin_parameters(
    info: PluginInfo,
    parameters: Mapping[str, Any] | None,
    context: Any = None,
) -> None:
    """Validate parameters against a plugin's declared contract."""

    parameters = dict(parameters or {})
    parameter_specs = info.parameter_specs
    unknown = sorted(set(parameters) - set(parameter_specs))
    if unknown:
        details = []
        for name in unknown:
            matches = difflib.get_close_matches(name, parameter_specs, n=1)
            suggestion = f" (did you mean {matches[0]!r}?)" if matches else ""
            details.append(f"{info.name}.{name}{suggestion}")
        raise ValueError("unknown plugin parameter(s): " + ", ".join(details))

    canonical_capacity = getattr(getattr(context, "spec", None), "state", None)
    canonical_capacity = getattr(canonical_capacity, "capacity", None)
    if canonical_capacity is not None and "capacity" in parameters:
        if parameters["capacity"] != canonical_capacity:
            raise ValueError(
                f"{info.name}.capacity conflicts with model.state.capacity={canonical_capacity}"
            )
        warn_user(f"{info.name}.capacity duplicates model.state.capacity and is deprecated", DeprecationWarning)

    for name, spec in parameter_specs.items():
        if spec.required and name not in parameters:
            raise ValueError(f"{info.name}.{name} is required")
        if name not in parameters:
            continue
        _validate_parameter_value(info, name, spec, parameters[name], context)


def _validate_parameter_value(
    info: PluginInfo,
    name: str,
    spec: ParameterSpec,
    value: Any,
    context: Any,
) -> None:
    if spec.allowed_values is not None and value not in spec.allowed_values:
        valid = ", ".join(str(item) for item in spec.allowed_values)
        raise ValueError(f"{info.name}.{name} must be one of: {valid}")

    if spec.type_label == "probability":
        values = np.asarray(value, dtype=float)
        if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{info.name}.{name} must be a probability")
    elif spec.type_label == "finite scalar":
        if isinstance(value, bool) or np.asarray(value).ndim != 0 or not np.isfinite(float(value)):
            raise ValueError(f"{info.name}.{name} must be a finite scalar")
    elif spec.type_label == "positive integer":
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{info.name}.{name} must be a positive integer")
    elif spec.type_label == "non-negative integer":
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{info.name}.{name} must be a non-negative integer")
    elif spec.type_label == "string":
        if not isinstance(value, str):
            raise ValueError(f"{info.name}.{name} must be a string")
    elif spec.type_label == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{info.name}.{name} must be a boolean")
    elif spec.type_label == "array":
        np.asarray(value)

    _validate_parameter_shape(info, name, spec, value, context)


def _validate_parameter_shape(
    info: PluginInfo,
    name: str,
    spec: ParameterSpec,
    value: Any,
    context: Any,
) -> None:
    if spec.shape is None or context is None:
        return
    arr = np.asarray(value)
    if spec.shape == "spatial_field":
        expected = tuple(context.lgca.dims)
    elif spec.shape == "spatial_vector_field":
        expected = tuple(context.lgca.dims) + (context.lgca.c.shape[0],)
    else:
        expected = tuple(spec.shape)
    if arr.shape != expected:
        raise ValueError(f"{info.name}.{name} must have shape {expected}; got {arr.shape}")


def interaction_coverage_table() -> list[dict[str, Any]]:
    """Return registry rows suitable for documentation and migration audits."""

    rows = []
    for plugin in list_plugins(kind="interaction"):
        rows.append(
            {
                "name": plugin.name,
                "aliases": plugin.aliases,
                "operator_kind": plugin.operator_kind,
                "backend_families": plugin.backend_families,
                "port_status": plugin.port_status,
                "test_status": plugin.test_status,
                "legacy_source": plugin.legacy_source,
                "parameters": plugin.parameters,
                "conservation_law": plugin.conservation_law.describe(),
                "description": plugin.description,
            }
        )
    return rows


def create_plugin(name: str, parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
    """Instantiate a plugin operator by name."""

    factory = default_registry.resolve(name)
    info = default_registry.describe(name)
    if info.deprecated:
        warn_user(f"The interaction name {name!r} is deprecated; {info.deprecated}", FutureWarning)
    return factory(parameters)


def _law_for_kind(kind: str) -> ConservationLaw:
    if kind == "birth_death":
        return ConservationLaw(False, False, False, ("particle number",))
    if kind == "phenotype_switch":
        return ConservationLaw(True, False, False, ("phenotype identity",))
    if kind == "reorientation":
        return ConservationLaw(True, True, False, ("channel occupancy",))
    if kind == "propagation":
        return ConservationLaw(True, True, None, ("position",))
    return ConservationLaw(None, None, None)


def _register_example_plugins() -> None:
    custom_rest_or_align_info = PluginInfo(
        name="custom.rest_or_align",
        operator_kind="reorientation",
        backend_families=("classical",),
        parameters={
            "beta": {"default": 2.0, "type_label": "finite scalar"},
            "alpha": {"default": 2.0, "type_label": "finite scalar"},
        },
        conservation_law=_law_for_kind("reorientation"),
        port_status="native",
        test_status="unit_tested",
        description="Alignment with an added resting-channel preference.",
    )

    def custom_rest_or_align_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .examples.custom_rest_or_align import RestOrAlignOperator

        return RestOrAlignOperator(custom_rest_or_align_info, parameters)

    register_plugin(custom_rest_or_align_info, custom_rest_or_align_factory)


# Plain-language meaning of built-in interaction parameters, shown by
# describe_plugin(). Entries keyed by (plugin, parameter) override the
# shared meaning of a parameter name.
_SWITCH = ("Steepness of the density-dependent switch between moving and resting. "
           "The probability that a moving cell starts resting is "
           "(1 + tanh(kappa * (rho - theta))) / 2, with rho the number of cells at the node "
           "divided by its capacity. Positive kappa makes crowded cells rest, negative kappa "
           "makes them move.")
_PARAMETER_MEANINGS = {
    "r_b": "Probability per time step that a cell divides.",
    "r_d": "Probability per time step that a cell dies.",
    "birth_rate": "Probability per time step that a cell divides into a free channel of its node; "
                  "one value per species or a single value for all.",
    "death_rate": "Probability per time step that a cell dies; one value per species or a single "
                  "value for all.",
    "beta": "Sensitivity to the directional cue: the weight of its score in the reorientation "
            "probability, P(state) ~ exp(beta * score). 0 gives a random walk.",
    "kappa": _SWITCH,
    "theta": "Relative density rho at which moving and resting are equally likely (see kappa).",
    "kappa_std": "Standard deviation of the random change of kappa that daughter cells inherit.",
    "theta_std": "Standard deviation of the random change of theta that daughter cells inherit.",
    "std": "Standard deviation of the random change of the birth probability r_b that daughter "
           "cells inherit (a normal distribution truncated to [0, a_max]).",
    "a_max": "Upper limit of the birth probability r_b after mutation.",
    "capacity": "No longer used: set StateSpec.capacity. Accepted when it equals it.",
    "gamma": "Preference of cells for the rest channel when they are distributed over the "
             "channels of a node (log-weight of the rest channel relative to a velocity channel).",
    "N": "Number of repetitions of the fast reaction in every time step.",
    "alpha": "Excitability of the medium.",
    "drb": "Amount by which a mutation raises or lowers a daughter cell's birth probability r_b.",
    "pmut": "Probability that a daughter cell's birth probability r_b mutates (by +drb or -drb).",
    "effect": "'passenger_mutation' leaves the birth probability unchanged; 'driver_mutation' "
              "multiplies it by fitness_increase.",
    "fitness_increase": "Factor by which a driver mutation multiplies the birth probability r_b.",
    "r_m": "Probability that a daughter cell acquires a mutation (founding a new family).",
    "p_d": "Probability that a daughter cell acquires a driver mutation, which raises r_b.",
    "p_p": "Probability that a daughter cell acquires a passenger mutation, which lowers r_b.",
    "s_d": "Mean increase of r_b by a driver mutation (exponentially distributed).",
    "s_p": "Mean decrease of r_b by a passenger mutation (exponentially distributed).",
    "mutation_matrix": "Probability that a daughter of species a belongs to species b "
                       "(entry [a][b]; rows sum to 1). The identity matrix means no mutation.",
    "include_center": "Count the cells at the node itself in its neighbourhood, in addition to "
                      "the neighbouring nodes.",
    "track_inheritance": "Record which family (lineage) every cell belongs to.",
    "r_int": "Interaction radius: the number of nodes around a node that count as its "
             "neighbourhood.",
    "gradient": "Gradient of the attractant signal at every node, with ghost nodes; defaults to "
                "that of a concentration peaked in the middle of the lattice.",
    "director": "Direction of the guiding fibres at every node, with ghost nodes; defaults to "
                "the first axis.",
    "rates": "Matrix of switch probabilities per time step: entry [a][b] is the probability "
             "that a cell of species a becomes species b. Off-diagonal row sums must be <= 1.",
}
_PLUGIN_PARAMETER_MEANINGS = {
    ("custom.rest_or_align", "alpha"): "Preference for resting over alignment.",
}


def _with_parameter_meanings(info: PluginInfo) -> PluginInfo:
    """Fill empty parameter descriptions of a built-in plugin from the glossary."""
    parameters = {}
    for name, spec in info.parameter_specs.items():
        if not spec.description:
            meaning = _PLUGIN_PARAMETER_MEANINGS.get((info.name, name), _PARAMETER_MEANINGS.get(name, ""))
            spec = replace(spec, description=meaning)
        parameters[name] = spec
    return replace(info, parameters=parameters)



def _describe_builtin_plugins() -> None:
    for name, (info, factory) in list(default_registry._plugins.items()):
        default_registry._plugins[name] = (_with_parameter_meanings(info), factory)

_register_example_plugins()
_describe_builtin_plugins()
