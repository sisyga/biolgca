"""Plugin registry and interaction operators.

The registry mirrors the Morpheus pattern of named plugins with metadata,
parameters, and lifecycle hooks. Built-in interactions are registered as native
operators with conservation metadata and compatibility tests against the
previous interaction semantics. The legacy adapter remains available for custom
or transitional interaction functions.
"""

from __future__ import annotations

import difflib
from ._warnings import warn_user
from dataclasses import replace
from typing import Any, Callable, Mapping

import numpy as np


__all__ = [
    "BirthDeathOperator",
    "ConservationLaw",
    "InteractionOperator",
    "LegacyInteractionOperator",
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
    LegacyInteractionOperator,
    ParameterSpec,
    PhenotypeSwitchOperator,
    PluginInfo,
    ReorientationOperator,
    ReorientationTerm,
)


_KIND_TO_BASE = {
    "birth_death": BirthDeathOperator,
    "phenotype_switch": PhenotypeSwitchOperator,
    "reorientation": ReorientationOperator,
}


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


def list_plugins(kind: str | None = None) -> list[PluginInfo]:
    """List plugins from the default registry."""

    return default_registry.list(kind=kind)


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


def resolve_operator_capacity(context, parameters, default):
    """Resolve an explicit override, canonical state capacity, then default.

    Parameter validation rejects conflicting explicit overrides. Factories must
    leave absent capacity parameters absent until this resolution step.
    """
    canonical = context.spec.state.capacity
    return parameters.get("capacity", canonical if canonical is not None else default)


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

    return default_registry.resolve(name)(parameters)


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


def _legacy_source(module: str, function: str) -> str:
    return f"{module}.{function}"


def _register_legacy(
    *,
    name: str,
    backend: str,
    module: str,
    function: str,
    legacy_interaction: str | None = None,
    operator_kind: str,
    aliases: tuple[str, ...] = (),
    default_parameters: Mapping[str, Any] | None = None,
    description: str = "",
) -> None:
    default_parameters = dict(default_parameters or {})
    info = PluginInfo(
        name=name,
        aliases=aliases,
        operator_kind=operator_kind,
        backend_families=(backend,),
        legacy_source=_legacy_source(module, function),
        parameters=default_parameters,
        conservation_law=_law_for_kind(operator_kind),
        description=description,
        test_status="smoke_tested",
    )
    adapter_cls = _KIND_TO_BASE.get(operator_kind, LegacyInteractionOperator)
    if adapter_cls is LegacyInteractionOperator:
        legacy_operator_cls = LegacyInteractionOperator
    else:
        class _LegacyOperator(adapter_cls, LegacyInteractionOperator):
            pass

        legacy_operator_cls = _LegacyOperator

    def factory(parameters: Mapping[str, Any] | None = None) -> LegacyInteractionOperator:
        merged_parameters = dict(default_parameters)
        merged_parameters.update(dict(parameters or {}))
        return legacy_operator_cls(
            info=info,
            legacy_interaction=legacy_interaction or function,
            parameters=merged_parameters,
            function_module=module,
            function_name=function,
        )

    register_plugin(info, factory)


def _register_native_plugins() -> None:
    only_propagation_info = PluginInfo(
        name="classical.only_propagation",
        aliases=("only_propagation", "propagation.default"),
        operator_kind="propagation",
        backend_families=("classical", "ib", "nove", "nove_ib", "multispecies"),
        legacy_source=_legacy_source("lgca.interactions", "only_propagation"),
        conservation_law=_law_for_kind("propagation"),
        port_status="native",
        test_status="unit_tested",
        description="Native no-op interaction marker that delegates movement to the pipeline propagation phase.",
    )

    def only_propagation_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeOnlyPropagationOperator

        return NativeOnlyPropagationOperator(only_propagation_info, parameters)

    register_plugin(only_propagation_info, only_propagation_factory)

    for name, mode, legacy_function, parameter_defaults, description in (
        (
            "multispecies.birth",
            "birth",
            "birth",
            {"r_b": 0.2, "gamma": 0.0},
            "Multispecies no-volume-exclusion birth with optional mutation.",
        ),
        (
            "multispecies.birthdeath",
            "birthdeath",
            "birthdeath",
            {"r_b": 0.2, "r_d": 0.02, "gamma": 0.0},
            "Multispecies no-volume-exclusion birth-death with optional mutation.",
        ),
    ):
        info = PluginInfo(
            name=name,
            operator_kind="birth_death",
            backend_families=("multispecies",),
            legacy_source=_legacy_source("lgca.ms_interactions", legacy_function),
            parameters={
                "r_b": {
                    "default": parameter_defaults["r_b"],
                    "validator": "probability scalar or per-species vector",
                },
                "gamma": {
                    "default": parameter_defaults["gamma"],
                    "validator": "finite scalar rest-channel bias",
                },
                "mutation_matrix": {
                    "default": "identity",
                    "validator": "row-stochastic n_species x n_species matrix",
                },
                **(
                    {
                        "r_d": {
                            "default": parameter_defaults["r_d"],
                            "validator": "probability scalar or per-species vector",
                        }
                    }
                    if "r_d" in parameter_defaults
                    else {}
                ),
            },
            conservation_law=_law_for_kind("birth_death"),
            port_status="native",
            test_status="unit_tested",
            description=description,
        )

        def factory(
            parameters: Mapping[str, Any] | None = None,
            *,
            info=info,
            mode=mode,
            parameter_defaults=parameter_defaults,
        ) -> InteractionOperator:
            from .pipeline import NativeMultispeciesBirthOperator

            merged_parameters = dict(parameter_defaults)
            merged_parameters.update(dict(parameters or {}))
            return NativeMultispeciesBirthOperator(
                info=info,
                mode=mode,
                parameters=merged_parameters,
            )

        register_plugin(info, factory)

    multispecies_go_or_grow_info = PluginInfo(
        name="multispecies.go_or_grow",
        operator_kind="birth_death",
        backend_families=("multispecies",),
        legacy_source=_legacy_source("lgca.ms_interactions", "go_or_grow"),
        parameters={
            "capacity": {
                "default": "current lattice capacity",
                "validator": "positive integer",
            },
            "r_b": {
                "default": 0.2,
                "validator": "probability scalar or per-species vector",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability scalar or per-species vector",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar or per-species vector",
            },
            "theta": {
                "default": 0.5,
                "validator": "finite scalar",
            },
            "kappa_std": {
                "default": "omitted",
                "validator": "non-negative finite scalar used to derive mutation matrix",
            },
            "mutation_matrix": {
                "default": "identity",
                "validator": "row-stochastic n_species x n_species matrix",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Multispecies no-volume-exclusion go-or-grow with species-level switching.",
    )

    def multispecies_go_or_grow_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeMultispeciesGoOrGrowOperator

        merged_parameters = {"r_b": 0.2, "r_d": 0.01, "kappa": 5.0, "theta": 0.5}
        merged_parameters.update(dict(parameters or {}))
        return NativeMultispeciesGoOrGrowOperator(
            multispecies_go_or_grow_info,
            merged_parameters,
        )

    register_plugin(multispecies_go_or_grow_info, multispecies_go_or_grow_factory)

    multispecies_excitable_medium_info = PluginInfo(
        name="multispecies.excitable_medium_ms",
        operator_kind="birth_death",
        backend_families=("multispecies",),
        legacy_source=_legacy_source("lgca.ms_interactions", "excitable_medium_ms"),
        parameters={
            "beta": {
                "default": 0.05,
                "validator": "finite scalar interaction coefficient",
            },
            "alpha": {
                "default": 1.0,
                "validator": "non-zero finite scalar excitability scale",
            },
            "N": {
                "default": 50,
                "validator": "non-negative integer fast-reaction repetitions",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Two-species volume-exclusion excitable-medium reaction operator.",
    )

    def multispecies_excitable_medium_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeMultispeciesExcitableMediumOperator

        merged_parameters = {"beta": 0.05, "alpha": 1.0, "N": 50}
        merged_parameters.update(dict(parameters or {}))
        return NativeMultispeciesExcitableMediumOperator(
            multispecies_excitable_medium_info,
            merged_parameters,
        )

    register_plugin(
        multispecies_excitable_medium_info,
        multispecies_excitable_medium_factory,
    )

    for name, mode, legacy_function, aliases, parameter_defaults, description in (
        (
            "classical.birth",
            "birth",
            "birth",
            ("birth",),
            {"r_b": 0.2},
            "Classical volume-exclusion birth step followed by random walk.",
        ),
        (
            "classical.birthdeath",
            "birthdeath",
            "birthdeath",
            ("birthdeath",),
            {"r_b": 0.2, "r_d": 0.05},
            "Classical volume-exclusion birth-death step followed by random walk.",
        ),
    ):
        info = PluginInfo(
            name=name,
            aliases=aliases,
            operator_kind="birth_death",
            backend_families=("classical",),
            legacy_source=_legacy_source("lgca.interactions", legacy_function),
            parameters={
                parameter: {
                    "default": default,
                    "validator": "probability",
                }
                for parameter, default in parameter_defaults.items()
            },
            conservation_law=_law_for_kind("birth_death"),
            port_status="native",
            test_status="unit_tested",
            description=description,
        )

        def factory(
            parameters: Mapping[str, Any] | None = None,
            *,
            info=info,
            mode=mode,
            parameter_defaults=parameter_defaults,
        ) -> InteractionOperator:
            from .pipeline import NativeClassicalBirthOperator

            merged_parameters = dict(parameter_defaults)
            merged_parameters.update(dict(parameters or {}))
            return NativeClassicalBirthOperator(
                info=info,
                mode=mode,
                parameters=merged_parameters,
            )

        register_plugin(info, factory)

    phenotype_switch_info = PluginInfo(
        name="phenotype_switch",
        aliases=("species_switch",),
        operator_kind="phenotype_switch",
        backend_families=("multispecies",),
        parameters={
            "rates": {
                "default": [[0.0, 0.0], [0.0, 0.0]],
                "validator": "square transition matrix with off-diagonal row sums <= 1",
            },
        },
        conservation_law=_law_for_kind("phenotype_switch"),
        port_status="native",
        test_status="unit_tested",
        description="Cells of a multispecies LGCA switch species at given rates; the number of cells "
                    "at each node is conserved and a node's cells are redistributed over its channels "
                    "when one of them switches.",
    )

    def phenotype_switch_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativePhenotypeSwitchOperator

        merged_parameters = {"rates": [[0.0, 0.0], [0.0, 0.0]]}
        merged_parameters.update(dict(parameters or {}))
        return NativePhenotypeSwitchOperator(merged_parameters)

    register_plugin(phenotype_switch_info, phenotype_switch_factory)

    for name, density_dependent, description in (
        (
            "nove.dd_alignment",
            True,
            "Density-dependent alignment for no-volume-exclusion LGCA.",
        ),
        (
            "nove.di_alignment",
            False,
            "Density-independent alignment for no-volume-exclusion LGCA.",
        ),
    ):
        info = PluginInfo(
            name=name,
            operator_kind="reorientation",
            backend_families=("nove",),
            legacy_source=_legacy_source("lgca.nove_interactions", name.split(".", 1)[1]),
            parameters={
                "beta": {
                    "default": 2.0,
                    "validator": "finite scalar alignment sensitivity",
                },
                "include_center": {
                    "default": False,
                    "validator": "boolean",
                },
            },
            conservation_law=_law_for_kind("reorientation"),
            port_status="native",
            test_status="unit_tested",
            description=description,
        )

        def factory(
            parameters: Mapping[str, Any] | None = None,
            *,
            info=info,
            density_dependent=density_dependent,
        ) -> InteractionOperator:
            from .pipeline import NativeNoVEAlignmentOperator

            merged_parameters = {"beta": 2.0, "include_center": False}
            merged_parameters.update(dict(parameters or {}))
            return NativeNoVEAlignmentOperator(
                info=info,
                density_dependent=density_dependent,
                parameters=merged_parameters,
            )

        register_plugin(info, factory)

    random_walk_info = PluginInfo(
        name="nove.random_walk",
        operator_kind="reorientation",
        backend_families=("nove",),
        legacy_source=_legacy_source("lgca.nove_interactions", "random_walk"),
        conservation_law=_law_for_kind("reorientation"),
        port_status="native",
        test_status="unit_tested",
        description="Uniform channel redistribution for no-volume-exclusion LGCA.",
    )

    def nove_random_walk_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeNoVERandomWalkOperator

        return NativeNoVERandomWalkOperator(random_walk_info, parameters)

    register_plugin(random_walk_info, nove_random_walk_factory)

    nove_go_or_rest_info = PluginInfo(
        name="nove.go_or_rest",
        operator_kind="reorientation",
        backend_families=("nove",),
        legacy_source=_legacy_source("lgca.nove_interactions", "go_or_rest"),
        parameters={
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar switching steepness",
            },
            "theta": {
                "default": 0.75,
                "validator": "finite scalar switching threshold",
            },
        },
        conservation_law=_law_for_kind("reorientation"),
        port_status="native",
        test_status="unit_tested",
        description=("Cells move between velocity and rest channels with a density-dependent "
                     "probability of resting; without volume exclusion."),
    )

    def nove_go_or_rest_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeNoVEGoOrRestOperator

        merged_parameters = {"kappa": 5.0, "theta": 0.75}
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEGoOrRestOperator(nove_go_or_rest_info, merged_parameters)

    register_plugin(nove_go_or_rest_info, nove_go_or_rest_factory)

    nove_go_or_grow_info = PluginInfo(
        name="nove.go_or_grow",
        operator_kind="birth_death",
        backend_families=("nove",),
        legacy_source=_legacy_source("lgca.nove_interactions", "go_or_grow"),
        parameters={
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar switching steepness",
            },
            "theta": {
                "default": 0.75,
                "validator": "finite scalar switching threshold",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="No-volume-exclusion go-or-grow switch with death and resting-cell birth.",
    )

    def nove_go_or_grow_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeNoVEGoOrGrowOperator

        merged_parameters = {"r_b": 0.2, "r_d": 0.01, "kappa": 5.0, "theta": 0.75}
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEGoOrGrowOperator(nove_go_or_grow_info, merged_parameters)

    register_plugin(nove_go_or_grow_info, nove_go_or_grow_factory)

    from .classical_operators import register_classical_random_walk

    register_classical_random_walk(register_plugin)

    classical_excitable_medium_info = PluginInfo(
        name="classical.excitable_medium",
        operator_kind="birth_death",
        backend_families=("classical",),
        legacy_source=_legacy_source("lgca.interactions", "excitable_medium"),
        parameters={
            "beta": {
                "default": 0.05,
                "validator": "finite scalar interaction coefficient",
            },
            "alpha": {
                "default": 1.0,
                "validator": "non-zero finite scalar excitability scale",
            },
            "N": {
                "default": 50,
                "validator": "non-negative integer fast-reaction repetitions",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Classical Barkley-style excitable-medium reaction operator.",
    )

    def classical_excitable_medium_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeClassicalExcitableMediumOperator

        merged_parameters = {"beta": 0.05, "alpha": 1.0, "N": 50}
        merged_parameters.update(dict(parameters or {}))
        return NativeClassicalExcitableMediumOperator(
            classical_excitable_medium_info,
            merged_parameters,
        )

    register_plugin(classical_excitable_medium_info, classical_excitable_medium_factory)

    ib_random_walk_info = PluginInfo(
        name="ib.random_walk",
        operator_kind="reorientation",
        backend_families=("ib",),
        legacy_source=_legacy_source("lgca.ib_interactions", "random_walk"),
        conservation_law=_law_for_kind("reorientation"),
        port_status="native",
        test_status="unit_tested",
        description="Identity-preserving channel permutation for volume-exclusion LGCA.",
    )

    def ib_random_walk_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeIdentityRandomWalkOperator

        return NativeIdentityRandomWalkOperator(ib_random_walk_info, parameters)

    register_plugin(ib_random_walk_info, ib_random_walk_factory)

    ib_birth_info = PluginInfo(
        name="ib.birth",
        operator_kind="birth_death",
        backend_families=("ib",),
        legacy_source=_legacy_source("lgca.ib_interactions", "birth"),
        parameters={
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "std": {
                "default": 0.01,
                "validator": "positive finite scalar daughter-rate mutation width",
            },
            "a_max": {
                "default": 1.0,
                "validator": "positive finite scalar birth-rate cap",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Identity-based volume-exclusion birth with inherited proliferation rates.",
    )

    def ib_birth_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeIdentityBirthOperator

        merged_parameters = {"r_b": 0.2, "std": 0.01, "a_max": 1.0}
        merged_parameters.update(dict(parameters or {}))
        return NativeIdentityBirthOperator(ib_birth_info, merged_parameters)

    register_plugin(ib_birth_info, ib_birth_factory)

    ib_birthdeath_info = PluginInfo(
        name="ib.birthdeath",
        operator_kind="birth_death",
        backend_families=("ib",),
        legacy_source=_legacy_source("lgca.ib_interactions", "birthdeath"),
        parameters={
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.02,
                "validator": "probability",
            },
            "std": {
                "default": 0.01,
                "validator": "non-negative finite scalar daughter-rate mutation width",
            },
            "a_max": {
                "default": 1.0,
                "validator": "positive finite scalar birth-rate cap",
            },
            "track_inheritance": {
                "default": False,
                "validator": "boolean",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Identity-based volume-exclusion birth-death with inherited proliferation rates.",
    )

    def ib_birthdeath_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeIdentityBirthDeathOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.02,
            "std": 0.01,
            "a_max": 1.0,
            "track_inheritance": False,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeIdentityBirthDeathOperator(ib_birthdeath_info, merged_parameters)

    register_plugin(ib_birthdeath_info, ib_birthdeath_factory)

    ib_birthdeath_discrete_info = PluginInfo(
        name="ib.birthdeath_discrete",
        operator_kind="birth_death",
        backend_families=("ib",),
        legacy_source=_legacy_source("lgca.ib_interactions", "birthdeath_discrete"),
        parameters={
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.02,
                "validator": "probability",
            },
            "drb": {
                "default": 0.01,
                "validator": "non-negative finite scalar discrete daughter-rate step",
            },
            "a_max": {
                "default": 1.0,
                "validator": "positive finite scalar birth-rate cap",
            },
            "pmut": {
                "default": 0.1,
                "validator": "probability",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Identity-based birth-death with discrete proliferation-rate mutation.",
    )

    def ib_birthdeath_discrete_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeIdentityBirthDeathDiscreteOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.02,
            "drb": 0.01,
            "a_max": 1.0,
            "pmut": 0.1,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeIdentityBirthDeathDiscreteOperator(
            ib_birthdeath_discrete_info,
            merged_parameters,
        )

    register_plugin(ib_birthdeath_discrete_info, ib_birthdeath_discrete_factory)

    ib_go_and_grow_mutations_info = PluginInfo(
        name="ib.go_and_grow_mutations",
        operator_kind="birth_death",
        backend_families=("ib",),
        legacy_source=_legacy_source("lgca.ib_interactions", "go_and_grow_mutations"),
        parameters={
            "effect": {
                "default": "passenger_mutation",
                "validator": "one of {'passenger_mutation', 'driver_mutation'}",
            },
            "r_b": {
                "default": 0.5,
                "validator": "probability",
            },
            "r_m": {
                "default": 0.001,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.02,
                "validator": "probability",
            },
            "fitness_increase": {
                "default": 1.1,
                "validator": "positive finite scalar driver mutation multiplier",
            },
            "r_int": {
                "default": "current lattice interaction radius",
                "validator": "positive integer",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        mutates_families=True,
        description=(
            "Identity-based go-and-grow interaction with passenger or driver "
            "family mutations."
        ),
    )

    def ib_go_and_grow_mutations_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeIdentityGoAndGrowMutationsOperator

        merged_parameters = {
            "effect": "passenger_mutation",
            "r_b": 0.5,
            "r_m": 0.001,
            "r_d": 0.02,
            "fitness_increase": 1.1,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeIdentityGoAndGrowMutationsOperator(
            ib_go_and_grow_mutations_info,
            merged_parameters,
        )

    register_plugin(ib_go_and_grow_mutations_info, ib_go_and_grow_mutations_factory)

    ib_go_or_grow_info = PluginInfo(
        name="ib.go_or_grow",
        operator_kind="birth_death",
        backend_families=("ib",),
        legacy_source=_legacy_source("lgca.ib_interactions", "go_or_grow"),
        parameters={
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar or one value per initial cell",
            },
            "theta": {
                "default": 0.75,
                "validator": "finite scalar or one value per initial cell",
            },
            "kappa_std": {
                "default": 0.2,
                "validator": "non-negative finite scalar",
            },
            "theta_std": {
                "default": 0.05,
                "validator": "non-negative finite scalar",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description=(
            "Identity-based go-or-grow switching with inherited kappa and theta traits."
        ),
    )

    def ib_go_or_grow_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeIdentityGoOrGrowOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.01,
            "kappa": 5.0,
            "theta": 0.75,
            "kappa_std": 0.2,
            "theta_std": 0.05,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeIdentityGoOrGrowOperator(
            ib_go_or_grow_info,
            merged_parameters,
        )

    register_plugin(ib_go_or_grow_info, ib_go_or_grow_factory)

    nove_ib_random_walk_info = PluginInfo(
        name="nove_ib.random_walk",
        operator_kind="reorientation",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "random_walk"),
        conservation_law=_law_for_kind("reorientation"),
        port_status="native",
        test_status="unit_tested",
        description="Identity-preserving uniform redistribution for no-volume-exclusion LGCA.",
    )

    def nove_ib_random_walk_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityRandomWalkOperator

        return NativeNoVEIdentityRandomWalkOperator(nove_ib_random_walk_info, parameters)

    register_plugin(nove_ib_random_walk_info, nove_ib_random_walk_factory)

    for name, mode, legacy_function, parameter_defaults, description in (
        (
            "nove_ib.birth",
            "birth",
            "birth",
            {"capacity": 8, "r_b": 0.2, "std": 0.01, "a_max": 1.0, "gamma": 0.0},
            "Identity no-volume-exclusion logistic birth with inherited proliferation rates.",
        ),
        (
            "nove_ib.birthdeath",
            "birthdeath",
            "birthdeath",
            {
                "capacity": 8,
                "r_b": 0.2,
                "r_d": 0.02,
                "std": 0.01,
                "a_max": 1.0,
                "gamma": 0.0,
            },
            "Identity no-volume-exclusion logistic birth-death with inherited proliferation rates.",
        ),
    ):
        info = PluginInfo(
            name=name,
            operator_kind="birth_death",
            backend_families=("nove_ib",),
            legacy_source=_legacy_source("lgca.nove_ib_interactions", legacy_function),
            parameters={
                "capacity": {
                    "default": parameter_defaults["capacity"],
                    "validator": "positive integer node capacity",
                },
                "r_b": {
                    "default": parameter_defaults["r_b"],
                    "validator": "probability",
                },
                "std": {
                    "default": parameter_defaults["std"],
                    "validator": "positive finite scalar daughter-rate mutation width",
                },
                "a_max": {
                    "default": parameter_defaults["a_max"],
                    "validator": "positive finite scalar birth-rate cap",
                },
                "gamma": {
                    "default": parameter_defaults["gamma"],
                    "validator": "finite scalar rest-channel bias",
                },
                **(
                    {
                        "r_d": {
                            "default": parameter_defaults["r_d"],
                            "validator": "probability",
                        }
                    }
                    if "r_d" in parameter_defaults
                    else {}
                ),
            },
            conservation_law=_law_for_kind("birth_death"),
            port_status="native",
            test_status="unit_tested",
            description=description,
        )

        def factory(
            parameters: Mapping[str, Any] | None = None,
            *,
            info=info,
            mode=mode,
            parameter_defaults=parameter_defaults,
        ) -> InteractionOperator:
            from .pipeline import NativeNoVEIdentityBirthDeathOperator

            merged_parameters = {key: value for key, value in parameter_defaults.items() if key != "capacity"}
            merged_parameters.update(dict(parameters or {}))
            return NativeNoVEIdentityBirthDeathOperator(
                info=info,
                mode=mode,
                parameters=merged_parameters,
            )

        register_plugin(info, factory)

    nove_ib_go_or_grow_info = PluginInfo(
        name="nove_ib.go_or_grow",
        operator_kind="birth_death",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "go_or_grow"),
        parameters={
            "capacity": {
                "default": 8,
                "validator": "positive integer node capacity",
            },
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar or one value per initial cell",
            },
            "theta": {
                "default": 0.5,
                "validator": "finite scalar or one value per initial cell",
            },
            "kappa_std": {
                "default": 0.2,
                "validator": "non-negative finite scalar",
            },
            "theta_std": {
                "default": 0.05,
                "validator": "non-negative finite scalar",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description=(
            "Identity no-volume-exclusion go-or-grow switching with inherited "
            "kappa and theta traits."
        ),
    )

    def nove_ib_go_or_grow_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityGoOrGrowOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.01,
            "kappa": 5.0,
            "theta": 0.5,
            "kappa_std": 0.2,
            "theta_std": 0.05,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEIdentityGoOrGrowOperator(
            nove_ib_go_or_grow_info,
            merged_parameters,
        )

    register_plugin(nove_ib_go_or_grow_info, nove_ib_go_or_grow_factory)

    nove_ib_go_or_grow_kappa_info = PluginInfo(
        name="nove_ib.go_or_grow_kappa",
        operator_kind="birth_death",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "go_or_grow_kappa"),
        parameters={
            "capacity": {
                "default": 8,
                "validator": "positive integer node capacity",
            },
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar or one value per initial cell",
            },
            "theta": {
                "default": 0.5,
                "validator": "finite scalar",
            },
            "kappa_std": {
                "default": 0.2,
                "validator": "non-negative finite scalar",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description=(
            "Identity no-volume-exclusion go-or-grow with neighbourhood-density "
            "switching and evolving kappa."
        ),
    )

    def nove_ib_go_or_grow_kappa_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityGoOrGrowKappaOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.01,
            "kappa": 5.0,
            "theta": 0.5,
            "kappa_std": 0.2,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEIdentityGoOrGrowKappaOperator(
            nove_ib_go_or_grow_kappa_info,
            merged_parameters,
        )

    register_plugin(nove_ib_go_or_grow_kappa_info, nove_ib_go_or_grow_kappa_factory)

    nove_ib_go_or_grow_kappa_chemo_info = PluginInfo(
        name="nove_ib.go_or_grow_kappa_chemo",
        operator_kind="birth_death",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "go_or_grow_kappa_chemo"),
        parameters={
            "capacity": {
                "default": 8,
                "validator": "positive integer node capacity",
            },
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar or one value per initial cell",
            },
            "theta": {
                "default": 0.5,
                "validator": "finite scalar",
            },
            "kappa_std": {
                "default": 0.2,
                "validator": "non-negative finite scalar",
            },
            "beta": {
                "default": 5.0,
                "validator": "finite scalar chemotactic sensitivity",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description=(
            "Identity no-volume-exclusion go-or-grow with neighbourhood-density "
            "switching, evolving kappa, and chemotactic migration."
        ),
    )

    def nove_ib_go_or_grow_kappa_chemo_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityGoOrGrowKappaChemoOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.01,
            "kappa": 5.0,
            "theta": 0.5,
            "kappa_std": 0.2,
            "beta": 5.0,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEIdentityGoOrGrowKappaChemoOperator(
            nove_ib_go_or_grow_kappa_chemo_info,
            merged_parameters,
        )

    register_plugin(
        nove_ib_go_or_grow_kappa_chemo_info,
        nove_ib_go_or_grow_kappa_chemo_factory,
    )

    nove_ib_go_or_grow_glioblastoma_info = PluginInfo(
        name="nove_ib.go_or_grow_glioblastoma",
        operator_kind="birth_death",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "go_or_grow_glioblastoma"),
        parameters={
            "capacity": {
                "default": 8,
                "validator": "positive integer node capacity",
            },
            "r_b": {
                "default": 0.2,
                "validator": "probability initial family birth rate",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "r_m": {
                "default": 0.001,
                "validator": "probability",
            },
            "fitness_increase": {
                "default": 1.1,
                "validator": "positive finite scalar driver mutation multiplier",
            },
            "theta": {
                "default": 0.5,
                "validator": "finite scalar",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar initial family switching slope",
            },
            "kappa_std": {
                "default": 0.2,
                "validator": "non-negative finite scalar",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        mutates_families=True,
        description=(
            "Identity no-volume-exclusion glioblastoma go-or-grow with family-level "
            "driver mutations and inherited switching sensitivity."
        ),
    )

    def nove_ib_go_or_grow_glioblastoma_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityGoOrGrowGlioblastomaOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.01,
            "r_m": 0.001,
            "fitness_increase": 1.1,
            "theta": 0.5,
            "kappa": 5.0,
            "kappa_std": 0.2,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEIdentityGoOrGrowGlioblastomaOperator(
            nove_ib_go_or_grow_glioblastoma_info,
            merged_parameters,
        )

    register_plugin(
        nove_ib_go_or_grow_glioblastoma_info,
        nove_ib_go_or_grow_glioblastoma_factory,
    )

    nove_ib_evo_steric_info = PluginInfo(
        name="nove_ib.evo_steric",
        operator_kind="birth_death",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "evo_steric"),
        parameters={
            "capacity": {
                "default": 512,
                "validator": "positive integer deme capacity",
            },
            "r_b": {
                "default": 0.1,
                "validator": "probability",
            },
            "r_m": {
                "default": 0.001,
                "validator": "probability",
            },
            "r_d": {
                "default": "0.98 * r_b",
                "validator": "probability",
            },
            "alpha": {
                "default": 2.0,
                "validator": "finite scalar steric interaction strength",
            },
            "gamma": {
                "default": 3.0,
                "validator": "finite scalar rest-channel bias",
            },
            "fitness_increase": {
                "default": 1.1,
                "validator": "positive finite scalar driver mutation multiplier",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        mutates_families=True,
        description=(
            "Identity no-volume-exclusion steric evolution with family-level "
            "driver mutations and steric channel redistribution."
        ),
    )

    def nove_ib_evo_steric_factory(
        parameters: Mapping[str, Any] | None = None,
    ) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityEvoStericOperator

        merged_parameters = {
            "r_b": 0.1,
            "r_m": 0.001,
            "alpha": 2.0,
            "gamma": 3.0,
            "fitness_increase": 1.1,
        }
        merged_parameters.update(dict(parameters or {}))
        if "r_d" not in merged_parameters:
            merged_parameters["r_d"] = 0.98 * merged_parameters["r_b"]
        return NativeNoVEIdentityEvoStericOperator(
            nove_ib_evo_steric_info,
            merged_parameters,
        )

    register_plugin(nove_ib_evo_steric_info, nove_ib_evo_steric_factory)

    cancer_dfe_info = PluginInfo(
        name="nove_ib.birthdeath_cancerdfe",
        operator_kind="birth_death",
        backend_families=("nove_ib",),
        legacy_source=_legacy_source("lgca.nove_ib_interactions", "birthdeath_cancerdfe"),
        parameters={
            "capacity": {
                "default": 8,
                "validator": "positive integer node capacity",
            },
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.02,
                "validator": "probability",
            },
            "p_d": {
                "default": 1.4e-5,
                "validator": "probability of driver mutation",
            },
            "p_p": {
                "default": 0.1,
                "validator": "probability of passenger mutation",
            },
            "s_d": {
                "default": "0.1 * r_b",
                "validator": "non-negative finite scalar driver scale",
            },
            "s_p": {
                "default": "0.001 * r_b",
                "validator": "non-negative finite scalar passenger scale",
            },
            "a_max": {
                "default": 1.0,
                "validator": "positive finite scalar birth-rate cap",
            },
            "gamma": {
                "default": 0.0,
                "validator": "finite scalar rest-channel bias",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Identity no-volume-exclusion birth-death with driver/passenger DFE mutations.",
    )

    def cancer_dfe_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeNoVEIdentityCancerDFEBirthDeathOperator

        merged_parameters = {
            "r_b": 0.2,
            "r_d": 0.02,
            "p_d": 1.4e-5,
            "p_p": 0.1,
            "a_max": 1.0,
            "gamma": 0.0,
        }
        merged_parameters.update(dict(parameters or {}))
        return NativeNoVEIdentityCancerDFEBirthDeathOperator(cancer_dfe_info, merged_parameters)

    register_plugin(cancer_dfe_info, cancer_dfe_factory)

    go_or_rest_info = PluginInfo(
        name="classical.go_or_rest",
        operator_kind="reorientation",
        backend_families=("classical",),
        legacy_source=_legacy_source("lgca.interactions", "go_or_rest"),
        parameters={
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar switching steepness",
            },
            "theta": {
                "default": 0.75,
                "validator": "finite scalar switching threshold",
            },
        },
        conservation_law=_law_for_kind("reorientation"),
        port_status="native",
        test_status="unit_tested",
        description=("Cells move between velocity and rest channels with a density-dependent "
                     "probability of resting; with volume exclusion."),
    )

    def go_or_rest_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeClassicalGoOrRestOperator

        merged_parameters = {"kappa": 5.0, "theta": 0.75}
        merged_parameters.update(dict(parameters or {}))
        return NativeClassicalGoOrRestOperator(go_or_rest_info, merged_parameters)

    register_plugin(go_or_rest_info, go_or_rest_factory)

    go_or_grow_info = PluginInfo(
        name="classical.go_or_grow",
        operator_kind="birth_death",
        backend_families=("classical",),
        legacy_source=_legacy_source("lgca.interactions", "go_or_grow"),
        parameters={
            "r_b": {
                "default": 0.2,
                "validator": "probability",
            },
            "r_d": {
                "default": 0.01,
                "validator": "probability",
            },
            "kappa": {
                "default": 5.0,
                "validator": "finite scalar switching steepness",
            },
            "theta": {
                "default": 0.75,
                "validator": "finite scalar switching threshold",
            },
        },
        conservation_law=_law_for_kind("birth_death"),
        port_status="native",
        test_status="unit_tested",
        description="Classical go-or-grow switch with death and rest-cell birth.",
    )

    def go_or_grow_factory(parameters: Mapping[str, Any] | None = None) -> InteractionOperator:
        from .pipeline import NativeClassicalGoOrGrowOperator

        merged_parameters = {"r_b": 0.2, "r_d": 0.01, "kappa": 5.0, "theta": 0.75}
        merged_parameters.update(dict(parameters or {}))
        return NativeClassicalGoOrGrowOperator(go_or_grow_info, merged_parameters)

    register_plugin(go_or_grow_info, go_or_grow_factory)

    for name, mode, legacy_function, aliases, description in (
        (
            "classical.alignment",
            "alignment",
            "alignment",
            (),
            "Neighbor-flux alignment for volume-exclusion classical LGCA.",
        ),
        (
            "classical.persistent_walk",
            "persistent_walk",
            "persistent_walk",
            (),
            "Local-flux persistent motion for volume-exclusion classical LGCA.",
        ),
        (
            "classical.aggregation",
            "aggregation",
            "aggregation",
            (),
            "Density-gradient aggregation for volume-exclusion classical LGCA.",
        ),
        (
            "classical.chemotaxis",
            "chemotaxis",
            "chemotaxis",
            (),
            "External-gradient chemotaxis for volume-exclusion classical LGCA.",
        ),
    ):
        parameter_contract = {
            "beta": {
                "default": 2.0,
                "validator": "finite scalar sensitivity",
            },
        }
        if mode == "chemotaxis":
            parameter_contract["gradient"] = {
                "default": None,
                "type_label": "array",
                "description": "Optional precomputed spatial gradient field.",
            }
        info = PluginInfo(
            name=name,
            aliases=aliases,
            operator_kind="reorientation",
            backend_families=("classical",),
            legacy_source=_legacy_source("lgca.interactions", legacy_function),
            parameters=parameter_contract,
            conservation_law=_law_for_kind("reorientation"),
            port_status="native",
            test_status="unit_tested",
            description=description,
        )

        def factory(
            parameters: Mapping[str, Any] | None = None,
            *,
            info=info,
            mode=mode,
        ) -> InteractionOperator:
            from .pipeline import NativeClassicalReorientationOperator

            merged_parameters = {"beta": 2.0}
            merged_parameters.update(dict(parameters or {}))
            return NativeClassicalReorientationOperator(
                info=info,
                mode=mode,
                parameters=merged_parameters,
            )

        register_plugin(info, factory)

    for name, mode, description in (
        (
            "classical.nematic",
            "nematic",
            "Neighbor nematic tensor alignment for volume-exclusion classical LGCA.",
        ),
        (
            "classical.contact_guidance",
            "contact_guidance",
            "Static director-field contact guidance for volume-exclusion classical LGCA.",
        ),
    ):
        info = PluginInfo(
            name=name,
            operator_kind="reorientation",
            backend_families=("classical",),
            legacy_source=_legacy_source("lgca.interactions", name.split(".", 1)[1]),
            parameters={
                "beta": {
                    "default": 2.0,
                    "validator": "finite scalar sensitivity",
                },
            },
            conservation_law=_law_for_kind("reorientation"),
            port_status="native",
            test_status="unit_tested",
            description=description,
        )

        def factory(
            parameters: Mapping[str, Any] | None = None,
            *,
            info=info,
            mode=mode,
        ) -> InteractionOperator:
            from .pipeline import NativeClassicalTensorReorientationOperator

            merged_parameters = {"beta": 2.0}
            merged_parameters.update(dict(parameters or {}))
            return NativeClassicalTensorReorientationOperator(
                info=info,
                mode=mode,
                parameters=merged_parameters,
            )

        register_plugin(info, factory)



def _register_example_plugins() -> None:
    custom_rest_or_align_info = PluginInfo(
        name="custom.rest_or_align",
        operator_kind="reorientation",
        backend_families=("classical",),
        legacy_source=_legacy_source(
            "lgca.examples.custom_rest_or_align", "rest_or_align"
        ),
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
        from .examples.custom_rest_or_align import NativeRestOrAlignOperator

        return NativeRestOrAlignOperator(custom_rest_or_align_info, parameters)

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
    "capacity": "Carrying capacity: the number of cells per node at which birth stops. "
                "Defaults to state.capacity.",
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
    "gradient": "Gradient of the attractant signal at every node; defaults to a built-in "
                "linear signal.",
    "rates": "Matrix of switch probabilities per time step: entry [a][b] is the probability "
             "that a cell of species a becomes species b. Off-diagonal row sums must be <= 1.",
}
_PLUGIN_PARAMETER_MEANINGS = {
    ("classical.birth", "r_b"): "Birth probability: a free channel of a node is filled with "
                                "probability r_b * n / K, where n is the number of cells at the "
                                "node and K the number of channels.",
    ("classical.birthdeath", "r_b"): "Birth probability: a free channel of a node is filled with "
                                     "probability r_b * n / K, where n is the number of cells at "
                                     "the node and K the number of channels.",
    ("classical.go_or_grow", "r_b"): "Probability per time step that a resting cell divides into "
                                     "a free rest channel.",
    ("nove.go_or_grow", "r_b"): "Division probability of a resting cell per time step at low "
                                "density; the probability is r_b * (1 - n / capacity) for n cells "
                                "at the node.",
    ("ib.go_or_grow", "r_b"): "Probability per time step that a resting cell divides.",
    ("nove_ib.evo_steric", "alpha"): "Strength with which cells avoid moving towards crowded "
                                     "neighbouring nodes.",
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

_register_native_plugins()
_register_example_plugins()
_describe_builtin_plugins()
