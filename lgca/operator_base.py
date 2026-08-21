"""Stable public contracts for interaction operators.

This module contains only extension-facing data and lifecycle types.  Numerical
implementations and the global registry live in separate modules so third-party
operators do not need to import the built-in plugin catalogue.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping


__all__ = [
    "BirthDeathOperator",
    "ConservationLaw",
    "InteractionOperator",
    "LegacyInteractionOperator",
    "ParameterSpec",
    "PhenotypeSwitchOperator",
    "PluginInfo",
    "ReorientationOperator",
    "ReorientationTerm",
]


@dataclass(frozen=True)
class ParameterSpec:
    """Machine-readable contract for a plugin parameter."""

    default: Any = None
    required: bool = False
    type_label: str | None = None
    shape: Any = None
    allowed_values: tuple[Any, ...] | None = None
    dependencies: tuple[str, ...] = ()
    validator: str | None = None
    description: str = ""

    @classmethod
    def from_metadata(cls, metadata: Any) -> "ParameterSpec":
        if isinstance(metadata, cls):
            return metadata
        if isinstance(metadata, Mapping):
            validator = metadata.get("validator")
            return cls(
                default=metadata.get("default"),
                required=bool(metadata.get("required", "default" not in metadata)),
                type_label=metadata.get("type_label") or _type_label_from_validator(validator),
                shape=metadata.get("shape"),
                allowed_values=_tuple_or_none(metadata.get("allowed_values")),
                dependencies=tuple(metadata.get("dependencies", ())),
                validator=validator,
                description=metadata.get("description", ""),
            )
        return cls(default=metadata, required=False)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "default": self.default,
            "required": self.required,
            "type_label": self.type_label,
            "shape": self.shape,
            "allowed_values": self.allowed_values,
            "dependencies": self.dependencies,
            "validator": self.validator,
            "description": self.description,
        }
        return {key: value for key, value in data.items() if value not in (None, "", ())}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _tuple_or_none(value):
    if value is None:
        return None
    return tuple(value)


def _type_label_from_validator(validator: str | None) -> str | None:
    if not validator:
        return None
    text = validator.lower()
    if "probability" in text:
        return "probability"
    if "positive integer" in text:
        return "positive integer"
    if "non-negative integer" in text:
        return "non-negative integer"
    if "finite scalar" in text and " or " not in text:
        return "finite scalar"
    return None


@dataclass(frozen=True)
class ConservationLaw:
    """Conservation contract advertised by an interaction plugin."""

    conserves_total_particles: bool | None
    conserves_phenotype_particles: bool | None
    conserves_momentum: bool | None
    changes: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = []
        if self.conserves_total_particles is True:
            parts.append("total mass")
        elif self.conserves_total_particles is False:
            parts.append("changes total mass")
        if self.conserves_phenotype_particles is True:
            parts.append("phenotype mass")
        elif self.conserves_phenotype_particles is False:
            parts.append("changes phenotype mass")
        if self.conserves_momentum is True:
            parts.append("momentum")
        elif self.conserves_momentum is False:
            parts.append("changes momentum")
        parts.extend(self.changes)
        return ", ".join(parts)


@dataclass(frozen=True)
class PluginInfo:
    """Public metadata for a registered plugin."""

    name: str
    operator_kind: str
    backend_families: tuple[str, ...]
    legacy_source: str | None = None
    aliases: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    conservation_law: ConservationLaw = field(
        default_factory=lambda: ConservationLaw(None, None, None)
    )
    port_status: str = "legacy_wrapper"
    test_status: str = "unverified"
    mutates_families: bool = False
    description: str = ""

    @property
    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """Return parameters normalized to :class:`ParameterSpec` objects."""

        return {
            name: ParameterSpec.from_metadata(metadata)
            for name, metadata in self.parameters.items()
        }


class InteractionOperator:
    """Base class for an explicitly registered interaction-phase operator."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        self.info = info
        self.parameters = dict(parameters or {})

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def operator_kind(self) -> str:
        return self.info.operator_kind

    @property
    def conservation_law(self) -> ConservationLaw:
        return self.info.conservation_law

    def validate(self, context) -> None:
        """Validate the operator against a model context."""

    def validate_parameter_contracts(self, context) -> None:
        """Validate user-supplied parameters against plugin metadata."""

        from .plugins import validate_plugin_parameters

        validate_plugin_parameters(self.info, self.parameters, context=context)

    def setup(self, context) -> None:
        """Prepare cached state before the first timestep."""

    def apply(self, context, step: int) -> None:
        """Apply the stochastic interaction sub-step."""

        raise NotImplementedError

    def dependencies(self) -> set[str]:
        """Return fields read by this operator."""

        return set()

    def outputs(self) -> set[str]:
        """Return fields written by this operator."""

        return {"nodes"}


class BirthDeathOperator(InteractionOperator):
    """Marker base for particle-number changing operators."""


class PhenotypeSwitchOperator(InteractionOperator):
    """Marker base for phenotype/species-changing operators."""


class ReorientationOperator(InteractionOperator):
    """Marker base for mass-preserving reorientation operators."""


class ReorientationTerm(InteractionOperator):
    """Marker base for terms combined by a reorientation sampler."""


class LegacyInteractionOperator(InteractionOperator):
    """Adapter that runs a legacy ``set_interaction`` interaction as a plugin."""

    def __init__(
        self,
        info: PluginInfo,
        legacy_interaction: str,
        parameters: Mapping[str, Any] | None = None,
        function_module: str | None = None,
        function_name: str | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.legacy_interaction = legacy_interaction
        self.function_module = function_module
        self.function_name = function_name
        self._interaction = None
        self._interaction_params: dict[str, Any] = {}

    def setup(self, context) -> None:
        lgca = context.lgca
        previous_interaction = getattr(lgca, "interaction", None)
        previous_params = dict(getattr(lgca, "interaction_params", {}))
        lgca.interaction_params = {}
        lgca.set_interaction(interaction=self.legacy_interaction, **self.parameters)
        if self.function_module is not None and self.function_name is not None:
            module = importlib.import_module(self.function_module)
            self._interaction = getattr(module, self.function_name)
        else:
            self._interaction = lgca.interaction
        self._interaction_params = dict(lgca.interaction_params)
        self._interaction_params.update(self.parameters)
        if previous_interaction is not None:
            lgca.interaction = previous_interaction
        lgca.interaction_params = previous_params

    def apply(self, context, step: int) -> None:
        if self._interaction is None:
            self.setup(context)
        lgca = context.lgca
        previous_interaction = getattr(lgca, "interaction", None)
        previous_params = dict(getattr(lgca, "interaction_params", {}))
        lgca.interaction = self._interaction
        lgca.interaction_params = dict(self._interaction_params)
        try:
            self._interaction(lgca)
        finally:
            if previous_interaction is not None:
                lgca.interaction = previous_interaction
            lgca.interaction_params = previous_params
