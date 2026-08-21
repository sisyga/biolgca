"""Composable interaction pipeline for LGCA models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .plugins import (
    BirthDeathOperator,
    ConservationLaw,
    InteractionOperator,
    PhenotypeSwitchOperator,
    PluginInfo,
    ReorientationOperator,
    create_plugin,
)


@dataclass(frozen=True)
class BirthDeathSpec:
    """Specification for a particle-number changing operator."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhenotypeSwitchSpec:
    """Specification for a phenotype/species-changing operator."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReorientationTermSpec:
    """Weighted term inside a reorientation operator."""

    name: str
    beta: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    species: int | None = None


@dataclass(frozen=True)
class ReorientationSpec:
    """Specification for a mass-preserving reorientation sampler."""

    terms: Sequence[ReorientationTermSpec] = field(default_factory=tuple)
    sampler: str = "boltzmann"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionPipelineSpec:
    """Ordered interaction phase followed by deterministic propagation."""

    operators: Sequence[Any] = field(default_factory=tuple)
    propagation: str | bool = "default"
    allow_custom_order: bool = False


@dataclass
class CompiledPipeline:
    """Executable interaction pipeline."""

    operators: list[InteractionOperator]
    propagation: str | bool = "default"

    @property
    def operator_names(self) -> list[str]:
        return [operator.name for operator in self.operators]

    @property
    def reorientation_term_names(self) -> list[str]:
        names = []
        for operator in self.operators:
            names.extend(getattr(operator, "term_names", []))
        return names

    def setup(self, context) -> None:
        for operator in self.operators:
            operator.validate_parameter_contracts(context)
            operator.validate(context)
            operator.setup(context)

    def describe_schedule(self) -> str:
        parts = []
        for operator in self.operators:
            law = operator.conservation_law.describe()
            suffix = f" ({operator.operator_kind}"
            if law:
                suffix += f"; {law}"
            suffix += f"; backend={','.join(operator.info.backend_families)}"
            if operator.info.legacy_source:
                suffix += f"; source={operator.info.legacy_source}"
            deps = ",".join(sorted(operator.dependencies())) or "-"
            outputs = ",".join(sorted(operator.outputs())) or "-"
            suffix += f"; inputs={deps}; outputs={outputs}"
            suffix += f"; status={operator.info.port_status}"
            suffix += ")"
            parts.append(f"{operator.name}{suffix}")
        if self.propagation not in (False, None, "none", "disabled"):
            parts.append("Propagation (deterministic)")
        return " -> ".join(parts)

    def execute_step(self, context, step: int, timing: list[dict[str, Any]] | None = None) -> None:
        lgca = context.lgca
        for operator in self.operators:
            if "boundary_nodes" in operator.dependencies():
                lgca.apply_boundaries()
                lgca.update_dynamic_fields()
            start = time.perf_counter()
            operator.apply(context, step)
            if "nodes" in operator.outputs():
                lgca.update_dynamic_fields()
            if timing is not None:
                timing.append(
                    {
                        "step": step,
                        "name": operator.name,
                        "kind": operator.operator_kind,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
        lgca.apply_boundaries()
        if self.propagation not in (False, None, "none", "disabled"):
            start = time.perf_counter()
            lgca.propagation()
            if timing is not None:
                timing.append(
                    {
                        "step": step,
                        "name": "Propagation",
                        "kind": "propagation",
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
            lgca.apply_boundaries()
        lgca.update_dynamic_fields()


class NativeOnlyPropagationOperator(InteractionOperator):
    """Native no-op marker for propagation-only legacy compatibility."""

    def apply(self, context, step: int) -> None:
        pass

    def outputs(self) -> set[str]:
        return set()


def compile_pipeline(spec: InteractionPipelineSpec | None, context) -> CompiledPipeline:
    """Compile a pipeline spec into executable operators."""

    spec = spec or InteractionPipelineSpec()
    operators = []
    for index, operator_spec in enumerate(spec.operators):
        try:
            operators.append(_compile_operator(operator_spec))
        except KeyError as exc:
            name = _operator_name(operator_spec)
            raise ValueError(f"dynamics.operators[{index}] unknown operator {name!r}") from exc
        except ValueError as exc:
            message = str(exc)
            separator = "" if message.startswith(".") else " "
            raise ValueError(f"dynamics.operators[{index}]{separator}{message}") from exc
    if not spec.allow_custom_order:
        _validate_operator_order(operators)
    pipeline = CompiledPipeline(operators=operators, propagation=spec.propagation)
    pipeline.setup(context)
    return pipeline


def _compile_operator(spec) -> InteractionOperator:
    if isinstance(spec, InteractionOperator):
        return spec
    if isinstance(spec, ReorientationSpec):
        return BoltzmannReorientationOperator(spec)
    if isinstance(spec, PhenotypeSwitchSpec) and spec.name == "phenotype_switch":
        return NativePhenotypeSwitchOperator(spec.parameters)
    if isinstance(spec, (BirthDeathSpec, PhenotypeSwitchSpec)):
        return create_plugin(spec.name, spec.parameters)
    if isinstance(spec, Mapping):
        if "name" not in spec:
            raise ValueError("requires a 'name'")
        return create_plugin(spec["name"], spec.get("parameters"))
    name = getattr(spec, "name", None)
    parameters = getattr(spec, "parameters", None)
    if name is None:
        raise ValueError("requires a 'name'")
    return create_plugin(name, parameters)


def _operator_name(spec) -> str:
    if isinstance(spec, Mapping):
        return str(spec.get("name", "<missing>"))
    return str(getattr(spec, "name", "<missing>"))


def _validate_operator_order(operators: Sequence[InteractionOperator]) -> None:
    order = {
        "birth_death": 0,
        "phenotype_switch": 1,
        "reorientation": 2,
        "propagation": 3,
    }
    highest = -1
    highest_kind = None
    for index, operator in enumerate(operators):
        current = order.get(operator.operator_kind, highest)
        if current < highest:
            raise ValueError(
                f"dynamics.operators[{index}] invalid operator order: "
                f"{operator.operator_kind} follows {highest_kind}"
            )
        highest = max(highest, current)
        highest_kind = operator.operator_kind


def _softmax_last_axis(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


class _ReorientationTerm:
    def __init__(self, spec: ReorientationTermSpec):
        self.name = spec.name
        self.beta = float(spec.beta)
        self.parameters = dict(spec.parameters)
        self.species = spec.species

    def validate(self, context) -> None:
        if self.species is not None and self.species >= context.spec.state.n_species:
            raise ValueError(f"species index {self.species} exceeds state.n_species")

    def score(self, candidates, node, lgca, coord):
        return np.zeros(candidates.shape[0], dtype=float)

    def dependencies(self) -> set[str]:
        return set()


class _UniformTerm(_ReorientationTerm):
    pass


class _RestingBiasTerm(_ReorientationTerm):
    def score(self, candidates, node, lgca, coord):
        return candidates[:, lgca.velocitychannels :].sum(axis=1)


class _ChemotaxisTerm(_ReorientationTerm):
    def __init__(self, spec: ReorientationTermSpec):
        super().__init__(spec)
        self.field_name = self.parameters.get("field")
        self.gradient = None

    def validate(self, context) -> None:
        super().validate(context)
        if not self.field_name:
            raise ValueError("chemotaxis requires parameter 'field'")
        if self.field_name not in context.fields:
            raise ValueError(f"state.fields.{self.field_name} is required by chemotaxis")
        field = np.asarray(context.fields[self.field_name], dtype=float)
        expected = tuple(context.lgca.dims)
        if field.shape != expected:
            raise ValueError(
                f"state.fields.{self.field_name} must have shape {expected}, got {field.shape}"
            )
        gradients = np.gradient(field)
        if isinstance(gradients, np.ndarray):
            gradients = [gradients]
        self.gradient = np.stack(gradients, axis=-1)

    def score(self, candidates, node, lgca, coord):
        if self.gradient is None:
            return np.zeros(candidates.shape[0], dtype=float)
        spatial = tuple(index - lgca.r_int for index in coord)
        gradient = self.gradient[spatial]
        flux = candidates[:, : lgca.velocitychannels] @ lgca.c.T
        return flux @ gradient

    def dependencies(self) -> set[str]:
        return set() if not self.field_name else {self.field_name}


class _NematicAlignmentTerm(_ReorientationTerm):
    def score(self, candidates, node, lgca, coord):
        source_nodes = getattr(lgca, "_reorientation_source_nodes", lgca.nodes)
        if getattr(lgca, "n_species", 1) > 1 and source_nodes.ndim == len(lgca.dims) + 2:
            source_channels = source_nodes.sum(axis=-2)
        else:
            source_channels = source_nodes
        neighbor_channels = lgca.nb_sum(source_channels[..., : lgca.velocitychannels])[coord]
        dot_sq = (lgca.c.T @ lgca.c) ** 2
        return candidates[:, : lgca.velocitychannels] @ dot_sq @ neighbor_channels

    def dependencies(self) -> set[str]:
        return {"boundary_nodes"}


class _PersistentWalkTerm(_ReorientationTerm):
    def score(self, candidates, node, lgca, coord):
        source_nodes = getattr(lgca, "_reorientation_source_nodes", lgca.nodes)
        if getattr(lgca, "n_species", 1) > 1 and source_nodes.ndim == len(lgca.dims) + 2:
            source_channels = source_nodes.sum(axis=-2)
        else:
            source_channels = source_nodes
        local_flux = source_channels[coord][..., : lgca.velocitychannels] @ lgca.c.T
        candidate_flux = candidates[:, : lgca.velocitychannels] @ lgca.c.T
        return candidate_flux @ local_flux


class _AggregationTerm(_ReorientationTerm):
    def score(self, candidates, node, lgca, coord):
        source_nodes = getattr(lgca, "_reorientation_source_nodes", lgca.nodes)
        if getattr(lgca, "n_species", 1) > 1 and source_nodes.ndim == len(lgca.dims) + 2:
            source_channels = source_nodes.sum(axis=-2)
        else:
            source_channels = source_nodes
        density = source_channels.sum(axis=-1)
        gradient = lgca.gradient(density)[coord]
        candidate_flux = candidates[:, : lgca.velocitychannels] @ lgca.c.T
        return candidate_flux @ gradient

    def dependencies(self) -> set[str]:
        return {"boundary_nodes", "cell_density"}


class _ContactGuidanceTerm(_ReorientationTerm):
    def __init__(self, spec: ReorientationTermSpec):
        super().__init__(spec)
        self.field_name = self.parameters.get("field", "director")
        self.director = None

    def validate(self, context) -> None:
        super().validate(context)
        if not self.field_name:
            raise ValueError("contact_guidance requires a non-empty parameter 'field'")
        if self.field_name not in context.fields:
            raise ValueError(f"state.fields.{self.field_name} is required by contact_guidance")
        field = np.asarray(context.fields[self.field_name], dtype=float)
        expected = tuple(context.lgca.dims) + (context.lgca.c.shape[0],)
        if field.shape != expected:
            raise ValueError(
                f"state.fields.{self.field_name} must have shape {expected}, got {field.shape}"
            )
        norm = np.linalg.norm(field, axis=-1, keepdims=True)
        self.director = np.divide(field, norm, out=np.zeros_like(field), where=norm > 0)

    def score(self, candidates, node, lgca, coord):
        if self.director is None:
            return np.zeros(candidates.shape[0], dtype=float)
        spatial = tuple(index - lgca.r_int for index in coord)
        director = self.director[spatial]
        if not np.any(director):
            return np.zeros(candidates.shape[0], dtype=float)
        channel_alignment = lgca.c.T @ director
        return candidates[:, : lgca.velocitychannels] @ (channel_alignment ** 2)

    def dependencies(self) -> set[str]:
        return {self.field_name} if self.field_name else set()


_REORIENTATION_TERMS = {
    "aggregation": _AggregationTerm,
    "alignment": _NematicAlignmentTerm,
    "chemotaxis": _ChemotaxisTerm,
    "contact_guidance": _ContactGuidanceTerm,
    "nematic": _NematicAlignmentTerm,
    "nematic_alignment": _NematicAlignmentTerm,
    "persistent_motion": _PersistentWalkTerm,
    "persistent_walk": _PersistentWalkTerm,
    "random_walk": _UniformTerm,
    "uniform": _UniformTerm,
    "resting_bias": _RestingBiasTerm,
}


class BoltzmannReorientationOperator(ReorientationOperator):
    """Native mass-preserving reorientation sampler."""

    def __init__(self, spec: ReorientationSpec):
        info = PluginInfo(
            name="reorientation.boltzmann",
            operator_kind="reorientation",
            backend_families=("classical", "multispecies"),
            conservation_law=ConservationLaw(True, True, False, ("channel occupancy",)),
            port_status="native",
            description="Boltzmann sampler over channel configurations.",
        )
        super().__init__(info=info, parameters=spec.parameters)
        self.sampler = spec.sampler
        self.terms = self._compile_terms(spec.terms)

    @property
    def term_names(self) -> list[str]:
        return [term.name for term in self.terms]

    def validate(self, context) -> None:
        if self.sampler != "boltzmann":
            raise ValueError(f"unsupported reorientation sampler {self.sampler!r}")
        if context.spec.state.identity_based:
            raise ValueError("native reorientation does not yet support identity-based states")
        if not context.spec.state.volume_exclusion:
            raise ValueError("native reorientation currently requires volume exclusion")
        for term in self.terms:
            term.validate(context)

    def setup(self, context) -> None:
        lgca = context.lgca
        if not hasattr(lgca, "permutations") and not hasattr(lgca, "_permutation_cache"):
            lgca.calc_permutations()

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        lgca._reorientation_source_nodes = lgca.nodes.copy()
        try:
            for spatial in np.ndindex(lgca.dims):
                coord = tuple(index + lgca.r_int for index in spatial)
                node = lgca._reorientation_source_nodes[coord]
                if getattr(lgca, "n_species", 1) > 1:
                    lgca.nodes[coord] = self._sample_multispecies_node(node, lgca, coord)
                else:
                    lgca.nodes[coord] = self._sample_node(node, lgca, coord, species=None)
        finally:
            del lgca._reorientation_source_nodes

    def dependencies(self) -> set[str]:
        deps = set()
        for term in self.terms:
            deps.update(term.dependencies())
        return deps

    @staticmethod
    def _compile_terms(term_specs: Sequence[ReorientationTermSpec]) -> list[_ReorientationTerm]:
        terms = []
        for index, term_spec in enumerate(term_specs):
            try:
                term_cls = _REORIENTATION_TERMS[term_spec.name]
            except KeyError as exc:
                raise ValueError(f".terms[{index}] unknown reorientation term {term_spec.name!r}") from exc
            terms.append(term_cls(term_spec))
        if not terms:
            terms.append(_UniformTerm(ReorientationTermSpec(name="random_walk")))
        return terms

    def _sample_multispecies_node(self, node, lgca, coord):
        new_node = np.zeros_like(node)
        for species in range(node.shape[0]):
            new_node[species] = self._sample_node(node[species], lgca, coord, species=species)
        return new_node

    def _sample_node(self, node, lgca, coord, species: int | None):
        n_particles = int(node.sum())
        if n_particles == 0:
            return np.zeros_like(node)
        if n_particles > lgca.K:
            raise ValueError("native reorientation requires at most one particle per channel")
        candidates = lgca.get_permutations(n_particles)
        scores = np.zeros(candidates.shape[0], dtype=float)
        for term in self.terms:
            if term.species is None or term.species == species:
                scores += term.beta * term.score(candidates, node, lgca, coord)
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        choice = lgca.rng.choice(candidates.shape[0], p=weights)
        return candidates[choice].astype(node.dtype)


class NativeClassicalRandomWalkOperator(ReorientationOperator):
    """Native volume-exclusion random walk matching the legacy permutation rule."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")

    def apply(self, context, step: int) -> None:
        context.lgca.nodes = context.lgca.rng.permuted(context.lgca.nodes, axis=-1)


class NativeIdentityRandomWalkOperator(ReorientationOperator):
    """Native identity-based volume-exclusion random walk."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")

    def apply(self, context, step: int) -> None:
        context.lgca.nodes = context.lgca.rng.permuted(context.lgca.nodes, axis=-1)


class NativeClassicalExcitableMediumOperator(BirthDeathOperator):
    """Native classical excitable-medium reaction operator."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.beta = 0.05
        self.alpha = 1.0
        self.repetitions = 50

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if context.lgca.restchannels < 1:
            raise ValueError(f"{self.name} requires at least one rest channel")
        self.beta = float(self.parameters.get("beta", 0.05))
        if not np.isfinite(self.beta):
            raise ValueError("beta must be a finite scalar")
        self.alpha = float(self.parameters.get("alpha", 1.0))
        if not np.isfinite(self.alpha) or self.alpha == 0.0:
            raise ValueError("alpha must be a non-zero finite scalar")
        repetitions = self.parameters.get("N", 50)
        if isinstance(repetitions, bool) or int(repetitions) != repetitions or repetitions < 0:
            raise ValueError("N must be a non-negative integer")
        self.repetitions = int(repetitions)

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        n_x = lgca.nodes[..., : lgca.velocitychannels].sum(-1)
        n_y = lgca.nodes[..., lgca.velocitychannels :].sum(-1)
        rho_x = n_x / lgca.velocitychannels
        rho_y = n_y / lgca.restchannels
        p_xp = rho_x**2 * (1 + (rho_y + self.beta) / self.alpha)
        p_xm = rho_x**3 + rho_x * (rho_y + self.beta) / self.alpha
        p_yp = rho_x
        p_ym = rho_y
        dn_y = (lgca.rng.random(n_y.shape) < p_yp).astype(np.int8)
        dn_y -= lgca.rng.random(n_y.shape) < p_ym
        for _ in range(self.repetitions):
            dn_x = (lgca.rng.random(n_x.shape) < p_xp).astype(np.int8)
            dn_x -= lgca.rng.random(n_x.shape) < p_xm
            n_x += dn_x
            rho_x = n_x / lgca.velocitychannels
            p_xp = rho_x**2 * (1 + (rho_y + self.beta) / self.alpha)
            p_xm = rho_x**3 + rho_x * (rho_y + self.beta) / self.alpha

        n_y += dn_y

        newnodes = np.zeros_like(lgca.nodes)
        v_idx = np.arange(lgca.velocitychannels)
        r_idx = np.arange(lgca.restchannels)
        newnodes[..., : lgca.velocitychannels] = (v_idx < n_x[..., None]).astype(
            lgca.nodes.dtype
        )
        newnodes[..., lgca.velocitychannels :] = (r_idx < n_y[..., None]).astype(
            lgca.nodes.dtype
        )
        newnodes[..., : lgca.velocitychannels] = lgca.rng.permuted(
            newnodes[..., : lgca.velocitychannels],
            axis=-1,
        )
        lgca.nodes = newnodes


class NativeNoVEIdentityRandomWalkOperator(ReorientationOperator):
    """Native identity-based no-volume-exclusion random walk."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            cells = [cell for channel in node for cell in channel]
            channeldist = lgca.rng.multinomial(len(cells), [1.0 / lgca.K] * lgca.K).cumsum()
            lgca.rng.shuffle(cells)
            lgca.nodes[coord] = [cells[: channeldist[0]]] + [
                cells[start:stop] for start, stop in zip(channeldist[:-1], channeldist[1:])
            ]


class NativeNoVEIdentityBirthDeathOperator(BirthDeathOperator):
    """Native identity no-volume-exclusion logistic birth/death operator."""

    def __init__(
        self,
        info: PluginInfo,
        *,
        mode: str,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.mode = mode
        self.capacity = 8
        self.r_b = 0.2
        self.r_d = 0.02
        self.std = 0.01
        self.a_max = 1.0
        self.gamma = 0.0
        self.channel_weights = None

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.capacity = int(self.parameters.get("capacity", 8))
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        if self.mode == "birthdeath":
            self.r_d = NativeClassicalBirthOperator._probability(
                "r_d", self.parameters.get("r_d", 0.02)
            )
        self.std = float(self.parameters.get("std", 0.01))
        if not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("std must be a positive finite scalar")
        self.a_max = float(self.parameters.get("a_max", 1.0))
        if not np.isfinite(self.a_max) or self.a_max <= 0.0:
            raise ValueError("a_max must be a positive finite scalar")
        self.gamma = float(self.parameters.get("gamma", 0.0))
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite")

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.props.update(r_b=[self.r_b] * (int(lgca.maxlabel) + 1))
        z = lgca.velocitychannels + np.exp(self.gamma) * lgca.restchannels
        self.channel_weights = np.array(
            [1.0 / z] * lgca.velocitychannels
            + [np.exp(self.gamma) / z] * lgca.restchannels
        )
        lgca.channel_weights = self.channel_weights

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, _split_cells_into_channels, trunc_gauss

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            density = lgca.cell_density[coord]
            rho = density / self.capacity
            cells = _cells_from_node(lgca.nodes[coord])
            newcells = cells.copy()
            for cell in cells:
                if self.mode == "birthdeath" and lgca.rng.random() < self.r_d:
                    newcells.remove(cell)

                r_b = lgca.props["r_b"][cell]
                if lgca.rng.random() < r_b * (1 - rho):
                    lgca.maxlabel += 1
                    newcells.append(lgca.maxlabel)
                    lgca.props["r_b"].append(
                        float(trunc_gauss(0, self.a_max, r_b, sigma=self.std, rng=lgca.rng))
                    )

            channeldist = lgca.rng.multinomial(len(newcells), self.channel_weights).cumsum()
            lgca.rng.shuffle(newcells)
            lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)


class NativeNoVEIdentityCancerDFEBirthDeathOperator(NativeNoVEIdentityBirthDeathOperator):
    """Native no-volume-exclusion identity birth-death with DFE mutations."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, mode="birthdeath", parameters=parameters)
        self.p_d = 1.4e-5
        self.p_p = 0.1
        self.s_d = None
        self.s_p = None

    def validate(self, context) -> None:
        super().validate(context)
        self.p_d = NativeClassicalBirthOperator._probability(
            "p_d", self.parameters.get("p_d", 1.4e-5)
        )
        self.p_p = NativeClassicalBirthOperator._probability(
            "p_p", self.parameters.get("p_p", 0.1)
        )
        self.s_d = float(self.parameters.get("s_d", 0.1 * self.r_b))
        if not np.isfinite(self.s_d) or self.s_d < 0.0:
            raise ValueError("s_d must be a non-negative finite scalar")
        self.s_p = float(self.parameters.get("s_p", 0.001 * self.r_b))
        if not np.isfinite(self.s_p) or self.s_p < 0.0:
            raise ValueError("s_p must be a non-negative finite scalar")

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, _split_cells_into_channels

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            density = lgca.cell_density[coord]
            rho = density / self.capacity
            cells = _cells_from_node(lgca.nodes[coord])
            newcells = cells.copy()
            for cell in cells:
                if lgca.rng.random() < self.r_d:
                    newcells.remove(cell)

                r_b = lgca.props["r_b"][cell]
                if lgca.rng.random() < r_b * (1 - rho):
                    lgca.maxlabel += 1
                    newcells.append(lgca.maxlabel)
                    passenger = 0.0
                    driver = 0.0
                    if lgca.rng.random() < self.p_p:
                        passenger = float(lgca.rng.exponential(scale=self.s_p))
                    if lgca.rng.random() < self.p_d:
                        driver = float(lgca.rng.exponential(scale=self.s_d))
                    lgca.props["r_b"].append(min(r_b - passenger + driver, self.a_max))

            channeldist = lgca.rng.multinomial(len(newcells), self.channel_weights).cumsum()
            lgca.rng.shuffle(newcells)
            lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)


class NativeNoVEIdentityGoOrGrowOperator(BirthDeathOperator):
    """Native no-volume-exclusion identity go-or-grow operator."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.capacity = 8
        self.r_b = 0.2
        self.r_d = 0.01
        self.kappa = None
        self.theta = None
        self.kappa_std = 0.2
        self.theta_std = 0.05

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.capacity = int(self.parameters.get("capacity", 8))
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.01)
        )
        self.kappa_std = float(self.parameters.get("kappa_std", 0.2))
        if not np.isfinite(self.kappa_std) or self.kappa_std < 0.0:
            raise ValueError("kappa_std must be a non-negative finite scalar")
        self.theta_std = float(self.parameters.get("theta_std", 0.05))
        if not np.isfinite(self.theta_std) or self.theta_std < 0.0:
            raise ValueError("theta_std must be a non-negative finite scalar")
        self.kappa = self._label_property(
            context,
            "kappa",
            self.parameters.get("kappa", 5.0),
        )
        self.theta = self._label_property(
            context,
            "theta",
            self.parameters.get("theta", 0.5),
        )

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.props.update(kappa=np.asarray(self.kappa, dtype=float))
        lgca.props.update(theta=np.asarray(self.theta, dtype=float))

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, tanh_switch

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        new_kappa_chunks = []
        new_theta_chunks = []
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            density = lgca.cell_density[coord]
            rho = density / self.capacity
            cells = np.asarray(_cells_from_node(node), dtype=int)

            notkilled = lgca.rng.random(size=density) < 1.0 - self.r_d
            cells = cells[notkilled]
            if len(cells) == 0:
                lgca.nodes[coord] = [[] for _ in range(lgca.K)]
                continue

            kappas = lgca.props["kappa"][cells]
            thetas = lgca.props["theta"][cells]
            switch = lgca.rng.random(len(cells)) < tanh_switch(
                rho=rho,
                kappa=kappas,
                theta=thetas,
            )
            restcells = list(cells[switch])
            velcells = list(cells[~switch])

            rho = len(cells) / self.capacity
            n_prolif = lgca.rng.binomial(len(restcells), max(self.r_b * (1 - rho), 0))
            if n_prolif > 0:
                proliferating = lgca.rng.choice(
                    restcells,
                    size=n_prolif,
                    replace=False,
                    shuffle=False,
                )
                lgca.maxlabel += n_prolif
                new_cells = np.arange(lgca.maxlabel - n_prolif + 1, lgca.maxlabel + 1)
                new_kappa_chunks.append(
                    lgca.rng.normal(
                        loc=lgca.props["kappa"][proliferating],
                        scale=self.kappa_std,
                    )
                )
                new_theta_chunks.append(
                    lgca.rng.normal(
                        loc=lgca.props["theta"][proliferating],
                        scale=self.theta_std,
                    )
                )
                restcells.extend(list(new_cells))

            new_node = [[] for _ in range(lgca.velocitychannels)]
            new_node.append(restcells)
            for cell in velcells:
                new_node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

            lgca.nodes[coord] = new_node
        if new_kappa_chunks:
            lgca.props["kappa"] = np.concatenate((lgca.props["kappa"], *new_kappa_chunks))
            lgca.props["theta"] = np.concatenate((lgca.props["theta"], *new_theta_chunks))

    @staticmethod
    def _label_property(context, name: str, value: Any) -> list[float]:
        label_count = int(context.lgca.maxlabel) + 1
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value] * label_count
        if len(values) != label_count:
            raise ValueError(f"{name} must be scalar or have one value per initial cell")
        values = [float(entry) for entry in values]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite numeric values")
        return values


class NativeNoVEIdentityGoOrGrowKappaOperator(BirthDeathOperator):
    """Native no-volume-exclusion go-or-grow with evolving kappa."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.capacity = 8
        self.r_b = 0.2
        self.r_d = 0.01
        self.kappa = None
        self.theta = 0.5
        self.kappa_std = 0.2

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.capacity = int(self.parameters.get("capacity", 8))
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.01)
        )
        self.theta = float(self.parameters.get("theta", 0.5))
        if not np.isfinite(self.theta):
            raise ValueError("theta must be a finite scalar")
        self.kappa_std = float(self.parameters.get("kappa_std", 0.2))
        if not np.isfinite(self.kappa_std) or self.kappa_std < 0.0:
            raise ValueError("kappa_std must be a non-negative finite scalar")
        self.kappa = NativeNoVEIdentityGoOrGrowOperator._label_property(
            context,
            "kappa",
            self.parameters.get("kappa", 5.0),
        )

    def setup(self, context) -> None:
        context.lgca.props.update(kappa=np.asarray(self.kappa, dtype=float))

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, _nb_sum, tanh_switch

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        nbdensity = _nb_sum(lgca, lgca.cell_density, add_center=True) / (
            (lgca.velocitychannels + 1) * self.capacity
        )
        new_kappa_chunks = []
        for coord in zip(*coords):
            density = lgca.cell_density[coord]
            nbdens = nbdensity[coord]
            cells = np.asarray(_cells_from_node(lgca.nodes[coord]), dtype=int)
            notkilled = lgca.rng.random(size=density) < 1.0 - self.r_d
            cells = cells[notkilled]
            if len(cells) == 0:
                lgca.nodes[coord] = [[] for _ in range(lgca.K)]
                continue

            kappas = lgca.props["kappa"][cells]
            switch = lgca.rng.random(len(cells)) < tanh_switch(
                rho=nbdens,
                kappa=kappas,
                theta=self.theta,
            )
            restcells = list(cells[switch])
            velcells = list(cells[~switch])

            rho = len(cells) / self.capacity
            n_prolif = lgca.rng.binomial(len(restcells), max(self.r_b * (1 - rho), 0))
            if n_prolif > 0:
                proliferating = lgca.rng.choice(restcells, n_prolif, replace=False)
                lgca.maxlabel += n_prolif
                new_cells = np.arange(lgca.maxlabel - n_prolif + 1, lgca.maxlabel + 1)
                new_kappa_chunks.append(
                    lgca.rng.normal(
                        loc=lgca.props["kappa"][proliferating],
                        scale=self.kappa_std,
                    )
                )
                restcells.extend(list(new_cells))

            new_node = [[] for _ in range(lgca.velocitychannels)]
            new_node.append(restcells)
            for cell in velcells:
                new_node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

            lgca.nodes[coord] = new_node
        if new_kappa_chunks:
            lgca.props["kappa"] = np.concatenate((lgca.props["kappa"], *new_kappa_chunks))


class NativeNoVEIdentityGoOrGrowKappaChemoOperator(NativeNoVEIdentityGoOrGrowKappaOperator):
    """Native no-volume-exclusion kappa go-or-grow with chemotactic migration."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.beta = 5.0

    def validate(self, context) -> None:
        super().validate(context)
        self.beta = float(self.parameters.get("beta", 5.0))
        if not np.isfinite(self.beta):
            raise ValueError("beta must be a finite scalar")

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, _nb_sum, tanh_switch

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        gradient = lgca.gradient(lgca.cell_density / self.capacity)
        nbdensity = _nb_sum(lgca, lgca.cell_density, add_center=True) / (
            lgca.velocitychannels * self.capacity
        )
        new_kappa_chunks = []
        for coord in zip(*coords):
            density = lgca.cell_density[coord]
            nbdens = nbdensity[coord]
            cells = np.asarray(_cells_from_node(lgca.nodes[coord]), dtype=int)
            notkilled = lgca.rng.random(size=density) < 1.0 - self.r_d
            cells = cells[notkilled]
            if len(cells) == 0:
                lgca.nodes[coord] = [[] for _ in range(lgca.K)]
                continue

            kappas = lgca.props["kappa"][cells]
            switch = lgca.rng.random(len(cells)) < tanh_switch(
                rho=nbdens,
                kappa=kappas,
                theta=self.theta,
            )
            restcells = list(cells[switch])
            velcells = list(cells[~switch])

            rho = len(cells) / self.capacity
            n_prolif = lgca.rng.binomial(len(restcells), max(self.r_b * (1 - rho), 0))
            if n_prolif > 0:
                proliferating = lgca.rng.choice(restcells, n_prolif, replace=False)
                lgca.maxlabel += n_prolif
                new_cells = np.arange(lgca.maxlabel - n_prolif + 1, lgca.maxlabel + 1)
                new_kappa_chunks.append(
                    lgca.rng.normal(
                        loc=lgca.props["kappa"][proliferating],
                        scale=self.kappa_std,
                    )
                )
                restcells.extend(list(new_cells))

            new_node = [[] for _ in range(lgca.velocitychannels)]
            new_node.append(restcells)
            if len(velcells) > 0:
                local_gradient = gradient[coord]
                weights = np.exp(self.beta * np.einsum("i,ij", local_gradient, lgca.c))
                weights /= weights.sum()
                sample = lgca.rng.multinomial(len(velcells), weights)
                lgca.rng.shuffle(velcells)
                for channel_index in range(lgca.velocitychannels):
                    new_node[channel_index].extend(velcells[: sample[channel_index]])
                    velcells = velcells[sample[channel_index] :]

            lgca.nodes[coord] = new_node
        if new_kappa_chunks:
            lgca.props["kappa"] = np.concatenate((lgca.props["kappa"], *new_kappa_chunks))


class NativeNoVEIdentityGoOrGrowGlioblastomaOperator(BirthDeathOperator):
    """Native no-volume-exclusion glioblastoma go-or-grow family dynamics."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.capacity = 8
        self.r_b = 0.2
        self.r_d = 0.01
        self.r_m = 0.001
        self.fitness_increase = 1.1
        self.theta = 0.5
        self.kappa = 5.0
        self.kappa_std = 0.2

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.capacity = int(self.parameters.get("capacity", 8))
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.01)
        )
        self.r_m = NativeClassicalBirthOperator._probability(
            "r_m", self.parameters.get("r_m", 0.001)
        )
        self.fitness_increase = float(self.parameters.get("fitness_increase", 1.1))
        if not np.isfinite(self.fitness_increase) or self.fitness_increase <= 0.0:
            raise ValueError("fitness_increase must be a positive finite scalar")
        self.theta = float(self.parameters.get("theta", 0.5))
        if not np.isfinite(self.theta):
            raise ValueError("theta must be a finite scalar")
        self.kappa = float(self.parameters.get("kappa", 5.0))
        if not np.isfinite(self.kappa):
            raise ValueError("kappa must be a finite scalar")
        self.kappa_std = float(self.parameters.get("kappa_std", 0.2))
        if not np.isfinite(self.kappa_std) or self.kappa_std < 0.0:
            raise ValueError("kappa_std must be a non-negative finite scalar")

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.init_families(type="homogeneous", mutation=True)
        if lgca.props.get("family"):
            lgca.props["family"][0] = 1
        lgca.family_props.update(r_b=[0.0] + [self.r_b] * lgca.maxfamily)
        lgca.family_props.update(kappa=[0.0] + [self.kappa] * lgca.maxfamily)

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, _nb_sum, tanh_switch

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        nbdensity = _nb_sum(lgca, lgca.cell_density, add_center=True) / (
            (lgca.velocitychannels + 1) * self.capacity
        )
        family_ids = np.asarray(lgca.props["family"], dtype=int)
        family_kappa = np.asarray(lgca.family_props["kappa"], dtype=float)

        for coord in zip(*coords):
            cells = np.asarray(_cells_from_node(lgca.nodes[coord]), dtype=int)
            if cells.size == 0:
                lgca.nodes[coord] = [[] for _ in range(lgca.K)]
                continue

            notkilled = lgca.rng.random(size=cells.size) < 1.0 - self.r_d
            cells = cells[notkilled]
            if cells.size == 0:
                lgca.nodes[coord] = [[] for _ in range(lgca.K)]
                continue

            families = family_ids[cells]
            kappas = family_kappa[families]
            switch = lgca.rng.random(cells.size) < tanh_switch(
                rho=nbdensity[coord],
                kappa=kappas,
                theta=self.theta,
            )
            restcells = list(map(int, cells[switch]))
            velcells = list(map(int, cells[~switch]))

            rho = cells.size / self.capacity
            newcells = []
            for cell in restcells:
                family = lgca.props["family"][cell]
                r_b = lgca.family_props["r_b"][family]
                if lgca.rng.random() < max(r_b * (1.0 - rho), 0.0):
                    lgca.maxlabel += 1
                    newcell = int(lgca.maxlabel)
                    newcells.append(newcell)
                    if lgca.rng.random() < self.r_m:
                        lgca.add_family(family)
                        new_family = int(lgca.maxfamily)
                        lgca.props["family"].append(new_family)
                        lgca.family_props["r_b"].append(
                            lgca.family_props["r_b"][family] * self.fitness_increase
                        )
                        lgca.family_props["kappa"].append(
                            float(
                                lgca.rng.normal(
                                    loc=lgca.family_props["kappa"][family],
                                    scale=self.kappa_std,
                                )
                            )
                        )
                    else:
                        lgca.props["family"].append(family)

            restcells.extend(newcells)

            new_node = [[] for _ in range(lgca.velocitychannels)]
            new_node.append(restcells)
            for cell in velcells:
                new_node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

            lgca.nodes[coord] = new_node


class NativeNoVEIdentityEvoStericOperator(BirthDeathOperator):
    """Native no-volume-exclusion steric evolution interaction."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.capacity = 512
        self.r_b = 0.1
        self.r_m = 0.001
        self.r_d = None
        self.alpha = 2.0
        self.gamma = 3.0
        self.fitness_increase = 1.1

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.capacity = int(self.parameters.get("capacity", 512))
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.1)
        )
        self.r_m = NativeClassicalBirthOperator._probability(
            "r_m", self.parameters.get("r_m", 0.001)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.98 * self.r_b)
        )
        self.alpha = float(self.parameters.get("alpha", 2.0))
        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be a finite scalar")
        self.gamma = float(self.parameters.get("gamma", 3.0))
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be a finite scalar")
        self.fitness_increase = float(self.parameters.get("fitness_increase", 1.1))
        if not np.isfinite(self.fitness_increase) or self.fitness_increase <= 0.0:
            raise ValueError("fitness_increase must be a positive finite scalar")

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.init_families(type="homogeneous", mutation=True)
        lgca.props["family"][0] = 1
        lgca.family_props.update(r_b=[0] + [self.r_b] * lgca.maxfamily)

    def apply(self, context, step: int) -> None:
        from .nove_ib_interactions import _cells_from_node, _split_cells_into_channels

        lgca = context.lgca
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        velchannelweights = -self.alpha * lgca.channel_weight(lgca.cell_density)
        channelweights = np.append(
            velchannelweights,
            np.full(lgca.cell_density.shape, self.gamma)[..., None],
            axis=-1,
        )
        channelprobs = np.exp(channelweights)
        channelprobs /= np.sum(channelprobs, axis=-1)[..., None]
        for coord in zip(*coords):
            density = lgca.cell_density[coord]
            rho = density / self.capacity
            cells = _cells_from_node(lgca.nodes[coord])
            newcells = cells.copy()
            for cell in cells:
                if lgca.rng.random() < self.r_d:
                    newcells.remove(cell)

                family = lgca.props["family"][cell]
                r_b = lgca.family_props["r_b"][family]

                if lgca.rng.random() < r_b * (1 - rho):
                    lgca.maxlabel += 1
                    newcells.append(lgca.maxlabel)
                    if lgca.rng.random() < self.r_m:
                        lgca.add_family(family)
                        lgca.props["family"].append(int(lgca.maxfamily))
                        lgca.family_props["r_b"].append(
                            lgca.family_props["r_b"][family] * self.fitness_increase
                        )
                    else:
                        lgca.props["family"].append(family)

            channelprob = channelprobs[coord]
            channeldist = lgca.rng.multinomial(len(newcells), channelprob).cumsum()
            lgca.rng.shuffle(newcells)

            lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)


class NativeIdentityBirthOperator(BirthDeathOperator):
    """Native identity-based birth operator matching legacy property updates."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.r_b = 0.2
        self.std = 0.01
        self.a_max = 1.0

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.std = float(self.parameters.get("std", 0.01))
        if self.std <= 0.0:
            raise ValueError("std must be positive")
        self.a_max = float(self.parameters.get("a_max", 1.0))
        if self.a_max <= 0.0:
            raise ValueError("a_max must be positive")

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.props.update(r_b=[0.0] + [self.r_b] * int(lgca.maxlabel))

    def apply(self, context, step: int) -> None:
        from .ib_interactions import trunc_gauss

        lgca = context.lgca
        relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
            lgca.cell_density[lgca.nonborder] < lgca.K
        )
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            r_bs = np.array([lgca.props["r_b"][label] for label in node])
            proliferating = lgca.rng.random(lgca.K) < r_bs
            for label in node[proliferating]:
                ind = lgca.rng.choice(lgca.K)
                if node[ind] == 0:
                    lgca.maxlabel += 1
                    node[ind] = lgca.maxlabel
                    r_b = lgca.props["r_b"][label]
                    lgca.props["r_b"].append(
                        float(trunc_gauss(0, self.a_max, r_b, sigma=self.std, rng=lgca.rng))
                    )
            lgca.nodes[coord] = node
        lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


class NativeIdentityBirthDeathOperator(BirthDeathOperator):
    """Native identity-based birth-death operator matching legacy ordering."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.r_b = 0.2
        self.r_d = 0.02
        self.std = 0.01
        self.a_max = 1.0
        self.track_inheritance = False

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.02)
        )
        self.std = float(self.parameters.get("std", 0.01))
        if self.std < 0.0:
            raise ValueError("std must be non-negative")
        self.a_max = float(self.parameters.get("a_max", 1.0))
        if self.a_max <= 0.0:
            raise ValueError("a_max must be positive")
        self.track_inheritance = bool(self.parameters.get("track_inheritance", False))

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.props.update(r_b=[0.0] + [self.r_b] * int(lgca.maxlabel))
        if self.track_inheritance:
            lgca.init_families(type="heterogeneous", mutation=False)

    def apply(self, context, step: int) -> None:
        from .ib_interactions import trunc_gauss

        lgca = context.lgca
        dying = (lgca.rng.random(size=lgca.nodes.shape) < self.r_d) & lgca.occupied

        relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
            lgca.cell_density[lgca.nonborder] < lgca.K
        )
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            occ = lgca.occupied[coord]
            r_bs = np.array([lgca.props["r_b"][label] for label in node])
            proliferating = (lgca.rng.random(lgca.K) * occ) < r_bs
            n_p = proliferating.sum()
            if n_p == 0:
                continue
            targetchannels = lgca.rng.choice(lgca.K, size=n_p, replace=False)
            for index, label in enumerate(node[proliferating]):
                channel = targetchannels[index]
                if node[channel] == 0:
                    lgca.maxlabel += 1
                    node[channel] = lgca.maxlabel
                    r_b = lgca.props["r_b"][label]
                    if self.std > 0:
                        lgca.props["r_b"].append(
                            float(trunc_gauss(0, self.a_max, r_b, sigma=self.std, rng=lgca.rng))
                        )
                    else:
                        lgca.props["r_b"].append(r_b)
                    if self.track_inheritance:
                        fam = lgca.props["family"][label]
                        lgca.props["family"].append(fam)
            lgca.nodes[coord] = node

        lgca.nodes[dying] = 0
        lgca.update_dynamic_fields()
        lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


class NativeIdentityBirthDeathDiscreteOperator(BirthDeathOperator):
    """Native identity birth-death with discrete proliferation-rate mutations."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.r_b = 0.2
        self.r_d = 0.02
        self.drb = 0.01
        self.a_max = 1.0
        self.pmut = 0.1

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.02)
        )
        self.pmut = NativeClassicalBirthOperator._probability(
            "pmut", self.parameters.get("pmut", 0.1)
        )
        self.drb = float(self.parameters.get("drb", 0.01))
        if not np.isfinite(self.drb) or self.drb < 0.0:
            raise ValueError("drb must be a non-negative finite scalar")
        self.a_max = float(self.parameters.get("a_max", 1.0))
        if not np.isfinite(self.a_max) or self.a_max <= 0.0:
            raise ValueError("a_max must be a positive finite scalar")

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.props.update(r_b=[0.0] + [self.r_b] * int(lgca.maxlabel))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        dying = (lgca.rng.random(size=lgca.nodes.shape) < self.r_d) & lgca.occupied
        lgca.nodes[dying] = 0
        lgca.update_dynamic_fields()

        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            occ = lgca.occupied[coord]
            r_bs = np.array([lgca.props["r_b"][label] for label in node])
            proliferating = (lgca.rng.random(lgca.K) * occ) < r_bs
            n_proliferating = proliferating.sum()
            if n_proliferating == 0:
                continue
            targetchannels = lgca.rng.choice(lgca.K, size=n_proliferating, replace=False)
            for index, label in enumerate(node[proliferating]):
                channel = targetchannels[index]
                if node[channel] == 0:
                    lgca.maxlabel += 1
                    node[channel] = lgca.maxlabel
                    r_b = lgca.props["r_b"][label]
                    if r_b < self.a_max:
                        lgca.props["r_b"].append(
                            lgca.rng.choice(
                                (r_b - self.drb, r_b + self.drb, r_b),
                                p=(self.pmut / 2, self.pmut / 2, 1 - self.pmut),
                            )
                        )
                    else:
                        lgca.props["r_b"].append(
                            lgca.rng.choice(
                                (r_b - self.drb, r_b),
                                p=(self.pmut / 2, 1 - self.pmut / 2),
                            )
                        )
            lgca.nodes[coord] = node

        lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


class NativeIdentityGoAndGrowMutationsOperator(BirthDeathOperator):
    """Native identity go-and-grow with explicit family mutation tracking."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.effect = "passenger_mutation"
        self.r_b = 0.5
        self.r_m = 0.001
        self.r_d = 0.02
        self.fitness_increase = 1.1

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.effect = str(self.parameters.get("effect", "passenger_mutation"))
        if self.effect not in {"passenger_mutation", "driver_mutation"}:
            raise ValueError(
                "effect must be 'passenger_mutation' or 'driver_mutation'"
            )
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.5)
        )
        self.r_m = NativeClassicalBirthOperator._probability(
            "r_m", self.parameters.get("r_m", 0.001)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.02)
        )
        self.fitness_increase = float(self.parameters.get("fitness_increase", 1.1))
        if not np.isfinite(self.fitness_increase) or self.fitness_increase <= 0.0:
            raise ValueError("fitness_increase must be a positive finite scalar")

    def setup(self, context) -> None:
        lgca = context.lgca
        if "r_int" in self.parameters:
            lgca.set_r_int(self.parameters["r_int"])
        lgca.init_families(type="homogeneous", mutation=True)
        if self.effect == "driver_mutation":
            lgca.family_props.update(r_b=[0] + [self.r_b] * lgca.maxfamily)

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        dying = (lgca.rng.random(size=lgca.nodes.shape) < self.r_d) & lgca.occupied
        lgca.nodes[dying] = 0
        lgca.update_dynamic_fields()

        relevant = (lgca.cell_density[lgca.nonborder] > 0) & (
            lgca.cell_density[lgca.nonborder] < lgca.K
        )
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            if self.effect == "driver_mutation":
                r_bs = np.array(
                    [lgca.family_props["r_b"][lgca.props["family"][label]] for label in node]
                )
            else:
                r_bs = self.r_b * node.astype(bool)
            proliferating = lgca.rng.random(lgca.K) < r_bs
            n_proliferating = proliferating.sum()
            if n_proliferating == 0:
                continue
            targetchannels = lgca.rng.choice(lgca.K, size=n_proliferating, replace=False)
            for index, label in enumerate(node[proliferating]):
                channel = targetchannels[index]
                if node[channel] == 0:
                    lgca.maxlabel += 1
                    node[channel] = lgca.maxlabel
                    family = lgca.props["family"][label]
                    mutation = lgca.rng.random() < self.r_m
                    if mutation:
                        lgca.add_family(family)
                        lgca.props["family"].append(int(lgca.maxfamily))
                        if self.effect == "driver_mutation":
                            lgca.family_props["r_b"].append(
                                lgca.family_props["r_b"][family] * self.fitness_increase
                            )
                    else:
                        lgca.props["family"].append(family)
            lgca.nodes[coord] = node

        lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


class NativeIdentityGoOrGrowOperator(BirthDeathOperator):
    """Native identity go-or-grow with inherited switching traits."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.r_b = 0.2
        self.r_d = 0.01
        self.kappa = None
        self.theta = None
        self.kappa_std = 0.2
        self.theta_std = 0.05

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if not context.spec.state.identity_based:
            raise ValueError(f"{self.name} requires state.identity_based=True")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.01)
        )
        self.kappa_std = float(self.parameters.get("kappa_std", 0.2))
        if not np.isfinite(self.kappa_std) or self.kappa_std < 0.0:
            raise ValueError("kappa_std must be a non-negative finite scalar")
        self.theta_std = float(self.parameters.get("theta_std", 0.05))
        if not np.isfinite(self.theta_std) or self.theta_std < 0.0:
            raise ValueError("theta_std must be a non-negative finite scalar")
        self.kappa = self._label_property(
            context,
            "kappa",
            self.parameters.get("kappa", 5.0),
        )
        self.theta = self._label_property(
            context,
            "theta",
            self.parameters.get("theta", 0.75),
        )

    def setup(self, context) -> None:
        lgca = context.lgca
        lgca.props.update(kappa=[0.0] + list(self.kappa))
        lgca.props.update(theta=[0.0] + list(self.theta))

    def apply(self, context, step: int) -> None:
        from .interactions import tanh_switch

        lgca = context.lgca
        dying = (lgca.rng.random(size=lgca.nodes.shape) < self.r_d) & lgca.occupied
        lgca.nodes[dying] = 0

        lgca.update_dynamic_fields()
        n_m = lgca.occupied[..., : lgca.velocitychannels].sum(-1)
        n_r = lgca.occupied[..., lgca.velocitychannels :].sum(-1)
        relevant = lgca.cell_density[lgca.nonborder] > 0
        coords = [axis_indices[relevant] for axis_indices in lgca.nonborder]
        for coord in zip(*coords):
            node = lgca.nodes[coord]
            vel = node[: lgca.velocitychannels]
            rest = node[lgca.velocitychannels :]
            rho = lgca.cell_density[coord] / lgca.K

            free_rest = lgca.restchannels - n_r[coord]
            free_vel = lgca.velocitychannels - n_m[coord]
            can_switch_to_rest = lgca.rng.permutation(vel[vel > 0])[:free_rest]
            can_switch_to_vel = lgca.rng.permutation(rest[rest > 0])[:free_vel]

            for cell in can_switch_to_rest:
                if lgca.rng.random() < tanh_switch(
                    rho,
                    kappa=lgca.props["kappa"][cell],
                    theta=lgca.props["theta"][cell],
                ):
                    rest[np.where(rest == 0)[0][0]] = cell
                    vel[np.where(vel == cell)[0][0]] = 0

            for cell in can_switch_to_vel:
                if lgca.rng.random() < 1 - tanh_switch(
                    rho,
                    kappa=lgca.props["kappa"][cell],
                    theta=lgca.props["theta"][cell],
                ):
                    vel[np.where(vel == 0)[0][0]] = cell
                    rest[np.where(rest == cell)[0][0]] = 0

            can_proliferate = lgca.rng.permutation(rest[rest > 0])[: (rest == 0).sum()]
            for cell in can_proliferate:
                if lgca.rng.random() < self.r_b:
                    lgca.maxlabel += 1
                    rest[np.where(rest == 0)[0][0]] = lgca.maxlabel
                    kappa = lgca.props["kappa"][cell]
                    if self.kappa_std == 0:
                        lgca.props["kappa"].append(kappa)
                    else:
                        lgca.props["kappa"].append(
                            lgca.rng.normal(loc=kappa, scale=self.kappa_std)
                        )
                    theta = lgca.props["theta"][cell]
                    if self.theta_std == 0:
                        lgca.props["theta"].append(theta)
                    else:
                        lgca.props["theta"].append(
                            lgca.rng.normal(loc=theta, scale=self.theta_std)
                        )

            v_channels = lgca.rng.permutation(vel)
            r_channels = lgca.rng.permutation(rest)
            lgca.nodes[coord] = np.hstack((v_channels, r_channels))

    @staticmethod
    def _label_property(context, name: str, value: Any) -> list[float]:
        lgca = context.lgca
        try:
            values = list(value)
        except TypeError:
            values = [value] * int(lgca.maxlabel)
        if len(values) != int(lgca.maxlabel):
            raise ValueError(f"{name} must be scalar or have one value per initial cell")
        values = [float(entry) for entry in values]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite numeric values")
        return values


class NativeClassicalReorientationOperator(ReorientationOperator):
    """Native classical reorientation matching legacy permutation sampling."""

    def __init__(
        self,
        info: PluginInfo,
        *,
        mode: str,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.mode = mode
        self.beta = 2.0
        self.gradient_field = None

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.beta = float(self.parameters.get("beta", 2.0))

    def setup(self, context) -> None:
        lgca = context.lgca
        if not hasattr(lgca, "permutations") and not hasattr(lgca, "_permutation_cache"):
            lgca.calc_permutations()
        if self.mode == "chemotaxis":
            self.gradient_field = self._chemotaxis_gradient(lgca)

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        field = self._field(lgca)
        newnodes = lgca.nodes.copy()
        nb_nodes = newnodes[lgca.nonborder]
        flux = field[lgca.nonborder]
        density = lgca.cell_density[lgca.nonborder]

        unique = np.unique(density)
        unique = unique[(unique > 0) & (unique < lgca.K)]
        for n_particles in unique:
            mask = density == n_particles
            j = lgca.get_flux_permutations(n_particles)
            weights = _softmax_last_axis(self.beta * (flux[mask] @ j))
            cumw = weights.cumsum(axis=1)
            rnd = lgca.rng.random(mask.sum())
            ind = (rnd[:, None] < cumw).argmax(axis=1)
            nb_nodes[mask] = lgca.get_permutations(n_particles)[ind]

        newnodes[lgca.nonborder] = nb_nodes
        lgca.nodes = newnodes

    def dependencies(self) -> set[str]:
        if self.mode == "alignment":
            return {"boundary_nodes"}
        if self.mode == "aggregation":
            return {"boundary_nodes", "cell_density"}
        if self.mode == "persistent_walk":
            return {"nodes"}
        return set()

    def _field(self, lgca):
        if self.mode == "alignment":
            return lgca.nb_sum(lgca.calc_flux(lgca.nodes))
        if self.mode == "persistent_walk":
            return lgca.calc_flux(lgca.nodes)
        if self.mode == "aggregation":
            return lgca.gradient(lgca.cell_density)
        if self.mode == "chemotaxis":
            return self.gradient_field
        raise ValueError(f"unsupported classical reorientation mode {self.mode!r}")

    def _chemotaxis_gradient(self, lgca):
        if "gradient" in self.parameters:
            gradient = np.asarray(self.parameters["gradient"], dtype=float)
            expected = lgca.nodes.shape[:-1] + (lgca.c.shape[0],)
            if gradient.shape == tuple(lgca.dims) + (lgca.c.shape[0],):
                pad_width = [(lgca.r_int, lgca.r_int)] * len(lgca.dims) + [(0, 0)]
                gradient = np.pad(gradient, pad_width=pad_width, mode="edge")
            if gradient.shape != expected:
                raise ValueError(f"gradient must have shape {expected}; got {gradient.shape}")
            return gradient

        if len(lgca.dims) == 1:
            source = lgca.l / 2
            r = abs(lgca.xcoords - source)
            lgca.concentration = np.exp(-2 * r / lgca.l)
            gradient = lgca.gradient(np.pad(lgca.concentration, 1, "reflect"))
            gradient /= gradient.max()
            return gradient

        if len(lgca.dims) == 2:
            x_source = lgca.xcoords.mean()
            y_source = lgca.ycoords.mean()
            rx = lgca.xcoords - x_source
            ry = lgca.ycoords - y_source
            r = np.sqrt(rx ** 2 + ry ** 2)
            lgca.concentration = np.exp(-2 * r / lgca.ly)
            return lgca.gradient(np.pad(lgca.concentration, 1, "reflect"))

        if len(lgca.dims) == 3:
            x_source = lgca.xcoords.mean()
            y_source = lgca.ycoords.mean()
            z_source = lgca.zcoords.mean()
            rx = lgca.xcoords - x_source
            ry = lgca.ycoords - y_source
            rz = lgca.zcoords - z_source
            r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
            lgca.concentration = np.exp(-2 * r / lgca.ly)
            return lgca.gradient(np.pad(lgca.concentration, 1, "reflect"))

        raise ValueError(f"{self.name} does not support {len(lgca.dims)} spatial dimensions")


class NativeClassicalTensorReorientationOperator(ReorientationOperator):
    """Native classical tensor-weighted reorientation matching legacy sampling."""

    def __init__(
        self,
        info: PluginInfo,
        *,
        mode: str,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.mode = mode
        self.beta = 2.0
        self.guiding_tensor = None

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.beta = float(self.parameters.get("beta", 2.0))
        if self.mode == "contact_guidance" and len(context.lgca.dims) != 2:
            raise ValueError("contact_guidance is not supported for this geometry")

    def setup(self, context) -> None:
        lgca = context.lgca
        if not hasattr(lgca, "permutations") and not hasattr(lgca, "_permutation_cache"):
            lgca.calc_permutations()
        if self.mode == "contact_guidance":
            director = self._director_field(context)
            eye = 0.5 * np.diag(np.ones(2))[None, ...]
            self.guiding_tensor = np.einsum("...i,...j->...ij", director, director) - eye

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        tensors = self._tensors(lgca)[lgca.nonborder]
        newnodes = lgca.nodes.copy()
        nb_nodes = newnodes[lgca.nonborder]
        density = lgca.cell_density[lgca.nonborder]

        unique = np.unique(density)
        unique = unique[(unique > 0) & (unique < lgca.K)]
        for n_particles in unique:
            mask = density == n_particles
            si = lgca.get_si_permutations(n_particles)
            weights = _softmax_last_axis(
                self.beta * np.einsum("nij,pij->np", tensors[mask], si)
            )
            cumw = weights.cumsum(axis=1)
            rnd = lgca.rng.random(mask.sum())
            ind = (rnd[:, None] < cumw).argmax(axis=1)
            nb_nodes[mask] = lgca.get_permutations(n_particles)[ind]

        newnodes[lgca.nonborder] = nb_nodes
        lgca.nodes = newnodes

    def dependencies(self) -> set[str]:
        if self.mode == "nematic":
            return {"boundary_nodes"}
        return set()

    def _tensors(self, lgca):
        if self.mode == "nematic":
            s = np.einsum("...k,kxy->...xy", lgca.nodes[..., : lgca.velocitychannels], lgca.cij)
            return lgca.nb_sum(s)
        if self.mode == "contact_guidance":
            return self.guiding_tensor
        raise ValueError(f"unsupported tensor reorientation mode {self.mode!r}")

    def _director_field(self, context):
        lgca = context.lgca
        if "director" in self.parameters:
            director = np.asarray(self.parameters["director"], dtype=float)
        elif "director" in context.fields:
            director = np.asarray(getattr(lgca, "director"), dtype=float)
        else:
            director = np.zeros(lgca.nodes.shape[:-1] + (2,), dtype=float)
            director[..., 0] = 1.0

        expected = lgca.nodes.shape[:-1] + (2,)
        if director.shape == tuple(lgca.dims) + (2,):
            pad_width = [(lgca.r_int, lgca.r_int)] * len(lgca.dims) + [(0, 0)]
            director = np.pad(director, pad_width=pad_width, mode="edge")
        if director.shape != expected:
            raise ValueError(f"director must have shape {expected}; got {director.shape}")
        return director


class NativeClassicalWettingOperator(ReorientationOperator):
    """Native wetting interaction matching the legacy adhesive-surface rule."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.beta = 2.0
        self.alpha = 2.0
        self.gamma = 2.0
        self.rho_0 = None
        self.n_crit = None

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if "ecm" not in context.fields and not hasattr(context.lgca, "ecm"):
            raise ValueError("state.fields.ecm is required by classical.wetting")
        self.beta = float(self.parameters.get("beta", 2.0))
        self.alpha = float(self.parameters.get("alpha", 2.0))
        self.gamma = float(self.parameters.get("gamma", 2.0))
        self.rho_0 = self.parameters.get("rho_0", context.lgca.restchannels // 2)

    def setup(self, context) -> None:
        lgca = context.lgca
        if not hasattr(lgca, "permutations") and not hasattr(lgca, "_permutation_cache"):
            lgca.calc_permutations()
        if lgca.r_int != 2:
            lgca.set_r_int(2)
        self.n_crit = (lgca.velocitychannels + 1) * self.rho_0

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        if hasattr(lgca, "spheroid"):
            birth = lgca.rng.random(lgca.nodes[lgca.spheroid].shape) < self.parameters["r_b"]
            ds = (1 - lgca.nodes[lgca.spheroid]) * birth
            lgca.nodes[lgca.spheroid, :] = np.add(
                lgca.nodes[lgca.spheroid, :], ds, casting="unsafe"
            )
            lgca.update_dynamic_fields()

        newnodes = lgca.nodes.copy()
        nb_nodes = newnodes[lgca.nonborder]

        nbs = lgca.nb_sum(lgca.cell_density).astype(float)
        nbs *= np.clip(1 - nbs / self.n_crit, a_min=0, a_max=None) / self.n_crit * 2
        g_adh = lgca.gradient(nbs)
        pressure = np.clip(lgca.cell_density - self.rho_0, a_min=0.0, a_max=None) / (
            lgca.K - self.rho_0
        )
        g_pressure = -lgca.gradient(pressure)

        resting = lgca.nodes[..., lgca.velocitychannels :].sum(-1)
        resting = lgca.nb_sum(resting) / lgca.velocitychannels / self.rho_0
        g = lgca.nb_sum(lgca.calc_flux(lgca.nodes))

        density = lgca.cell_density[lgca.nonborder]
        flux = g[lgca.nonborder]
        rest_nb = resting[lgca.nonborder]
        g_adh_nb = g_adh[lgca.nonborder]
        g_press_nb = g_pressure[lgca.nonborder]
        ecm_nb = lgca.ecm[lgca.nonborder]

        unique = np.unique(density)
        unique = unique[(unique > 0) & (unique < lgca.K)]
        for n_particles in unique:
            mask = density == n_particles
            perms = lgca.get_permutations(n_particles)
            restc = perms[:, lgca.velocitychannels :].sum(-1)
            j = lgca.get_flux_permutations(n_particles)
            weights = _softmax_last_axis(
                self.beta * (flux[mask] @ j) / lgca.velocitychannels / 2
                + self.beta * rest_nb[mask, None] * restc
                + self.beta * np.einsum("nd,dp->np", g_adh_nb[mask], j)
                + restc * ecm_nb[mask, None]
                + self.gamma * np.einsum("nd,dp->np", g_press_nb[mask], j)
            )
            cumw = weights.cumsum(axis=1)
            rnd = lgca.rng.random(mask.sum())
            ind = (rnd[:, None] < cumw).argmax(axis=1)
            nb_nodes[mask] = perms[ind]

        newnodes[lgca.nonborder] = nb_nodes
        lgca.nodes = newnodes
        lgca.ecm -= self.alpha * lgca.ecm * lgca.cell_density / lgca.K


class NativeClassicalBirthOperator(BirthDeathOperator):
    """Native classical birth/birth-death operator matching legacy ordering."""

    def __init__(
        self,
        info: PluginInfo,
        *,
        mode: str,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.mode = mode
        self.r_b = 0.2
        self.r_d = 0.05

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        self.r_b = self._probability("r_b", self.parameters.get("r_b", 0.2))
        if self.mode == "birthdeath":
            self.r_d = self._probability("r_d", self.parameters.get("r_d", 0.05))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        birth = lgca.rng.random(lgca.nodes.shape) < self.r_b * lgca.cell_density[..., None] / lgca.K
        if self.mode == "birth":
            np.add(lgca.nodes, (1 - lgca.nodes) * birth, out=lgca.nodes, casting="unsafe")
        elif self.mode == "birthdeath":
            death = lgca.rng.random(lgca.nodes.shape) < self.r_d
            ds = (1 - lgca.nodes) * birth - lgca.nodes * death
            np.add(lgca.nodes, ds, out=lgca.nodes, casting="unsafe")
            lgca.update_dynamic_fields()
        else:
            raise ValueError(f"unsupported classical birth mode {self.mode!r}")
        lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)

    @staticmethod
    def _probability(name: str, value) -> float:
        probability = float(value)
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"{name} must be a probability")
        return probability


class NativeClassicalGoOrRestOperator(PhenotypeSwitchOperator):
    """Native classical moving/resting switch matching the legacy rule."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.kappa = 5.0
        self.theta = 0.75

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if context.lgca.restchannels < 1:
            raise ValueError(f"{self.name} requires at least one rest channel")
        self.kappa = float(self.parameters.get("kappa", 5.0))
        self.theta = float(self.parameters.get("theta", 0.75))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        n_m = lgca.nodes[..., : lgca.velocitychannels].sum(-1)
        n_r = lgca.nodes[..., lgca.velocitychannels :].sum(-1)
        m_to_r_capacity = np.minimum(n_m, lgca.restchannels - n_r)
        r_to_m_capacity = np.minimum(n_r, lgca.velocitychannels - n_m)

        rho = lgca.cell_density / lgca.K
        prob = 0.5 * (1 + np.tanh(self.kappa * (rho - self.theta)))
        moving_to_rest = lgca.rng.binomial(m_to_r_capacity, prob)
        resting_to_moving = lgca.rng.binomial(r_to_m_capacity, 1 - prob)
        n_m = n_m + resting_to_moving - moving_to_rest
        n_r = n_r + moving_to_rest - resting_to_moving

        newnodes = np.zeros_like(lgca.nodes)
        v_idx = np.arange(lgca.velocitychannels)
        r_idx = np.arange(lgca.restchannels)
        newnodes[..., : lgca.velocitychannels] = (v_idx < n_m[..., None]).astype(lgca.nodes.dtype)
        newnodes[..., lgca.velocitychannels :] = (r_idx < n_r[..., None]).astype(lgca.nodes.dtype)
        newnodes[..., : lgca.velocitychannels] = lgca.rng.permuted(
            newnodes[..., : lgca.velocitychannels], axis=-1
        )
        lgca.nodes = newnodes


class NativeClassicalGoOrGrowOperator(BirthDeathOperator):
    """Native classical go-or-grow switch matching the legacy composite rule."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.r_b = 0.2
        self.r_d = 0.01
        self.kappa = 5.0
        self.theta = 0.75

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if context.lgca.restchannels < 1:
            raise ValueError(f"{self.name} requires at least one rest channel")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.01)
        )
        self.kappa = float(self.parameters.get("kappa", 5.0))
        self.theta = float(self.parameters.get("theta", 0.75))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        n_m = lgca.nodes[..., : lgca.velocitychannels].sum(-1)
        n_r = lgca.nodes[..., lgca.velocitychannels :].sum(-1)
        m_to_r_capacity = np.minimum(n_m, lgca.restchannels - n_r)
        r_to_m_capacity = np.minimum(n_r, lgca.velocitychannels - n_m)

        rho = lgca.cell_density / lgca.K
        prob = 0.5 * (1 + np.tanh(self.kappa * (rho - self.theta)))
        moving_to_rest = lgca.rng.binomial(m_to_r_capacity, prob)
        resting_to_moving = lgca.rng.binomial(r_to_m_capacity, 1 - prob)
        n_m = n_m + resting_to_moving - moving_to_rest
        n_r = n_r + moving_to_rest - resting_to_moving
        n_m -= lgca.rng.binomial(n_m, self.r_d)
        n_r -= lgca.rng.binomial(n_r, self.r_d)
        birth_capacity = np.minimum(n_r, lgca.restchannels - n_r)
        n_r += lgca.rng.binomial(birth_capacity, self.r_b)

        newnodes = np.zeros_like(lgca.nodes)
        v_idx = np.arange(lgca.velocitychannels)
        r_idx = np.arange(lgca.restchannels)
        newnodes[..., : lgca.velocitychannels] = (v_idx < n_m[..., None]).astype(lgca.nodes.dtype)
        newnodes[..., lgca.velocitychannels :] = (r_idx < n_r[..., None]).astype(lgca.nodes.dtype)
        newnodes[..., : lgca.velocitychannels] = lgca.rng.permuted(
            newnodes[..., : lgca.velocitychannels], axis=-1
        )
        lgca.nodes = newnodes


class NativePhenotypeSwitchOperator(PhenotypeSwitchOperator):
    """Sample an atomic, particle-conserving multispecies state transition.

    At each lattice site, ``s`` is the complete ``(n_species, K)`` channel
    state and the interaction constructs one equally shaped and typed ``s'``.
    It guarantees ``N(s') == N(s)`` and, for Boolean states, at most one
    particle per species/channel slot. If at least one phenotype changes, all
    channel positions are resampled; if none changes, the state is unchanged.
    Volume-exclusion capacity is enforced while targets are sampled, so
    collisions cannot merge or delete particles.
    """

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        info = PluginInfo(
            name="phenotype_switch",
            aliases=("species_switch",),
            operator_kind="phenotype_switch",
            backend_families=("multispecies",),
            conservation_law=ConservationLaw(True, False, True, ("species identity",)),
            port_status="native",
            description="Channel-preserving stochastic species transition operator.",
        )
        super().__init__(info=info, parameters=parameters)
        self.rates = None

    def validate(self, context) -> None:
        n_species = context.spec.state.n_species
        if n_species < 2:
            raise ValueError("phenotype switch requires state.n_species >= 2")
        if context.spec.state.identity_based:
            raise ValueError("phenotype switch does not yet support identity-based states")
        if "rates" not in self.parameters:
            raise ValueError("missing required parameter 'rates'")
        rates = np.asarray(self.parameters["rates"], dtype=float)
        if rates.shape != (n_species, n_species):
            raise ValueError("rates must have shape (n_species, n_species)")
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("rates must contain finite non-negative values")
        off_diag = rates.copy()
        np.fill_diagonal(off_diag, 0.0)
        if np.any(off_diag.sum(axis=1) > 1.0):
            raise ValueError("row sums of phenotype switch rates must be <= 1")
        self.rates = off_diag

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        rates = self.rates
        if rates is None:
            self.validate(context)
            rates = self.rates
        for spatial in np.ndindex(lgca.dims):
            coord = tuple(index + lgca.r_int for index in spatial)
            state = lgca.nodes[coord + (slice(None), slice(None))]
            lgca.nodes[coord + (slice(None), slice(None))] = self._sample_state(
                state, rates, lgca.rng
            )

    @staticmethod
    def _sample_state(state, rates, rng):
        """Construct one admissible complete state without sequential writes."""
        state = np.asarray(state)
        n_species, n_channels = state.shape
        probabilities = np.asarray(rates, dtype=float).copy()
        np.fill_diagonal(
            probabilities, 1.0 - probabilities.sum(axis=1)
        )

        if state.dtype == bool:
            sources = np.repeat(np.arange(n_species), state.sum(axis=1))
            if sources.size == 0:
                return state.copy()
            rng.shuffle(sources)
            remaining = np.full(n_species, n_channels, dtype=int)
            targets = np.empty(sources.size, dtype=int)
            changed = False
            for index, source in enumerate(sources):
                available = remaining > 0
                constrained = probabilities[source] * available
                if constrained.sum() == 0.0:
                    if available[source]:
                        target = int(source)
                    else:
                        target = int(rng.choice(np.flatnonzero(available)))
                else:
                    constrained /= constrained.sum()
                    target = int(rng.choice(n_species, p=constrained))
                targets[index] = target
                remaining[target] -= 1
                changed |= target != source

            if not changed:
                return state.copy()

            result = np.zeros_like(state)
            for target, count in enumerate(np.bincount(targets, minlength=n_species)):
                if count:
                    channels = rng.choice(n_channels, size=count, replace=False)
                    result[target, channels] = True
            return result

        source_counts = state.sum(axis=1).astype(int)
        flows = np.zeros((n_species, n_species), dtype=int)
        for source, count in enumerate(source_counts):
            if count:
                flows[source] = rng.multinomial(count, probabilities[source])
        if not (flows - np.diag(np.diag(flows))).any():
            return state.copy()

        result = np.zeros_like(state)
        channel_probabilities = np.full(n_channels, 1.0 / n_channels)
        for target, count in enumerate(flows.sum(axis=0)):
            if count:
                result[target] = rng.multinomial(count, channel_probabilities)
        return result


class NativeNoVEAlignmentOperator(ReorientationOperator):
    """Native no-volume-exclusion alignment reorientation."""

    def __init__(
        self,
        info: PluginInfo,
        *,
        density_dependent: bool,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.density_dependent = density_dependent
        self.beta = 2.0
        self.include_center = False

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if context.lgca.restchannels != 0:
            raise ValueError(f"{self.name} requires state.restchannels=0")
        self.beta = float(self.parameters.get("beta", 2.0))
        self.include_center = bool(self.parameters.get("include_center", False))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        g = lgca.calc_flux(lgca.nodes)
        if self.include_center:
            g += lgca.nb_sum(g)
        else:
            g = lgca.nb_sum(g)

        if not self.density_dependent:
            if self.include_center:
                nsum = lgca.nb_sum(lgca.cell_density)[..., None] + lgca.cell_density[..., None]
            else:
                nsum = lgca.nb_sum(lgca.cell_density)[..., None]
            np.maximum(nsum, 1, out=nsum)
            g = g / nsum

        weights = _softmax_last_axis(self.beta * np.einsum("...i,ij->...j", g, lgca.c))
        newnodes = lgca.nodes.copy()
        density = lgca.cell_density[lgca.nonborder]
        newnodes[lgca.nonborder] = lgca.rng.multinomial(density, weights[lgca.nonborder])
        lgca.nodes = newnodes


class NativeNoVERandomWalkOperator(ReorientationOperator):
    """Native no-volume-exclusion uniform channel redistribution."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        newnodes = lgca.nodes.copy()
        weights = np.full(lgca.K, 1 / lgca.K)
        density = lgca.cell_density[lgca.nonborder]
        newnodes[lgca.nonborder] = lgca.rng.multinomial(density, weights)
        lgca.nodes = newnodes


class NativeNoVEGoOrRestOperator(PhenotypeSwitchOperator):
    """Native no-volume-exclusion moving/resting switch."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.kappa = 5.0
        self.theta = 0.75

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if context.lgca.restchannels < 1:
            raise ValueError(f"{self.name} requires at least one rest channel")
        self.kappa = float(self.parameters.get("kappa", 5.0))
        self.theta = float(self.parameters.get("theta", 0.75))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        nb_nodes = lgca.nodes[lgca.nonborder]
        n_m = nb_nodes[..., : lgca.velocitychannels].sum(-1)
        n_r = nb_nodes[..., lgca.velocitychannels :].sum(-1)
        rho = (n_m + n_r) / lgca.capacity

        prob = 0.5 * (1 + np.tanh(self.kappa * (rho - self.theta)))
        moving_to_rest = lgca.rng.binomial(n_m, prob)
        resting_to_moving = lgca.rng.binomial(n_r, 1 - prob)
        n_m = n_m + resting_to_moving - moving_to_rest
        n_r = n_r + moving_to_rest - resting_to_moving

        weights = np.full(lgca.velocitychannels, 1 / lgca.velocitychannels)
        v_channels = lgca.rng.multinomial(n_m, weights)
        r_channels = n_r[..., None]
        nb_new = np.concatenate((v_channels, r_channels), axis=-1)
        lgca.nodes[lgca.nonborder] = nb_new.astype(lgca.nodes.dtype)


class NativeNoVEGoOrGrowOperator(BirthDeathOperator):
    """Native no-volume-exclusion go-or-grow interaction."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.r_b = 0.2
        self.r_d = 0.01
        self.kappa = 5.0
        self.theta = 0.75

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 1:
            raise ValueError(f"{self.name} does not support multispecies states")
        if context.lgca.restchannels < 1:
            raise ValueError(f"{self.name} requires at least one rest channel")
        self.r_b = NativeClassicalBirthOperator._probability(
            "r_b", self.parameters.get("r_b", 0.2)
        )
        self.r_d = NativeClassicalBirthOperator._probability(
            "r_d", self.parameters.get("r_d", 0.01)
        )
        self.kappa = float(self.parameters.get("kappa", 5.0))
        self.theta = float(self.parameters.get("theta", 0.75))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        nb_nodes = lgca.nodes[lgca.nonborder]
        n_m = nb_nodes[..., : lgca.velocitychannels].sum(-1)
        n_r = nb_nodes[..., lgca.velocitychannels :].sum(-1)
        rho = (n_m + n_r) / lgca.capacity

        prob = 0.5 * (1 + np.tanh(self.kappa * (rho - self.theta)))
        moving_to_rest = lgca.rng.binomial(n_m, prob)
        resting_to_moving = lgca.rng.binomial(n_r, 1 - prob)
        n_m = n_m + resting_to_moving - moving_to_rest
        n_r = n_r + moving_to_rest - resting_to_moving

        n_m -= lgca.rng.binomial(n_m, self.r_d)
        n_r -= lgca.rng.binomial(n_r, self.r_d)
        birth_prob = np.clip(self.r_b * (1 - rho), 0, 1)
        n_r += lgca.rng.binomial(n_r, birth_prob)

        weights = np.full(lgca.velocitychannels, 1 / lgca.velocitychannels)
        v_channels = lgca.rng.multinomial(n_m, weights)
        r_channels = n_r[..., None]
        nb_new = np.concatenate((v_channels, r_channels), axis=-1)
        lgca.nodes[lgca.nonborder] = nb_new.astype(lgca.nodes.dtype)


class NativeMultispeciesBirthOperator(BirthDeathOperator):
    """Native multispecies no-volume-exclusion birth/birth-death operator."""

    def __init__(
        self,
        info: PluginInfo,
        *,
        mode: str,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(info=info, parameters=parameters)
        self.mode = mode
        self.r_b = None
        self.r_d = None
        self.capacity = None
        self.mutation_matrix = None
        self.channel_weights = None

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species < 2:
            raise ValueError(f"{self.name} requires state.n_species >= 2")
        lgca = context.lgca
        self.capacity = self.parameters.get("capacity", getattr(lgca, "capacity", lgca.K))
        self.r_b = self._species_probability_vector(context, "r_b", 0.2)
        self.r_d = self._species_probability_vector(context, "r_d", 0.02)
        self.mutation_matrix = self._resolve_mutation_matrix(context)
        gamma = float(self.parameters.get("gamma", 0.0))
        z = lgca.velocitychannels + np.exp(gamma) * lgca.restchannels
        self.channel_weights = np.array(
            [1.0 / z] * lgca.velocitychannels
            + [np.exp(gamma) / z] * lgca.restchannels
        )

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        species_counts = lgca.nodes.sum(axis=-1).astype(np.int64)
        total = species_counts.sum(axis=-1)
        rho = total / self.capacity
        birth_prob = np.clip(self.r_b * (1 - rho[..., None]), 0, 1)

        if self.mode == "birth":
            births_by_parent = lgca.rng.binomial(species_counts, birth_prob)
            final_counts = species_counts + self._offspring_by_species(lgca, births_by_parent)
        elif self.mode == "birthdeath":
            deaths = lgca.rng.binomial(species_counts, self.r_d)
            survivors = species_counts - deaths
            births_by_parent = lgca.rng.binomial(species_counts, birth_prob)
            final_counts = survivors + self._offspring_by_species(lgca, births_by_parent)
        else:
            raise ValueError(f"unsupported multispecies birth mode {self.mode!r}")

        lgca.nodes = lgca.rng.multinomial(
            np.asarray(final_counts, dtype=np.int64),
            self.channel_weights,
        ).astype(lgca.nodes.dtype)

    def _species_probability_vector(self, context, name: str, default: float) -> np.ndarray:
        value = self.parameters.get(name, default)
        rates = np.asarray(value, dtype=float)
        if rates.ndim == 0:
            rates = np.full(context.spec.state.n_species, float(rates))
        if rates.shape != (context.spec.state.n_species,):
            raise ValueError(f"{name} must be scalar or have length n_species")
        if np.any((rates < 0.0) | (rates > 1.0)):
            raise ValueError(f"{name} entries must be probabilities")
        return rates

    def _resolve_mutation_matrix(self, context) -> np.ndarray:
        from .ms_interactions import mutation_matrix_from_trait_bins

        if "mutation_matrix" in self.parameters:
            matrix = np.asarray(self.parameters["mutation_matrix"], dtype=float)
        elif "std" in self.parameters:
            matrix = mutation_matrix_from_trait_bins(self.r_b, float(self.parameters["std"]))
        else:
            matrix = np.eye(context.spec.state.n_species)
        expected = (context.spec.state.n_species, context.spec.state.n_species)
        if matrix.shape != expected:
            raise ValueError(f"mutation_matrix must have shape {expected}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("mutation_matrix must contain finite numeric values")
        if np.any(matrix < 0):
            raise ValueError("mutation_matrix entries must be non-negative")
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("mutation_matrix rows must sum to 1")
        return matrix

    def _offspring_by_species(self, lgca, births_by_parent):
        offspring_by_parent = lgca.rng.multinomial(
            np.asarray(births_by_parent, dtype=np.int64),
            self.mutation_matrix,
        )
        return offspring_by_parent.sum(axis=-2)


class NativeMultispeciesGoOrGrowOperator(BirthDeathOperator):
    """Native multispecies no-volume-exclusion go-or-grow operator."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.capacity = None
        self.r_b = None
        self.r_d = None
        self.kappa = None
        self.theta = 0.5
        self.mutation_matrix = None

    def validate(self, context) -> None:
        if context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=False")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species < 2:
            raise ValueError(f"{self.name} requires state.n_species >= 2")
        if context.lgca.restchannels != 1:
            raise ValueError(f"{self.name} requires exactly one rest channel")
        self.capacity = int(self.parameters.get("capacity", getattr(context.lgca, "capacity", context.lgca.K)))
        if self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.r_b = self._species_probability_vector(context, "r_b", 0.2)
        self.r_d = self._species_probability_vector(context, "r_d", 0.01)
        self.kappa = self._species_vector(context, "kappa", 5.0)
        self.theta = float(self.parameters.get("theta", 0.5))
        if not np.isfinite(self.theta):
            raise ValueError("theta must be a finite scalar")
        self.mutation_matrix = self._resolve_mutation_matrix(context)

    def apply(self, context, step: int) -> None:
        from .interactions import tanh_switch

        lgca = context.lgca
        newnodes = np.zeros_like(lgca.nodes)
        velocity_weights = np.full(lgca.velocitychannels, 1 / lgca.velocitychannels)
        n_m = lgca.nodes[..., : lgca.velocitychannels].sum(axis=-1).astype(np.int64)
        n_r = lgca.nodes[..., lgca.velocitychannels :].sum(axis=-1).astype(np.int64)
        total = n_m.sum(axis=-1) + n_r.sum(axis=-1)
        rho = total / self.capacity

        n_m -= lgca.rng.binomial(n_m, self.r_d)
        n_r -= lgca.rng.binomial(n_r, self.r_d)

        switch_prob = tanh_switch(rho[..., None], kappa=self.kappa, theta=self.theta)
        moving_to_rest = lgca.rng.binomial(n_m, switch_prob)
        rest_to_moving = lgca.rng.binomial(n_r, 1 - switch_prob)
        n_m = n_m + rest_to_moving - moving_to_rest
        n_r = n_r + moving_to_rest - rest_to_moving

        post_death_total = n_m.sum(axis=-1) + n_r.sum(axis=-1)
        birth_prob = np.clip(
            self.r_b * (1 - post_death_total[..., None] / self.capacity),
            0,
            1,
        )
        births_by_parent = lgca.rng.binomial(n_r, birth_prob)
        n_r += self._offspring_by_species(lgca, births_by_parent)

        newnodes[..., : lgca.velocitychannels] = lgca.rng.multinomial(
            n_m,
            velocity_weights,
        )
        newnodes[..., lgca.velocitychannels] = n_r.astype(lgca.nodes.dtype)
        lgca.nodes = newnodes

    def _species_probability_vector(self, context, name: str, default: float) -> np.ndarray:
        values = self._species_vector(context, name, default)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} entries must be probabilities")
        return values

    def _species_vector(self, context, name: str, default: float) -> np.ndarray:
        value = self.parameters.get(name, default)
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(context.spec.state.n_species, float(values))
        if values.shape != (context.spec.state.n_species,):
            raise ValueError(f"{name} must be scalar or have length n_species")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite numeric values")
        return values

    def _resolve_mutation_matrix(self, context) -> np.ndarray:
        from .ms_interactions import mutation_matrix_from_trait_bins

        if "mutation_matrix" in self.parameters:
            matrix = np.asarray(self.parameters["mutation_matrix"], dtype=float)
        elif "kappa_std" in self.parameters:
            matrix = mutation_matrix_from_trait_bins(
                self.kappa,
                float(self.parameters["kappa_std"]),
            )
        else:
            matrix = np.eye(context.spec.state.n_species)
        expected = (context.spec.state.n_species, context.spec.state.n_species)
        if matrix.shape != expected:
            raise ValueError(f"mutation_matrix must have shape {expected}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("mutation_matrix must contain finite numeric values")
        if np.any(matrix < 0):
            raise ValueError("mutation_matrix entries must be non-negative")
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("mutation_matrix rows must sum to 1")
        return matrix

    def _offspring_by_species(self, lgca, births_by_parent):
        offspring_by_parent = lgca.rng.multinomial(
            np.asarray(births_by_parent, dtype=np.int64),
            self.mutation_matrix,
        )
        return offspring_by_parent.sum(axis=-2)


class NativeMultispeciesExcitableMediumOperator(BirthDeathOperator):
    """Native two-species volume-exclusion excitable-medium operator."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info=info, parameters=parameters)
        self.beta = 0.05
        self.alpha = 1.0
        self.repetitions = 50

    def validate(self, context) -> None:
        if not context.spec.state.volume_exclusion:
            raise ValueError(f"{self.name} requires state.volume_exclusion=True")
        if context.spec.state.identity_based:
            raise ValueError(f"{self.name} does not support identity-based states")
        if context.spec.state.n_species != 2:
            raise ValueError(f"{self.name} requires state.n_species == 2")
        if context.lgca.restchannels < 1:
            raise ValueError(f"{self.name} requires at least one rest channel")
        self.beta = float(self.parameters.get("beta", 0.05))
        if not np.isfinite(self.beta):
            raise ValueError("beta must be a finite scalar")
        self.alpha = float(self.parameters.get("alpha", 1.0))
        if not np.isfinite(self.alpha) or self.alpha == 0.0:
            raise ValueError("alpha must be a non-zero finite scalar")
        repetitions = self.parameters.get("N", 50)
        if isinstance(repetitions, bool) or int(repetitions) != repetitions or repetitions < 0:
            raise ValueError("N must be a non-negative integer")
        self.repetitions = int(repetitions)

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        n_x = lgca.nodes[..., 1, : lgca.velocitychannels].sum(-1)
        n_y = lgca.nodes[..., 0, lgca.velocitychannels :].sum(-1)

        rho_x = n_x / lgca.velocitychannels
        rho_y = n_y / lgca.restchannels
        p_xp = rho_x**2 * (1 + (rho_y + self.beta) / self.alpha)
        p_xm = rho_x**3 + rho_x * (rho_y + self.beta) / self.alpha
        p_yp = rho_x
        p_ym = rho_y

        dn_y = (lgca.rng.random(n_y.shape) < p_yp).astype(np.int8)
        dn_y -= lgca.rng.random(n_y.shape) < p_ym

        for _ in range(self.repetitions):
            dn_x = (lgca.rng.random(n_x.shape) < p_xp).astype(np.int8)
            dn_x -= lgca.rng.random(n_x.shape) < p_xm
            n_x += dn_x
            n_x = np.clip(n_x, 0, lgca.velocitychannels)
            rho_x = n_x / lgca.velocitychannels
            p_xp = rho_x**2 * (1 + (rho_y + self.beta) / self.alpha)
            p_xm = rho_x**3 + rho_x * (rho_y + self.beta) / self.alpha

        n_y += dn_y
        n_y = np.clip(n_y, 0, lgca.restchannels)

        newnodes = np.zeros_like(lgca.nodes)
        v_idx = np.arange(lgca.velocitychannels)
        r_idx = np.arange(lgca.restchannels)
        newnodes[..., 1, : lgca.velocitychannels] = (
            v_idx < n_x[..., None]
        ).astype(lgca.nodes.dtype)
        newnodes[..., 0, lgca.velocitychannels :] = (
            r_idx < n_y[..., None]
        ).astype(lgca.nodes.dtype)
        newnodes[..., 1, : lgca.velocitychannels] = lgca.rng.permuted(
            newnodes[..., 1, : lgca.velocitychannels],
            axis=-1,
        )
        lgca.nodes = newnodes


class NativeBirthDeathOperator(BirthDeathOperator):
    """Native volume-exclusion birth/death operator."""

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        info = PluginInfo(
            name="birth_death",
            aliases=("birthdeath_native",),
            operator_kind="birth_death",
            backend_families=("classical", "multispecies"),
            parameters={
                "birth_rate": {
                    "default": 0.0,
                    "type_label": "probability",
                    "validator": "probability scalar or per-species vector",
                },
                "death_rate": {
                    "default": 0.0,
                    "type_label": "probability",
                    "validator": "probability scalar or per-species vector",
                },
                "capacity": {
                    "default": "n_species * K",
                    "type_label": "positive integer",
                    "validator": "positive integer",
                },
            },
            conservation_law=ConservationLaw(False, False, False, ("particle number",)),
            port_status="native",
            description="Local birth/death process with volume-exclusion capacity.",
        )
        super().__init__(info=info, parameters=parameters)
        self.birth_rate = None
        self.death_rate = None
        self.capacity = None

    def validate(self, context) -> None:
        if context.spec.state.identity_based:
            raise ValueError("birth_death does not yet support identity-based states")
        if not context.spec.state.volume_exclusion:
            raise ValueError("birth_death currently requires volume exclusion")
        n_species = context.spec.state.n_species
        self.birth_rate = self._rates("birth_rate", n_species)
        self.death_rate = self._rates("death_rate", n_species)
        capacity = self.parameters.get("capacity", n_species * context.lgca.K)
        if int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        for spatial in np.ndindex(lgca.dims):
            coord = tuple(index + lgca.r_int for index in spatial)
            if getattr(lgca, "n_species", 1) > 1:
                lgca.nodes[coord] = self._apply_multispecies_node(lgca.nodes[coord], lgca.rng)
            else:
                lgca.nodes[coord] = self._apply_species_node(
                    lgca.nodes[coord], self.birth_rate[0], self.death_rate[0], lgca.rng, self.capacity
                )

    def _rates(self, name: str, n_species: int) -> np.ndarray:
        value = self.parameters.get(name, 0.0)
        rates = np.asarray(value, dtype=float)
        if rates.ndim == 0:
            rates = np.full(n_species, float(rates))
        if rates.shape != (n_species,):
            raise ValueError(f"{name} must be scalar or have length n_species")
        if np.any((rates < 0.0) | (rates > 1.0)):
            raise ValueError(f"{name} entries must be probabilities")
        return rates

    def _apply_multispecies_node(self, node, rng):
        new_node = node.copy()
        remaining_capacity = self.capacity
        for species in range(new_node.shape[0]):
            remaining_capacity -= int(new_node[species].sum())

        for species in range(new_node.shape[0]):
            before = int(new_node[species].sum())
            new_node[species] = self._apply_death(new_node[species], self.death_rate[species], rng)
            remaining_capacity += before - int(new_node[species].sum())

        for species in range(new_node.shape[0]):
            before = int(new_node[species].sum())
            new_node[species] = self._apply_birth(
                new_node[species], self.birth_rate[species], rng, remaining_capacity
            )
            remaining_capacity -= int(new_node[species].sum()) - before
        return new_node

    def _apply_species_node(self, node, birth_rate, death_rate, rng, capacity):
        after_death = self._apply_death(node.copy(), death_rate, rng)
        remaining_capacity = capacity - int(after_death.sum())
        return self._apply_birth(after_death, birth_rate, rng, remaining_capacity)

    @staticmethod
    def _apply_death(node, death_rate, rng):
        occupied = np.flatnonzero(node)
        if occupied.size == 0 or death_rate == 0.0:
            return node
        survivors = rng.random(occupied.size) >= death_rate
        node[occupied[~survivors]] = False
        return node

    @staticmethod
    def _apply_birth(node, birth_rate, rng, remaining_capacity):
        occupied = np.flatnonzero(node)
        empty = np.flatnonzero(~node)
        if occupied.size == 0 or empty.size == 0 or remaining_capacity <= 0 or birth_rate == 0.0:
            return node
        n_births = rng.binomial(occupied.size, birth_rate)
        n_births = min(int(n_births), int(remaining_capacity), empty.size)
        if n_births > 0:
            chosen = rng.choice(empty, size=n_births, replace=False)
            node[chosen] = True
        return node
