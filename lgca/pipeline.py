"""Composable interaction pipeline for LGCA models."""

from __future__ import annotations

import time
from ._warnings import warn_user
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .base import _sampling_totals
from .identity_kernels import inherit_missing_properties
from .plugins import (
    BirthDeathOperator,
    ConservationLaw,
    InteractionOperator,
    PhenotypeSwitchOperator,
    PluginInfo,
    ReorientationOperator,
    create_plugin,
    resolve_operator_capacity,
)


@dataclass(frozen=True)
class BirthDeathSpec:
    """A registered interaction that creates or removes cells.

    Equivalent to ``{"name": name, "parameters": parameters}`` in
    :attr:`InteractionPipelineSpec.operators`.

    Attributes
    ----------
    name : str
        Registered name, e.g. ``"birth_death"`` or ``"classical.go_or_grow"``;
        see :func:`lgca.plugins.list_plugins`.
    parameters : mapping, default={}
        Interaction parameters; omitted ones take their documented defaults.
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhenotypeSwitchSpec:
    """A registered interaction in the phenotype-switching phase of a time step.

    ``PhenotypeSwitchSpec(name="phenotype_switch", parameters={"rates": R})``
    lets cells of a multispecies LGCA change their species: each cell of
    species ``a`` becomes species ``b`` with probability ``R[a][b]`` per time
    step (the diagonal is ignored). The number of cells at a node is
    conserved; a switch into a species whose channels at that node are full is
    rejected, and when a cell switches, the node's cells are redistributed over
    its channels. A phenotype switch needs several species (or an
    identity-based model); moving cells between velocity and rest channels of
    one species, as ``classical.go_or_rest`` does, is a reorientation.

    Attributes
    ----------
    name : str
        Registered name, e.g. ``"phenotype_switch"``.
    parameters : mapping, default={}
        Interaction parameters.
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReorientationTermSpec:
    """One directional cue in a :class:`ReorientationSpec`.

    Each term scores every candidate channel state ``s'`` of a node. The
    sampler adds the scores weighted by ``beta``; see :class:`ReorientationSpec`.
    ``J(s')`` is the flux of the candidate state (the sum of the velocity
    vectors of its occupied channels) and gradients are in lattice units.

    ``"random_walk"`` (alias ``"uniform"``)
        Score 0: all states are equally likely.
    ``"persistent_walk"`` (alias ``"persistent_motion"``)
        ``J(s) · J(s')``: cells keep the direction they had at this node.
    ``"polar_alignment"``
        ``J_nb · J(s')``, with ``J_nb`` the flux of the neighbouring nodes:
        cells move in the direction of their neighbours.
    ``"nematic_alignment"`` (alias ``"nematic"``)
        Rewards sharing an axis with neighbouring cells; opposite directions
        count the same.
    ``"aggregation"``
        ``∇ρ · J(s')``: cells move up the gradient of the cell density ``ρ``.
    ``"chemotaxis"``
        ``∇f · J(s')`` for the scalar field named by ``parameters["field"]``
        in :attr:`StateSpec.fields <lgca.model.StateSpec.fields>`.
    ``"contact_guidance"``
        ``Σ (d · c_i)²`` over occupied velocity channels ``i``: cells move along
        the axis of the director field ``d`` named by ``parameters["field"]``
        (default ``"director"``).
    ``"resting_bias"``
        Number of cells in rest channels: cells prefer to rest.

    Attributes
    ----------
    name : str
        One of the names above; :func:`list_reorientation_terms` lists them.
    beta : float, default=1.0
        Weight (sensitivity) of the term; 0 switches it off, negative values
        reverse the preference.
    parameters : mapping, default={}
        ``{"field": name}`` for ``chemotaxis`` and ``contact_guidance``.
    species : int, optional
        Apply the term only to cells of this species (zero-based index).
    """

    name: str
    beta: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    species: int | None = None


@dataclass(frozen=True)
class ReorientationSpec:
    """Stochastic reorientation that combines several directional cues.

    At every node, the cells choose a new channel state ``s'`` among all states
    with the same number of cells, with probability
    ``P(s') ∝ exp(Σ_k beta_k · G_k(s'))``, where ``G_k`` are the scores of the
    terms. All terms thus act in one decision instead of one after another.
    The number of cells is conserved. Supported for classical models with
    volume exclusion, including several species.

    Attributes
    ----------
    terms : sequence of ReorientationTermSpec, default=()
        The cues. Without terms, the reorientation is a random walk.
    sampler : str, default="boltzmann"
        The sampling rule; only ``"boltzmann"`` is available.
    parameters : mapping, default={}
        Reserved; must be empty.

    Examples
    --------
    >>> ReorientationSpec(terms=[
    ...     ReorientationTermSpec(name="polar_alignment", beta=1.5),
    ...     ReorientationTermSpec(name="chemotaxis", beta=5, parameters={"field": "signal"}),
    ... ])  # doctest: +SKIP
    """

    terms: Sequence[ReorientationTermSpec] = field(default_factory=tuple)
    sampler: str = "boltzmann"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionPipelineSpec:
    """The interactions of one time step, followed by propagation.

    Every time step applies the operators in the order they are listed and
    then moves the cells along their velocity channels (propagation). The
    order is part of the model: birth before reorientation is a different
    model from reorientation before birth. Two reorientation operators in a
    row are two independent random decisions; directional cues that should
    compete in one decision belong as terms of one :class:`ReorientationSpec`.

    Attributes
    ----------
    operators : sequence, default=()
        Each entry is one of

        - a mapping ``{"name": ..., "parameters": {...}}`` naming a registered
          interaction (see :func:`lgca.plugins.list_plugins`);
        - a :class:`ReorientationSpec` combining directional cues;
        - a :class:`BirthDeathSpec` or :class:`PhenotypeSwitchSpec`;
        - an :class:`~lgca.operator_base.InteractionOperator` instance (Python
          only; it cannot be saved to a model file).
    propagation : bool or str, default="default"
        ``"default"`` or ``True`` moves the cells after the interactions;
        ``False``, ``None``, ``"none"`` or ``"disabled"`` keeps them in place,
        e.g. to test an interaction on its own.
    allow_custom_order : None
        Deprecated and ignored: operators always run in the listed order.
    """

    operators: Sequence[Any] = field(default_factory=tuple)
    propagation: str | bool = "default"
    allow_custom_order: bool | None = None

    def __post_init__(self):
        if self.allow_custom_order is not None:
            warn_user("allow_custom_order is deprecated and ignored: operators always run "
                      "in the order they are listed", DeprecationWarning)


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
            deps = ",".join(sorted(operator.dependencies())) or "-"
            outputs = ",".join(sorted(operator.outputs())) or "-"
            suffix += f"; inputs={deps}; outputs={outputs}"
            suffix += ")"
            parts.append(f"{operator.name}{suffix}")
        if self.propagation not in (False, None, "none", "disabled"):
            parts.append("Propagation (deterministic)")
        return " -> ".join(parts)

    def execute_step(
        self,
        context,
        step: int,
        timing: dict[tuple[str, str], dict[str, Any]] | None = None,
        timing_trace: list[dict[str, Any]] | None = None,
        timing_trace_limit: int = 0,
    ) -> None:
        lgca = context.lgca
        lgca._validate_evolution()
        for operator in self.operators:
            if "boundary_nodes" in operator.dependencies():
                lgca.apply_boundaries()
                lgca.update_dynamic_fields()
            start = time.perf_counter()
            operator.apply(context, step)
            if "nodes" in operator.outputs():
                lgca.update_dynamic_fields()
            elapsed = time.perf_counter() - start
            if timing is not None:
                _record_timing(timing, operator.name, operator.operator_kind, elapsed)
            if timing_trace is not None and len(timing_trace) < timing_trace_limit:
                timing_trace.append(
                    {"step": step, "name": operator.name,
                     "kind": operator.operator_kind, "elapsed_seconds": elapsed}
                )
        lgca.apply_boundaries()
        if self.propagation not in (False, None, "none", "disabled"):
            start = time.perf_counter()
            lgca.propagation()
            elapsed = time.perf_counter() - start
            if timing is not None:
                _record_timing(timing, "Propagation", "propagation", elapsed)
            if timing_trace is not None and len(timing_trace) < timing_trace_limit:
                timing_trace.append(
                    {"step": step, "name": "Propagation", "kind": "propagation",
                     "elapsed_seconds": elapsed}
                )
            lgca.apply_boundaries()
        lgca.update_dynamic_fields()


def _record_timing(timing, name: str, kind: str, elapsed: float) -> None:
    """Update a constant-space timing summary for one pipeline phase."""
    key = (name, kind)
    aggregate = timing.get(key)
    if aggregate is None:
        timing[key] = {
            "name": name,
            "kind": kind,
            "count": 1,
            "total_seconds": elapsed,
            "min_seconds": elapsed,
            "max_seconds": elapsed,
        }
        return
    aggregate["count"] += 1
    aggregate["total_seconds"] += elapsed
    aggregate["min_seconds"] = min(aggregate["min_seconds"], elapsed)
    aggregate["max_seconds"] = max(aggregate["max_seconds"], elapsed)


class NativeOnlyPropagationOperator(InteractionOperator):
    """Native no-op marker for propagation-only legacy compatibility."""

    def apply(self, context, step: int) -> None:
        pass

    def outputs(self) -> set[str]:
        return set()


def compile_pipeline(spec: InteractionPipelineSpec | None, context) -> CompiledPipeline:
    """Compile a pipeline spec into executable operators."""

    spec = spec or InteractionPipelineSpec()
    if not (spec.propagation is None or isinstance(spec.propagation, bool)
            or isinstance(spec.propagation, str) and spec.propagation in {"default", "none", "disabled"}):
        raise ValueError("dynamics.propagation must be a boolean, null, 'default', 'none', or 'disabled'")
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
    _reject_single_species_phenotype_switch(operators, context)
    growth = [operator for operator in operators if operator.operator_kind == "birth_death"]
    shared_property_operators = {"ib.birth", "ib.birthdeath", "ib.birthdeath_discrete",
                                "ib.go_or_grow", "ib.go_and_grow_mutations"}
    if context.spec.state.identity_based and len(growth) > 1:
        if any(operator.name not in shared_property_operators for operator in growth):
            raise ValueError(
                "Identity growth composition requires a shared daughter-property lifecycle; "
                "currently supported for ib.birth, ib.birthdeath, ib.birthdeath_discrete, "
                "ib.go_or_grow and ib.go_and_grow_mutations. Use one growth operator "
                "for other identity backends."
            )
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


def _reject_single_species_phenotype_switch(operators: Sequence[InteractionOperator], context) -> None:
    """A classical phenotype is a species, so one species has nothing to switch to."""
    state = context.spec.state
    if state.identity_based or state.n_species != 1:
        return
    for index, operator in enumerate(operators):
        if operator.operator_kind == "phenotype_switch":
            raise ValueError(
                f"dynamics.operators[{index}] {operator.name!r} is a phenotype switch, which "
                "moves cells between species; this classical model has one species. Set "
                "state.n_species to the number of phenotypes, or use an identity-based model "
                "to change individual cell parameters."
            )


def _softmax_last_axis(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


_MAX_CANDIDATE_BATCH_BYTES = 32 * 1024 ** 2


def _candidate_batches(mask, candidates):
    """Bound score/softmax/cumulative temporaries conservatively to 32 MiB."""
    bytes_per_site = int(candidates) * np.dtype(float).itemsize * 6
    batch_size = max(1, _MAX_CANDIDATE_BATCH_BYTES // bytes_per_site)
    if bytes_per_site > _MAX_CANDIDATE_BATCH_BYTES:
        raise ValueError(f"One candidate calculation requires about {bytes_per_site:,} bytes; "
                         "reduce channels or use a non-enumerating interaction")
    sites = np.flatnonzero(mask)
    for start in range(0, len(sites), batch_size):
        yield np.unravel_index(sites[start:start + batch_size], mask.shape)


class _ReorientationTerm:
    def __init__(self, spec: ReorientationTermSpec):
        self.name = spec.name
        if self.name == "alignment":
            warn_user("The composed 'alignment' alias means nematic_alignment; "
                          "use 'nematic_alignment' or 'polar_alignment' explicitly", DeprecationWarning)
        if isinstance(spec.beta, (bool, str)) or np.asarray(spec.beta).ndim != 0 or not np.isfinite(spec.beta):
            raise ValueError("beta must be a finite numeric scalar")
        self.beta = float(spec.beta)
        self.parameters = dict(spec.parameters)
        self.species = spec.species
        if self.species is not None and (isinstance(self.species, bool)
                or not isinstance(self.species, (int, np.integer)) or self.species < 0):
            raise ValueError("species must be a nonnegative integer index")
        allowed = {"field"} if self.name in {"chemotaxis", "contact_guidance"} else set()
        unknown = set(self.parameters) - allowed
        if unknown:
            raise ValueError(f"parameters contains unknown keys: {sorted(unknown)}")

    def validate(self, context) -> None:
        if self.species is not None and self.species >= context.spec.state.n_species:
            raise ValueError(f"species index {self.species} exceeds state.n_species")

    def score(self, candidates, node, lgca, coord):
        coords = tuple(np.asarray([index]) for index in coord)
        return np.broadcast_to(self.score_batch(_candidate_features(candidates, lgca), lgca, coords),
                               (1, len(candidates)))[0]

    def score_batch(self, features, lgca, coords):
        return 0.0

    def prepare(self, lgca, source_channels):
        """Prepare spatial fields once from the frozen operator input."""

    def dependencies(self) -> set[str]:
        return set()


class _UniformTerm(_ReorientationTerm):
    pass


class _RestingBiasTerm(_ReorientationTerm):
    def score_batch(self, features, lgca, coords):
        return features["rest"]


def _physical_field_gradient(lgca, field):
    """Differentiate a prescribed field in physical lattice coordinates.

    Centered interior and one-sided edge differences do not wrap the prescribed
    field across particle boundaries. Singleton axes have zero derivative.
    The coordinate Jacobian removes the hexagonal row staggering and spacing.
    """
    ndim = len(lgca.dims)
    coordinates = [getattr(lgca, name) for name in ("xcoords", "ycoords", "zcoords")[:ndim]]
    derivatives = np.zeros(field.shape + (ndim,))
    jacobian = np.zeros(field.shape + (ndim, ndim))
    for axis, size in enumerate(lgca.dims):
        if size == 1:
            jacobian[..., axis, axis] = 1
            continue
        derivatives[..., axis] = np.gradient(field, axis=axis)
        for component, coordinate in enumerate(coordinates):
            jacobian[..., axis, component] = np.gradient(coordinate, axis=axis)
    return np.linalg.solve(jacobian, derivatives[..., None])[..., 0]


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
        self.gradient = _physical_field_gradient(context.lgca, field)

    def prepare(self, lgca, source_channels):
        field = np.asarray(getattr(lgca, self.field_name), dtype=float)
        field = field[lgca.nonborder]
        if field.shape != tuple(lgca.dims) or not np.isfinite(field).all():
            raise ValueError(f"Field {self.field_name!r} must contain finite scalar values on the lattice")
        self.gradient = _physical_field_gradient(lgca, field)

    def score_batch(self, features, lgca, coords):
        spatial = tuple(index - lgca.r_int for index in coords)
        return self.gradient[spatial] @ features["flux"].T

    def dependencies(self) -> set[str]:
        return set() if not self.field_name else {self.field_name}


class _NematicAlignmentTerm(_ReorientationTerm):
    def prepare(self, lgca, source_channels):
        self.neighbor_channels = lgca.nb_sum(
            source_channels[..., : lgca.velocitychannels].astype(np.int64)
        )

    def score_batch(self, features, lgca, coords):
        return self.neighbor_channels[coords] @ features["nematic"].T

    def dependencies(self) -> set[str]:
        return {"boundary_nodes"}


class _PersistentWalkTerm(_ReorientationTerm):
    def prepare(self, lgca, source_channels):
        self.local_flux = source_channels[..., : lgca.velocitychannels] @ lgca.c.T

    def score_batch(self, features, lgca, coords):
        return self.local_flux[coords] @ features["flux"].T


class _PolarAlignmentTerm(_PersistentWalkTerm):
    def prepare(self, lgca, source_channels):
        neighbors = lgca.nb_sum(source_channels[..., :lgca.velocitychannels].astype(np.int64))
        self.local_flux = neighbors @ lgca.c.T

    def dependencies(self) -> set[str]:
        return {"boundary_nodes"}


class _AggregationTerm(_ReorientationTerm):
    def prepare(self, lgca, source_channels):
        density = source_channels.sum(axis=-1)
        self.gradient = lgca.gradient(density)

    def score_batch(self, features, lgca, coords):
        return self.gradient[coords] @ features["flux"].T

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

    def prepare(self, lgca, source_channels):
        field = np.asarray(getattr(lgca, self.field_name), dtype=float)[lgca.nonborder]
        expected = tuple(lgca.dims) + (lgca.c.shape[0],)
        if field.shape != expected or not np.isfinite(field).all():
            raise ValueError(f"Field {self.field_name!r} must contain finite vectors of shape {expected}")
        norm = np.linalg.norm(field, axis=-1, keepdims=True)
        self.director = np.divide(field, norm, out=np.zeros_like(field), where=norm > 0)

    def score_batch(self, features, lgca, coords):
        spatial = tuple(index - lgca.r_int for index in coords)
        alignment = self.director[spatial] @ lgca.c
        return alignment**2 @ features["channels"].T

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
    "polar_alignment": _PolarAlignmentTerm,
    "random_walk": _UniformTerm,
    "uniform": _UniformTerm,
    "resting_bias": _RestingBiasTerm,
}


def list_reorientation_terms() -> tuple[str, ...]:
    """Return the supported reorientation-term names in deterministic order."""

    return tuple(sorted(_REORIENTATION_TERMS))


def _candidate_features(candidates, lgca):
    """Cache candidate-only quantities for one occupancy group."""
    feature_bytes = len(candidates) * (2 * lgca.velocitychannels + lgca.c.shape[0] + 1) * 8
    if feature_bytes > _MAX_CANDIDATE_BATCH_BYTES:
        raise ValueError(f"Candidate features require {feature_bytes:,} bytes; "
                         "reduce channels or use a non-enumerating interaction")
    channels = candidates[:, :lgca.velocitychannels].astype(float)
    return {"channels": channels, "flux": channels @ lgca.c.T,
            "nematic": channels @ (lgca.c.T @ lgca.c)**2,
            "rest": candidates[:, lgca.velocitychannels:].sum(axis=1)}


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
        for index, term in enumerate(self.terms):
            try:
                term.validate(context)
            except ValueError as exc:
                raise ValueError(f"terms[{index}] {exc}") from exc

    def setup(self, context) -> None:
        lgca = context.lgca
        if not hasattr(lgca, "permutations") and not hasattr(lgca, "_permutation_cache"):
            lgca.calc_permutations()

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        lgca._reorientation_source_nodes = lgca.nodes.copy()
        try:
            source_channels = lgca._reorientation_source_nodes
            if getattr(lgca, "n_species", 1) > 1:
                source_channels = source_channels.sum(axis=-2)
            for term in self.terms:
                term.prepare(lgca, source_channels)
            self._sample_batches(lgca)
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
            try:
                terms.append(term_cls(term_spec))
            except (TypeError, ValueError) as exc:
                raise ValueError(f".terms[{index}] {exc}") from exc
        if not terms:
            terms.append(_UniformTerm(ReorientationTermSpec(name="random_walk")))
        return terms

    def _sample_batches(self, lgca):
        nodes = lgca._reorientation_source_nodes[lgca.nonborder]
        counts = nodes.sum(axis=-1)
        sampled = np.zeros_like(nodes)
        # Preserve the former spatial-then-species categorical draw order,
        # including full sites, while grouping computation by occupancy.
        draws = np.zeros(counts.shape)
        draws[counts > 0] = lgca.rng.random(np.count_nonzero(counts))
        ndim = len(lgca.dims)
        multispecies = getattr(lgca, "n_species", 1) > 1
        for count in np.unique(counts[counts > 0]):
            candidates = lgca.get_permutations(int(count))
            features = _candidate_features(candidates, lgca)
            for indices in _candidate_batches(counts == count, len(candidates)):
                coords = tuple(axis + lgca.r_int for axis in indices[:ndim])
                scores = np.zeros((len(indices[0]), len(candidates)))
                species = indices[-1] if multispecies else np.zeros(len(indices[0]), dtype=int)
                for term in self.terms:
                    if term.beta == 0:
                        continue
                    contribution = term.beta * term.score_batch(features, lgca, coords)
                    if term.species is not None:
                        contribution = contribution * (species == term.species)[:, None]
                    scores += contribution
                probabilities = _softmax_last_axis(scores)
                cumulative = np.cumsum(probabilities, axis=-1)
                cumulative /= cumulative[:, -1:]
                choices = (cumulative <= draws[indices][:, None]).sum(axis=-1)
                sampled[indices] = candidates[choices]
        lgca.nodes[lgca.nonborder] = sampled


from .classical_operators import NativeClassicalRandomWalkOperator


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
        self.capacity = int(resolve_operator_capacity(context, self.parameters, 8))
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
        from .identity_kernels import apply_nove_identity_birth

        apply_nove_identity_birth(
            context.lgca,
            capacity=self.capacity,
            a_max=self.a_max,
            std=self.std,
            channel_weights=self.channel_weights,
            r_d=self.r_d if self.mode == "birthdeath" else None,
        )

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
        self.capacity = int(resolve_operator_capacity(context, self.parameters, 8))
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
        self.capacity = int(resolve_operator_capacity(context, self.parameters, 8))
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
        self.capacity = int(resolve_operator_capacity(context, self.parameters, 8))
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
        self.capacity = int(resolve_operator_capacity(context, self.parameters, 512))
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
        from .identity_kernels import apply_identity_birth

        apply_identity_birth(context.lgca, a_max=self.a_max, std=self.std)


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
        from .identity_kernels import apply_identity_birthdeath

        apply_identity_birthdeath(context.lgca, r_d=self.r_d, a_max=self.a_max, std=self.std)

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
                    inherit_missing_properties(lgca, label)
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
                    inherit_missing_properties(lgca, label)
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
                    inherit_missing_properties(lgca, cell)

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
            for batch in _candidate_batches(mask, len(lgca.get_permutations(n_particles))):
                weights = _softmax_last_axis(self.beta * (flux[batch] @ j))
                cumw = weights.cumsum(axis=1)
                rnd = lgca.rng.random(len(batch[0]))
                ind = (rnd[:, None] < cumw).argmax(axis=1)
                nb_nodes[batch] = lgca.get_permutations(n_particles)[ind]

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
            for batch in _candidate_batches(mask, len(lgca.get_permutations(n_particles))):
                weights = _softmax_last_axis(
                    self.beta * np.einsum("nij,pij->np", tensors[batch], si)
                )
                cumw = weights.cumsum(axis=1)
                rnd = lgca.rng.random(len(batch[0]))
                ind = (rnd[:, None] < cumw).argmax(axis=1)
                nb_nodes[batch] = lgca.get_permutations(n_particles)[ind]

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


class NativeClassicalGoOrRestOperator(ReorientationOperator):
    """Moving/resting redistribution of each node's cells, matching the legacy rule."""

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
    Each particle retains its source capacity until processed in random order.
    A sampled switch into a full species is rejected and the particle stays
    in its source species, even when its configured stay probability is zero.
    Collisions therefore cannot merge particles or create forbidden switches.
    """

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        info = PluginInfo(
            name="phenotype_switch",
            aliases=("species_switch",),
            operator_kind="phenotype_switch",
            backend_families=("multispecies",),
            parameters={
                "rates": {
                    "required": True,
                    "type_label": "array",
                    "description": "Off-diagonal phenotype transition probabilities.",
                }
            },
            conservation_law=ConservationLaw(True, False, False, ("species identity", "channel occupancy")),
            port_status="native",
            description="Atomic phenotype transition with collision-safe channel resampling.",
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
            counts = state.sum(axis=1).astype(int)
            targets = np.empty(sources.size, dtype=int)
            changed = False
            for index, source in enumerate(sources):
                target = int(rng.choice(n_species, p=probabilities[source]))
                if target != source and counts[target] >= n_channels:
                    target = int(source)
                targets[index] = target
                counts[source] -= 1
                counts[target] += 1
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

    def dependencies(self) -> set[str]:
        return {"boundary_nodes", "cell_density"}

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
        density = _sampling_totals(lgca.nodes[lgca.nonborder])
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
        density = _sampling_totals(lgca.nodes[lgca.nonborder])
        newnodes[lgca.nonborder] = lgca.rng.multinomial(density, weights)
        lgca.nodes = newnodes


class NativeNoVEGoOrRestOperator(ReorientationOperator):
    """Moving/resting redistribution of each node's cells without volume exclusion."""

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
        n_m = _sampling_totals(nb_nodes[..., : lgca.velocitychannels])
        n_r = _sampling_totals(nb_nodes[..., lgca.velocitychannels :])
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
        n_m = _sampling_totals(nb_nodes[..., : lgca.velocitychannels])
        n_r = _sampling_totals(nb_nodes[..., lgca.velocitychannels :])
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
        from .ms_interactions import _sample_offspring_by_species

        return _sample_offspring_by_species(
            lgca.rng, births_by_parent, self.mutation_matrix
        )


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
        from .ms_interactions import _sample_offspring_by_species

        return _sample_offspring_by_species(
            lgca.rng, births_by_parent, self.mutation_matrix
        )


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
    """Native volume-exclusion birth/death operator.

    Deaths precede births. Species compete for shared birth capacity in a fresh
    uniformly random order at each site, without a species-index priority.
    """

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
        canonical = context.spec.state.capacity
        capacity = self.parameters.get("capacity", canonical if canonical is not None else n_species * context.lgca.K)
        if int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        if getattr(lgca, "n_species", 1) == 1:
            interior = lgca.nodes[lgca.nonborder]
            lgca.nodes[lgca.nonborder] = self._apply_single_species_lattice(
                interior,
                self.birth_rate[0],
                self.death_rate[0],
                lgca.rng,
                self.capacity,
            )
            return
        interior = lgca.nodes[lgca.nonborder]
        lgca.nodes[lgca.nonborder] = self._apply_multispecies_lattice(interior, lgca.rng)

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

    def _apply_multispecies_lattice(self, nodes, rng):
        """Apply turnover to all sites at once; nodes have shape ``dims + (n_species, K)``.

        At every site, cells die independently, then each species draws its
        births from a binomial distribution in its cell number. Species claim
        the site's remaining capacity in a random order, and births occupy
        uniformly chosen free channels of their species.
        """
        n_species = nodes.shape[-2]
        after_death = nodes.copy()
        if np.any(self.death_rate > 0.0):
            after_death &= rng.random(nodes.shape) >= self.death_rate[:, None]
        counts = after_death.sum(axis=-1)
        if not np.any(self.birth_rate > 0.0):
            return after_death

        wanted = np.minimum(rng.binomial(counts, self.birth_rate), nodes.shape[-1] - counts)
        remaining = np.maximum(self.capacity - counts.sum(axis=-1), 0)
        births = np.zeros_like(wanted)
        order = np.argsort(rng.random(counts.shape), axis=-1)
        for position in range(n_species):
            species = order[..., position:position + 1]
            granted = np.minimum(np.take_along_axis(wanted, species, axis=-1)[..., 0], remaining)
            np.put_along_axis(births, species, granted[..., None], axis=-1)
            remaining -= granted
        if not np.any(births):
            return after_death

        scores = rng.random(nodes.shape)
        scores[after_death] = np.inf
        ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1)
        return after_death | (ranks < births[..., None])

    @staticmethod
    def _apply_single_species_lattice(nodes, birth_rate, death_rate, rng, capacity):
        """Apply independent local turnover without a Python loop over sites."""

        after_death = nodes.copy()
        if death_rate > 0.0:
            after_death &= rng.random(nodes.shape) >= death_rate
        particle_counts = after_death.sum(axis=-1)
        if birth_rate == 0.0:
            return after_death

        births = rng.binomial(particle_counts, birth_rate)
        empty_counts = (~after_death).sum(axis=-1)
        remaining_capacity = np.maximum(capacity - particle_counts, 0)
        births = np.minimum(births, np.minimum(empty_counts, remaining_capacity))
        if not np.any(births):
            return after_death

        scores = rng.random(nodes.shape)
        scores[after_death] = np.inf
        order = np.argsort(scores, axis=-1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(nodes.shape[-1]), axis=-1)
        after_death |= ranks < births[..., None]
        return after_death
