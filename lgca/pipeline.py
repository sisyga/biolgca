"""Composable interaction pipeline for LGCA models."""

from __future__ import annotations

import difflib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._warnings import warn_user
from .lattice_state import channel_mask, occupations
from .plugins import (
    ConservationLaw,
    InteractionOperator,
    PluginInfo,
    ReorientationOperator,
    create_plugin,
)


@dataclass(frozen=True)
class BirthDeathSpec:
    """A registered interaction that creates or removes cells.

    Equivalent to ``{"name": name, "parameters": parameters}`` in
    :attr:`InteractionPipelineSpec.operators`.

    Attributes
    ----------
    name : str
        Registered name, e.g. ``"birth_death"`` or ``"go_or_grow.growth"``;
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
    step (the diagonal is ignored); the rates may respond to cues of the
    surroundings (:mod:`lgca.switching`). The number of cells at a node is
    conserved; a switch into a species without a free channel at the node
    fails, and switched cells go to free channels of their new species. A
    phenotype switch needs several species (identity-based models change
    traits with ``trait_switch``); moving cells between velocity and rest
    channels of one species, as ``go_or_rest`` does, is a reorientation.

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
        cells move in the direction of their neighbours. Parameters
        ``include_center`` and ``normalize`` (divide by the number of cells).
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
    ``"resting"``
        A cell alone at its node rests with the switching probability
        ``parameters["probability"]`` (:mod:`lgca.switching`; default the
        go-or-grow switch of the density): go-or-rest as a term.

    New terms are written with :func:`lgca.reorientation_term`; the built-in
    terms above are defined the same way in :mod:`lgca.builtin_rules`, and
    each of them is also an operator of its own (one cue), e.g.
    ``{"name": "chemotaxis", "parameters": {"beta": 2.0, "field": "signal"}}``.

    Attributes
    ----------
    name : str
        One of the names above or of a term defined with
        :func:`lgca.reorientation_term`; :func:`list_reorientation_terms`
        lists them.
    beta : float, default=1.0
        Weight (sensitivity) of the term; 0 switches it off, negative values
        reverse the preference.
    parameters : mapping, default={}
        ``{"field": name}`` for ``chemotaxis`` and ``contact_guidance``.
    species : int, optional
        Apply the term only to cells of this species (zero-based index).
    sensed_species : int or list of int, optional
        The species whose cells the term senses, for terms computed from the
        cells, e.g. ``polar_alignment`` with ``sensed_species=1``: align with
        the cells of species 1 only. Default: all cells.
    trait : str, optional
        Identity-based models: the name of a cell trait that scales the term
        for every cell, e.g. an alignment strength per cell. A cell's score
        in channel ``i`` is then ``beta * trait * w_i``.
    """

    name: str
    beta: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    species: int | None = None
    trait: str | None = None
    sensed_species: int | Sequence[int] | None = None


@dataclass(frozen=True)
class ReorientationSpec:
    """Stochastic reorientation that combines several directional cues.

    At every node, the cells choose a new channel state ``s'`` among all states
    with the same number of cells, with probability
    ``P(s') ∝ exp(Σ_k beta_k · G_k(s'))``, where ``G_k`` are the scores of the
    terms. All terms thus act in one decision instead of one after another.
    The number of cells of each species is conserved.

    This is the rule with volume exclusion. Without it, every cell chooses
    its channel ``i`` independently with ``P(i) ∝ exp(Σ_k beta_k · w_ki)``,
    where ``w_ki`` is the score of one cell in channel ``i``. Identity-based
    models update their cell numbers in the same way and then place the
    node's cells on the occupied channels at random. Supported for every
    model family, with one or several species.

    Terms with a ``trait`` give every cell of an identity-based model its own
    weight, so the cells of a node are no longer interchangeable. Without
    volume exclusion each cell still chooses its channel on its own, exactly.
    With volume exclusion, the labelled state of every node is sampled with
    a Metropolis chain that starts from a random arrangement of the node's
    cells and proposes to swap the contents of a channel holding a cell with
    another channel, ``sweeps * K`` times per node, so the chain grows with
    the number of channels ``K``. With a strong cue, ten sweeps reached the
    Boltzmann distribution within sampling noise from 5 channels (square) to
    27 (Moore) and with 6 rest channels on hex; five did not.

    Attributes
    ----------
    terms : sequence of ReorientationTermSpec, default=()
        The cues. Without terms, the reorientation is a random walk.
    sampler : str, default="boltzmann"
        The sampling rule; only ``"boltzmann"`` is available.
    parameters : mapping, default={}
        ``{"species": [0]}``: only the cells of these species move; the
        others keep their channels. ``{"channels": "velocity"}``: only the
        cells in these channels move, and only among them.
        ``{"sweeps": 10}``: length of the Metropolis chain for terms with a
        ``trait`` in identity-based models with volume exclusion.

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
        if self.propagation not in (False, None, "none", "disabled") and getattr(lgca, "enable_propagation", True):
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
    pipeline = CompiledPipeline(operators=operators, propagation=spec.propagation)
    pipeline.setup(context)
    return pipeline


def _compile_operator(spec) -> InteractionOperator:
    if isinstance(spec, InteractionOperator):
        return spec
    if isinstance(spec, ReorientationSpec):
        return BoltzmannReorientationOperator(spec)
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
        if isinstance(spec.beta, (bool, str)) or np.asarray(spec.beta).ndim != 0 or not np.isfinite(spec.beta):
            raise ValueError("beta must be a finite numeric scalar")
        self.beta = float(spec.beta)
        self.parameters = dict(spec.parameters)
        self.species = spec.species
        if self.species is not None and (isinstance(self.species, bool)
                or not isinstance(self.species, (int, np.integer)) or self.species < 0):
            raise ValueError("species must be a nonnegative integer index")
        self.trait = spec.trait
        if self.trait is not None and (not isinstance(self.trait, str) or not self.trait):
            raise ValueError("trait must be the name of a cell trait")
        self.sensed_species = spec.sensed_species

    def validate(self, context) -> None:
        from .lattice_state import _species_indices

        if self.species is not None and self.species >= context.spec.state.n_species:
            raise ValueError(f"species index {self.species} exceeds state.n_species")
        if self.sensed_species is not None:
            _species_indices(self.sensed_species, context.spec.state.n_species)
        if self.trait is not None:
            if not context.spec.state.identity_based:
                raise ValueError(f"trait={self.trait!r} needs an identity-based model, whose cells "
                                 "have traits")
            if self.trait not in context.lgca.props:
                raise ValueError(f"the cells have no trait {self.trait!r}; declare it with "
                                 f"StateSpec(traits={{{self.trait!r}: ...}})")

    def score(self, candidates, node, lgca, coord):
        """Scores of ``candidates`` (channel states) at the padded coordinate ``coord``."""
        spatial = tuple(index - lgca.r_int for index in coord)
        return np.asarray(candidates, dtype=float) @ self.weights[spatial]

    def prepare(self, lgca):
        """Compute the channel weights once per step from the state before the reorientation."""

    def dependencies(self) -> set[str]:
        return set()


_COUPLINGS = ("flux", "nematic", "rest", "channels")


class _FieldTerm(_ReorientationTerm):
    """A term whose score couples a field of the lattice state to the candidate state.

    The field comes from ``definition.function(state, **parameters)`` with a
    :class:`~lgca.lattice_state.LatticeState` of the state before the
    reorientation, and is recomputed once per time step. Every coupling scores
    a channel state linearly, ``G(s) = Σ_i s_i w_i``; :attr:`weights` holds the
    weights ``w_i``, shape ``dims + (K,)``.
    """

    def __init__(self, spec: ReorientationTermSpec, definition):
        super().__init__(spec)
        from .plugins import validate_plugin_parameters

        try:
            validate_plugin_parameters(definition.info, self.parameters)
        except ValueError as exc:
            raise ValueError(f"parameters: {exc}") from exc
        self.definition = definition
        self.coupling = definition.coupling
        self.field = None
        self.weights = None
        self.cell_weights = None  # (labels, values of shape (cells, K)) for terms with a weight per cell
        self.step = 0
        self.fields_read: set[str] = set()

    def validate(self, context) -> None:
        super().validate(context)
        self.prepare(context.lgca)

    def prepare(self, lgca):
        from .lattice_state import LatticeState
        from .rules import CellWeights

        state = LatticeState(lgca, step=self.step)
        if self.sensed_species is not None:
            state = state.sensing(self.sensed_species)
        try:
            field = self.definition.function(state, **self.parameters)
        except KeyError as exc:
            raise ValueError(f"{self.name}: {exc.args[0] if exc.args else exc}") from exc
        self.fields_read |= state.fields_read
        dims, velocity, d = state.dims, state.velocitychannels, state.c.shape[0]
        if isinstance(field, CellWeights):
            self._prepare_cells(field, state)
            return
        self.cell_weights = None
        field = np.asarray(field, dtype=float)
        shapes = {"flux": [dims + (d,)], "nematic": [dims + (d, d)], "rest": [dims],
                  "channels": [dims + (velocity,), dims + (state.K,)]}[self.coupling]
        for shape in shapes:
            try:
                field = np.broadcast_to(field, shape)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"{self.name} must return an array that broadcasts to "
                             f"{' or '.join(map(str, shapes))} for coupling {self.coupling!r}, "
                             f"got shape {field.shape}")
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{self.name} returned values that are not finite")
        self.field = field
        weights = np.zeros(dims + (state.K,))
        if self.coupling == "flux":  # g · c_i
            weights[..., :velocity] = field @ state.c
        elif self.coupling == "nematic":  # c_i · Q c_i
            weights[..., :velocity] = np.einsum("...ab,ai,bi->...i", field, state.c, state.c)
        elif self.coupling == "rest":
            weights[..., velocity:] = field[..., None]
        else:
            weights[..., :field.shape[-1]] = field
        self.weights = weights

    def _prepare_cells(self, field, state):
        if not state.identity_based:
            raise ValueError(f"{self.name} returned CellWeights, which need an identity-based model")
        if self.coupling not in ("rest", "channels"):
            raise ValueError(f"{self.name} returned CellWeights; only terms with coupling 'rest' or "
                             f"'channels' can have a weight per cell")
        values, velocity, K = field.values, state.velocitychannels, state.K
        weights = np.zeros((len(values), K))
        if self.coupling == "rest" and values.shape == (len(values),):
            weights[:, velocity:] = values[:, None]
        elif self.coupling == "channels" and values.ndim == 2 and values.shape[1] in (velocity, K):
            weights[:, :values.shape[1]] = values
        else:
            expected = "(cells,)" if self.coupling == "rest" else f"(cells, {velocity}) or (cells, {K})"
            raise ValueError(f"{self.name} returned CellWeights of shape {values.shape}; coupling "
                             f"{self.coupling!r} needs {expected}")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"{self.name} returned values that are not finite")
        self.field = None
        self.weights = None
        self.cell_weights = (field.labels, weights)

    def dependencies(self) -> set[str]:
        return set(self.fields_read)


def _match_cells(term, cells):
    """The per-cell weights of ``term`` in the order of ``cells``, matched by label."""
    labels, weights = term.cell_weights
    if len(labels) == len(cells) and np.array_equal(labels, cells.label):
        return weights
    order = np.argsort(labels, kind="stable")
    position = np.searchsorted(labels[order], cells.label)
    position = np.minimum(position, len(labels) - 1)
    if len(labels) == 0 or np.any(labels[order][position] != cells.label):
        raise ValueError(f"{term.name} returned CellWeights that do not name every cell")
    return weights[order[position]]


# name -> term definition (see lgca.rules.reorientation_term); filled by lgca.builtin_rules
_REORIENTATION_TERMS: dict[str, Any] = {}
_TERM_ALIASES: dict[str, str] = {}


def register_reorientation_term(definition, replace: bool = False) -> None:
    """Register a term definition under its name and aliases.

    Registering a name again from the module that registered it replaces the
    entry (re-running a notebook cell); other modules need ``replace=True``.
    """
    names = (definition.name,) + tuple(definition.aliases)
    for name in names:
        canonical = _TERM_ALIASES.get(name, name)
        previous = _REORIENTATION_TERMS.get(canonical)
        if previous is not None and not replace and previous.module != definition.module:
            raise ValueError(f"Reorientation term {name!r} is already registered by module "
                             f"{previous.module!r}. Choose another name or pass replace=True.")
    _REORIENTATION_TERMS[definition.name] = definition
    for alias in definition.aliases:
        _TERM_ALIASES[alias] = definition.name


def list_reorientation_terms() -> tuple[str, ...]:
    """Return the supported reorientation-term names and aliases in deterministic order."""

    return tuple(sorted(set(_REORIENTATION_TERMS) | set(_TERM_ALIASES)))


def _candidate_matrix(candidates):
    """Candidate channel states as floats, for one matrix product with the channel weights."""
    matrix_bytes = candidates.size * np.dtype(float).itemsize
    if matrix_bytes > _MAX_CANDIDATE_BATCH_BYTES:
        raise ValueError(f"Candidate states require {matrix_bytes:,} bytes; "
                         "reduce channels or use a model without volume exclusion")
    return candidates.astype(float)


class BoltzmannReorientationOperator(ReorientationOperator):
    """Reorientation that combines the scores of several terms in one decision.

    Every term gives a weight ``w_i`` per channel, and the weights of all terms
    add up, ``w_i = Σ_k beta_k w_ik``. With volume exclusion, a node's cells
    of each species move to the channel state ``s'`` with
    ``P(s') ∝ exp(Σ_i s'_i w_i)``, among the states with as many cells.
    Without volume exclusion, every cell moves to channel ``i`` independently
    with ``P(i) ∝ exp(w_i)``, a multinomial draw per node and species. In
    identity-based models, the cell numbers are updated this way and the
    node's cells are then assigned to the occupied channels at random.
    """

    def __init__(self, spec: ReorientationSpec):
        info = PluginInfo(
            name="reorientation.boltzmann",
            operator_kind="reorientation",
            backend_families=("classical", "multispecies", "nove", "ib", "nove_ib"),
            conservation_law=ConservationLaw(True, True, False, ("channel occupancy",)),
            parameters={"sweeps": {"default": 10, "description": (
                "Metropolis proposals per node, in units of the channel number, for terms with a trait "
                "in identity-based models with volume exclusion.")},
                        "channels": {"default": "all", "description": (
                "The channels that take part: 'all', 'velocity', 'rest' or channel indices. Only cells "
                "in these channels move, and only among them.")},
                        "species": {"default": None, "description": (
                "The species whose cells move, an index or a list; the others keep their channels. "
                "Default: all.")}},
            port_status="native",
            description="Boltzmann sampler over channel configurations.",
        )
        super().__init__(info=info, parameters=spec.parameters)
        self.sampler = spec.sampler
        self.terms = self._compile_terms(spec.terms)
        self.sweeps = self.parameters.get("sweeps", 10)
        if isinstance(self.sweeps, bool) or not isinstance(self.sweeps, (int, np.integer)) or self.sweeps < 1:
            raise ValueError(".parameters.sweeps must be a positive integer")
        self.channels = self.parameters.get("channels", "all")
        self.species = self.parameters.get("species")

    @property
    def term_names(self) -> list[str]:
        return [term.name for term in self.terms]

    def validate(self, context) -> None:
        from .lattice_state import _species_indices

        if self.sampler != "boltzmann":
            raise ValueError(f"unsupported reorientation sampler {self.sampler!r}")
        if self.species is not None:
            try:
                _species_indices(self.species, context.spec.state.n_species)
            except ValueError as exc:
                raise ValueError(f".parameters.{exc}") from exc
        for index, term in enumerate(self.terms):
            try:
                term.validate(context)
            except ValueError as exc:
                raise ValueError(f"terms[{index}] {exc}") from exc

    def setup(self, context) -> None:
        lgca = context.lgca
        if (context.spec.state.volume_exclusion and not hasattr(lgca, "permutations")
                and not hasattr(lgca, "_permutation_cache")):
            lgca.calc_permutations()

    def apply(self, context, step: int) -> None:
        from .ib_base import IBLGCA_base
        from .nove_ib_base import NoVE_IBLGCA_base

        lgca = context.lgca
        for term in self.terms:
            term.step = step
            term.prepare(lgca)
        mask = channel_mask(self.channels, lgca.K, lgca.velocitychannels)
        mask = None if mask.all() else mask
        # only these species move; with one species (identity-based models) that is all cells
        still = self._still_species(getattr(lgca, "n_species", 1))
        if any((term.trait is not None or term.cell_weights is not None) and term.beta != 0
               for term in self.terms):
            self._apply_traits(lgca, step, not isinstance(lgca, NoVE_IBLGCA_base), mask)
            return
        weights = self._channel_weights(lgca)
        if isinstance(lgca, NoVE_IBLGCA_base):
            self._apply_nove_identity(lgca, weights, mask)
        elif isinstance(lgca, IBLGCA_base):
            self._apply_identity(lgca, weights, mask)
        elif lgca.nodes.dtype == bool:
            lgca._reorientation_source_nodes = lgca.nodes.copy()
            try:
                if still is None and mask is None:
                    self._sample_batches(lgca, weights)
                else:
                    source = lgca._reorientation_source_nodes[lgca.nonborder]
                    nodes = source
                    if still is not None:  # species that stay have no cells to move: no draws for them
                        nodes = source.copy()
                        nodes[..., still, :] = False
                    self._sample_batches(lgca, weights, nodes=nodes, mask=mask)
                    if still is not None:
                        interior = lgca.nodes[lgca.nonborder]
                        interior[..., still, :] = source[..., still, :]
                        lgca.nodes[lgca.nonborder] = interior
            finally:
                del lgca._reorientation_source_nodes
        else:
            interior = lgca.nodes[lgca.nonborder]
            counts = interior if interior.ndim == weights.ndim else interior[..., None, :]
            moving = counts
            if still is not None:
                moving = counts.copy()
                moving[..., still, :] = 0
            if mask is None:
                sampled = self._sample_independent(lgca, moving.sum(axis=-1), weights)
            else:
                sampled = moving.copy()
                sampled[..., mask] = self._sample_independent(lgca, moving[..., mask].sum(axis=-1),
                                                              weights[..., mask])
            if still is not None:
                sampled[..., still, :] = counts[..., still, :]
            lgca.nodes[lgca.nonborder] = sampled.reshape(interior.shape).astype(lgca.nodes.dtype)

    def _still_species(self, n_species):
        """The species whose cells keep their channels, or None if all move."""
        if self.species is None:
            return None
        from .lattice_state import _species_indices

        still = np.setdiff1d(np.arange(n_species), _species_indices(self.species, n_species))
        return still if len(still) else None

    def dependencies(self) -> set[str]:
        deps = set()
        for term in self.terms:
            deps.update(term.dependencies())
        return deps

    @staticmethod
    def _compile_terms(term_specs: Sequence[ReorientationTermSpec]) -> list[_ReorientationTerm]:
        terms = []
        for index, term_spec in enumerate(term_specs):
            if term_spec.name == "alignment":
                warn_user("The composed 'alignment' alias means nematic_alignment; "
                          "use 'nematic_alignment' or 'polar_alignment' explicitly", DeprecationWarning)
            definition = _REORIENTATION_TERMS.get(_TERM_ALIASES.get(term_spec.name, term_spec.name))
            if definition is None:
                matches = difflib.get_close_matches(term_spec.name, list_reorientation_terms(), n=1)
                hint = f" (did you mean {matches[0]!r}?)" if matches else ""
                raise ValueError(f".terms[{index}] unknown reorientation term {term_spec.name!r}{hint}")
            try:
                terms.append(_FieldTerm(term_spec, definition))
            except (TypeError, ValueError) as exc:
                raise ValueError(f".terms[{index}] {exc}") from exc
        if not terms:
            terms.append(_FieldTerm(ReorientationTermSpec(name="random_walk"),
                                    _REORIENTATION_TERMS["random_walk"]))
        return terms

    def _channel_weights(self, lgca):
        """Summed channel weights of all terms, shape ``dims + (n_species, K)``."""
        n_species = getattr(lgca, "n_species", 1)
        weights = np.zeros(tuple(lgca.dims) + (n_species, lgca.K))
        for term in self.terms:
            if term.beta == 0:
                continue
            if term.species is None:
                weights += term.beta * term.weights[..., None, :]
            else:
                weights[..., term.species, :] += term.beta * term.weights
        return weights

    @staticmethod
    def _sample_independent(lgca, number, weights):
        """Every cell picks channel ``i`` with ``P(i) ∝ exp(w_i)``: a multinomial per node and species."""
        return lgca.rng.multinomial(np.asarray(number, dtype=np.int64), _softmax_last_axis(weights))

    def _sample_batches(self, lgca, weights, nodes=None, mask=None):
        """Sample new channel states with volume exclusion and write them to the interior.

        With a channel ``mask``, only the cells in these channels move, among them.
        """
        if nodes is None:
            nodes = lgca._reorientation_source_nodes[lgca.nonborder]
        if mask is not None:
            sampled = nodes.copy()
            sampled[..., mask] = self._sample_states(lgca, weights[..., mask], nodes[..., mask], subset=True)
            lgca.nodes[lgca.nonborder] = sampled
            return sampled
        return self._sample_states(lgca, weights, nodes)

    def _sample_states(self, lgca, weights, nodes, subset=False):
        """New channel states of ``nodes`` (all their channels, or a subset of the model's)."""
        counts = nodes.sum(axis=-1)
        sampled = np.zeros_like(nodes)
        # Preserve the former spatial-then-species categorical draw order,
        # including full sites, while grouping computation by occupancy.
        draws = np.zeros(counts.shape)
        draws[counts > 0] = lgca.rng.random(np.count_nonzero(counts))
        ndim = len(lgca.dims)
        multispecies = counts.ndim > ndim
        for count in np.unique(counts[counts > 0]):
            candidates = occupations(nodes.shape[-1], int(count)) if subset else lgca.get_permutations(int(count))
            matrix = _candidate_matrix(candidates)
            for indices in _candidate_batches(counts == count, len(candidates)):
                species = indices[-1] if multispecies else np.zeros(len(indices[0]), dtype=int)
                scores = weights[indices[:ndim] + (species,)] @ matrix.T
                probabilities = _softmax_last_axis(scores)
                cumulative = np.cumsum(probabilities, axis=-1)
                cumulative /= cumulative[:, -1:]
                choices = (cumulative <= draws[indices][:, None]).sum(axis=-1)
                sampled[indices] = candidates[choices]
        if not subset:
            lgca.nodes[lgca.nonborder] = sampled
        return sampled

    def _apply_identity(self, lgca, weights, mask=None):
        """Sample the occupied channels, then place the node's labelled cells on them at random."""
        from .lattice_state import place_labels

        labels = lgca.nodes[lgca.nonborder]
        occupied = self._sample_batches(lgca, weights, nodes=labels > 0, mask=mask)
        channels = np.ones(lgca.K, dtype=bool) if mask is None else mask
        lgca.nodes[lgca.nonborder] = place_labels(labels, occupied, channels, lgca.rng)

    def _cell_scores(self, lgca, cells):
        """The score of every cell in every channel, shape ``(cells, K)``.

        A cell's score in channel ``i`` is ``Σ_k beta_k strength_k w_ki``, with
        ``w_ki`` the weights of term ``k`` at the cell's node (or of the cell,
        for terms with :class:`~lgca.rules.CellWeights`) and ``strength_k`` the
        cell's trait that scales the term (1 without one).
        """
        from .cells import trait_array

        scores = np.zeros((len(cells), lgca.K))
        for term in self.terms:
            if term.beta == 0:
                continue
            if term.cell_weights is None:
                weights = term.weights.reshape(-1, lgca.K)[cells.index]
            else:
                weights = _match_cells(term, cells)
            strength = term.beta
            if term.trait is not None:
                strength = strength * trait_array(lgca, term.trait).values[cells.label].astype(float)[:, None]
            scores += strength * weights
        return scores

    def _apply_traits(self, lgca, step, volume_exclusion, mask=None):
        """Reorientation with a weight per cell: exact without volume exclusion, Metropolis with it.

        With a channel ``mask``, only the cells in these channels move, among them.
        """
        from .lattice_state import LatticeState

        state = LatticeState(lgca, step=step, kind="reorientation")
        cells = state.cells
        channels = np.arange(lgca.K) if mask is None else np.flatnonzero(mask)
        moving = np.ones(len(cells), dtype=bool) if mask is None else mask[cells.channel]
        scores = self._cell_scores(lgca, cells)[moving][:, channels]
        if volume_exclusion:
            new = self._metropolis(cells.index[moving], scores, len(channels), lgca.rng)
        else:  # every cell draws its channel from its own weights
            cumulative = np.cumsum(_softmax_last_axis(scores), axis=-1)
            draws = lgca.rng.random(len(scores)) * cumulative[:, -1]
            new = np.minimum((cumulative <= draws[:, None]).sum(axis=-1), len(channels) - 1)
        channel = cells.channel.copy()
        channel[moving] = channels[new]
        cells.channel = channel
        state._cells_changed()
        state.commit()

    def _metropolis(self, index, scores, K, rng):
        """New channels of the cells: a Metropolis chain per node, all nodes at once.

        The chain starts from a random arrangement of each node's cells and
        proposes to swap the contents of a channel holding a cell (chosen
        uniformly among the node's cells) with another channel. The number
        of cells is fixed, so the proposal is symmetric.
        """
        n = len(index)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        nodes, row = np.unique(index, return_inverse=True)
        per_row = np.bincount(row)
        # position of each cell among the cells of its node
        order = np.argsort(row, kind="stable")
        starts = np.r_[0, np.cumsum(per_row)[:-1]]
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n) - np.repeat(starts, per_row)
        members = np.full((len(nodes), per_row.max()), -1, dtype=np.int64)
        members[row, rank] = np.arange(n)
        # random start: the node's cells on the first channels of a random permutation
        permutation = np.argsort(rng.random((len(nodes), K)), axis=-1)
        channel = permutation[row, rank]
        occupant = np.full((len(nodes), K), -1, dtype=np.int64)
        occupant[row, channel] = np.arange(n)
        # the score of a cell in a channel; row -1: an empty channel, score 0
        padded = np.concatenate([scores, np.zeros((1, K))])
        rows = np.arange(len(nodes))
        for _ in range(int(self.sweeps) * K):
            cell = members[rows, (rng.random(len(nodes)) * per_row).astype(np.int64)]
            here = channel[cell]
            there = (here + rng.integers(1, K, len(nodes))) % K
            other = occupant[rows, there]
            delta = (padded[cell, there] - padded[cell, here]) + (padded[other, here] - padded[other, there])
            accept = rng.random(len(nodes)) < np.exp(np.minimum(delta, 0))
            r, c, o, h, t = rows[accept], cell[accept], other[accept], here[accept], there[accept]
            channel[c] = t
            moved = o >= 0
            channel[o[moved]] = h[moved]
            occupant[r, t] = c
            occupant[r, h] = o
        return channel

    def _apply_nove_identity(self, lgca, weights, mask=None):
        """Sample cell numbers per channel, then place the node's cells on them in random order."""
        from .lattice_state import LatticeState

        state = LatticeState(lgca, kind="reorientation")
        before = state.counts[..., 0, :]
        if mask is None:
            after = self._sample_independent(lgca, before.sum(axis=-1)[..., None], weights)[..., 0, :]
            mask = np.ones(lgca.K, dtype=bool)
        else:
            after = before.copy()
            after[..., mask] = self._sample_independent(lgca, before[..., mask].sum(axis=-1),
                                                        weights[..., 0, :][..., mask])
        state._place_cells(after, mask)
        state._counts = after[..., None, :]
        state.commit()
