"""Check a new interaction on every lattice and model family it claims to support.

:func:`check_interaction` runs an interaction on small seeded models of every
declared geometry and family, with one and two species and with periodic and
reflecting boundaries, and checks the properties every interaction must have.
It is meant as a one-line test::

    def test_crowding_death():
        check_interaction(crowding_death, parameters={"r_d": 0.2})
"""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._warnings import is_user_code
from .lattice_state import ContractWarning

__all__ = ["ContractWarning", "InteractionCheckError", "InteractionReport", "check_interaction",
           "strict_contracts"]

_DIMS = {"lin": (16,), "square": (8, 6), "hex": (8, 6), "cubic": (4, 4, 3), "moore": (4, 3, 3)}
_LARGE_DIMS = {"lin": (4000,), "square": (60, 60), "hex": (60, 60), "cubic": (16, 16, 16), "moore": (12, 12, 12)}
_VELOCITIES = {"lin": 2, "square": 4, "hex": 6, "cubic": 6, "moore": 26}
_BOUNDARIES = ("periodic", "reflecting")
_IDENTITY = ("ib", "nove_ib")
_STEPS = 3


class InteractionCheckError(AssertionError):
    """Raised by :func:`check_interaction` when a check fails; the message is the report.

    If the rule raised, the first of its errors is the ``__cause__``, with the traceback that leads to
    the line of the rule.
    """


@contextlib.contextmanager
def strict_contracts():
    """Turn :class:`~lgca.lattice_state.ContractWarning` into an error, e.g. in tests of rules.

    A rule that does what its kind does not allow then raises before the
    operation changes anything, instead of warning once and going on as the
    wider kind. Equivalent to ``warnings.simplefilter("error", ContractWarning)``
    within the block; a test suite can also set it for every test, e.g. with
    ``filterwarnings = ["error::lgca.ContractWarning"]`` in pytest's
    configuration.

    Examples
    --------
    >>> from lgca.testing import strict_contracts
    >>> with strict_contracts():                 # doctest: +SKIP
    ...     model.step()
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContractWarning)
        yield


@dataclass
class InteractionReport:
    """Outcome of :func:`check_interaction`, one row per model checked."""

    name: str
    rows: list[tuple[str, list[str]]] = field(default_factory=list)
    growth: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> list[str]:
        failed = [f"{case}: {problem}" for case, problems in self.rows for problem in problems]
        return failed + self.growth

    def __str__(self) -> str:
        checked = len(self.rows)
        outcome = "all passed" if self.passed else f"{len(self.failures)} problems"
        lines = [f"check_interaction({self.name!r}): {checked} models checked, {outcome}"]
        lines += [f"  - {failure}" for failure in self.failures]
        return "\n".join(lines)


def check_interaction(
    rule,
    parameters: Mapping[str, Any] | None = None,
    *,
    geometries: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    n_species: Sequence[int] | None = None,
    density: float | None = None,
    seed: int = 0,
    expected_growth=None,
    prepare=None,
    traits: Mapping[str, Any] | None = None,
    fields: Mapping[str, Any] | None = None,
    raise_on_failure: bool = True,
) -> InteractionReport:
    """Run an interaction on small models and check what every interaction must do.

    For every geometry, family and number of species the rule supports, and
    for periodic and reflecting boundaries, the rule is applied for a few
    steps to a random seeded state. The checks are:

    - the state keeps its shape and type, and changes to ghost nodes do not
      reach the lattice through the boundary conditions;
    - channels hold non-negative whole numbers of cells, at most one per
      channel and species with volume exclusion;
    - the rule uses only operations its kind allows, under
      :func:`strict_contracts`, and the conservation law of the kind holds
      at every node (a reorientation keeps the cells of each species, a
      phenotype switch the cells), and so does momentum if the rule
      declares it;
    - the same seed gives the same result.

    A reorientation term (:func:`lgca.reorientation_term <lgca.rules.reorientation_term>`) is
    checked as the one term of a :class:`~lgca.pipeline.ReorientationSpec`, with ``beta`` and its
    parameters from ``parameters``: as a reorientation, which must give the same result for the same
    seed.

    Parameters
    ----------
    rule : Interaction, ReorientationCue or str
        A rule made with :func:`lgca.interaction <lgca.rules.interaction>`, a reorientation term, or
        the name of a registered interaction.
    parameters : mapping, optional
        Parameters of the rule.
    geometries, families, n_species : sequence, optional
        Restrict the models checked. Defaults: what the rule declares, and one
        and two species (two and three for a classical phenotype switch;
        identity-based models have one). Raises ``ValueError`` if no model
        is left to check.
    density : float, optional
        Mean number of cells per node of the random initial states. Default:
        40% of the channels of all species.
    seed : int, default=0
        Seed of the models.
    expected_growth : float or sequence of float, optional
        Expected relative change of the number of cells (of each species) in
        one step at the given density, e.g. ``-0.2`` for a death probability
        of 0.2. It is compared with one step on a large lattice.
    prepare : callable, optional
        ``prepare(state)`` arranges each random initial state before the
        rule runs, for rules that expect a layout, e.g. resting cells only
        in rest channels. It receives a
        :class:`~lgca.lattice_state.LatticeState` and assigns
        ``state.counts``.
    traits : mapping, optional
        Initial cell traits of identity-based models, as in
        ``StateSpec(traits=...)``, for rules that read traits.
    fields : mapping, optional
        Fields of the models, as in ``StateSpec(fields=...)``, for rules
        that read fields with ``state.field``: a number (the same value at
        every node), or a function of the lattice shape ``dims`` that returns
        the array, e.g. ``{"oxygen": lambda dims: np.linspace(0, 1, dims[0])}``
        for a gradient along the first axis (it broadcasts).
    raise_on_failure : bool, default=True
        Raise :class:`InteractionCheckError` if a check fails.

    Returns
    -------
    InteractionReport
    """
    from .pipeline import ReorientationSpec
    from .plugins import describe_plugin
    from .rules import Interaction, ReorientationCue

    entry = None
    if isinstance(rule, ReorientationCue):  # a term runs as the one term of a reorientation
        info = rule.info
        declared_geometries = tuple(_DIMS)
        declared = [(family, count) for family in info.backend_families for count in (1, 2)
                    if family not in _IDENTITY or count == 1]
        entry = ReorientationSpec(terms=[rule(**dict(parameters or {}))])
        kind, momentum = "reorientation", False
    elif isinstance(rule, Interaction):
        info = rule.info
        declared_geometries = rule.geometries
        # identity-based models have one species; a classical phenotype is a species, so it needs two
        declared = [(family, count) for family in rule.families
                    for count in ((rule.n_species,) if rule.n_species is not None
                                  else (1,) if family in _IDENTITY
                                  else (2, 3) if rule.kind == "phenotype_switch" else (1, 2))
                    if family not in _IDENTITY or count == 1]
    else:
        # built-in plugins name their families: "multispecies" means classical with several species
        info = describe_plugin(str(rule))
        declared_geometries = tuple(_DIMS)
        backends = {"classical": [("classical", 1)], "nove": [("nove", 1)],
                    "multispecies": [("classical", 2)], "ib": [("ib", 1)], "nove_ib": [("nove_ib", 1)]}
        declared = [case for backend in info.backend_families for case in backends.get(backend, [])]
    if entry is None:
        kind = info.operator_kind
        momentum = info.conservation_law.conserves_momentum is True
        entry = {"name": info.name, "parameters": dict(parameters or {})}
    geometries = tuple(declared_geometries if geometries is None else geometries)
    models = list(dict.fromkeys(declared))
    if families is not None or n_species is not None:
        families = dict.fromkeys(family for family, _ in declared) if families is None else families
        explicit = n_species is not None
        n_species = dict.fromkeys(count for _, count in declared) if n_species is None else n_species
        models = [(family, count) for family in families for count in n_species
                  if explicit or family not in _IDENTITY or count == 1]
    if not geometries or not models:  # an empty report would pass without checking anything
        raise ValueError(f"check_interaction({info.name!r}): no model to check; the rule declares the "
                         f"families {', '.join(dict.fromkeys(f for f, _ in declared)) or 'none'} with "
                         f"{', '.join(map(str, dict.fromkeys(c for _, c in declared))) or 'no'} species, "
                         f"and the geometries {', '.join(declared_geometries) or 'none'}; choose "
                         "families, n_species and geometries among them")

    report = InteractionReport(info.name)
    errors = []  # the exceptions the rule raised: the first becomes the cause of InteractionCheckError
    for geometry in geometries:
        for family, species in models:
            for boundary in _BOUNDARIES:
                case = f"{geometry}, {family}, {species} species, {boundary}"
                setup = _Setup(geometry, family, species, boundary, density, seed, traits=traits, fields=fields)
                report.rows.append((case, _check_case(entry, setup, kind, momentum, prepare, errors)))
    if expected_growth is not None:
        report.growth = _check_growth(entry, geometries[0], models, density, seed, expected_growth,
                                      prepare, traits, fields)
    if raise_on_failure and not report.passed:
        raise InteractionCheckError(str(report)) from (errors[0] if errors else None)
    return report


@dataclass(frozen=True)
class _Setup:
    geometry: str
    family: str
    n_species: int
    boundary: str
    density: float | None
    seed: int
    large: bool = False
    traits: Any = None
    fields: Any = None

    def spec(self, entry):
        from .model import Description, ModelSpec, SpaceSpec, StateSpec, TimeSpec
        from .pipeline import InteractionPipelineSpec

        channels = _VELOCITIES[self.geometry] + 1
        density = 0.4 * channels * self.n_species if self.density is None else self.density
        dims = (_LARGE_DIMS if self.large else _DIMS)[self.geometry]
        fields = {name: value(dims) if callable(value) else value for name, value in (self.fields or {}).items()}
        fields = {name: _field_on(value, dims) for name, value in fields.items()}
        name = entry["name"] if isinstance(entry, Mapping) else entry.terms[0].name
        return ModelSpec(
            description=Description(title=f"check {name}"),
            space=SpaceSpec(geometry=self.geometry, dims=dims, boundary=self.boundary),
            state=StateSpec(density=density, restchannels=1, volume_exclusion=self.family in ("classical", "ib"),
                            identity_based=self.family in _IDENTITY, n_species=self.n_species,
                            traits=dict(self.traits or {}) if self.family in _IDENTITY else {}, fields=fields),
            time=TimeSpec(steps=_STEPS, seed=self.seed),
            dynamics=InteractionPipelineSpec(operators=[entry], propagation=False),
        )


def _build(entry, setup, prepare=None):
    from .lattice_state import LatticeState
    from .model import build_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", ContractWarning)  # e.g. a stack narrower than its operators
        compiled = build_model(setup.spec(entry))
    if prepare is not None:
        state = LatticeState(compiled.lgca)
        prepare(state)
        state.commit()
    return compiled


def _check_case(entry, setup, kind, momentum, prepare, errors=None) -> list[str]:
    try:
        with strict_contracts():
            first = _run(entry, setup, kind, momentum, prepare)
            second = _run(entry, setup, kind, momentum, prepare)
    except _Problems as problems:
        return problems.messages
    except Exception as exc:  # noqa: BLE001 - any error of the rule is reported, and kept (see errors)
        from .cells import UnknownTraitError
        from .lattice_state import UnknownFieldError

        if errors is not None:
            errors.append(exc)
        # the models are check_interaction's: what they lack is given to it, not to a StateSpec
        hint = ("; give it with check_interaction(..., fields={...})" if isinstance(exc, UnknownFieldError)
                else "; give it with check_interaction(..., traits={...})" if isinstance(exc, UnknownTraitError)
                else "")
        return [f"{type(exc).__name__}: {exc}{_where(exc)}{hint}"]
    if not np.array_equal(first, second):
        return ["the same seed gave different results; draw random numbers only from state.rng"]
    return []


def _field_on(value, dims):
    """A field value on the lattice: a number, an array of shape ``dims``, or a 1-D profile along axis 0."""
    if np.ndim(value) == 0:
        return value
    if np.ndim(value) == 1:  # the docstring's gradient: it runs along the first axis
        value = np.reshape(value, (-1,) + (1,) * (len(dims) - 1))
    return np.broadcast_to(value, dims)


def _where(exc) -> str:
    """`` (file:line)`` of the innermost frame of ``exc`` outside lgca, e.g. the line of the rule."""
    found, tb = None, exc.__traceback__
    while tb is not None:
        if is_user_code(tb.tb_frame.f_code.co_filename):
            found = tb
        tb = tb.tb_next
    return "" if found is None else f" ({os.path.basename(found.tb_frame.f_code.co_filename)}:{found.tb_lineno})"


class _Problems(Exception):
    def __init__(self, messages):
        super().__init__("; ".join(messages))
        self.messages = messages


def _run(entry, setup, kind, momentum, prepare):
    compiled = _build(entry, setup, prepare)
    lgca, operator = compiled.lgca, compiled.pipeline.operators[0]
    spatial = len(lgca.dims)
    ghosts = np.ones(lgca.nodes.shape[:spatial], dtype=bool)
    ghosts[lgca.nonborder] = False
    for step in range(1, _STEPS + 1):
        lgca.apply_boundaries()
        lgca.update_dynamic_fields()
        before = lgca.nodes.copy()
        operator.apply(compiled.context, step)
        problems = _compare(lgca, before, kind, momentum)
        if not problems:
            problems = _ghost_leaks(lgca, before, ghosts)
        if problems:
            raise _Problems([f"step {step}: {problem}" for problem in problems])
        lgca.propagation()
    return lgca.nodes.copy()


def _ghost_leaks(lgca, before, ghosts) -> list[str]:
    """Apply the boundary conditions with and without the rule's changes to ghost nodes.

    Ghost nodes are overwritten by periodic boundaries but are added to the
    lattice by reflecting ones, so changing them can create or remove cells.
    """
    after = lgca.nodes.copy()
    restored = after.copy()
    restored[ghosts] = before[ghosts]
    lgca.nodes = restored
    lgca.apply_boundaries()
    expected = lgca.nodes.copy()
    lgca.nodes = after
    lgca.apply_boundaries()
    if not np.array_equal(lgca.nodes[lgca.nonborder], expected[lgca.nonborder]):
        return [("changes to ghost nodes reach the lattice through the boundary conditions; "
                 "change only the interior (state.commit() does this)")]
    return []


def _compare(lgca, before, kind, momentum) -> list[str]:
    nodes = lgca.nodes
    if nodes.shape != before.shape:
        return [f"the state changed shape from {before.shape} to {nodes.shape}"]
    if nodes.dtype != before.dtype:
        return [f"the state changed type from {before.dtype} to {nodes.dtype}"]
    problems = []
    old, new = before[lgca.nonborder], nodes[lgca.nonborder]
    if _is_identity(lgca):
        problems += _label_problems(old, new, kind)
        old, new = lgca._channel_counts(old), lgca._channel_counts(new)
    elif nodes.dtype != bool and np.any(nodes < 0):
        problems.append("a channel holds a negative number of cells")
    if new.ndim == len(lgca.dims) + 1:
        old, new = old[..., None, :], new[..., None, :]
    old, new = old.astype(np.int64), new.astype(np.int64)
    law = {"reorientation": ((-1,), "the number of cells of each species"),
           "phenotype_switch": ((-2, -1), "the number of cells")}.get(kind)
    if law is not None:
        axes, what = law
        changed = np.any(np.atleast_1d(old.sum(axis=axes) != new.sum(axis=axes)).reshape(lgca.dims + (-1,)),
                         axis=-1)
        if np.any(changed):
            problems.append(f"a {kind} must keep {what} at every node; it changed at {int(changed.sum())} nodes")
    if momentum:
        velocity = lgca.velocitychannels
        c = np.asarray(lgca.c, dtype=float)
        flux_old = old[..., :velocity].sum(axis=-2) @ c.T
        flux_new = new[..., :velocity].sum(axis=-2) @ c.T
        changed = np.any(~np.isclose(flux_old, flux_new), axis=-1)
        if np.any(changed):
            problems.append(f"momentum is declared conserved but changed at {int(changed.sum())} nodes")
    return problems


def _is_identity(lgca):
    from .ib_base import IBLGCA_base

    return isinstance(lgca, IBLGCA_base)


def _labels_per_node(nodes):
    """Sorted labels of every node of an interior array of labels or label lists."""
    flat = nodes.reshape(-1, nodes.shape[-1])
    if flat.dtype == object:
        return [sorted(label for channel in node for label in channel) for node in flat]
    return [sorted(node[node > 0].tolist()) for node in flat]


def _label_problems(old, new, kind) -> list[str]:
    """Labels must stay unique; a reorientation or phenotype switch keeps each node's cells."""
    before, after = _labels_per_node(old), _labels_per_node(new)
    labels = [label for node in after for label in node]
    problems = [] if len(labels) == len(set(labels)) else ["two cells carry the same label"]
    if kind in ("reorientation", "phenotype_switch") and before != after:
        moved = sum(a != b for a, b in zip(before, after))
        problems.append(f"a {kind} must keep the cells (labels) of every node; they changed at {moved} nodes")
    return problems


def _cells_per_species(lgca, n_species):
    interior = lgca.nodes[lgca.nonborder]
    if _is_identity(lgca):
        interior = lgca._channel_counts(interior)
    interior = np.asarray(interior, dtype=np.int64)
    return interior.reshape(-1, n_species, lgca.K).sum(axis=(0, 2))


def _check_growth(entry, geometry, models, density, seed, expected, prepare, traits=None, fields=None) -> list[str]:
    problems = []
    for family, species in models:
        setup = _Setup(geometry, family, species, "periodic", density, seed, large=True, traits=traits,
                       fields=fields)
        compiled = _build(entry, setup, prepare)
        lgca = compiled.lgca
        lgca.apply_boundaries()
        lgca.update_dynamic_fields()
        start = _cells_per_species(lgca, species)
        compiled.pipeline.operators[0].apply(compiled.context, 1)
        measured = (_cells_per_species(lgca, species) - start) / np.maximum(start, 1)
        target = np.broadcast_to(np.asarray(expected, dtype=float), measured.shape)
        tolerance = 4 * np.sqrt(np.maximum(np.abs(target), 1e-3) / np.maximum(start, 1))
        for index in np.flatnonzero(np.abs(measured - target) > tolerance):
            problems.append(
                f"growth ({geometry}, {family}, {species} species, species {index}): measured "
                f"{measured[index]:+.4f} per step, expected {target[index]:+.4f} "
                f"(tolerance {tolerance[index]:.4f}, {start[index]} cells)")
    return problems
