"""The interaction names of earlier biolgca versions, as stacks of the rules.

``get_lgca(interaction="alignment", beta=2)`` looks the name up here for the
family of the model (classical, without volume exclusion, identity-based,
several species) and runs the stack of rules that replaces the legacy
interaction function, with the legacy parameters and defaults. The stacks are
also registered under the prefixed names of earlier model files, e.g.
``"classical.alignment"`` or ``"nove_ib.go_or_grow"``; these names are
deprecated and warn, since the rules without prefix work in every family.

The rules reproduce the legacy interactions in distribution, but not every
detail of their random draws, so seeded runs give other numbers than earlier
versions. Where the legacy functions differed from the rules, the rules
apply:

- growth is logistic, ``r_b n (1 - n / capacity) - r_d n`` per node, with
  birth and death decided at the same time (``birth_death``); the legacy
  ``classical.birth`` only matched this in the mean;
- go-or-grow switches, lets cells die and resting cells divide, then moves
  the migrating cells (``go_or_rest``, ``go_or_grow.growth``,
  ``random_walk``), in every family; the identity-based legacy functions let
  cells die first, the multispecies one switched on the density before the
  deaths;
- ``nematic`` and ``contact_guidance`` score with traceless tensors, which
  equal the legacy tensors in 2D (see ``nematic_alignment``);
- the default concentration field of ``chemotaxis`` takes its gradient with
  the model's ghost nodes, not with values mirrored at the edge;
- the research models of :mod:`lgca.research_models` list their own
  differences.
"""

from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr

from ._warnings import warn_user
from .plugins import _with_parameter_meanings, register_plugin
from .rules import Stack

logger = logging.getLogger("lgca")

__all__ = ["LegacyInteraction", "compile_legacy_interaction", "legacy_family", "legacy_interaction", "legacy_names",
           "mutation_matrix_from_trait_bins"]


@dataclass(frozen=True)
class _Legacy:
    """A legacy interaction: the stack that replaces it and how get_lgca sets the model up."""

    stack: Stack
    capacity: float | None = None  # the legacy default capacity of identity-based models without VE


# family -> get_lgca name -> translation
_NAMES: dict[str, dict[str, _Legacy]] = {family: {} for family in
                                         ("classical", "nove", "ib", "nove_ib", "multispecies")}
_DEFAULTS = {"classical": "random_walk", "nove": "dd_alignment", "ib": "random_walk",
             "nove_ib": "random_walk", "multispecies": "random_walk"}
_FAMILIES = {"classical": ("classical",), "nove": ("nove",), "ib": ("ib",), "nove_ib": ("nove_ib",)}


def _legacy(prefix, name, *, kind, replacement, families=None, traits=(), aliases=(), names=(),
            capacity=None):
    """Register ``function(state, **legacy parameters)`` under ``prefix.name`` and as a get_lgca name.

    ``replacement`` completes the deprecation warning of the prefixed name,
    ``aliases`` are further deprecated model-file names, ``names`` further
    get_lgca names.
    """

    def decorate(function):
        full = f"{prefix}.{name}"

        @functools.wraps(function)
        def checked(state, **values):
            _check_probabilities(values)
            return function(state, **values)

        rule = Stack(checked, kind=kind, families=families or _FAMILIES[prefix], traits=traits, name=full)
        info = _with_parameter_meanings(replace(rule.info, aliases=tuple(aliases), deprecated=replacement))
        register_plugin(info, rule._factory)
        entry = _Legacy(rule, capacity)
        for key in (name, *names):
            _NAMES[prefix][key] = entry
        return rule

    return decorate


def legacy_family(lgca) -> str:
    """The family whose legacy names apply to ``lgca``: classical, nove, ib, nove_ib or multispecies."""
    from .ib_base import IBLGCA_base
    from .nove_base import NoVE_LGCA_base

    ve = not isinstance(lgca, NoVE_LGCA_base)
    if isinstance(lgca, IBLGCA_base):
        return "ib" if ve else "nove_ib"
    if getattr(lgca, "n_species", 1) > 1:
        return "multispecies"
    return "classical" if ve else "nove"


def legacy_names(family: str, ndim: int | None = None) -> list[str]:
    """The interaction names get_lgca accepts for a family, and a lattice of ``ndim`` dimensions."""
    names = set(_NAMES[family])
    if family == "multispecies":
        names |= set(_NAMES["classical"]) | set(_NAMES["nove"])
    if ndim == 1:
        names.discard("contact_guidance")
    return sorted(names)


def legacy_interaction(family: str, name: str | None, ve: bool = True):
    """The translation of a get_lgca interaction name, its stack and legacy defaults."""
    name = _DEFAULTS[family] if name is None else str(name).replace(" ", "_")
    tables = [_NAMES[family]]
    if family == "multispecies":  # the rules without prefix work for several species
        tables.append(_NAMES["classical" if ve else "nove"])
    written_for = "classical" if ve else "nove"
    for table in tables:
        if name in table and (family != "multispecies" or written_for in table[name].stack.families):
            return table[name]
    raise ValueError(f"Unknown interaction {name!r}. Implemented interactions: "
                     f"{', '.join(legacy_names(family))}")


def compile_legacy_interaction(lgca, name, parameters) -> None:
    """Give a model built with get_lgca the pipeline of a legacy interaction name.

    ``parameters`` are the keyword arguments of get_lgca; those the stack
    takes are its parameters. Sets ``lgca.interaction``,
    ``lgca.interaction_params`` and the compiled model that
    ``lgca.timestep()`` runs.
    """
    from .model import CompiledModel, ModelContext, ModelSpec, SpaceSpec, StateSpec
    from .nove_base import NoVE_LGCA_base
    from .pipeline import InteractionPipelineSpec, compile_pipeline
    from .rules import StackOperator

    family = legacy_family(lgca)
    ve = not isinstance(lgca, NoVE_LGCA_base)
    name = _DEFAULTS[family] if name is None else str(name).replace(" ", "_")
    legacy = legacy_interaction(family, name, ve=ve)
    # the capacity is the model's (StateSpec.capacity below), not a parameter of the stack
    given = {key: value for key, value in parameters.items()
             if key in legacy.stack.parameters and key != "capacity"}
    # defaults of None stand for values derived from other parameters
    defaults = {key: value for key, value in legacy.stack.defaults().items()
                if key != "capacity" and value is not None}
    for key in sorted(set(defaults) - set(given)):
        logger.info("%s: %s set to %r (default)", name, key, defaults[key])
    lgca.interaction_params = {**defaults, **given}
    capacity = None
    if not ve:  # the crowding scale: the model's, or the default of the legacy interaction
        own = getattr(lgca, "capacity", lgca.K)
        capacity = own if "capacity" in parameters or legacy.capacity is None else legacy.capacity
        lgca.interaction_params["capacity"] = capacity
    lgca._validate_interaction_params()
    spec = ModelSpec(
        space=SpaceSpec(geometry=lgca.geometry, dims=tuple(lgca.dims), boundary=lgca.bc),
        state=StateSpec(restchannels=lgca.restchannels, volume_exclusion=ve,
                        identity_based=family in ("ib", "nove_ib"), n_species=getattr(lgca, "n_species", 1),
                        capacity=capacity),
        dynamics=InteractionPipelineSpec(operators=[StackOperator(legacy.stack, given)],
                                         propagation=getattr(lgca, "enable_propagation", True)))
    context = ModelContext(lgca=lgca, spec=spec, fields={}, metadata={})
    pipeline = compile_pipeline(spec.dynamics, context)
    lgca._compiled_model = CompiledModel(lgca=lgca, spec=spec, context=context, pipeline=pipeline,
                                         metadata={"operator_names": pipeline.operator_names})
    lgca.interaction = LegacyInteraction(name, lgca._compiled_model)


class LegacyInteraction:
    """``lgca.interaction`` of a model built with get_lgca: applies the stack once, without propagation."""

    def __init__(self, name: str, compiled):
        self.__name__ = name
        self._compiled = compiled

    def __call__(self, lgca) -> None:
        compiled = self._compiled
        for operator in compiled.pipeline.operators:
            operator.apply(compiled.context, compiled._step + 1)
            lgca.update_dynamic_fields()

    def __repr__(self) -> str:
        return f"<legacy interaction {self.__name__!r}>"


_PROBABILITIES = ("r_b", "r_d", "r_m", "p_d", "p_p", "pmut")


def _check_probabilities(values):
    """The rates of the legacy interactions are probabilities, as the legacy code checked."""
    for name in _PROBABILITIES:
        value = values.get(name)
        if value is None or isinstance(value, str):
            continue
        rates = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
            raise ValueError(f"{name} must be a probability between 0 and 1, got {value!r}")


def _attach_field(state, name, values):
    """Store a field on the model, as StateSpec.fields does."""
    setattr(state._lgca, name, values)


def _check_capacity(state, capacity):
    if capacity is not None and capacity != state.capacity:
        raise ValueError(f"capacity={capacity!r} as an interaction parameter is no longer used; set "
                         f"StateSpec.capacity (it is {state.capacity})")


def _check_rest_channels(state):
    if state.volume_exclusion and state.restchannels < 2:
        warn_user("Not enough rest channels - system will die out: a resting cell fills a rest channel, "
                  "and its daughter needs another free one")


def _walk(gamma=0.0):
    """A random walk, with a preference ``gamma`` for rest channels."""
    if gamma:
        return {"name": "resting_bias", "parameters": {"beta": gamma}}
    return {"name": "random_walk"}


def _birth_rate_mutation(std, a_max):
    """The legacy mutation of a daughter's birth rate: a normal change, truncated to [0, a_max]."""
    return {"r_b": {"distribution": "normal", "scale": std, "bounds": [0, a_max], "at_bounds": "redraw"}}


def mutation_matrix_from_trait_bins(traits, std):
    """Mutation matrix of species that stand for trait values, from a normal change of the trait.

    Entry ``[a][b]`` is the probability that a daughter of species ``a``
    (trait ``traits[a]``) gets a trait closest to ``traits[b]`` after a
    normal change with standard deviation ``std``.
    """
    traits = np.asarray(traits, dtype=float)
    if traits.ndim != 1:
        raise ValueError("traits must be one-dimensional.")
    if not np.all(np.isfinite(traits)):
        raise ValueError("traits must contain finite numeric values.")
    if std < 0:
        raise ValueError("std must be non-negative.")
    if std == 0:
        return np.eye(len(traits))
    order = np.argsort(traits)
    ordered = traits[order]
    if np.any(np.diff(ordered) <= 0):
        raise ValueError("traits must contain distinct values to derive a mutation kernel.")
    edges = np.concatenate(([-np.inf], (ordered[:-1] + ordered[1:]) / 2, [np.inf]))
    rows = ndtr((edges[1:][None, :] - traits[:, None]) / std) - ndtr((edges[:-1][None, :] - traits[:, None]) / std)
    matrix = np.empty_like(rows)
    matrix[:, order] = rows
    return matrix / matrix.sum(axis=1, keepdims=True)


def _species_mutation(state, mutation_matrix, traits, std):
    if mutation_matrix is not None:
        return mutation_matrix
    if std is not None:
        return mutation_matrix_from_trait_bins(np.broadcast_to(traits, (state.n_species,)), std)
    return None


# ---------------------------------------------------------------- classical, with volume exclusion

@_legacy("classical", "random_walk", kind="reorientation", families=("classical", "nove"),
         replacement="use 'random_walk'")
def _classical_random_walk(state):
    """Cells move to random channels of their node."""
    return [{"name": "random_walk"}]


@_legacy("classical", "only_propagation", kind="reorientation",
         families=("classical", "nove", "ib", "nove_ib"), aliases=("only_propagation", "propagation.default"),
         replacement="leave out the interaction")
def _only_propagation(state):
    """No interaction: cells only move along their channels."""
    return []


@_legacy("classical", "alignment", kind="reorientation", replacement="use 'polar_alignment'")
def _alignment(state, beta=2.0):
    """Cells move in the direction of the cells at the neighbouring nodes."""
    return [{"name": "polar_alignment", "parameters": {"beta": beta}}]


@_legacy("classical", "persistent_walk", kind="reorientation", names=("persistent_motion",),
         replacement="use 'persistent_walk'")
def _persistent_walk(state, beta=2.0):
    """Cells keep the direction they had at this node."""
    return [{"name": "persistent_walk", "parameters": {"beta": beta}}]


@_legacy("classical", "aggregation", kind="reorientation", replacement="use 'aggregation'")
def _aggregation(state, beta=2.0):
    """Cells move up the gradient of the cell density."""
    return [{"name": "aggregation", "parameters": {"beta": beta}}]


@_legacy("classical", "nematic", kind="reorientation", replacement="use 'nematic_alignment'")
def _nematic(state, beta=2.0):
    """Cells share an axis with the cells at the neighbouring nodes."""
    return [{"name": "nematic_alignment", "parameters": {"beta": beta}}]


@_legacy("classical", "chemotaxis", kind="reorientation",
         replacement="use 'chemotaxis' with a field in StateSpec.fields")
def _chemotaxis(state, beta=2.0, gradient=None):
    """Cells move up a gradient: the given one, or by default that of a concentration peaked in the middle.

    ``gradient`` has the shape of the lattice with ghost nodes, ``lgca.nodes.shape[:-1] + (d,)``.
    """
    if gradient is not None:
        _attach_field(state, "gradient", np.asarray(gradient, dtype=float))
        return [{"name": "directed_motion", "parameters": {"beta": beta, "field": "gradient"}}]
    lgca = state._lgca
    coordinates = [np.asarray(getattr(lgca, axis), dtype=float) for axis in ("xcoords", "ycoords", "zcoords")[
        :len(state.dims)]]
    if coordinates[0].shape != state.dims:
        coordinates = [values[lgca.nonborder] for values in coordinates]
    r = np.sqrt(sum((values - values.mean()) ** 2 for values in coordinates))
    length = lgca.l if len(state.dims) == 1 else lgca.ly
    _attach_field(state, "concentration", np.exp(-2 * r / length))
    return [{"name": "chemotaxis", "parameters": {"beta": beta, "field": "concentration"}}]


@_legacy("classical", "contact_guidance", kind="reorientation",
         replacement="use 'contact_guidance' with a field in StateSpec.fields")
def _contact_guidance(state, beta=2.0, director=None):
    """Cells move along the axis of a director field: the given one, or by default the first axis.

    ``director`` has the shape of the lattice with ghost nodes, ``lgca.nodes.shape[:-1] + (d,)``.
    """
    if len(state.dims) == 1:
        raise ValueError("contact_guidance is not supported for 1D lattices: every direction lies on the one axis")
    if director is None:
        director = np.zeros(state.dims + (state.c.shape[0],))
        director[..., 0] = 1
    _attach_field(state, "director", np.asarray(director, dtype=float))
    return [{"name": "contact_guidance", "parameters": {"beta": beta, "field": "director"}}]


@_legacy("classical", "go_or_rest", kind="reorientation", replacement="use 'go_or_rest'")
def _classical_go_or_rest(state, kappa=5.0, theta=0.75):
    """Moving cells start resting on crowded nodes, resting cells start moving on sparse ones."""
    return [{"name": "go_or_rest", "parameters": {"kappa": kappa, "theta": theta}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}}]


@_legacy("classical", "go_or_grow", kind="birth_death",
         replacement="list 'go_or_rest', 'go_or_grow.growth' and 'random_walk' (channels='velocity')")
def _classical_go_or_grow(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.75):
    """Cells switch between moving and resting, die, and resting cells divide."""
    _check_rest_channels(state)
    return [{"name": "go_or_rest", "parameters": {"kappa": kappa, "theta": theta}},
            {"name": "go_or_grow.growth", "parameters": {"r_b": r_b, "r_d": r_d}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}}]


@_legacy("classical", "birth", kind="birth_death", aliases=("birth",), names=("go_and_grow",),
         replacement="list 'birth_death' and 'random_walk'")
def _classical_birth(state, r_b=0.2):
    """Cells divide logistically and move to random channels."""
    return [{"name": "birth_death", "parameters": {"birth_rate": r_b}}, {"name": "random_walk"}]


@_legacy("classical", "birthdeath", kind="birth_death", aliases=("birthdeath",),
         replacement="list 'birth_death' and 'random_walk'")
def _classical_birthdeath(state, r_b=0.2, r_d=0.05):
    """Cells die and divide logistically, then move to random channels."""
    return [{"name": "birth_death", "parameters": {"birth_rate": r_b, "death_rate": r_d}},
            {"name": "random_walk"}]


@_legacy("classical", "excitable_medium", kind="birth_death", replacement="use 'excitable_medium'")
def _excitable_medium(state, beta=0.05, alpha=1.0, N=50):
    """Excitable medium after Barkley."""
    return [{"name": "excitable_medium", "parameters": {"alpha": alpha, "beta": beta, "N": N}}]


# ---------------------------------------------------------------- classical, without volume exclusion

@_legacy("nove", "random_walk", kind="reorientation", replacement="use 'random_walk'")
def _nove_random_walk(state):
    """Cells move to random channels of their node."""
    return [{"name": "random_walk"}]


@_legacy("nove", "dd_alignment", kind="reorientation", replacement="use 'polar_alignment'")
def _dd_alignment(state, beta=2.0, include_center=False):
    """Cells move with the flux of their neighbourhood (density-dependent alignment)."""
    return [{"name": "polar_alignment", "parameters": {"beta": beta, "include_center": include_center}}]


@_legacy("nove", "di_alignment", kind="reorientation", replacement="use 'polar_alignment' (normalize=True)")
def _di_alignment(state, beta=2.0, include_center=False):
    """Cells move with the mean direction of their neighbourhood (density-independent alignment)."""
    return [{"name": "polar_alignment", "parameters": {"beta": beta, "include_center": include_center,
                                                       "normalize": True}}]


@_legacy("nove", "go_or_rest", kind="reorientation", replacement="use 'go_or_rest'")
def _nove_go_or_rest(state, kappa=5.0, theta=0.75):
    """Moving cells start resting on crowded nodes, resting cells start moving on sparse ones."""
    return [{"name": "go_or_rest", "parameters": {"kappa": kappa, "theta": theta}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}}]


@_legacy("nove", "go_or_grow", kind="birth_death",
         replacement="list 'go_or_rest', 'go_or_grow.growth' and 'random_walk' (channels='velocity')")
def _nove_go_or_grow(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.75):
    """Cells switch between moving and resting, die, and resting cells divide."""
    return _classical_go_or_grow.function(state, r_b=r_b, r_d=r_d, kappa=kappa, theta=theta)


# ---------------------------------------------------------------- several species

@_legacy("multispecies", "birth", kind="birth_death", families=("nove",),
         replacement="list 'birth_death' (mutation_matrix) and 'random_walk'")
def _multispecies_birth(state, r_b=0.2, gamma=0.0, mutation_matrix=None, std=None):
    """Cells of every species divide logistically; daughters may change species.

    ``std`` derives the mutation matrix from a normal change of the birth
    rate, with the species standing for the values of ``r_b``.
    """
    matrix = _species_mutation(state, mutation_matrix, r_b, std)
    return [{"name": "birth_death", "parameters": {"birth_rate": r_b, "mutation_matrix": matrix}}, _walk(gamma)]


@_legacy("multispecies", "birthdeath", kind="birth_death", families=("nove",),
         replacement="list 'birth_death' (mutation_matrix) and 'random_walk'")
def _multispecies_birthdeath(state, r_b=0.2, r_d=0.02, gamma=0.0, mutation_matrix=None, std=None):
    """Cells of every species die and divide logistically; daughters may change species."""
    matrix = _species_mutation(state, mutation_matrix, r_b, std)
    return [{"name": "birth_death", "parameters": {"birth_rate": r_b, "death_rate": r_d,
                                                   "mutation_matrix": matrix}}, _walk(gamma)]


@_legacy("multispecies", "go_or_grow", kind="birth_death", families=("nove",),
         replacement="list 'birth_death', 'go_or_rest', 'birth_death' (channels='rest') and 'random_walk'")
def _multispecies_go_or_grow(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.5, kappa_std=None,
                             mutation_matrix=None, capacity=None):
    """Go-or-grow with a switch steepness ``kappa`` per species; daughters may change species.

    ``kappa_std`` derives the mutation matrix from a normal change of
    ``kappa``, with the species standing for its values.
    """
    _check_capacity(state, capacity)
    matrix = _species_mutation(state, mutation_matrix, kappa, kappa_std)
    return [{"name": "birth_death", "parameters": {"death_rate": r_d}},
            {"name": "go_or_rest", "parameters": {"kappa": kappa, "theta": theta}},
            {"name": "birth_death", "parameters": {"birth_rate": r_b, "channels": "rest",
                                                   "mutation_matrix": matrix}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}}]


@_legacy("multispecies", "excitable_medium_ms", kind="birth_death", families=("classical",),
         replacement="use 'excitable_medium'")
def _excitable_medium_ms(state, beta=0.05, alpha=1.0, N=50):
    """Excitable medium after Barkley: activators (species 0) move, inhibitors (species 1) rest."""
    return [{"name": "excitable_medium", "parameters": {"alpha": alpha, "beta": beta, "N": N}}]


# ---------------------------------------------------------------- identity-based

@_legacy("ib", "random_walk", kind="reorientation", replacement="use 'random_walk'")
def _ib_random_walk(state):
    """Cells move to random channels of their node."""
    return [{"name": "random_walk"}]


@_legacy("ib", "birth", kind="birth_death", traits="r_b",
         replacement="list 'birth_death' (birth_rate='r_b', mutation) and 'random_walk'")
def _ib_birth(state, r_b=0.2, std=0.01, a_max=1.0):
    """Cells divide logistically with their own birth rate, which daughters inherit with a normal change."""
    return [{"name": "birth_death", "parameters": {"birth_rate": "r_b",
                                                   "mutation": _birth_rate_mutation(std, a_max)}},
            {"name": "random_walk"}]


@_legacy("ib", "birthdeath", kind="birth_death", traits="r_b",
         replacement="list 'birth_death' (birth_rate='r_b', mutation) and 'random_walk'")
def _ib_birthdeath(state, r_b=0.2, r_d=0.02, std=0.01, a_max=1.0, track_inheritance=False):
    """Cells die and divide logistically; daughters inherit the birth rate with a normal change.

    With ``track_inheritance`` every initial cell founds a family, which its
    descendants belong to.
    """
    if track_inheritance:
        state._lgca.init_families(type="heterogeneous", mutation=False)
    return [{"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": r_d,
                                                   "mutation": _birth_rate_mutation(std, a_max)}},
            {"name": "random_walk"}]


@_legacy("ib", "go_or_grow", kind="birth_death", traits=("kappa", "theta"),
         replacement="list 'go_or_rest', 'go_or_grow.growth' and 'random_walk' with traits")
def _ib_go_or_grow(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.75, kappa_std=0.2, theta_std=0.05):
    """Go-or-grow with a switch ``kappa`` and ``theta`` per cell, which daughters inherit with normal changes."""
    _check_rest_channels(state)
    return [{"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": "theta"}},
            {"name": "go_or_grow.growth", "parameters": {"r_b": r_b, "r_d": r_d, "mutation": {
                "kappa": kappa_std, "theta": theta_std}}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}}]


@_legacy("ib", "birthdeath_discrete", kind="birth_death", replacement="use 'birthdeath_discrete'")
def _ib_birthdeath_discrete(state, r_b=0.2, r_d=0.02, drb=0.01, a_max=1.0, pmut=0.1):
    """See :func:`lgca.research_models.birthdeath_discrete`."""
    return [{"name": "birthdeath_discrete", "parameters": {"r_b": r_b, "r_d": r_d, "drb": drb, "a_max": a_max,
                                                           "pmut": pmut}}]


@_legacy("ib", "go_and_grow_mutations", kind="birth_death", replacement="use 'go_and_grow_mutations'")
def _ib_go_and_grow_mutations(state, r_b=0.5, r_d=0.02, r_m=1e-3, effect="passenger_mutation",
                              fitness_increase=1.1):
    """See :func:`lgca.research_models.go_and_grow_mutations`."""
    return [{"name": "go_and_grow_mutations", "parameters": {
        "r_b": r_b, "r_d": r_d, "r_m": r_m, "effect": effect, "fitness_increase": fitness_increase}}]


# ---------------------------------------------------------------- identity-based, without volume exclusion

@_legacy("nove_ib", "random_walk", kind="reorientation", names=("diffusion",), replacement="use 'random_walk'")
def _nove_ib_random_walk(state):
    """Cells move to random channels of their node."""
    return [{"name": "random_walk"}]


@_legacy("nove_ib", "birth", kind="birth_death", traits="r_b", capacity=8,
         replacement="list 'birth_death' (birth_rate='r_b', mutation) and 'random_walk'")
def _nove_ib_birth(state, r_b=0.2, std=0.01, a_max=1.0, gamma=0.0, capacity=None):
    """Cells divide logistically with their own birth rate, which daughters inherit with a normal change."""
    _check_capacity(state, capacity)
    return [{"name": "birth_death", "parameters": {"birth_rate": "r_b",
                                                   "mutation": _birth_rate_mutation(std, a_max)}},
            _walk(gamma)]


@_legacy("nove_ib", "birthdeath", kind="birth_death", traits="r_b", capacity=8,
         replacement="list 'birth_death' (birth_rate='r_b', mutation) and 'random_walk'")
def _nove_ib_birthdeath(state, r_b=0.2, r_d=0.02, std=0.01, a_max=1.0, gamma=0.0, capacity=None):
    """Cells die and divide logistically; daughters inherit the birth rate with a normal change."""
    _check_capacity(state, capacity)
    return [{"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": r_d,
                                                   "mutation": _birth_rate_mutation(std, a_max)}},
            _walk(gamma)]


@_legacy("nove_ib", "go_or_grow", kind="birth_death", traits=("kappa", "theta"), capacity=8,
         replacement="list 'go_or_rest', 'go_or_grow.growth' and 'random_walk' with traits")
def _nove_ib_go_or_grow(state, r_b=0.2, r_d=0.01, kappa=5.0, theta=0.5, kappa_std=0.2, theta_std=0.05,
                        capacity=None):
    """Go-or-grow with a switch ``kappa`` and ``theta`` per cell, which daughters inherit with normal changes."""
    _check_capacity(state, capacity)
    return _ib_go_or_grow.function(state, r_b=r_b, r_d=r_d, kappa=kappa, theta=theta, kappa_std=kappa_std,
                                   theta_std=theta_std)


def _research(name, legacy_name=None, capacity=8):
    """A research model of lgca.research_models under its legacy prefixed name."""
    from . import research_models

    model = getattr(research_models, name)
    parameters = model.defaults()

    def translate(state, capacity=None, **values):
        _check_capacity(state, capacity)
        return [{"name": name, "parameters": {**parameters, **values}}]

    # the parameters of the research model, plus the capacity that the legacy code took
    translate.__signature__ = inspect.Signature(
        [inspect.Parameter("state", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, default=value) for key, value in parameters.items()]
        + [inspect.Parameter("capacity", inspect.Parameter.KEYWORD_ONLY, default=None)])
    translate.__name__ = f"_nove_ib_{name}"
    translate.__doc__ = model.__doc__
    _legacy("nove_ib", legacy_name or name, kind="birth_death", capacity=capacity,
            replacement=f"use '{name}'")(translate)


_research("birthdeath_cancerdfe")
_research("go_or_grow_kappa")
_research("go_or_grow_kappa_chemo")
_research("go_or_grow_glioblastoma")
_research("evo_steric", capacity=512)
_NAMES["nove_ib"]["steric_evolution"] = _NAMES["nove_ib"]["evo_steric"]
_NAMES["ib"]["go_and_grow"] = _NAMES["ib"]["birthdeath"]

for _table in _NAMES.values():
    _table["only_propagation"] = _NAMES["classical"]["only_propagation"]
