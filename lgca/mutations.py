"""Mutations of cell traits when identity-based cells divide.

A mutation is an event: a daughter mutates with some probability and then
its traits change by random effects. Growth rules such as ``birth_death`` and
``go_or_grow.growth`` take a ``mutation`` parameter::

    "mutation": {
        "probability": 0.01,                               # per daughter, default 1
        "traits": {
            "r_b": {"value": 1.1, "operation": "multiply"},  # a fixed effect
            "kappa": {"distribution": "normal", "scale": 0.2},  # added by default
        },
    }

With ``new_family=True`` every daughter that mutates founds a new family.
A list of such blocks gives independent kinds of mutation, each with its own
probability, applied in the listed order (e.g. driver and passenger
mutations). A dict of traits without ``"traits"`` is short for one block with
probability 1, e.g. ``{"kappa": 0.2}``; a number as an effect is the standard
deviation of a normal change.

An effect is one of

``{"distribution": name, **parameters}``
    a draw from ``numpy.random.Generator.<name>``, e.g. ``"normal"`` with
    ``loc`` and ``scale``, ``"exponential"`` with ``scale``, ``"gamma"``,
    ``"lognormal"``, ``"uniform"`` or ``"choice"`` with ``a`` (and ``p``);
``{"value": v}``
    the same effect ``v`` for every mutating daughter;
``{"function": name, **parameters}``
    a function registered with :func:`mutation_effect`, called as
    ``function(rng, size, **parameters)``; in Python the function itself may
    be passed instead of the dict.

and may add ``"operation"`` (``"add"``, the default, ``"subtract"`` or
``"multiply"``: how the effect changes the trait) and ``"bounds"``
(``[low, high]``, either may be ``None``) with ``"at_bounds"``: ``"clip"``
(default) sets values beyond a bound to it, ``"redraw"`` draws the effect
again, which truncates the distribution (a normal change then follows a
truncated normal).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["mutation_effect"]

_EFFECTS: dict[str, Callable] = {}
_DISTRIBUTIONS = frozenset({
    "beta", "binomial", "chisquare", "choice", "exponential", "f", "gamma", "geometric", "gumbel",
    "hypergeometric", "laplace", "logistic", "lognormal", "logseries", "negative_binomial",
    "noncentral_chisquare", "noncentral_f", "normal", "pareto", "poisson", "power", "rayleigh",
    "standard_cauchy", "standard_exponential", "standard_gamma", "standard_normal", "standard_t",
    "triangular", "uniform", "vonmises", "wald", "weibull", "zipf",
})
_OPERATIONS = ("add", "subtract", "multiply")
_REDRAWS = 100


def mutation_effect(function: Callable | None = None, *, name: str | None = None):
    """Register ``function(rng, size, **parameters)`` as a distribution of mutation effects.

    The function returns ``size`` effects, which change a trait like the draws
    of a NumPy distribution (see :mod:`lgca.mutations`). Model files refer to
    it by name, ``{"function": "driver_effect", "scale": 0.1}``.

    Examples
    --------
    >>> import numpy as np
    >>> from lgca import mutation_effect
    >>> @mutation_effect
    ... def mostly_small(rng, size, scale=0.01, large=0.1, p_large=0.05):
    ...     '''Small effects, and now and then a large one.'''
    ...     return np.where(rng.random(size) < p_large, large, rng.exponential(scale, size))
    """
    def register(function):
        _EFFECTS[name or function.__name__] = function
        return function

    return register(function) if function is not None else register


@dataclass(frozen=True)
class _Effect:
    draw: Callable  # draw(rng, size) -> effects
    operation: str
    low: float
    high: float
    redraw: bool

    def apply(self, rng, values):
        new = self._changed(values, self.draw(rng, len(values)))
        if not self.redraw:
            return np.clip(new, self.low, self.high)
        for _ in range(_REDRAWS):
            outside = (new < self.low) | (new > self.high)
            if not outside.any():
                return new
            new[outside] = self._changed(values[outside], self.draw(rng, int(outside.sum())))
        raise ValueError(f"mutation effects keep falling outside the bounds [{self.low}, {self.high}] "
                         f"after {_REDRAWS} draws; use 'at_bounds': 'clip'")

    def _changed(self, values, effects):
        effects = np.asarray(effects, dtype=float)
        if self.operation == "add":
            return values + effects
        if self.operation == "subtract":
            return values - effects
        return values * effects


@dataclass(frozen=True)
class _Mutation:
    probability: float
    traits: dict[str, _Effect]


def parse_mutation(mutation) -> list[_Mutation]:
    """The kinds of mutation in a ``mutation`` parameter (see the module docstring)."""
    if mutation is None:
        return []
    blocks = [mutation] if isinstance(mutation, Mapping) else list(mutation)
    parsed = []
    for index, block in enumerate(blocks):
        where = "mutation" if isinstance(mutation, Mapping) else f"mutation[{index}]"
        if not isinstance(block, Mapping):
            raise TypeError(f"{where} must be a dict with 'traits' (and 'probability'), got {block!r}")
        if "traits" not in block:  # a dict of traits: one kind of mutation, always
            block = {"traits": block}
        unknown = set(block) - {"probability", "traits"}
        if unknown:
            raise ValueError(f"{where} has unknown keys {sorted(unknown)}; a kind of mutation has "
                             f"'probability' and 'traits'")
        probability = float(block.get("probability", 1.0))
        if not 0 <= probability <= 1:
            raise ValueError(f"{where}['probability'] must be a probability, got {probability}")
        traits = block["traits"] or {}
        if not isinstance(traits, Mapping):
            raise TypeError(f"{where}['traits'] must map trait names to effects, got {traits!r}")
        parsed.append(_Mutation(probability, {name: _effect(effect, f"{where}[{name!r}]")
                                              for name, effect in traits.items()}))
    return parsed


def apply_mutations(state, daughters, mutations) -> np.ndarray:
    """Mutate the daughters (positions in ``state.cells``); returns the mask of daughters that mutated."""
    cells, rng = state.cells, state.rng
    mutated = np.zeros(len(daughters), dtype=bool)
    for mutation in mutations:
        events = rng.random(len(daughters)) < mutation.probability
        mutated |= events
        which = daughters[events]
        for name, effect in mutation.traits.items():
            values = np.asarray(cells[name][which], dtype=float)
            cells.set_trait(which, name, effect.apply(rng, values))
    return mutated


def _effect(spec, where) -> _Effect:
    if callable(spec):
        return _Effect(spec, "add", -np.inf, np.inf, False)
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        spec = {"distribution": "normal", "scale": spec}
    if not isinstance(spec, Mapping):
        raise TypeError(f"{where} must be a number (standard deviation of a normal change), a dict "
                         f"or a function, got {spec!r}")
    spec = dict(spec)
    operation = spec.pop("operation", "add")
    if operation not in _OPERATIONS:
        raise ValueError(f"{where}['operation'] must be one of {_OPERATIONS}, got {operation!r}")
    low, high = spec.pop("bounds", None) or (None, None)
    low, high = -np.inf if low is None else float(low), np.inf if high is None else float(high)
    if not low <= high:
        raise ValueError(f"{where}['bounds'] must be [low, high] with low <= high")
    at_bounds = spec.pop("at_bounds", "clip")
    if at_bounds not in ("clip", "redraw"):
        raise ValueError(f"{where}['at_bounds'] must be 'clip' or 'redraw', got {at_bounds!r}")
    kinds = [key for key in ("distribution", "value", "function") if key in spec]
    if len(kinds) != 1:
        raise ValueError(f"{where} needs exactly one of 'distribution', 'value' or 'function', got {spec!r}")
    kind, source = kinds[0], spec.pop(kinds[0])
    draw = _draw(kind, source, spec, where)
    return _Effect(draw, operation, low, high, at_bounds == "redraw")


def _draw(kind, source, parameters: dict[str, Any], where) -> Callable:
    if kind == "value":
        if parameters:
            raise ValueError(f"{where}: a fixed 'value' takes no parameters, got {sorted(parameters)}")
        value = float(source)
        return lambda rng, size: np.full(size, value)
    if kind == "distribution":
        if source not in _DISTRIBUTIONS:
            raise ValueError(f"{where}['distribution'] must name a distribution of numpy.random.Generator, "
                             f"e.g. 'normal' or 'exponential', got {source!r}")
        return lambda rng, size: getattr(rng, source)(size=size, **parameters)
    function = source if callable(source) else _EFFECTS.get(source)
    if function is None:
        raise ValueError(f"{where}['function'] {source!r} is not registered; decorate it with "
                         f"lgca.mutation_effect (registered: {sorted(_EFFECTS) or 'none'})")
    return lambda rng, size: function(rng, size, **parameters)

