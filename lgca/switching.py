"""Switching probabilities that respond to the surroundings of a cell.

The probabilities of switches, e.g. the rates of ``phenotype_switch`` and the
events of ``trait_switch`` and of mutations, are numbers or responses to
cues::

    {"max": 0.1, "cues": [
        {"name": "density", "kappa": 5.0, "theta": 0.5},
        {"name": "field", "field": "signal", "kappa": -2.0, "theta": 1.0},
    ]}

stands for the probability

    p = max * (1 + tanh(Σ_k kappa_k (c_k - theta_k))) / 2,

where ``c_k`` is the value of cue ``k`` at the cell's node. With one cue this
is the switch of go-or-grow (``go_or_rest``): ``theta`` is the value of the
cue at which half of the maximal probability is reached, ``kappa`` the
steepness; with ``kappa > 0`` cells switch more where the cue is large, with
``kappa < 0`` less. ``max`` defaults to 1.

The cues are values of the lattice state at every node:

``"density"``
    cells at the node over the capacity (``K`` with volume exclusion,
    ``StateSpec.capacity`` without); ``scope="neighbourhood"`` averages over
    the node and its neighbours.
``"field"``
    the value of a field of ``StateSpec.fields`` (``field=`` its name).
``"gradient"``
    the length of the gradient of such a field.
``"flux"``
    the length of the flux of the node's cells, the sum of their
    velocities, divided by their number (``normalize=False``: not divided),
    so 1 where all move in the same direction; ``scope="neighbourhood"``
    includes the neighbouring nodes.

Every cue takes ``sensed_species`` (the species whose cells it reads, e.g.
``[1]``; default all). In identity-based models ``kappa`` and ``theta`` may
name cell traits, so that every cell responds with its own sensitivity, and
``{"name": "trait", "trait": "age"}`` is a cue with a value per cell. Cues of
your own are functions of the lattice state that return a value per node,
registered with :func:`switch_cue`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Probability", "list_switch_cues", "parse_probability", "switch_cue"]

_CUES: dict[str, Callable] = {}
_RESPONSE = ("name", "kappa", "theta", "sensed_species")


def switch_cue(function: Callable | None = None, *, name: str | None = None):
    """Register ``function(state, **parameters)``, a value per node, as a cue of switching probabilities.

    The function returns an array of shape ``state.dims``. Model files refer
    to it by name, ``{"name": "crowding", "kappa": 4.0, "theta": 0.5}``.

    Examples
    --------
    >>> import numpy as np
    >>> from lgca import switch_cue
    >>> @switch_cue
    ... def crowded_neighbours(state):
    ...     '''Cells at the neighbouring nodes.'''
    ...     return state.neighbor_sum(state.density)
    """

    def register(function):
        _CUES[name or function.__name__] = function
        return function

    return register if function is None else register(function)


def list_switch_cues() -> tuple[str, ...]:
    """The names of the registered cues, and ``"trait"``."""
    return tuple(sorted({*_CUES, "trait"}))


@switch_cue(name="density")
def _density(state, scope="node"):
    cells = state.density.astype(float)
    if scope == "neighbourhood":
        cells = (cells + state.neighbor_sum(cells)) / (state.velocitychannels + 1)
    elif scope != "node":
        raise ValueError(f"scope must be 'node' or 'neighbourhood', got {scope!r}")
    return cells / state.capacity


@switch_cue(name="field")
def _field(state, field):
    return np.asarray(state.field(field), dtype=float)


@switch_cue(name="gradient")
def _gradient(state, field):
    return np.linalg.norm(state.gradient(field), axis=-1)


@switch_cue(name="flux")
def _flux(state, scope="node", normalize=True):
    flux, cells = state.flux, state.density.astype(float)
    if scope == "neighbourhood":
        flux, cells = flux + state.neighbor_sum(flux), cells + state.neighbor_sum(cells)
    elif scope != "node":
        raise ValueError(f"scope must be 'node' or 'neighbourhood', got {scope!r}")
    length = np.linalg.norm(flux, axis=-1)
    return length / np.maximum(cells, 1) if normalize else length


@dataclass(frozen=True)
class _Cue:
    name: str
    kappa: Any
    theta: Any
    sensed_species: Any
    parameters: dict


@dataclass(frozen=True)
class Probability:
    """A switching probability: a number, or ``max`` times the response to cues (see the module)."""

    max: float
    cues: tuple[_Cue, ...] = ()

    @property
    def constant(self) -> bool:
        return not self.cues

    def nodes(self, state) -> np.ndarray | float:
        """The probability at every node, shape ``state.dims`` (a number without cues)."""
        if self.constant:
            return self.max
        total = np.zeros(state.dims)
        for cue in self.cues:
            if isinstance(cue.kappa, str) or isinstance(cue.theta, str) or cue.name == "trait":
                raise ValueError(f"the cue {cue.name!r} reads cell traits, which only identity-based models have")
            total += cue.kappa * (_values(state, cue) - cue.theta)
        return self.max * (1 + np.tanh(total)) / 2

    def cells(self, state, which) -> np.ndarray | float:
        """The probability of the cells at positions ``which`` of ``state.cells``."""
        if self.constant:
            return self.max
        cells = state.cells
        total = np.zeros(len(which))
        for cue in self.cues:
            if cue.name == "trait":
                values = np.asarray(cells[cue.parameters["trait"]][which], dtype=float)
            else:
                values = _values(state, cue).reshape(-1)[cells.index[which]]
            kappa = cells[cue.kappa][which] if isinstance(cue.kappa, str) else cue.kappa
            theta = cells[cue.theta][which] if isinstance(cue.theta, str) else cue.theta
            total += np.asarray(kappa, dtype=float) * (values - np.asarray(theta, dtype=float))
        return self.max * (1 + np.tanh(total)) / 2


def parse_probability(spec, where: str = "probability") -> Probability:
    """A probability from a number or ``{"max": ..., "cues": [...]}`` (see the module)."""
    if isinstance(spec, Probability):
        return spec
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"max", "cues"}
        if unknown:
            raise ValueError(f"{where} has unknown keys {sorted(unknown)}; a probability that responds to "
                             f"cues has 'max' and 'cues'")
        cues = spec.get("cues") or []
        if isinstance(cues, Mapping) or not isinstance(cues, (list, tuple)):
            raise TypeError(f"{where}['cues'] must be a list of cues, e.g. [{{'name': 'density', 'kappa': 5, "
                            f"'theta': 0.5}}]")
        return Probability(_probability(spec.get("max", 1.0), f"{where}['max']"),
                           tuple(_cue(cue, f"{where}['cues'][{index}]") for index, cue in enumerate(cues)))
    return Probability(_probability(spec, where))


def _probability(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{where} must be a number or a response to cues ({{'max': ..., 'cues': [...]}}), "
                        f"got {value!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"{where} must be a probability, got {value}")
    return float(value)


def _cue(spec, where) -> _Cue:
    if not isinstance(spec, Mapping) or "name" not in spec:
        raise TypeError(f"{where} must be a dict with a 'name', e.g. {{'name': 'density', 'kappa': 5, "
                        f"'theta': 0.5}}")
    name = spec["name"]
    if name not in _CUES and name != "trait":
        raise ValueError(f"{where}: unknown cue {name!r}; the cues are {', '.join(list_switch_cues())}")
    if name == "trait" and not isinstance(spec.get("trait"), str):
        raise ValueError(f"{where}: the cue 'trait' needs the name of a trait, e.g. {{'trait': 'age'}}")
    for key in ("kappa", "theta"):
        value = spec.get(key, 1.0 if key == "kappa" else 0.0)
        if not isinstance(value, str) and (isinstance(value, bool) or not np.isfinite(float(value))):
            raise ValueError(f"{where}['{key}'] must be a number or the name of a cell trait, got {value!r}")
    return _Cue(name, spec.get("kappa", 1.0), spec.get("theta", 0.0), spec.get("sensed_species"),
                {key: value for key, value in spec.items() if key not in _RESPONSE})


def _values(state, cue):
    """The cue at every node, shape ``state.dims``."""
    view = state if cue.sensed_species is None else state.sensing(cue.sensed_species)
    try:
        values = np.asarray(_CUES[cue.name](view, **cue.parameters), dtype=float)
    except TypeError as exc:
        raise ValueError(f"cue {cue.name!r}: {exc}") from exc
    if values.shape != state.dims:
        raise ValueError(f"the cue {cue.name!r} must give one value per node, shape {state.dims}, "
                         f"got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"the cue {cue.name!r} gave values that are not finite")
    return values
