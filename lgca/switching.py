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

The Boltzmann form gives a switch a weight instead::

    {"rate": 0.05, "cues": [{"name": "density", "beta": 3.0}]}

stands for the weight ``w = rate * exp(Σ_k beta_k c_k)`` against staying,
whose weight is 1: a single switch (an event of ``trait_switch`` or of
mutations) happens with probability ``w / (1 + w)``, and a cell of species
``a`` that can switch to several species (a row of the ``phenotype_switch``
rates) becomes ``b`` with probability ``w_ab / (1 + Σ_b' w_ab')``. ``rate``
is the weight where the cues are 0, and about the probability of a switch
while the weights are small; the ``beta_k`` say how strongly the cues favour
the switch. Unlike the probabilities above, the weights of a row need not
sum to at most 1. For two states the two forms agree: ``rate = exp(-2 kappa
theta)`` and ``beta = 2 kappa`` give ``w / (1 + w) = (1 + tanh(kappa (c -
theta))) / 2``, with ``max`` 1.

The Hill form multiplies saturating responses::

    {"max": 0.1, "hill": [{"name": "field", "field": "oxygen", "K": 0.2, "n": 1}]}

stands for ``p = max * Π_k c_k^n_k / (K_k^n_k + c_k^n_k)``: the cue gives
half the maximum at ``c = K``, ``n`` sets the steepness (``n = 1``, the
default, is Michaelis–Menten), and a negative ``n`` gives the decreasing
response ``K^|n| / (K^|n| + c^|n|)``, e.g. death that rises as a field runs
out. Several cues multiply, as independent limiting factors. Cue values must
not be negative; ``K`` must be positive and ``n`` not 0. In a row of
``phenotype_switch`` rates it is a probability like the tanh form. A
probability uses one form: ``"cues"`` with ``"max"`` (tanh), ``"cues"`` with
``"rate"`` (Boltzmann) or ``"hill"``.

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
registered with :func:`switch_cue`. In the Boltzmann form ``beta`` may name a
trait as well, in the Hill form ``K`` and ``n``, and in the tanh and Hill forms
``max``, e.g. a maximal division rate that evolves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Probability", "list_switch_cues", "parse_probability", "switch_cue"]

_CUES: dict[str, Callable] = {}
LOG_ODDS_LIMIT = 500.0  # log odds of probabilities 0 and 1; exp(-500) is about 7e-218
_RESPONSE = ("name", "kappa", "theta", "beta", "sensed_species")
_HILL = ("name", "K", "n", "sensed_species")
_FORMS = ("a probability uses one form: 'cues' with 'max' (tanh), 'cues' with 'rate' (Boltzmann) or "
          "'hill'")


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
    """A switching probability (see the module).

    A number, or ``max`` times the response to cues; or, in the Boltzmann
    form (``rate`` given), the weight ``rate * exp(Σ beta c)`` of switching
    against staying. A cue's ``kappa`` holds its ``beta`` then, and its
    ``theta`` is 0. In the Hill form (``hill`` true) a cue's ``kappa`` holds
    its ``n`` and its ``theta`` its ``K``.
    """

    max: float | str
    cues: tuple[_Cue, ...] = ()
    rate: float | None = None
    hill: bool = False

    @property
    def constant(self) -> bool:
        return not self.cues

    @property
    def boltzmann(self) -> bool:
        return self.rate is not None

    @property
    def reads_traits(self) -> bool:
        """Whether ``max`` or a cue reads cell traits (identity-based models only)."""
        return isinstance(self.max, str) or any(
            isinstance(cue.kappa, str) or isinstance(cue.theta, str) or cue.name == "trait" for cue in self.cues)

    def nodes(self, state) -> np.ndarray | float:
        """The probability at every node, shape ``state.dims`` (a number without cues)."""
        return self._probability(self.drive(state), self._node_max())

    def cells(self, state, which) -> np.ndarray | float:
        """The probability of the cells at positions ``which`` of ``state.cells``."""
        return self._probability(self.cell_drive(state, which), self._cell_max(state, which))

    def log_odds(self, state) -> np.ndarray | float:
        """``log(p / (1 - p))`` at every node, bounded to ``±LOG_ODDS_LIMIT`` where p is 0 or 1."""
        return self._log_odds(self.drive(state), self._node_max())

    def cell_log_odds(self, state, which) -> np.ndarray | float:
        """``log(p / (1 - p))`` of the cells at positions ``which`` of ``state.cells``."""
        return self._log_odds(self.cell_drive(state, which), self._cell_max(state, which))

    def _node_max(self):
        if isinstance(self.max, str):
            # a model mismatch rather than a wrong type, as for the traits of cues
            raise ValueError(f"the maximal probability {self.max!r} names a cell trait, which only "  # noqa: TRY004
                             "identity-based models have")
        return self.max

    def _cell_max(self, state, which):
        if not isinstance(self.max, str):
            return self.max
        values = np.asarray(state.cells[self.max][which], dtype=float)
        if values.size and not (np.all(values >= 0) and np.all(values <= 1)):
            raise ValueError(f"the trait {self.max!r} is the maximal probability, so it must lie in [0, 1]; its "
                             f"values range from {values.min():.3g} to {values.max():.3g}")
        return values

    def cell_drive(self, state, which) -> np.ndarray | float:
        """The drive (see :meth:`drive`) of the cells at positions ``which`` of ``state.cells``."""
        if self.constant:
            return 0.0
        cells = state.cells
        total = np.zeros(len(which))
        for cue in self.cues:
            if cue.name == "trait":
                values = np.asarray(cells[cue.parameters["trait"]][which], dtype=float)
            else:
                values = _values(state, cue).reshape(-1)[cells.index[which]]
            kappa = cells[cue.kappa][which] if isinstance(cue.kappa, str) else cue.kappa
            theta = cells[cue.theta][which] if isinstance(cue.theta, str) else cue.theta
            total += self._response(cue, values, np.asarray(kappa, dtype=float), np.asarray(theta, dtype=float))
        return total

    def drive(self, state) -> np.ndarray | float:
        """``Σ kappa (c - theta)`` at every node; 0 without cues.

        ``Σ beta c`` in the Boltzmann form, and in the Hill form the logarithm
        of the product of the responses, ``Σ log(c^n / (K^n + c^n))``.
        """
        if self.constant:
            return 0.0
        total = np.zeros(state.dims)
        for cue in self.cues:
            if isinstance(cue.kappa, str) or isinstance(cue.theta, str) or cue.name == "trait":
                raise ValueError(f"the cue {cue.name!r} reads cell traits, which only identity-based models have")
            total += self._response(cue, _values(state, cue), cue.kappa, cue.theta)
        return total

    def _response(self, cue, values, kappa, theta):
        """A cue's term of the drive."""
        if not self.hill:
            return kappa * (values - theta)
        # log(c^n / (K^n + c^n)) = log σ(n (log c - log K)), which also holds for n < 0
        if np.any(values < 0):
            raise ValueError(f"the cue {cue.name!r} has negative values (down to {np.min(values):g}); the Hill "
                             f"form needs values >= 0")
        if np.any(theta <= 0) or np.any(kappa == 0):
            raise ValueError(f"the cue {cue.name!r} of the Hill form needs K > 0 and n != 0 for every cell")
        with np.errstate(divide="ignore"):
            return -np.logaddexp(0.0, -kappa * (np.log(values) - np.log(theta)))

    def log_weight(self, state) -> np.ndarray | float:
        """``log rate + Σ beta c`` at every node (Boltzmann form); ``-inf`` where the rate is 0."""
        with np.errstate(divide="ignore"):
            return np.log(self.rate) + self.drive(state)

    def _log_odds(self, drive, maximum):
        with np.errstate(divide="ignore"):
            if self.boltzmann:
                odds = np.log(self.rate) + drive
            elif self.hill and not self.constant:  # log p - log(1 - p), p = max exp(drive)
                log_p = np.log(maximum) + np.asarray(drive, dtype=float)
                odds = log_p - np.log(-np.expm1(np.minimum(log_p, 0.0)))
            elif self.constant:
                odds = np.log(maximum) - np.log1p(-maximum) + np.zeros_like(drive, dtype=float)
            elif np.ndim(maximum) == 0 and maximum == 1:  # (1 + tanh(x)) / 2 has the log odds 2 x
                odds = 2 * np.asarray(drive, dtype=float)
            else:  # p = max σ(2 x)
                log_sigma = -np.logaddexp(0.0, -2 * np.asarray(drive, dtype=float))
                odds = np.log(maximum) + log_sigma - np.log1p(-maximum * np.exp(log_sigma))
        odds = np.clip(odds, -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT)
        return float(odds) if np.ndim(odds) == 0 else odds

    def _probability(self, drive, maximum):
        if self.boltzmann:  # w / (1 + w), computed without overflow
            with np.errstate(divide="ignore"):
                return _logistic(np.log(self.rate) + drive)
        if self.constant:  # a number, or the trait of every cell
            return maximum
        if self.hill:
            return maximum * np.exp(drive)
        return maximum * (1 + np.tanh(drive)) / 2


def _logistic(x):
    x = np.asarray(x, dtype=float)
    result = np.exp(-np.logaddexp(0.0, -x))
    return float(result) if result.ndim == 0 else result


def choice_probabilities(log_weights) -> list:
    """The probabilities ``w_b / (1 + Σ_b' w_b')`` of switches with log weights ``log w_b``.

    Staying has weight 1. The log weights are numbers or arrays of the same
    shape; the result is one per switch.
    """
    stacked = np.stack(np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in log_weights]))
    largest = np.maximum(stacked.max(axis=0), 0.0)
    scaled = np.exp(stacked - largest)
    total = np.exp(-largest) + scaled.sum(axis=0)
    return list(scaled / total)


def parse_probability(spec, where: str = "probability") -> Probability:
    """A probability from a number or ``{"max": ..., "cues": [...]}`` (see the module)."""
    if isinstance(spec, Probability):
        return spec
    if isinstance(spec, Mapping) and "hill" in spec:
        return _hill(spec, where)
    if isinstance(spec, Mapping):
        boltzmann = "rate" in spec
        if boltzmann and "max" in spec:
            raise ValueError(f"{where} has 'max' and 'rate'; give 'max' for the tanh form, 'rate' for the "
                             f"Boltzmann form")
        unknown = set(spec) - {"rate" if boltzmann else "max", "cues"}
        if unknown:
            raise ValueError(f"{where} has unknown keys {sorted(unknown)}; {_FORMS}")
        cues = spec.get("cues") or []
        if isinstance(cues, Mapping) or not isinstance(cues, (list, tuple)):
            raise TypeError(f"{where}['cues'] must be a list of cues, e.g. [{{'name': 'density', 'kappa': 5, "
                            f"'theta': 0.5}}]")
        cues = tuple(_cue(cue, f"{where}['cues'][{index}]", boltzmann) for index, cue in enumerate(cues))
        if boltzmann:
            return Probability(1.0, cues, _rate(spec["rate"], f"{where}['rate']"))
        return Probability(_maximum(spec.get("max", 1.0), f"{where}['max']"), cues)
    return Probability(_probability(spec, where))


def _maximum(value, where):
    """``max``: a probability, or in identity-based models the name of a trait."""
    return value if isinstance(value, str) else _probability(value, where)


def _rate(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{where} must be a number, the weight of the switch where the cues are 0, "
                        f"got {value!r}")
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"{where} must be a finite number >= 0, got {value}")
    return float(value)


def _probability(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{where} must be a number or a response to cues ({{'max': ..., 'cues': [...]}}), "
                        f"got {value!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"{where} must be a probability, got {value}")
    return float(value)


def _hill(spec, where) -> Probability:
    unknown = set(spec) - {"max", "hill"}
    if unknown:
        raise ValueError(f"{where} has {sorted(unknown)} next to 'hill'; {_FORMS}")
    cues = spec["hill"] or []
    if isinstance(cues, Mapping) or not isinstance(cues, (list, tuple)):
        raise TypeError(f"{where}['hill'] must be a list of cues, e.g. [{{'name': 'field', 'field': 'oxygen', "
                        f"'K': 0.2, 'n': 1}}]")
    cues = tuple(_hill_cue(cue, f"{where}['hill'][{index}]") for index, cue in enumerate(cues))
    return Probability(_maximum(spec.get("max", 1.0), f"{where}['max']"), cues, hill=True)


def _hill_cue(spec, where) -> _Cue:
    name = _cue_name(spec, where, "{'name': 'field', 'field': 'oxygen', 'K': 0.2}")
    wrong = [key for key in ("kappa", "theta", "beta") if key in spec]
    if wrong:
        raise ValueError(f"{where} has {wrong}: the Hill form ('hill') has 'K', the cue value of half the "
                         f"maximum, and 'n', the steepness")
    if "K" not in spec:
        raise ValueError(f"{where} needs 'K', the value of the cue at which the response is half its maximum")
    K, n = spec["K"], spec.get("n", 1.0)
    if not isinstance(K, str) and (isinstance(K, bool) or not float(K) > 0 or not np.isfinite(float(K))):
        raise ValueError(f"{where}['K'] must be a number > 0 or the name of a cell trait, got {K!r}")
    if not isinstance(n, str) and (isinstance(n, bool) or float(n) == 0 or not np.isfinite(float(n))):
        raise ValueError(f"{where}['n'] must be a number other than 0 (negative for a decreasing response) or "
                         f"the name of a cell trait, got {n!r}")
    return _Cue(name, n if isinstance(n, str) else float(n), K if isinstance(K, str) else float(K),
                spec.get("sensed_species"), {key: value for key, value in spec.items() if key not in _HILL})


def _cue_name(spec, where, example):
    if not isinstance(spec, Mapping) or "name" not in spec:
        raise TypeError(f"{where} must be a dict with a 'name', e.g. {example}")
    name = spec["name"]
    if name not in _CUES and name != "trait":
        raise ValueError(f"{where}: unknown cue {name!r}; the cues are {', '.join(list_switch_cues())}")
    if name == "trait" and not isinstance(spec.get("trait"), str):
        raise ValueError(f"{where}: the cue 'trait' needs the name of a trait, e.g. {{'trait': 'age'}}")
    return name


def _cue(spec, where, boltzmann=False) -> _Cue:
    name = _cue_name(spec, where, "{'name': 'density', 'kappa': 5, 'theta': 0.5}")
    used, other = (("beta",), ("kappa", "theta")) if boltzmann else (("kappa", "theta"), ("beta",))
    wrong = [key for key in other if key in spec]
    if wrong:
        form = ("the Boltzmann form ('rate') weighs cues with 'beta'" if boltzmann else
                "the tanh form ('max') weighs cues with 'kappa' and 'theta'; 'beta' belongs to the "
                "Boltzmann form ('rate')")
        raise ValueError(f"{where} has {wrong}: {form}")
    for key in used:
        value = spec.get(key, 0.0 if key == "theta" else 1.0)
        if not isinstance(value, str) and (isinstance(value, bool) or not np.isfinite(float(value))):
            raise ValueError(f"{where}['{key}'] must be a number or the name of a cell trait, got {value!r}")
    coefficient = spec.get("beta" if boltzmann else "kappa", 1.0)
    return _Cue(name, coefficient, 0.0 if boltzmann else spec.get("theta", 0.0), spec.get("sensed_species"),
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
