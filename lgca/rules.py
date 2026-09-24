"""Write an interaction as a Python function of the lattice state.

The :func:`interaction` decorator turns a function ``rule(state, **parameters)``
into a registered interaction. The function receives a
:class:`~lgca.lattice_state.LatticeState` and changes it with its operations;
the decorator builds the plugin metadata from the signature and docstring,
checks that the model is one the rule was written for, and after every call
checks the conservation law of the rule's kind.

Examples
--------
>>> import numpy as np
>>> from lgca import interaction
>>> @interaction(kind="birth_death", families=("classical", "nove"), name="crowding_death")
... def crowding_death(state, r_d=0.1):
...     '''Each cell dies with probability r_d * density / capacity.'''
...     state.remove_cells(np.minimum(r_d * state.density / state.capacity, 1))
>>> crowding_death(r_d=0.2)
{'name': 'crowding_death', 'parameters': {'r_d': 0.2}}
"""

from __future__ import annotations

import inspect
import re
from dataclasses import replace
from typing import Any, Callable, Iterable

import numpy as np

from .lattice_state import LatticeState
from .operator_base import InteractionOperator, PluginInfo
from .plugins import _law_for_kind, register_plugin, validate_plugin_parameters

__all__ = ["Interaction", "interaction"]

KINDS = ("birth_death", "phenotype_switch", "reorientation")
FAMILIES = {"classical": "with volume exclusion", "nove": "without volume exclusion"}
GEOMETRIES = ("lin", "square", "hex", "cubic", "moore")
CONSERVED = ("momentum",)


def interaction(
    function: Callable | None = None,
    *,
    kind: str,
    families: str | Iterable[str],
    geometries: str | Iterable[str] | None = None,
    conserves: str | Iterable[str] = (),
    n_species: int | None = None,
    name: str | None = None,
    register: bool = True,
):
    """Turn ``function(state, **parameters)`` into a registered interaction.

    Parameters
    ----------
    kind : {"birth_death", "phenotype_switch", "reorientation"}
        What the rule does. A reorientation must keep the number of cells of
        each species at every node, a phenotype switch the number of cells at
        every node; both are checked after every call.
    families : str or sequence of str
        Model families the rule is written for: ``"classical"`` (with volume
        exclusion, at most one cell per channel and species) and/or ``"nove"``
        (without volume exclusion). Rules work for any number of species. A
        model of another family is rejected when the model is built.
    geometries : str or sequence of str, optional
        Lattices the rule is written for, from ``"lin"``, ``"square"``,
        ``"hex"``, ``"cubic"`` and ``"moore"``. Default: all.
    conserves : str or sequence of str, default=()
        Further conserved quantities, checked after every call. ``"momentum"``
        keeps the sum of the cell velocities at every node, as in the HPP
        collision rule.
    n_species : int, optional
        Number of species the rule needs, e.g. 2 for a rule that switches
        migrating and resting cells. Default: any number.
    name : str, optional
        Name in model files. Default: the function name, prefixed with its
        module unless it is defined in a notebook or script.
    register : bool, default=True
        Register the rule so that model files can refer to it by name.
        Registering again from the same module (re-running a notebook cell)
        replaces the earlier rule.

    Returns
    -------
    Interaction
        Call it with parameters to get an entry for
        ``InteractionPipelineSpec(operators=[...])``, or with a
        :class:`~lgca.lattice_state.LatticeState` to apply the rule directly.

    Notes
    -----
    Parameters are the keyword arguments after ``state``: a default makes a
    parameter optional, no default makes it required. The first paragraph of
    the docstring becomes the description, and a numpydoc ``Parameters``
    section describes the parameters. Use only ``state.rng`` for random
    numbers, so that runs are reproducible from their seed.
    """

    def decorate(function):
        rule = Interaction(function, kind=kind, families=families, geometries=geometries,
                           conserves=conserves, n_species=n_species, name=name)
        if register:
            register_plugin(rule.info, rule._factory)
        return rule

    return decorate if function is None else decorate(function)


class Interaction:
    """An interaction written as a function of a :class:`~lgca.lattice_state.LatticeState`.

    Created by :func:`interaction`. Calling it with keyword parameters returns
    an operator entry, ``{"name": ..., "parameters": {...}}``; calling it with
    a state as first argument applies the rule to that state.
    """

    def __init__(self, function, *, kind, families, geometries=None, conserves=(), n_species=None,
                 name=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
        self.function = function
        self.kind = kind
        self.families = _names("families", families, tuple(FAMILIES), hint=(
            "identity-based models are not supported by decorated rules yet"))
        self.geometries = GEOMETRIES if geometries is None else _names("geometries", geometries, GEOMETRIES)
        self.conserves = _names("conserves", conserves, CONSERVED, allow_empty=True)
        if n_species is not None and (isinstance(n_species, bool) or int(n_species) != n_species
                                      or n_species < 1):
            raise ValueError(f"n_species must be a positive integer, got {n_species!r}")
        self.n_species = None if n_species is None else int(n_species)
        module = getattr(function, "__module__", None)
        self.name = name or (function.__name__ if module in (None, "__main__")
                             else f"{module}.{function.__name__}")
        self.parameters = _parameters(function)
        law = _law_for_kind(kind)
        if "momentum" in self.conserves:
            law = replace(law, conserves_momentum=True)
        self.info = PluginInfo(
            name=self.name,
            operator_kind=kind,
            backend_families=self.families,
            parameters=self.parameters,
            conservation_law=law,
            port_status="native",
            description=_summary(function),
        )
        self.__doc__ = function.__doc__
        self.__name__ = function.__name__
        self.__module__ = module
        self._factory = _factory_for(self, module)

    def __call__(self, *args, **parameters):
        if args:
            if isinstance(args[0], LatticeState):
                return self.function(*args, **parameters)
            raise TypeError(f"{self.name}() takes parameters by keyword, e.g. "
                            f"{self.name}({', '.join(f'{key}=...' for key in self.parameters)})")
        validate_plugin_parameters(self.info, parameters)
        return {"name": self.name, "parameters": dict(parameters)}

    def __repr__(self) -> str:
        return f"<interaction {self.name!r} ({self.kind}; {', '.join(self.families)})>"

    def __str__(self) -> str:
        return str(self.info)

    def _repr_pretty_(self, printer, cycle) -> None:
        printer.text(str(self))


class FunctionInteractionOperator(InteractionOperator):
    """Pipeline operator that runs a decorated rule on a :class:`LatticeState`."""

    def __init__(self, rule: Interaction, parameters=None):
        super().__init__(info=rule.info, parameters=parameters)
        self.rule = rule
        self._capacity = None

    def validate(self, context) -> None:
        state = context.spec.state
        rule = self.rule
        if state.identity_based:
            raise ValueError(f"{rule.name} is a decorated rule for classical models; "
                             "identity-based models are not supported yet")
        family = "classical" if state.volume_exclusion else "nove"
        if family not in rule.families:
            raise ValueError(
                f"{rule.name} is written for {_family_list(rule.families)}, but this model is "
                f"{FAMILIES[family]}. If the rule works there too, add {family!r} to families=")
        if rule.n_species is not None and state.n_species != rule.n_species:
            raise ValueError(f"{rule.name} needs n_species={rule.n_species}, but the model has "
                             f"{state.n_species}")
        geometry = context.lgca.geometry
        if geometry not in rule.geometries:
            raise ValueError(f"{rule.name} is written for the geometries {', '.join(rule.geometries)}, "
                             f"not {geometry!r}")
        self._capacity = state.capacity

    def apply(self, context, step: int) -> None:
        state = LatticeState(context.lgca, step=step, capacity=self._capacity, kind=self.operator_kind)
        before = state.flux if "momentum" in self.rule.conserves else None
        self.rule.function(state, **self.parameters)
        if before is not None:
            changed = np.any(~np.isclose(state.flux, before), axis=-1)
            if np.any(changed):
                raise ValueError(f"{self.rule.name} declares that it conserves momentum, but the sum of "
                                 f"the cell velocities changed at {int(changed.sum())} nodes")
        state.commit()


def _factory_for(rule, module):
    def factory(parameters=None):
        return FunctionInteractionOperator(rule, parameters)

    # the registry lets the defining module replace its own rule (re-run notebook cells)
    factory.__module__ = module
    return factory


def _names(label, values, allowed, allow_empty=False, hint=""):
    values = (values,) if isinstance(values, str) else tuple(values)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        extra = f"; {hint}" if hint else ""
        raise ValueError(f"{label} must be chosen from {', '.join(allowed)}, got {unknown}{extra}")
    if not values and not allow_empty:
        raise ValueError(f"{label} must name at least one of {', '.join(allowed)}")
    return values


def _family_list(families):
    return " and ".join(f"models {FAMILIES[family]}" for family in families)


def _parameters(function) -> dict[str, dict[str, Any]]:
    signature = inspect.signature(function)
    arguments = list(signature.parameters.values())
    if not arguments or arguments[0].kind not in (arguments[0].POSITIONAL_ONLY,
                                                  arguments[0].POSITIONAL_OR_KEYWORD):
        raise TypeError(f"{function.__name__} must take the lattice state as its first argument")
    descriptions = _parameter_descriptions(function.__doc__ or "")
    parameters = {}
    for argument in arguments[1:]:
        if argument.kind in (argument.VAR_POSITIONAL, argument.VAR_KEYWORD):
            raise TypeError(f"{function.__name__}: parameters must be named, not *{argument.name}")
        entry = {"required": argument.default is argument.empty}
        if argument.default is not argument.empty:
            entry["default"] = argument.default
        if argument.name in descriptions:
            entry["description"] = descriptions[argument.name]
        parameters[argument.name] = entry
    return parameters


def _summary(function) -> str:
    doc = inspect.cleandoc(function.__doc__ or "")
    return " ".join(doc.split("\n\n")[0].split())


def _parameter_descriptions(doc: str) -> dict[str, str]:
    """Descriptions from a numpydoc ``Parameters`` section."""
    lines = inspect.cleandoc(doc).splitlines()
    try:
        start = next(index for index, line in enumerate(lines[:-1])
                     if line.strip() == "Parameters" and set(lines[index + 1].strip()) == {"-"})
    except StopIteration:
        return {}
    descriptions, current = {}, None
    for line in lines[start + 2:]:
        if line.strip() and set(line.strip()) == {"-"}:
            descriptions.pop(current, None)  # that line was the title of the next section
            break
        if line and not line.startswith(" "):
            current = re.split(r"\s*:", line, maxsplit=1)[0].strip()
            descriptions[current] = ""
        elif current and line.strip():
            descriptions[current] = f"{descriptions[current]} {line.strip()}".strip()
    return {key: value for key, value in descriptions.items() if value}
