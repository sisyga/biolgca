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
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from .lattice_state import LatticeState
from .operator_base import InteractionOperator, PluginInfo
from .plugins import _law_for_kind, register_plugin, validate_plugin_parameters

__all__ = ["Interaction", "ReorientationCue", "interaction", "register_single_cue", "reorientation_term"]

KINDS = ("birth_death", "phenotype_switch", "reorientation")
FAMILIES = {"classical": "with volume exclusion", "nove": "without volume exclusion",
            "ib": "identity-based with volume exclusion", "nove_ib": "identity-based without volume exclusion"}
GEOMETRIES = ("lin", "square", "hex", "cubic", "moore")
CONSERVED = ("momentum",)
_TERM_FAMILIES = ("classical", "nove", "ib", "nove_ib")


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
        exclusion, at most one cell per channel and species), ``"nove"``
        (without volume exclusion), and the identity-based families ``"ib"``
        and ``"nove_ib"``, whose cells carry labels. Classical rules work for
        any number of species. A model of another family is rejected when the
        model is built. In identity-based models, rules can read the state and
        use the operations that keep track of labels (so far
        :meth:`~lgca.lattice_state.LatticeState.shuffle_cells`).
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
        self.families = _names("families", families, tuple(FAMILIES))
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
        info = rule.info
        if (parameters or {}).get("new_family") is True:  # daughters found families: track them
            info = replace(info, mutates_families=True)
        super().__init__(info=info, parameters=parameters)
        self.rule = rule
        self.capacity = None  # state.capacity of the rule: StateSpec.capacity or the LatticeState default
        self._capacity = None  # StateSpec.capacity, None if the model sets none

    def setup(self, context) -> None:
        lgca = context.lgca
        if self.info.mutates_families and "family" not in lgca.props:
            lgca.init_families(type="homogeneous", mutation=True)

    def validate(self, context) -> None:
        state = context.spec.state
        rule = self.rule
        family = ("ib" if state.volume_exclusion else "nove_ib") if state.identity_based else (
            "classical" if state.volume_exclusion else "nove")
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
        lgca = context.lgca
        self._capacity = state.capacity  # None: the LatticeState default
        if state.capacity is not None:
            self.capacity = state.capacity
        elif state.volume_exclusion:
            self.capacity = state.n_species * lgca.K
        else:
            self.capacity = getattr(lgca, "capacity", lgca.K)

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


def reorientation_term(
    function: Callable | None = None,
    *,
    coupling: str,
    name: str | None = None,
    aliases: str | Iterable[str] = (),
    register: bool = True,
):
    """Turn ``function(state, **parameters)`` into a term of the Boltzmann reorientation.

    The function returns a field on the lattice, computed from the state
    before the reorientation; ``coupling`` says how the field scores a
    candidate channel state ``s'`` of a node:

    ``"flux"``
        a vector per node, shape ``dims + (d,)``; score ``g · J(s')`` with
        ``J(s')`` the sum of the velocities of the candidate's cells. Cells
        move along ``g``.
    ``"nematic"``
        a symmetric tensor per node, shape ``dims + (d, d)``; score
        ``Σ_i n_i c_i · Q c_i`` over occupied velocity channels ``i``. Cells
        move along the main axis of ``Q``, in either direction.
    ``"rest"``
        a number per node, shape ``dims``; score times the number of the
        candidate's cells in rest channels. Positive values make cells rest.
    ``"channels"``
        a weight per channel, shape ``dims + (velocitychannels,)`` or
        ``dims + (K,)``; score ``Σ_i w_i n_i``.

    Arrays that broadcast to these shapes are accepted, e.g. one vector for
    the whole lattice. The terms of a :class:`~lgca.pipeline.ReorientationSpec`
    act in one decision, ``P(s') ∝ exp(Σ_k beta_k G_k(s'))``. Every coupling
    scores a channel state as the sum of the scores of its cells, so terms
    also work without volume exclusion, where each cell chooses its channel on
    its own, and in identity-based models; see
    :class:`~lgca.pipeline.ReorientationSpec`.

    Parameters
    ----------
    coupling : {"flux", "nematic", "rest", "channels"}
        How the field scores a candidate state.
    name : str, optional
        Name in model files. Default: the function name.
    aliases : str or sequence of str, default=()
        Further names.
    register : bool, default=True
        Make the term available by name in :class:`~lgca.pipeline.ReorientationTermSpec`.

    Returns
    -------
    ReorientationCue
        Call it with ``beta=`` and parameters to get a
        :class:`~lgca.pipeline.ReorientationTermSpec`.

    Examples
    --------
    >>> import numpy as np
    >>> from lgca import reorientation_term
    >>> @reorientation_term(coupling="flux")
    ... def drift(state, direction=(1.0, 0.0)):
    ...     '''Cells move in a fixed direction.'''
    ...     return np.asarray(direction, dtype=float)
    >>> drift(beta=2.0, direction=[0, 1])
    ReorientationTermSpec(name='drift', beta=2.0, parameters={'direction': [0, 1]}, species=None, trait=None)
    """

    def decorate(function):
        cue = ReorientationCue(function, coupling=coupling, name=name, aliases=aliases)
        if register:
            from .pipeline import register_reorientation_term

            register_reorientation_term(cue)
        return cue

    return decorate if function is None else decorate(function)


class ReorientationCue:
    """A term of the Boltzmann reorientation defined by a field and a coupling.

    Created by :func:`reorientation_term`. Calling it with ``beta``,
    optionally ``species`` or ``trait`` and the parameters of the function returns a
    :class:`~lgca.pipeline.ReorientationTermSpec`.
    """

    def __init__(self, function, *, coupling, name=None, aliases=()):
        from .pipeline import _COUPLINGS

        if coupling not in _COUPLINGS:
            raise ValueError(f"coupling must be one of {', '.join(_COUPLINGS)}, got {coupling!r}")
        self.function = function
        self.coupling = coupling
        self.name = name or function.__name__
        self.aliases = (aliases,) if isinstance(aliases, str) else tuple(aliases)
        self.module = getattr(function, "__module__", None)
        parameters = _parameters(function)
        reserved = {"beta", "species", "trait"} & set(parameters)
        if reserved:
            raise TypeError(f"{self.name}: {', '.join(sorted(reserved))} are set on the term, "
                            "not by the function; rename the parameter")
        self.info = PluginInfo(name=self.name, operator_kind="reorientation_term",
                               backend_families=_TERM_FAMILIES, aliases=self.aliases,
                               parameters=parameters, description=_summary(function))
        self.__doc__ = function.__doc__
        self.__name__ = function.__name__

    def __call__(self, beta: float = 1.0, species: int | None = None, trait: str | None = None, **parameters):
        from .pipeline import ReorientationTermSpec

        validate_plugin_parameters(self.info, parameters)
        return ReorientationTermSpec(name=self.name, beta=beta, parameters=dict(parameters), species=species,
                                     trait=trait)

    def __repr__(self) -> str:
        return f"<reorientation term {self.name!r} (coupling {self.coupling})>"

    def __str__(self) -> str:
        text = str(self.info).replace(f"Phase: reorientation_term. Model families: {', '.join(_TERM_FAMILIES)}.",
                                      f"Coupling: {self.coupling}. Term of ReorientationSpec.")
        return text

    def _repr_pretty_(self, printer, cycle) -> None:
        printer.text(str(self))


def register_single_cue(cue: ReorientationCue, aliases: str | Iterable[str] = ()) -> None:
    """Register a reorientation term as an operator of its own.

    ``{"name": cue.name, "parameters": {"beta": ..., **term_parameters}}`` then
    stands for a :class:`~lgca.pipeline.ReorientationSpec` with this one term,
    in every model family. ``trait`` and ``sweeps`` work as in the spec.
    """
    parameters = {
        "beta": {"default": 1.0, "description": "Sensitivity to the cue; negative values reverse it."},
        "trait": {"default": None, "description": (
            "Identity-based models: a cell trait that scales beta for every cell.")},
        "sweeps": {"default": 10, "description": (
            "Metropolis proposals per node, in units of the channel number, for a trait with volume "
            "exclusion.")},
        **cue.info.parameters,
    }
    info = PluginInfo(name=cue.name, operator_kind="reorientation",
                      backend_families=("classical", "multispecies", "nove", "ib", "nove_ib"),
                      aliases=(aliases,) if isinstance(aliases, str) else tuple(aliases),
                      parameters=parameters, conservation_law=_law_for_kind("reorientation"),
                      port_status="native", description=f"{cue.info.description} (one cue; see "
                                                         "ReorientationSpec to combine cues)")

    def factory(parameters=None):
        from .pipeline import (
            BoltzmannReorientationOperator,
            ReorientationSpec,
            ReorientationTermSpec,
        )

        values = dict(parameters or {})
        beta, trait = values.pop("beta", 1.0), values.pop("trait", None)
        sampler = {"sweeps": values.pop("sweeps")} if "sweeps" in values else {}
        term = ReorientationTermSpec(cue.name, beta=beta, parameters=values, trait=trait)
        operator = BoltzmannReorientationOperator(ReorientationSpec(terms=[term], parameters=sampler))
        operator.info = replace(operator.info, name=cue.name, description=info.description)
        return operator

    factory.__module__ = cue.module
    register_plugin(info, factory)


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
