"""Fields: reaction–diffusion equations on the lattice, coupled to the cells.

A field is a named array of ``StateSpec.fields``, one value per node, such as
the concentration of oxygen or of a chemoattractant. The pipeline operator
``pde`` updates one field by one LGCA time step of

.. math::

    \\partial_t c = D \\Delta c - \\nabla \\cdot (v c) + P(x) - L(x)\\, c,

where ``D`` is the diffusion coefficient, ``v`` the advection velocity, ``P``
the production per node and step (a constant, a map, and secretion by cells)
and ``L`` the loss rate (decay and uptake by cells). Parameters are in lattice
units per LGCA step: ``D`` in nodes² per step, ``v`` in nodes per step, rates
per step.

The Laplacian uses the neighbourhood of the cells,
``Δc(x) ≈ w Σ_i (c(x + c_i) − c(x))`` over the velocity channels ``c_i`` with
``w = 2d / Σ_i |c_i|²``, exact for quadratic functions on every lattice. It is
assembled once per model as a sparse matrix. Advection is upwinded along the
same channels.

The field keeps its place as an attribute of the model, with ghost nodes filled
according to its boundary condition after every update, so the cues that read
fields (``chemotaxis``, the ``field`` and ``gradient`` cues of
:mod:`lgca.switching`, ``state.field``) see the updated values.

Examples
--------
Cells that secrete a signal, which diffuses and decays:

>>> from lgca.fields import PDESpec
>>> signal = PDESpec(field="signal", diffusion=1.0, decay=0.1, cells=[{"production": 0.5}])
>>> signal.parameters["cells"]
[{'production': 0.5}]
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse import linalg as spla

from ._warnings import warn_user
from .lattice_state import LatticeState, _species_indices, channel_mask
from .operator_base import FieldOperator, ParameterSpec, PluginInfo
from .plugins import _law_for_kind, register_plugin

__all__ = ["PDEOperator", "PDESpec", "laplacian", "list_reactions", "reaction"]

SOLVERS = ("explicit", "implicit", "steady")
_EXPLICIT_METHODS = ("RK23", "RK45", "DOP853", "BDF", "Radau")
_SOLVER_OPTIONS = {
    "explicit": {"method": "RK45", "rtol": 1e-4, "atol": 1e-6, "warn_evaluations": 200},
    "implicit": {"substeps": 1, "backend": "auto", "rtol": 1e-6, "max_iterations": 20},
    "steady": {"backend": "auto", "rtol": 1e-6, "max_iterations": 20},
}
_BACKENDS = ("auto", "direct", "cg", "amg")
# without pyamg, the steady solver factors matrices of up to this many nodes and iterates above
_DIRECT_LIMIT = 20_000
_SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
_PER_SIDE_GEOMETRIES = ("lin", "square", "cubic")
_CELL_TERM_KEYS = {"production", "uptake", "saturation", "n", "species", "channels"}
_REACTIONS: dict[str, Any] = {}


def reaction(function=None, *, name: str | None = None):
    """Register ``function(state, c, **parameters)`` as a reaction of the ``pde`` operator.

    The function gets the lattice state (the cells as the previous operator
    left them, and every field through ``state.field``) and the field's
    current values ``c`` (shape ``state.dims``, read-only), and returns
    ``(production, loss_rate)``: numbers or arrays of shape ``state.dims``,
    both non-negative, entering the equation as ``+ production - loss_rate
    * c``. Writing every reaction this way keeps the field non-negative in
    the implicit and steady solvers. Terms that depend on ``c`` are iterated
    to convergence (``solver_options`` ``rtol`` and ``max_iterations``).

    The operator uses it by name, with the parameters after ``c``:
    ``PDESpec(field="activator", reactions=[{"name": "activation", "rate": 2.0}])``.
    Several fields that react with each other are updated one after another,
    in the order of their operators (operator splitting, first order in
    time).

    Examples
    --------
    Self-activation that saturates, lost in proportion to an inhibitor:

    >>> from lgca.fields import reaction
    >>> @reaction
    ... def activation(state, c, rate=1.0, inhibitor="inhibitor"):
    ...     return rate * c**2 / (1 + c**2), 0.1 * state.field(inhibitor)
    """

    def register(function):
        parameters = list(inspect.signature(function).parameters)
        if len(parameters) < 2:
            raise TypeError(f"a reaction takes the lattice state and the field's values, "
                            f"function(state, c, **parameters); {function.__name__} takes {parameters}")
        _REACTIONS[name or function.__name__] = function
        return function

    return register if function is None else register(function)


def list_reactions() -> tuple[str, ...]:
    """The names of the registered reactions."""
    return tuple(sorted(_REACTIONS))


@dataclasses.dataclass(frozen=True)
class PDESpec:
    """A field updated by a reaction–diffusion equation, one LGCA step per call.

    Equivalent to ``{"name": "pde", "parameters": {...}}`` in
    :attr:`~lgca.pipeline.InteractionPipelineSpec.operators`; ``parameters``
    gives that mapping. The operator changes no cells, only the field.

    Attributes
    ----------
    field : str
        Name of a field of ``StateSpec.fields`` with one value per node (a
        number is a uniform initial value).
    diffusion : float, default=0
        Diffusion coefficient ``D`` in nodes² per step.
    decay : float, default=0
        Loss rate per step everywhere.
    production : float or str, default=0
        Production per node and step independent of the cells: a number or
        the name of a field, e.g. a map of vessels.
    cells : sequence of mapping, default=()
        Secretion and uptake by cells, each a mapping:
        ``{"production": r}`` (every cell adds ``r`` per step),
        ``{"uptake": r}`` (loss rate ``r`` per cell, so a node with ``n``
        cells loses ``r n c`` per step) or ``{"uptake": r, "saturation": K,
        "n": 1}`` (every cell takes up ``r cⁿ / (Kⁿ + cⁿ)`` per step, at most
        ``r``). Optional keys: ``"species"`` (an index or a list; default all)
        and ``"channels"`` (``"all"``, ``"rest"`` or ``"velocity"``). In
        identity-based models ``r`` may name a cell trait, so every cell
        secretes or consumes at its own rate.
    reactions : sequence of mapping, default=()
        Reactions registered with :func:`reaction`, each ``{"name": ...}``
        with its parameters, e.g. ``[{"name": "activation", "rate": 2.0}]``.
    advection : sequence of float or str, optional
        Velocity ``v`` of the term ``-∇·(v c)`` in nodes per step: a vector
        with one component per dimension (Cartesian, also on hexagonal
        lattices), or the name of a field of ``StateSpec.fields`` of shape
        ``dims + (d,)``, a velocity per node (read at every step). Upwinding
        keeps ``c ≥ 0`` and conserves the total, and adds a numerical
        diffusion of about ``|v| / 2``; the result is accurate where
        ``D`` is large against it.
    boundary : str or mapping, optional
        ``"periodic"`` (the default, and the only choice, on periodic
        lattices), ``"no_flux"`` (the default otherwise), ``{"value": c_b}``
        (fixed value beyond the edge) or, on 1D, square and cubic lattices,
        one condition per side, e.g. ``{"x-": {"value": 1.0}, "x+":
        {"value": 0.0}, "default": "no_flux"}``.
    solver : {"implicit", "explicit", "steady"}, default="implicit"
        ``"implicit"``: backward Euler, stable for any ``D`` and keeping
        ``c ≥ 0``; first order in time (use substeps for accuracy).
        ``"explicit"``: :func:`scipy.integrate.solve_ivp` with error
        control, accurate while the field changes slowly.
        ``"steady"``: the field is at equilibrium with the current cells at
        every call (quasi-steady), for fields much faster than the cells;
        it is also solved when the model is built. Something must remove
        the field (decay, uptake or a fixed value at the boundary).
    solver_options : mapping, default={}
        Implicit and steady: ``backend`` (``"auto"``, ``"direct"``,
        ``"cg"``, ``"amg"``), ``rtol`` (1e-6, for the iterative solver and
        the iteration of saturating uptake and reactions), ``max_iterations``
        (20);
        implicit also ``substeps`` (1). Explicit: ``method`` (``"RK45"``;
        also ``"RK23"``, ``"DOP853"``, ``"BDF"``, ``"Radau"``), ``rtol``
        (1e-4), ``atol`` (1e-6), ``warn_evaluations`` (200).

    Notes
    -----
    The ``"auto"`` backend factors a matrix that does not depend on the cells
    once (no uptake). Otherwise the implicit solver uses conjugate gradients
    with a Jacobi preconditioner, and the steady solver conjugate gradients
    preconditioned by an algebraic multigrid hierarchy (pyamg) that is
    rebuilt only when the number of iterations has doubled; without pyamg
    (Python 3.14) it factors lattices of up to 20 000 nodes and uses the
    Jacobi preconditioner above. With advection the matrix is not symmetric:
    BiCGSTAB replaces conjugate gradients, and the multigrid hierarchy is
    pyamg's approximate ideal restriction (AIR) instead of smoothed
    aggregation.

    Examples
    --------
    >>> PDESpec(field="oxygen", diffusion=50.0, cells=[{"uptake": 0.2}], boundary={"value": 1.0}).parameters
    {'field': 'oxygen', 'diffusion': 50.0, 'cells': [{'uptake': 0.2}], 'boundary': {'value': 1.0}}
    """

    name: ClassVar[str] = "pde"

    field: str
    diffusion: float = 0.0
    decay: float = 0.0
    production: float | str = 0.0
    cells: Sequence[Mapping[str, Any]] = ()
    reactions: Sequence[Mapping[str, Any]] = ()
    advection: Sequence[float] | str | None = None
    boundary: str | Mapping[str, Any] | None = None
    solver: str = "implicit"
    solver_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def parameters(self) -> dict[str, Any]:
        """The parameters of the ``pde`` operator; defaults are left out."""
        parameters = {"field": self.field}
        defaults = {"diffusion": 0.0, "decay": 0.0, "production": 0.0, "advection": None, "boundary": None,
                    "solver": "implicit"}
        for name in ("diffusion", "decay", "production", "cells", "reactions", "advection", "boundary",
                     "solver", "solver_options"):
            value = getattr(self, name)
            if name in ("cells", "reactions", "solver_options"):
                if value:  # plain lists and dicts for model files; the operator checks the entries
                    parameters[name] = (dict(value) if isinstance(value, Mapping)
                                        else [dict(term) if isinstance(term, Mapping) else term for term in value]
                                        if isinstance(value, (list, tuple)) else value)
            elif name == "advection" and isinstance(value, (tuple, np.ndarray)):
                parameters[name] = np.asarray(value).tolist()  # a list in model files
            elif value != defaults[name]:
                parameters[name] = value
        return parameters


def laplacian(lgca, boundary=None) -> tuple[sp.csr_matrix, np.ndarray]:
    """The discrete Laplacian of a field on the nodes of ``lgca``.

    Returns the sparse matrix ``A`` and the vector ``s`` with ``Δc ≈ A c + s``
    for the interior values ``c`` flattened in C order; ``s`` holds the
    contributions of fixed values beyond the edge. ``boundary`` as in
    :class:`PDESpec`.

    Examples
    --------
    >>> from lgca import get_lgca
    >>> lgca = get_lgca(geometry="lin", dims=4, bc="reflecting", density=0, interaction="only_propagation")
    >>> A, s = laplacian(lgca, {"x-": {"value": 1.0}, "x+": "no_flux"})
    >>> A.toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -1.]])
    >>> s
    array([1., 0., 0., 0.])
    """
    sides = _parse_boundary(boundary, lgca)
    return _assemble_laplacian(lgca, sides)


class PDEOperator(FieldOperator):
    """The ``pde`` operator: one LGCA step of a reaction–diffusion equation (see :class:`PDESpec`)."""

    def __init__(self, info: PluginInfo, parameters: Mapping[str, Any] | None = None):
        super().__init__(info, parameters)
        parameters = self.parameters
        self.field = parameters.get("field")
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("pde.field must be the name of a field of StateSpec.fields")
        self.diffusion = _non_negative(parameters.get("diffusion", 0.0), "pde.diffusion")
        self.decay = _non_negative(parameters.get("decay", 0.0), "pde.decay")
        production = parameters.get("production", 0.0)
        self.production = production if isinstance(production, str) else _non_negative(production,
                                                                                        "pde.production")
        cells = parameters.get("cells", ())
        if isinstance(cells, (Mapping, str, bytes)) or not isinstance(cells, Sequence):
            raise TypeError("pde.cells must be a list of cell terms, e.g. [{'uptake': 0.1}]")
        self.cell_terms = [_CellTerm.parse(term, index) for index, term in enumerate(cells)]
        reactions = parameters.get("reactions", ())
        if isinstance(reactions, (Mapping, str, bytes)) or not isinstance(reactions, Sequence):
            raise TypeError("pde.reactions must be a list of reactions, e.g. [{'name': 'activation'}]")
        self.reactions = [_Reaction.parse(entry, index) for index, entry in enumerate(reactions)]
        self.advection = _advection(parameters.get("advection"))
        self.boundary = parameters.get("boundary")
        self.solver = parameters.get("solver", "implicit")
        if self.solver not in SOLVERS:
            raise ValueError(f"pde.solver must be one of {', '.join(SOLVERS)}, got {self.solver!r}")
        self.options = _solver_options(self.solver, parameters.get("solver_options") or {})
        self.statistics: dict[str, Any] = {"solver": self.solver, "calls": 0}
        self._warned = False

    # --------------------------------------------------------------- pipeline

    def dependencies(self) -> set[str]:
        read = {self.field}
        read.update(name for name in (self.production, self.advection) if isinstance(name, str))
        return read

    def outputs(self) -> set[str]:
        return {f"field:{self.field}"}

    def setup(self, context) -> None:
        lgca = context.lgca
        fields = context.spec.state.fields
        if self.field not in fields:
            known = ", ".join(sorted(fields)) or "none"
            raise ValueError(f"pde.field {self.field!r} is not a field of StateSpec.fields (declared: {known}); "
                             f"add it, e.g. fields={{{self.field!r}: 0.0}}")
        self._dims = tuple(int(size) for size in lgca.dims)
        self._interior = tuple(slice(lgca.r_int, lgca.r_int + size) for size in self._dims)
        values = np.asarray(getattr(lgca, self.field))
        if values.shape not in (self._dims, np.shape(lgca.cell_density)):
            raise ValueError(f"pde.field {self.field!r} must have one value per node, shape {self._dims}; "
                             f"got shape {values.shape}")
        if isinstance(self.production, str):
            if self.production not in fields:
                raise ValueError(f"pde.production names the field {self.production!r}, which is not in "
                                 "StateSpec.fields")
            production = np.asarray(getattr(lgca, self.production), dtype=float)
            if production.shape not in (self._dims, np.shape(lgca.cell_density)):
                raise ValueError(f"pde.production field {self.production!r} must have shape {self._dims}")
        self._sides = _parse_boundary(self.boundary, lgca)
        self._fixed = any(isinstance(condition, float) for pair in self._sides for condition in pair)
        self.laplacian, self._laplacian_source = _assemble_laplacian(lgca, self._sides)
        n = self.laplacian.shape[0]
        if isinstance(self.advection, str):
            self._velocity_shape = self._dims + (len(self._dims),)
            if self.advection not in fields:
                raise ValueError(f"pde.advection names the field {self.advection!r}, which is not in "
                                 "StateSpec.fields")
        elif self.advection is not None and len(self.advection) != len(self._dims):
            raise ValueError(f"pde.advection must have one component per dimension ({len(self._dims)}), got "
                             f"{list(self.advection)}")
        for term in self.cell_terms:
            term.setup(lgca, context.spec.state)
        self._backend = self._resolve_backend(n)
        self._velocity = None
        self._assemble(lgca)
        if self.solver == "steady":
            self._check_steady_state_exists()
        self.statistics.update({"solver": self.solver, "calls": 0})
        if self.solver != "explicit":
            self.statistics["backend"] = self._backend
        context.metadata.setdefault("fields", {})[self.field] = self.statistics

    def _assemble(self, lgca):
        """The linear part without cells, ``A c + b``, with the current velocity; resets the solvers."""
        n = self.laplacian.shape[0]
        self._A = (self.diffusion * self.laplacian - self.decay * sp.identity(n, format="csr")).tocsr()
        self._b = self.diffusion * self._laplacian_source
        if self.advection is not None:
            velocity = self._read_velocity(lgca)
            advection, source = _assemble_advection(lgca, self._sides, velocity.reshape(n, -1))
            self._A = (self._A + advection).tocsr()
            self._b = self._b + source
            self._velocity = velocity
        self._lu = self._lu_matrix = self._amg = None
        self._base_dt = None
        if self.solver == "steady":
            self._steady_base = (-self._A).tocsr()

    def _read_velocity(self, lgca):
        if not isinstance(self.advection, str):
            return np.broadcast_to(np.asarray(self.advection, dtype=float), self._dims + (len(self._dims),))
        values = np.asarray(getattr(lgca, self.advection), dtype=float)
        if values.shape[len(self._dims):] != (len(self._dims),):
            raise ValueError(f"pde.advection field {self.advection!r} must hold a velocity per node, shape "
                             f"{self._velocity_shape}; got {values.shape}")
        if values.shape[:len(self._dims)] != self._dims:
            values = values[self._interior]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"pde.advection field {self.advection!r} must be finite")
        return values

    @property
    def _symmetric(self):
        return self.advection is None

    def _resolve_backend(self, n):
        backend = self.options.get("backend")
        if backend == "amg" and _pyamg() is None:
            raise ValueError("pde.solver_options backend 'amg' needs pyamg, which is not installed (it has no "
                             "wheels for Python 3.14 yet); use backend 'auto'")
        if backend != "auto":
            return backend
        if self.solver == "implicit":  # I - dt A is well conditioned: Jacobi suffices
            return "cg"
        return "amg" if _pyamg() is not None else "direct" if n <= _DIRECT_LIMIT else "cg"

    def _check_steady_state_exists(self):
        if self.reactions:  # a reaction may remove the field; a singular problem fails when solved
            return
        fixed = self._fixed
        uptake = any(term.kind == "uptake" for term in self.cell_terms)
        if self.diffusion == 0 and self.decay == 0:
            raise ValueError(f"pde {self.field!r}: the steady solver without diffusion needs decay, so that "
                             "every node, also one without cells, has a steady state")
        if self.decay == 0 and not fixed and not uptake:
            raise ValueError(f"pde {self.field!r}: the steady solver needs something that removes the field "
                             "(decay, uptake by cells, or a fixed value at the boundary); with periodic or "
                             "no-flux boundaries and no loss there is no unique steady state")

    def attach_field(self, lgca) -> None:
        """Store the field as a float array with ghost nodes filled by its boundary condition."""
        values = np.asarray(getattr(lgca, self.field), dtype=float)
        if values.shape != self._dims:
            values = values[self._interior]
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"the initial values of field {self.field!r} must be finite and non-negative")
        setattr(lgca, self.field, self._pad(values))
        if self.solver == "steady":  # the first operators see the field at equilibrium
            self._update(lgca, step=0)

    def apply(self, context, step: int) -> None:
        self._update(context.lgca, step)
        self.statistics["calls"] += 1

    def _update(self, lgca, step):
        with _single_threaded_blas():
            self._advance(lgca, step)

    def _advance(self, lgca, step):
        if isinstance(self.advection, str) and not np.array_equal(self._read_velocity(lgca), self._velocity):
            self._assemble(lgca)  # the velocity field has changed
        stored = getattr(lgca, self.field)
        c = np.array(stored[self._interior], dtype=float).ravel()
        production, loss, saturating = self._sources(lgca, step)
        if self.solver == "explicit":
            c = self._explicit(c, production, loss, saturating)
        elif self.solver == "implicit":
            c = self._implicit(c, production, loss, saturating)
        else:
            c = self._steady(c, production, loss, saturating)
        stored[...] = self._pad(c.reshape(self._dims))

    # ---------------------------------------------------------------- sources

    def _sources(self, lgca, step):
        """Production and linear loss rate per node from the cells and the map, and the terms that
        depend on the field: functions of its values that return (production, loss rate)."""
        n = self._A.shape[0]
        production = self._b.copy()
        if isinstance(self.production, str):
            values = np.asarray(getattr(lgca, self.production), dtype=float)
            if values.shape != self._dims:
                values = values[self._interior]
            production += values.ravel()
        elif self.production:
            production += self.production
        loss = np.zeros(n)
        nonlinear = []
        if self.cell_terms or self.reactions:
            state = LatticeState(lgca, step=step)
            for term in self.cell_terms:
                weights = term.weights(state)
                if term.kind == "production":
                    production += weights
                elif term.saturation is None:
                    loss += weights
                elif np.any(weights):
                    nonlinear.append(functools.partial(_saturating_uptake, weights, term.saturation, term.n))
            for entry in self.reactions:
                nonlinear.append(functools.partial(entry.evaluate, state, self._dims))
        return production, loss, nonlinear

    # ---------------------------------------------------------------- solvers

    def _explicit(self, c, production, loss, nonlinear):
        options = self.options
        A = self._A

        def rhs(_t, values):
            change = A @ values + production - loss * values
            for term in nonlinear:
                made, lost = term(values)
                change += made - lost * values
            return change

        method = options["method"]
        implicit = {"jac": (A - sp.diags(loss)).tocsc()} if method in ("BDF", "Radau") else {}
        # t_eval keeps only the final state: solve_ivp stores every step otherwise
        solution = solve_ivp(rhs, (0.0, 1.0), c, method=method, t_eval=[1.0], rtol=options["rtol"],
                             atol=options["atol"], **implicit)
        if not solution.success:
            raise RuntimeError(f"pde {self.field!r}: the explicit solver failed ({solution.message}); "
                               "try solver='implicit'")
        evaluations = int(solution.nfev)
        stats = self.statistics
        stats["method"] = method
        stats["rhs_evaluations"] = stats.get("rhs_evaluations", 0) + evaluations
        stats["max_rhs_evaluations"] = max(stats.get("max_rhs_evaluations", 0), evaluations)
        if evaluations > options["warn_evaluations"] and not self._warned:
            self._warned = True
            warn_user(f"pde {self.field!r}: the explicit solver needed {evaluations} evaluations for one "
                      "step, a sign of a field that changes fast compared with the cells; "
                      "solver='implicit' is faster")
        c = solution.y[:, -1]
        if c.min() < -options["atol"]:
            raise RuntimeError(f"pde {self.field!r}: the explicit solver produced negative values (down to "
                               f"{c.min():.3g}); use solver='implicit', which keeps the field non-negative")
        return np.maximum(c, 0.0)

    def _implicit(self, c, production, loss, nonlinear):
        dt = 1.0 / self.options["substeps"]
        if getattr(self, "_base_dt", None) != dt:
            self._base = (sp.identity(self._A.shape[0], format="csr") - dt * self._A).tocsr()
            self._base_dt = dt
        for _ in range(self.options["substeps"]):
            c = self._solve_system(self._base, dt, c + dt * production, c, loss, nonlinear)
        return c

    def _steady(self, c, production, loss, nonlinear):
        if (self.decay == 0 and not np.any(self._b) and not self.reactions
                and not np.any(loss) and not nonlinear):
            raise RuntimeError(f"pde {self.field!r}: no cells take up the field, and nothing else removes it, "
                               "so it has no steady state; add decay or a fixed value at the boundary")
        with warnings.catch_warnings():  # a singular matrix is reported below
            warnings.simplefilter("ignore", spla.MatrixRankWarning)
            c = self._solve_system(self._steady_base, 1.0, production, c, loss, nonlinear)
        if not np.all(np.isfinite(c)):
            raise RuntimeError(f"pde {self.field!r}: the steady problem has no unique solution; something must "
                               "remove the field (decay, uptake, a reaction's loss rate or a fixed boundary "
                               "value)")
        return c

    def _solve_system(self, base, scale, right, start, loss, nonlinear):
        """Solve ``(base + scale diag(L)) c = right + scale P``.

        ``L`` is the loss rate of uptake by cells plus that of the terms that depend on c (saturating
        uptake, reactions), which also add their production ``P``. These are evaluated at the current
        iterate and the linear problem solved again (Picard iteration), starting from ``start``,
        until c changes by less than ``rtol`` or the terms no longer change.
        """
        if not nonlinear:
            return self._solve_linear(base, scale, right, start, loss)
        options = self.options
        previous = start
        made, lost = _evaluate(nonlinear, previous, loss)
        for iteration in range(1, options["max_iterations"] + 1):
            new = self._solve_linear(base, scale, right + scale * made, previous, lost)
            change = np.max(np.abs(new - previous), initial=0.0)
            previous = new
            new_made, new_lost = _evaluate(nonlinear, new, loss)
            if (change <= options["rtol"] * max(np.max(np.abs(new), initial=0.0), 1e-300)
                    or (np.array_equal(new_made, made) and np.array_equal(new_lost, lost))):
                break
            made, lost = new_made, new_lost
        else:
            if not self._warned:
                self._warned = True
                warn_user(f"pde {self.field!r}: the terms that depend on the field (saturating uptake, "
                          f"reactions) did not converge in {options['max_iterations']} iterations; raise "
                          "solver_options 'max_iterations'"
                          + (" or use substeps" if self.solver == "implicit" else ""))
        stats = self.statistics
        stats["max_iterations_used"] = max(stats.get("max_iterations_used", 0), iteration)
        return previous

    def _solve_linear(self, base, scale, right, start, loss):
        if not np.any(loss):  # the matrix does not depend on the cells
            if self.solver == "steady" and self.decay == 0 and not self._fixed:
                # nothing removes the field: -A is singular (SuperLU may miss it by rounding)
                raise RuntimeError(f"pde {self.field!r}: the steady problem has no unique solution; something "
                                   "must remove the field (decay, uptake, a reaction's loss rate or a fixed "
                                   "boundary value)")
            return self._solve_constant(base, right, start)
        return self._solve(base + sp.diags(scale * loss), right, start)

    def _solve_constant(self, matrix, right, start):
        """Solve with a matrix that does not depend on the cells: factor it once."""
        if self.options["backend"] not in ("auto", "direct"):
            return self._solve(matrix, right, start)
        if self._lu_matrix is not matrix:
            try:
                self._lu = spla.splu(matrix.tocsc())
            except RuntimeError as exc:  # SuperLU: "Factor is exactly singular"
                raise RuntimeError(f"pde {self.field!r}: the problem has no unique solution ({exc}); something "
                                   "must remove the field (decay, uptake, a reaction's loss rate or a fixed "
                                   "boundary value)") from None
            self._lu_matrix = matrix
        return np.maximum(self._lu.solve(right), 0.0)

    def _solve(self, matrix, right, start):
        if self._backend == "direct":
            values = spla.spsolve(matrix.tocsc(), right)
        elif self._backend == "amg":
            values = self._solve_amg(matrix, right, start)
        else:  # conjugate gradients (BiCGSTAB with advection), Jacobi preconditioner
            values, _ = self._cg(matrix, right, start, sp.diags(1.0 / matrix.diagonal()))
        # the exact solution is non-negative; rounding may leave tiny negative values where c is nearly zero
        return np.maximum(values, 0.0)

    def _solve_amg(self, matrix, right, start):
        """CG (BiCGSTAB with advection) preconditioned by a multigrid hierarchy of an earlier matrix.

        The hierarchy is rebuilt when a solve needs twice the iterations of the first solve after the
        last rebuild, or fails to converge; the cells change the matrix only a little per step.
        """
        if self._amg is None or self._amg_stale:
            # with advection the matrix is not symmetric: approximate ideal restriction (AIR), made for
            # upwinded advection, needed 2-4 iterations where smoothed aggregation needed up to 945
            pyamg = _pyamg()
            self._amg = (pyamg.smoothed_aggregation_solver(matrix.tocsr()) if self._symmetric
                         else pyamg.air_solver(matrix.tocsr()))
            self._amg_reference, self._amg_stale = None, False
            self.statistics["amg_setups"] = self.statistics.get("amg_setups", 0) + 1
        # an old hierarchy gets a few times its first number of iterations before it is rebuilt
        limit = None if self._amg_reference is None else 4 * self._amg_reference + 10
        values, iterations = self._cg(matrix, right, start, self._amg.aspreconditioner(cycle="V"),
                                      maxiter=limit)
        if values is None:  # an old hierarchy that no longer converges: rebuild and solve again
            self._amg_stale = True
            return self._solve_amg(matrix, right, start)
        if self._amg_reference is None:
            self._amg_reference = max(iterations, 1)
        elif iterations > 2 * self._amg_reference:
            self._amg_stale = True
        return values

    def _cg(self, matrix, right, start, preconditioner, maxiter=None):
        """Values and number of iterations; values None if a given ``maxiter`` was reached."""
        count = [0]

        def counted(_):
            count[0] += 1

        method = spla.cg if self._symmetric else spla.bicgstab
        values, info = _krylov(method, matrix, right, x0=start, rtol=self.options["rtol"], M=preconditioner,
                               callback=counted, maxiter=maxiter or 10 * len(right))
        stats = self.statistics
        stats["max_linear_iterations"] = max(stats.get("max_linear_iterations", 0), count[0])
        if info != 0:
            if maxiter is not None:
                return None, count[0]
            raise RuntimeError(f"pde {self.field!r}: the iterative solver did not converge "
                               f"(scipy.sparse.linalg.{method.__name__} returned {info}); try solver_options "
                               "{'backend': 'direct'}")
        return values, count[0]

    # ------------------------------------------------------------- boundaries

    def _pad(self, values):
        """The interior values with ghost nodes filled according to the boundary condition."""
        width = int(self._interior[0].start)
        for axis, sides in enumerate(self._sides):
            if sides[0] == "periodic":
                pad = [(0, 0)] * values.ndim
                pad[axis] = (width, width)
                values = np.pad(values, pad, mode="wrap")
                continue
            for side, condition in enumerate(sides):
                pad = [(0, 0)] * values.ndim
                pad[axis] = (width, 0) if side == 0 else (0, width)
                if condition is None:
                    values = np.pad(values, pad, mode="edge")
                else:
                    values = np.pad(values, pad, mode="constant", constant_values=condition)
        return values


class _Reaction:
    """One entry of ``pde.reactions``: a registered reaction and its parameters."""

    def __init__(self, name, function, parameters, path):
        self.name, self.function, self.parameters, self.path = name, function, parameters, path

    @classmethod
    def parse(cls, entry, index):
        path = f"pde.reactions[{index}]"
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise TypeError(f"{path} must be a mapping with the name of a reaction, e.g. {{'name': 'activation', "
                            "'rate': 2.0}")
        name = entry["name"]
        if name not in _REACTIONS:
            raise ValueError(f"{path}: no reaction {name!r} is registered (registered: "
                             f"{', '.join(list_reactions()) or 'none'}); register it with lgca.fields.reaction "
                             "before the model is built")
        function = _REACTIONS[name]
        parameters = {key: value for key, value in entry.items() if key != "name"}
        signature = inspect.signature(function)
        try:
            signature.bind(None, None, **parameters)
        except TypeError as exc:
            accepted = list(signature.parameters)[2:]
            raise ValueError(f"{path}: the reaction {name!r} takes the parameters {accepted}; {exc}") from None
        return cls(name, function, parameters, path)

    def evaluate(self, state, dims, c):
        """Production and loss rate per node, flattened, at the field values c."""
        values = c.reshape(dims).view()
        values.flags.writeable = False
        result = self.function(state, values, **self.parameters)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"the reaction {self.name!r} must return (production, loss_rate), got "
                            f"{type(result).__name__}")
        terms = []
        for label, value in zip(("production", "loss rate"), result):
            try:
                value = np.broadcast_to(np.asarray(value, dtype=float), dims)
            except ValueError:
                raise ValueError(f"the {label} of the reaction {self.name!r} must be a number or one value per "
                                 f"node, shape {dims}; got shape {np.shape(value)}") from None
            if not np.all(np.isfinite(value)) or value.min(initial=0.0) < 0:
                raise ValueError(f"the {label} of the reaction {self.name!r} must be finite and non-negative; "
                                 f"its smallest value is {np.min(value):.3g}. Write a loss as a loss rate "
                                 "times c")
            terms.append(value.ravel())
        return terms[0], terms[1]


class _CellTerm:
    """One entry of ``pde.cells``: secretion or uptake by cells."""

    def __init__(self, kind, rate, saturation, n, species, channels, path):
        self.kind, self.rate, self.saturation, self.n = kind, rate, saturation, n
        self.species, self.channels, self.path = species, channels, path

    @classmethod
    def parse(cls, term, index):
        path = f"pde.cells[{index}]"
        if not isinstance(term, Mapping):
            raise TypeError(f"{path} must be a mapping such as {{'uptake': 0.1}}")
        unknown = sorted(set(term) - _CELL_TERM_KEYS)
        if unknown:
            raise ValueError(f"{path} has unknown keys {unknown}; allowed: {sorted(_CELL_TERM_KEYS)}")
        kinds = [kind for kind in ("production", "uptake") if kind in term]
        if len(kinds) != 1:
            raise ValueError(f"{path} needs exactly one of 'production' and 'uptake'")
        kind = kinds[0]
        rate = term[kind]
        if not isinstance(rate, str):
            rate = _non_negative(rate, f"{path}.{kind}")
        saturation = term.get("saturation")
        n = term.get("n", 1)
        if kind == "production" and ("saturation" in term or "n" in term):
            raise ValueError(f"{path}: 'saturation' and 'n' belong to uptake terms")
        if "n" in term and saturation is None:
            raise ValueError(f"{path}: 'n' needs a 'saturation' constant")
        if saturation is not None:
            saturation = _non_negative(saturation, f"{path}.saturation")
            if saturation == 0:
                raise ValueError(f"{path}.saturation must be positive")
            n = _non_negative(n, f"{path}.n")
            if n < 1:
                raise ValueError(f"{path}.n must be at least 1")
        return cls(kind, rate, saturation, n, term.get("species"), term.get("channels", "all"), path)

    def setup(self, lgca, state_spec):
        velocity = int(lgca.velocitychannels)
        n_species = 1 if state_spec.identity_based else int(state_spec.n_species)
        self._species = (np.arange(n_species) if self.species is None
                         else _species_indices(self.species, n_species))
        K = int(lgca.K)
        self._mask = channel_mask(self.channels, K, velocity)
        if isinstance(self.rate, str):
            if not state_spec.identity_based:
                raise ValueError(f"{self.path}: the rate {self.rate!r} names a cell trait, which needs an "
                                 "identity-based model; give a number")
            if self.rate not in lgca.props:
                known = ", ".join(sorted(lgca.props)) or "none"
                raise ValueError(f"{self.path}: the cells have no trait {self.rate!r} (traits: {known}); "
                                 "set it in StateSpec.traits")
        self._n_nodes = prod(int(size) for size in lgca.dims)

    def weights(self, state):
        """Sum of the rates of the selected cells per node, flattened."""
        if state.identity_based:
            cells = state.cells
            selected = self._mask[cells.channel]
            index = cells.index[selected]
            if isinstance(self.rate, str):
                rates = np.asarray(cells[self.rate], dtype=float)[selected]
                if rates.size and (not np.all(np.isfinite(rates)) or rates.min() < 0):
                    raise ValueError(f"{self.path}: the trait {self.rate!r} must be finite and non-negative; "
                                     f"its smallest value is {rates.min():.3g}")
                return np.bincount(index, weights=rates, minlength=self._n_nodes)
            return self.rate * np.bincount(index, minlength=self._n_nodes).astype(float)
        counts = state.counts[..., self._species, :][..., self._mask]
        return self.rate * counts.sum(axis=(-2, -1), dtype=np.int64).ravel().astype(float)


# ------------------------------------------------------------------- helpers

def _saturating_uptake(weights, K, n, c):
    """Production and loss rate of saturating uptake at the field values c."""
    return 0.0, weights * _hill_rate(c, K, n)


def _evaluate(nonlinear, c, loss):
    """Production and total loss rate of the terms that depend on the field, at the values c."""
    made, lost = np.zeros_like(c), loss.copy()
    for term in nonlinear:
        production, rate = term(c)
        made += production
        lost += rate
    return made, lost


def _hill_rate(c, K, n):
    """The Hill uptake as a loss rate: c^(n-1) / (K^n + c^n), finite at c = 0."""
    c = np.maximum(c, 0.0)
    return c**(n - 1) / (K**n + c**n)


def _krylov(method, matrix, right, *, x0, rtol, M, callback, maxiter):
    """scipy.sparse.linalg.cg or bicgstab with a relative tolerance; SciPy < 1.12 calls it ``tol``."""
    tolerance = {"rtol": rtol} if "rtol" in _CG_PARAMETERS else {"tol": rtol}
    return method(matrix, right, x0=x0, atol=0.0, M=M, maxiter=maxiter, callback=callback, **tolerance)


_CG_PARAMETERS = inspect.signature(spla.cg).parameters


def _single_threaded_blas():
    """Limit NumPy's and SciPy's BLAS to one thread while a field is updated.

    The solvers spend their time in vector operations (dot products, norms) of lattice size, for
    which waking the BLAS threads costs more than the arithmetic: with 16 threads, a steady solve
    at 200² took 75 ms instead of 21 ms, an implicit step 18 ms instead of 3.6 ms.
    """
    return _blas_controller().limit(limits=1, user_api="blas")


@functools.cache
def _blas_controller():
    from threadpoolctl import ThreadpoolController

    _pyamg()  # load every BLAS library the solvers use before the controller looks for them
    return ThreadpoolController()


@functools.cache
def _pyamg():
    """The pyamg module, or None where it is not installed (Python 3.14 until it has wheels)."""
    try:
        import pyamg
    except ImportError:
        return None
    return pyamg


def _non_negative(value, path):
    if isinstance(value, bool) or np.ndim(value) != 0:
        raise ValueError(f"{path} must be a non-negative number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{path} must be a non-negative number") from None
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{path} must be a non-negative number, got {value}")
    return value


def _advection(value):
    """None, the name of a velocity field, or a vector of finite numbers."""
    if value is None or (isinstance(value, str) and value):
        return value
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (Sequence, np.ndarray)):
        raise TypeError(f"pde.advection must be a velocity vector, e.g. [0.5, 0.0], or the name of a field "
                        f"of velocities; got {value!r}")
    vector = []
    for component in value:
        if isinstance(component, (bool, str)) or np.ndim(component) != 0:
            raise ValueError(f"pde.advection must be a vector of numbers, got {value!r}")
        component = float(component)
        if not np.isfinite(component):
            raise ValueError(f"pde.advection must be finite, got {value!r}")
        vector.append(component)
    return tuple(vector)


def _solver_options(solver, options):
    if not isinstance(options, Mapping):
        raise TypeError("pde.solver_options must be a mapping")
    defaults = _SOLVER_OPTIONS[solver]
    unknown = sorted(set(options) - set(defaults))
    if unknown:
        raise ValueError(f"pde.solver_options {unknown} do not apply to solver {solver!r}; "
                         f"its options are {sorted(defaults)}")
    merged = {**defaults, **options}
    if solver == "explicit":
        if merged["method"] == "LSODA":
            raise ValueError("pde.solver_options method 'LSODA' is not supported: SciPy gives it no sparse "
                             "Jacobian, and its dense one of a whole lattice exhausts memory; use 'BDF' or "
                             "solver='implicit'")
        if merged["method"] not in _EXPLICIT_METHODS:
            raise ValueError(f"pde.solver_options method must be one of {', '.join(_EXPLICIT_METHODS)}")
        for name in ("rtol", "atol"):
            if _non_negative(merged[name], f"pde.solver_options {name}") == 0:
                raise ValueError(f"pde.solver_options {name} must be positive")
        _positive_integer(merged["warn_evaluations"], "pde.solver_options warn_evaluations")
    else:
        if solver == "implicit":
            _positive_integer(merged["substeps"], "pde.solver_options substeps")
        _positive_integer(merged["max_iterations"], "pde.solver_options max_iterations")
        if merged["backend"] not in _BACKENDS:
            raise ValueError(f"pde.solver_options backend must be one of {', '.join(_BACKENDS)}")
        if _non_negative(merged["rtol"], "pde.solver_options rtol") == 0:
            raise ValueError("pde.solver_options rtol must be positive")
    return merged


def _positive_integer(value, path):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{path} must be a positive integer")


def _wrapped_axes(lgca) -> tuple[int, ...]:
    """Axes along which the lattice wraps around (as in LatticeState._pad)."""
    ndim = len(lgca.dims)
    return tuple({"periodic": range(ndim), "inflow": range(1, ndim)}.get(lgca.bc, ()))


def _parse_boundary(boundary, lgca):
    """Per axis a pair (lower side, upper side): "periodic", None (no flux) or a fixed value."""
    ndim = len(lgca.dims)
    wrapped = _wrapped_axes(lgca)
    periodic_lattice = len(wrapped) == ndim
    if boundary is None:
        boundary = "periodic" if periodic_lattice else "no_flux"
    if isinstance(boundary, str) and boundary == "periodic":
        if not periodic_lattice:
            raise ValueError(f"pde.boundary 'periodic' needs a periodic lattice; this one has boundary "
                             f"{lgca.bc!r}. Use 'no_flux' or a fixed value, {{'value': ...}}")
    elif periodic_lattice:
        raise ValueError(f"pde.boundary {boundary!r}: the lattice is periodic, so the field must be periodic "
                         "too; leave out boundary or set it to 'periodic'")
    if isinstance(boundary, str) and boundary == "periodic":
        return [("periodic", "periodic")] * ndim
    if isinstance(boundary, Mapping) and set(boundary) != {"value"}:
        allowed = set(_SIDES[:2 * ndim]) | {"default"}
        unknown = sorted(set(boundary) - allowed)
        if unknown:
            raise ValueError(f"pde.boundary has unknown keys {unknown}; per side, use {sorted(allowed)}, "
                             "or {'value': c} for all sides")
        if lgca.geometry not in _PER_SIDE_GEOMETRIES:
            raise ValueError(f"pde.boundary per side is available on 1D, square and cubic lattices, not on "
                             f"{lgca.geometry!r}; give one condition for all sides")
        default = _condition(boundary.get("default", "no_flux"), "pde.boundary.default")
        sides = []
        for axis in range(ndim):
            pair = []
            for side in _SIDES[2 * axis:2 * axis + 2]:
                if axis in wrapped:
                    if side in boundary:
                        raise ValueError(f"pde.boundary.{side}: the lattice wraps around along this axis")
                    pair.append("periodic")
                else:
                    pair.append(_condition(boundary[side], f"pde.boundary.{side}") if side in boundary
                                else default)
            sides.append(tuple(pair))
        return sides
    condition = _condition(boundary, "pde.boundary")
    return [("periodic", "periodic") if axis in wrapped else (condition, condition) for axis in range(ndim)]


def _condition(value, path):
    """None for no flux, a float for a fixed value."""
    if isinstance(value, str) and value == "no_flux":
        return None
    if isinstance(value, Mapping) and set(value) == {"value"}:
        return _non_negative(value["value"], f"{path}.value")
    raise ValueError(f"{path} must be 'no_flux' or {{'value': c}}, got {value!r}")


def _neighbours(lgca, sides):
    """The neighbour of every node through every velocity channel, and the value beyond each edge.

    Returns ``(rows, columns, fixed)``, flattened over nodes and channels (channel ``i`` of node ``x``
    at ``x * b + i``, pairing with ``lgca.c[:, i]``): the node, its neighbour (negative beyond the
    edge) and, for neighbours beyond the edge, the fixed value there (NaN for no flux).
    """
    dims = tuple(int(size) for size in lgca.dims)
    n = prod(dims)
    width = int(lgca.r_int)
    # interior nodes are numbered; ghost nodes hold their node (wrapped) or -1 - (2 axis + side)
    codes = np.arange(n, dtype=float).reshape(dims)
    for axis, (lower, _) in enumerate(sides):
        pad = [(0, 0)] * len(dims)
        pad[axis] = (width, width)
        if lower == "periodic":
            codes = np.pad(codes, pad, mode="wrap")
        else:
            codes = np.pad(codes, pad, mode="constant", constant_values=(-1 - 2 * axis, -2 - 2 * axis))
    interior = tuple(slice(width, width + size) for size in dims)
    neighbours = np.rint(lgca.channel_weight(codes)[interior]).astype(np.int64).reshape(n, -1)
    rows = np.repeat(np.arange(n), neighbours.shape[1])
    columns = neighbours.ravel()
    values = np.array([np.nan if condition is None else condition
                       for pair in sides for condition in (pair if pair[0] != "periodic" else (None, None))])
    fixed = np.full(len(columns), np.nan)
    fixed[columns < 0] = values[-1 - columns[columns < 0]]
    return rows, columns, fixed


def _channel_vectors(lgca):
    """The velocity channels as columns, shape ``(d, b)``, and the weight ``w = 2d / Σ|c_i|²``."""
    d = len(lgca.dims)
    c = np.asarray(lgca.c, dtype=float).reshape(d, -1)
    return c, 2 * d / float(np.sum(c**2))


def _assemble_laplacian(lgca, sides):
    """Sparse Laplacian over the velocity channels, and the source of fixed values beyond the edge."""
    n = prod(int(size) for size in lgca.dims)
    _, w = _channel_vectors(lgca)
    rows, columns, fixed = _neighbours(lgca, sides)
    inside = columns >= 0
    held = ~inside & ~np.isnan(fixed)  # a fixed value beyond the edge; no flux drops the face
    diagonal = -w * np.bincount(rows[inside | held], minlength=n).astype(float)
    source = w * np.bincount(rows[held], weights=fixed[held], minlength=n)
    matrix = sp.coo_matrix((np.full(inside.sum(), w), (rows[inside], columns[inside])), shape=(n, n))
    return (matrix + sp.diags(diagonal)).tocsr(), source


def _assemble_advection(lgca, sides, velocity):
    """``-∇·(v c) ≈ M c + s`` by first-order upwinding over the velocity channels.

    The flux through the face between ``x`` and its neighbour ``x + c_i`` is ``w (v·c_i) c`` with
    ``c`` taken at the upwind node and ``v`` averaged over the two nodes (``velocity`` has shape
    ``(n, d)``). Each face is shared by its two nodes, so the total is conserved; the off-diagonal
    entries are non-negative, so the implicit and steady solutions stay non-negative. No flux drops
    the faces beyond the edge; beyond a fixed value, inflow brings that value and outflow leaves.
    """
    n = prod(int(size) for size in lgca.dims)
    c, w = _channel_vectors(lgca)
    rows, columns, fixed = _neighbours(lgca, sides)
    along = velocity @ c  # v·c_i at every node, shape (n, b)
    flow = w * along.ravel()  # outward flow through each face
    inside = np.flatnonzero(columns >= 0)
    flow[inside] = w * (along.ravel()[inside] + along[columns[inside], inside % c.shape[1]]) / 2
    inside = columns >= 0
    beyond = ~inside & ~np.isnan(fixed)
    open_face = inside | beyond
    outflow = open_face & (flow > 0)
    inflow = flow < 0
    # (an empty bincount is of integers, even with weights)
    diagonal = -np.bincount(rows[outflow], weights=flow[outflow], minlength=n).astype(float)
    source = np.bincount(rows[inflow & beyond], weights=-flow[inflow & beyond] * fixed[inflow & beyond],
                         minlength=n).astype(float)
    taken = inflow & inside
    matrix = sp.coo_matrix((-flow[taken], (rows[taken], columns[taken])), shape=(n, n))
    return (matrix + sp.diags(diagonal)).tocsr(), source


# ---------------------------------------------------------------- registration

_FAMILIES = ("classical", "nove", "ib", "nove_ib")

PDE_INFO = PluginInfo(
    name="pde",
    operator_kind="field",
    backend_families=_FAMILIES,
    parameters={
        "field": ParameterSpec(required=True, type_label="string",
                               description="Name of the field in StateSpec.fields that the equation updates."),
        "diffusion": ParameterSpec(default=0.0, type_label="finite scalar",
                                   description="Diffusion coefficient D in nodes² per time step."),
        "decay": ParameterSpec(default=0.0, type_label="finite scalar",
                               description="Loss rate per time step everywhere."),
        "production": ParameterSpec(default=0.0,
                                    description="Production per node and time step independent of the "
                                                "cells: a number or the name of a field."),
        "cells": ParameterSpec(default=[],
                               description="Secretion and uptake by cells: a list of mappings such as "
                                           "{'production': r}, {'uptake': r} or {'uptake': r, "
                                           "'saturation': K, 'n': 1}, optionally with 'species' and "
                                           "'channels'. In identity-based models r may name a cell trait."),
        "reactions": ParameterSpec(default=[],
                                   description="Reactions registered with lgca.fields.reaction: a list of "
                                               "mappings {'name': ..., parameter: value, ...}."),
        "advection": ParameterSpec(default=None,
                                   description="Velocity v of the advection term -div(v c) in nodes per "
                                               "time step: a vector with one component per dimension, or "
                                               "the name of a field of velocities, shape dims + (d,)."),
        "boundary": ParameterSpec(default=None,
                                  description="'periodic' (periodic lattices), 'no_flux' (default "
                                              "otherwise), {'value': c} or, on 1D, square and cubic "
                                              "lattices, one condition per side ('x-', 'x+', ...)."),
        "solver": ParameterSpec(default="implicit", allowed_values=SOLVERS,
                                description="'implicit' (backward Euler, stable, keeps c >= 0), "
                                            "'explicit' (scipy.integrate.solve_ivp with error control) or "
                                            "'steady' (the field at equilibrium with the cells at every "
                                            "step, for fields much faster than the cells)."),
        "solver_options": ParameterSpec(default={},
                                        description="Implicit and steady: backend ('auto', 'direct', "
                                                    "'cg', 'amg'), rtol, max_iterations; implicit also "
                                                    "substeps. Explicit: method, rtol, atol, "
                                                    "warn_evaluations."),
    },
    conservation_law=_law_for_kind("field"),
    port_status="native",
    test_status="unit_tested",
    description="Updates a field by one time step of a reaction-advection-diffusion equation, "
                "dc/dt = D Laplace(c) - div(v c) + P - L c, with secretion and uptake by the cells "
                "and reactions of your own.",
)


def _pde_factory(parameters: Mapping[str, Any] | None = None) -> PDEOperator:
    return PDEOperator(PDE_INFO, parameters)


register_plugin(PDE_INFO, _pde_factory)
