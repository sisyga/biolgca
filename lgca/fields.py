"""Fields: reaction–diffusion equations on the lattice, coupled to the cells.

A field is a named array of ``StateSpec.fields``, one value per node, such as
the concentration of oxygen or of a chemoattractant. The pipeline operator
``pde`` updates one field by one LGCA time step of

.. math::

    \\partial_t c = D \\Delta c + P(x) - L(x)\\, c,

where ``D`` is the diffusion coefficient, ``P`` the production per node and
step (a constant, a map, and secretion by cells) and ``L`` the loss rate
(decay and uptake by cells). Parameters are in lattice units per LGCA step:
``D`` in nodes² per step, rates per step.

The Laplacian uses the neighbourhood of the cells,
``Δc(x) ≈ w Σ_i (c(x + c_i) − c(x))`` over the velocity channels ``c_i`` with
``w = 2d / Σ_i |c_i|²``, exact for quadratic functions on every lattice. It is
assembled once per model as a sparse matrix.

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
import inspect
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

__all__ = ["PDEOperator", "PDESpec", "laplacian"]

SOLVERS = ("explicit", "implicit")
_EXPLICIT_METHODS = ("RK23", "RK45", "DOP853", "BDF", "Radau")
_SOLVER_OPTIONS = {
    "explicit": {"method": "RK45", "rtol": 1e-4, "atol": 1e-6, "warn_evaluations": 200},
    "implicit": {"substeps": 1, "backend": "auto", "rtol": 1e-6, "max_iterations": 20},
}
_BACKENDS = ("auto", "direct", "cg")
_SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
_PER_SIDE_GEOMETRIES = ("lin", "square", "cubic")
_CELL_TERM_KEYS = {"production", "uptake", "saturation", "n", "species", "channels"}


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
    boundary : str or mapping, optional
        ``"periodic"`` (the default, and the only choice, on periodic
        lattices), ``"no_flux"`` (the default otherwise), ``{"value": c_b}``
        (fixed value beyond the edge) or, on 1D, square and cubic lattices,
        one condition per side, e.g. ``{"x-": {"value": 1.0}, "x+":
        {"value": 0.0}, "default": "no_flux"}``.
    solver : {"implicit", "explicit"}, default="implicit"
        ``"implicit"``: backward Euler, stable for any ``D`` and keeping
        ``c ≥ 0``; first order in time (use substeps for accuracy).
        ``"explicit"``: :func:`scipy.integrate.solve_ivp` with error
        control, accurate while the field changes slowly.
    solver_options : mapping, default={}
        Implicit: ``substeps`` (1), ``backend`` (``"auto"``, ``"direct"``,
        ``"cg"``), ``rtol`` (1e-6, for the iterative solver and the
        iteration of saturating uptake), ``max_iterations`` (20).
        Explicit: ``method`` (``"RK45"``; also ``"RK23"``, ``"DOP853"``,
        ``"BDF"``, ``"Radau"``), ``rtol`` (1e-4), ``atol`` (1e-6),
        ``warn_evaluations`` (200).

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
    boundary: str | Mapping[str, Any] | None = None
    solver: str = "implicit"
    solver_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def parameters(self) -> dict[str, Any]:
        """The parameters of the ``pde`` operator; defaults are left out."""
        parameters = {"field": self.field}
        defaults = {"diffusion": 0.0, "decay": 0.0, "production": 0.0, "boundary": None,
                    "solver": "implicit"}
        for name in ("diffusion", "decay", "production", "cells", "boundary", "solver", "solver_options"):
            value = getattr(self, name)
            if name in ("cells", "solver_options"):
                if value:
                    parameters[name] = ([dict(term) for term in value] if name == "cells" else dict(value))
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
        self.boundary = parameters.get("boundary")
        self.solver = parameters.get("solver", "implicit")
        if self.solver == "steady":
            raise ValueError("pde.solver 'steady' is not available yet; use 'implicit' with a large "
                             "diffusion coefficient, or several substeps")
        if self.solver not in SOLVERS:
            raise ValueError(f"pde.solver must be one of {', '.join(SOLVERS)}, got {self.solver!r}")
        self.options = _solver_options(self.solver, parameters.get("solver_options") or {})
        self.statistics: dict[str, Any] = {"solver": self.solver, "calls": 0}
        self._warned = False

    # --------------------------------------------------------------- pipeline

    def dependencies(self) -> set[str]:
        read = {self.field}
        if isinstance(self.production, str):
            read.add(self.production)
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
        self.laplacian, boundary_source = _assemble_laplacian(lgca, self._sides)
        n = self.laplacian.shape[0]
        # the linear part without cells: A c + b
        self._A = (self.diffusion * self.laplacian - self.decay * sp.identity(n, format="csr")).tocsr()
        self._b = self.diffusion * boundary_source
        for term in self.cell_terms:
            term.setup(lgca, context.spec.state)
        self._lu = None
        self.statistics.update({"solver": self.solver, "calls": 0})
        context.metadata.setdefault("fields", {})[self.field] = self.statistics

    def attach_field(self, lgca) -> None:
        """Store the field as a float array with ghost nodes filled by its boundary condition."""
        values = np.asarray(getattr(lgca, self.field), dtype=float)
        if values.shape != self._dims:
            values = values[self._interior]
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"the initial values of field {self.field!r} must be finite and non-negative")
        setattr(lgca, self.field, self._pad(values))

    def apply(self, context, step: int) -> None:
        lgca = context.lgca
        stored = getattr(lgca, self.field)
        c = np.array(stored[self._interior], dtype=float).ravel()
        production, loss, saturating = self._sources(lgca, step)
        if self.solver == "explicit":
            c = self._explicit(c, production, loss, saturating)
        else:
            c = self._implicit(c, production, loss, saturating)
        stored[...] = self._pad(c.reshape(self._dims))
        self.statistics["calls"] += 1

    # ---------------------------------------------------------------- sources

    def _sources(self, lgca, step):
        """Production and linear loss rate per node from the cells and the map; saturating terms."""
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
        saturating = []
        if self.cell_terms:
            state = LatticeState(lgca, step=step)
            for term in self.cell_terms:
                weights = term.weights(state)
                if term.kind == "production":
                    production += weights
                elif term.saturation is None:
                    loss += weights
                else:
                    saturating.append((weights, term.saturation, term.n))
        return production, loss, saturating

    # ---------------------------------------------------------------- solvers

    def _explicit(self, c, production, loss, saturating):
        options = self.options
        A = self._A

        def rhs(_t, values):
            change = A @ values + production - loss * values
            for weights, K, n in saturating:
                change -= weights * _hill(values, K, n)
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

    def _implicit(self, c, production, loss, saturating):
        options = self.options
        dt = 1.0 / options["substeps"]
        base = self._implicit_base(dt)
        constant = not saturating and not np.any(loss)
        iterations_used = 0
        for _ in range(options["substeps"]):
            right = c + dt * production
            if constant:
                c = self._solve_constant(base, right)
                continue
            if not saturating:
                c = self._solve(base + sp.diags(dt * loss), right, c)
                continue
            previous = c
            for iteration in range(1, options["max_iterations"] + 1):
                total = loss + sum(weights * _hill_rate(previous, K, n) for weights, K, n in saturating)
                new = self._solve(base + sp.diags(dt * total), right, previous)
                change = np.max(np.abs(new - previous), initial=0.0)
                previous = new
                if change <= options["rtol"] * max(np.max(np.abs(new), initial=0.0), 1e-300):
                    break
            else:
                if not self._warned:
                    self._warned = True
                    warn_user(f"pde {self.field!r}: saturating uptake did not converge in "
                              f"{options['max_iterations']} iterations; raise solver_options "
                              "'max_iterations' or use substeps")
            iterations_used = max(iterations_used, iteration)
            c = previous
        if saturating:
            stats = self.statistics
            stats["max_iterations_used"] = max(stats.get("max_iterations_used", 0), iterations_used)
        return c

    def _implicit_base(self, dt):
        if getattr(self, "_base_dt", None) != dt:
            n = self._A.shape[0]
            self._base = (sp.identity(n, format="csr") - dt * self._A).tocsr()
            self._base_dt = dt
            self._lu = None
        return self._base

    def _solve_constant(self, matrix, right):
        """Solve with a matrix that does not change between steps: factor it once."""
        if self.options["backend"] == "cg":
            return self._solve(matrix, right, right)
        if self._lu is None:
            self._lu = spla.splu(matrix.tocsc())
        return self._lu.solve(right)

    def _solve(self, matrix, right, start):
        if self.options["backend"] == "direct":
            return spla.spsolve(matrix.tocsc(), right)
        # symmetric positive definite: conjugate gradients, Jacobi preconditioner, previous field as start
        preconditioner = sp.diags(1.0 / matrix.diagonal())
        values, info = _cg(matrix, right, x0=start, rtol=self.options["rtol"], M=preconditioner)
        if info != 0:
            raise RuntimeError(f"pde {self.field!r}: the iterative solver did not converge "
                               f"(scipy.sparse.linalg.cg returned {info}); try solver_options "
                               "{'backend': 'direct'}")
        # rounding of the iterative solver may leave tiny negative values where c is nearly zero
        return np.maximum(values, 0.0)

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

def _hill(c, K, n):
    c = np.maximum(c, 0.0)
    return c**n / (K**n + c**n)


def _hill_rate(c, K, n):
    """The Hill uptake as a loss rate: c^(n-1) / (K^n + c^n), finite at c = 0."""
    c = np.maximum(c, 0.0)
    return c**(n - 1) / (K**n + c**n)


def _cg(matrix, right, *, x0, rtol, M):
    """scipy.sparse.linalg.cg with a relative tolerance; SciPy < 1.12 calls it ``tol``."""
    if "rtol" in _CG_PARAMETERS:
        return spla.cg(matrix, right, x0=x0, rtol=rtol, atol=0.0, M=M, maxiter=10 * len(right))
    return spla.cg(matrix, right, x0=x0, tol=rtol, atol=0.0, M=M, maxiter=10 * len(right))


_CG_PARAMETERS = inspect.signature(spla.cg).parameters


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


def _assemble_laplacian(lgca, sides):
    """Sparse Laplacian over the velocity channels, and the source of fixed values beyond the edge."""
    dims = tuple(int(size) for size in lgca.dims)
    n = prod(dims)
    width = int(lgca.r_int)
    c = np.asarray(lgca.c, dtype=float).reshape(len(dims), -1)
    w = 2 * len(dims) / float(np.sum(c**2))
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
    inside = columns >= 0
    diagonal = -w * np.bincount(rows[inside], minlength=n).astype(float)
    source = np.zeros(n)
    ghost = -1 - columns[~inside]
    values = np.array([np.nan if condition is None else condition
                       for pair in sides for condition in (pair if pair[0] != "periodic" else (None, None))])
    fixed = values[ghost]
    held = ~np.isnan(fixed)
    ghost_rows = rows[~inside][held]
    diagonal -= w * np.bincount(ghost_rows, minlength=n)
    source += w * np.bincount(ghost_rows, weights=fixed[held], minlength=n)
    matrix = sp.coo_matrix((np.full(inside.sum(), w), (rows[inside], columns[inside])), shape=(n, n))
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
        "boundary": ParameterSpec(default=None,
                                  description="'periodic' (periodic lattices), 'no_flux' (default "
                                              "otherwise), {'value': c} or, on 1D, square and cubic "
                                              "lattices, one condition per side ('x-', 'x+', ...)."),
        "solver": ParameterSpec(default="implicit", allowed_values=SOLVERS,
                                description="'implicit' (backward Euler, stable, keeps c >= 0) or "
                                            "'explicit' (scipy.integrate.solve_ivp with error control)."),
        "solver_options": ParameterSpec(default={},
                                        description="Implicit: substeps, backend ('auto', 'direct', 'cg'), "
                                                    "rtol, max_iterations. Explicit: method, rtol, atol, "
                                                    "warn_evaluations."),
    },
    conservation_law=_law_for_kind("field"),
    port_status="native",
    test_status="unit_tested",
    description="Updates a field by one time step of a reaction-diffusion equation, "
                "dc/dt = D Laplace(c) + P - L c, with secretion and uptake by the cells.",
)


def _pde_factory(parameters: Mapping[str, Any] | None = None) -> PDEOperator:
    return PDEOperator(PDE_INFO, parameters)


register_plugin(PDE_INFO, _pde_factory)
