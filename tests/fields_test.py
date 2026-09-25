"""Fields updated by the pde operator: the Laplacian, boundaries, solvers and the coupling to cells."""

import gc
import json
import subprocess
import sys

import jsonschema
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

from lgca import get_lgca
from lgca.fields import PDESpec, laplacian
from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    describe_model_graph,
    load_model_spec,
    load_model_spec_schema,
    run_model,
    save_model_spec,
)
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import FieldRecorder, Schedule, estimate_recording_bytes

GEOMETRIES = {"lin": 20, "square": (10, 8), "hex": (10, 8), "cubic": (6, 5, 4), "moore": (6, 5, 4)}
FAMILIES = {"classical": {}, "nove": {"volume_exclusion": False, "capacity": 10},
            "ib": {"identity_based": True},
            "nove_ib": {"identity_based": True, "volume_exclusion": False, "capacity": 10}}


def _coordinates(lgca):
    names = ("xcoords", "ycoords", "zcoords")[:len(lgca.dims)]
    return [np.asarray(getattr(lgca, name), dtype=float) for name in names]


def _build(operators, *, geometry="square", dims=(10, 8), boundary="reflecting", fields=None, steps=1,
           seed=1, observers=(), propagation=False, **state):
    if "nodes" not in state:
        state.setdefault("density", 0.0)
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims, boundary=boundary),
        state=StateSpec(fields={"u": 0.0} if fields is None else fields, **state),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=propagation),
        analysis=AnalysisSpec(observers=list(observers))))


def _field(compiled, name="u"):
    return compiled.lgca.__dict__[name][compiled.lgca.nonborder]


# ------------------------------------------------------------------ Laplacian

@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_laplacian_is_exact_for_quadratics(geometry):
    lgca = get_lgca(geometry=geometry, dims=GEOMETRIES[geometry], bc="reflecting", density=0,
                    interaction="only_propagation")
    A, _ = laplacian(lgca)
    coordinates = _coordinates(lgca)
    square = sum(x**2 for x in coordinates)
    inner = np.ones(lgca.dims, dtype=bool)  # nodes whose neighbours are all inside
    for axis in range(len(lgca.dims)):
        index = [slice(None)] * len(lgca.dims)
        for edge in (0, -1):
            index[axis] = edge
            inner[tuple(index)] = False
    result = (A @ square.ravel()).reshape(lgca.dims)
    np.testing.assert_allclose(result[inner], 2 * len(lgca.dims), rtol=1e-12)
    if len(lgca.dims) > 1:  # a mixed term has no Laplacian
        mixed = coordinates[0] * coordinates[1]
        np.testing.assert_allclose((A @ mixed.ravel()).reshape(lgca.dims)[inner], 0, atol=1e-10)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
def test_laplacian_is_symmetric_and_conserves(geometry, boundary):
    lgca = get_lgca(geometry=geometry, dims=GEOMETRIES[geometry], bc=boundary, density=0,
                    interaction="only_propagation")
    A, source = laplacian(lgca)
    assert abs(A - A.T).max() == 0
    np.testing.assert_allclose(np.asarray(A.sum(axis=0)).ravel(), 0, atol=1e-12)
    assert not source.any()


def test_fixed_values_enter_as_a_source():
    lgca = get_lgca(geometry="square", dims=(3, 3), bc="reflecting", density=0, interaction="only_propagation")
    A, source = laplacian(lgca, {"x-": {"value": 2.0}, "y+": {"value": 1.0}})
    constant = np.full(9, 1.0)
    # with the field equal to the fixed values' mean nothing but the edges contribute
    result = (A @ constant + source).reshape(3, 3)
    np.testing.assert_allclose(result[0], [1, 1, 1])  # x- neighbours hold 2, the field 1
    np.testing.assert_allclose(result[1:, :], 0)


def test_steady_state_with_fixed_values_converges_at_second_order():
    """c'' = (k / D) c with c = 1 beyond both ends: the discrete solution against the cosh profile."""
    errors = []
    for length in (20, 40, 80):
        lam = 4.0 / length  # the same continuous problem on every lattice
        lgca = get_lgca(geometry="lin", dims=length, bc="reflecting", density=0, interaction="only_propagation")
        A, source = laplacian(lgca, {"value": 1.0})
        c = spla.spsolve((A - lam**2 * sp.identity(length)).tocsc(), -source)
        x = np.arange(length)
        middle = (length - 1) / 2
        exact = np.cosh(lam * (x - middle)) / np.cosh(lam * (middle + 1))
        errors.append(np.abs(c - exact).max())
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 3.8), errors


# ------------------------------------------------------------------ solvers

SOLVERS = [("implicit", {}), ("implicit", {"backend": "cg"}), ("implicit", {"backend": "direct"}),
           ("explicit", {}), ("explicit", {"method": "BDF"})]


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
@pytest.mark.parametrize("solver,options", SOLVERS)
def test_diffusion_conserves_the_total(geometry, boundary, solver, options):
    rng = np.random.default_rng(3)
    initial = rng.random(GEOMETRIES[geometry] if geometry != "lin" else (GEOMETRIES[geometry],))
    compiled = _build([PDESpec(field="u", diffusion=0.7, solver=solver, solver_options=options)],
                      geometry=geometry, dims=GEOMETRIES[geometry], boundary=boundary, fields={"u": initial})
    for _ in range(3):
        compiled.step()
    c = _field(compiled)
    assert c.sum() == pytest.approx(initial.sum(), rel=1e-6 if options.get("backend") == "cg" else 1e-12)
    assert c.std() < initial.std()


@pytest.mark.parametrize("substeps", [1, 4])
def test_implicit_decay_is_the_discrete_exponential(substeps):
    compiled = _build([PDESpec(field="u", decay=0.3, solver_options={"substeps": substeps})],
                      fields={"u": 2.0})
    for _ in range(3):
        compiled.step()
    np.testing.assert_allclose(_field(compiled), 2.0 * (1 + 0.3 / substeps) ** (-3 * substeps), rtol=1e-14)


def test_explicit_decay_follows_the_exponential():
    compiled = _build([PDESpec(field="u", decay=0.3, solver="explicit", solver_options={"rtol": 1e-8,
                                                                                        "atol": 1e-12})],
                      fields={"u": 2.0})
    for _ in range(3):
        compiled.step()
    np.testing.assert_allclose(_field(compiled), 2.0 * np.exp(-0.9), rtol=1e-7)


def test_implicit_approaches_the_explicit_solution_with_substeps():
    initial = np.zeros((20, 20))
    initial[10, 10] = 100.0
    results = {}
    for name, solver, options in [("reference", "explicit", {"rtol": 1e-9, "atol": 1e-12,
                                                             "warn_evaluations": 10_000}),
                                  ("one", "implicit", {}), ("many", "implicit", {"substeps": 16})]:
        compiled = _build([PDESpec(field="u", diffusion=0.5, decay=0.05, solver=solver, solver_options=options)],
                          dims=(20, 20), fields={"u": initial})
        compiled.step()
        results[name] = _field(compiled)
    one = np.abs(results["one"] - results["reference"]).max()
    many = np.abs(results["many"] - results["reference"]).max()
    assert many < one / 8  # first order in time: 16 substeps, about 16 times smaller


def test_implicit_long_time_limit_is_the_steady_state():
    compiled = _build([PDESpec(field="u", diffusion=2.0, decay=0.1, boundary={"value": 1.0},
                               solver_options={"substeps": 1})], geometry="lin", dims=30)
    for _ in range(400):
        compiled.step()
    A, source = laplacian(compiled.lgca, {"value": 1.0})
    steady = spla.spsolve((2.0 * A - 0.1 * sp.identity(30)).tocsc(), -2.0 * source)
    np.testing.assert_allclose(_field(compiled), steady, rtol=1e-10)


def test_implicit_stays_non_negative_under_strong_uptake():
    nodes = np.zeros((20, 20, 5), dtype=bool)
    nodes[5:15, 5:15] = True
    compiled = _build([PDESpec(field="u", diffusion=5.0, cells=[{"uptake": 50.0}],
                               solver_options={"backend": "cg"})],
                      dims=(20, 20), fields={"u": 1.0}, nodes=nodes, restchannels=1)
    for _ in range(5):
        compiled.step()
    c = _field(compiled)
    assert c.min() >= 0 and c[10, 10] < 1e-6


def test_explicit_solver_warns_when_the_field_is_too_fast():
    compiled = _build([PDESpec(field="u", diffusion=200.0, solver="explicit")], dims=(10, 10),
                      fields={"u": np.random.default_rng(0).random((10, 10))})
    with pytest.warns(UserWarning, match="solver='implicit' is faster"):
        compiled.step()
    assert compiled.metadata["fields"]["u"]["max_rhs_evaluations"] > 200


def test_negative_explicit_values_are_an_error():
    compiled = _build([PDESpec(field="u", decay=40.0, solver="explicit",
                               solver_options={"method": "RK23", "rtol": 0.5, "atol": 1e-9})],
                      geometry="lin", dims=10, fields={"u": 1.0})
    with pytest.raises(RuntimeError, match="negative values"):
        compiled.step()


@pytest.mark.parametrize("parameters,message", [
    ({"solver": "explicit", "solver_options": {"method": "LSODA"}}, "LSODA"),
    ({"solver": "steady", "solver_options": {"substeps": 2}}, "do not apply"),
    ({"solver_options": {"method": "RK45"}}, "do not apply"),
    ({"diffusion": -1.0}, "non-negative"),
    ({"cells": [{"uptake": 1.0, "production": 1.0}]}, "exactly one"),
    ({"cells": [{"uptake": 1.0, "n": 2}]}, "saturation"),
    ({"cells": [{"uptake": "rate"}]}, "identity-based"),
    ({"boundary": "periodic"}, "needs a periodic lattice"),
    ({"boundary": {"x-": "open"}}, "no_flux"),
])
def test_invalid_parameters_are_explained(parameters, message):
    with pytest.raises(ValueError, match=message):
        _build([{"name": "pde", "parameters": {"field": "u", **parameters}}])


def test_periodic_lattice_needs_a_periodic_field_and_sides_need_a_simple_lattice():
    with pytest.raises(ValueError, match="must be periodic"):
        _build([PDESpec(field="u", boundary="no_flux")], boundary="periodic")
    with pytest.raises(ValueError, match="per side"):
        _build([PDESpec(field="u", boundary={"x-": {"value": 1.0}})], geometry="hex")
    with pytest.raises(ValueError, match="not a field"):
        _build([PDESpec(field="oxygen")])


# ------------------------------------------------------------------ steady solver

def _disc(size):
    nodes = np.zeros((size, size, 5), dtype=bool)
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    nodes[(x - size / 2) ** 2 + (y - size / 2) ** 2 < (size / 4) ** 2] = True
    return nodes


def _steady_residual(compiled, diffusion, decay, uptake=0.0, saturation=None, production=0.0):
    """D Δc + b - decay c - uptake(c) + P at every node, from the field the model stores."""
    lgca = compiled.lgca
    c = _field(compiled).ravel()
    pde = next(operator for operator in compiled.pipeline.operators if operator.name == "pde")
    A, source = laplacian(lgca, pde.boundary)
    n = _counts(compiled).sum(axis=(-2, -1)).ravel()
    taken = uptake * n * (c if saturation is None else c / (saturation + c))
    return diffusion * (A @ c + source) - decay * c - taken + production


@pytest.mark.parametrize("backend", ["auto", "direct", "cg", "amg"])
@pytest.mark.parametrize("saturation", [None, 0.2])
def test_steady_solves_the_equation(backend, saturation):
    term = {"uptake": 0.3} if saturation is None else {"uptake": 0.3, "saturation": saturation}
    compiled = _build([PDESpec(field="u", diffusion=4.0, decay=0.01, cells=[term], boundary={"value": 1.0},
                               solver="steady", solver_options={"backend": backend, "rtol": 1e-10})],
                      dims=(30, 30), nodes=_disc(30), restchannels=1, fields={"u": 0.0})
    # solved when built, before any step
    residual = _steady_residual(compiled, 4.0, 0.01, 0.3, saturation)
    assert np.abs(residual).max() < 1e-7
    assert _field(compiled).min() >= 0 and _field(compiled)[15, 15] < 0.5


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_steady_backends_agree_on_every_lattice(geometry):
    dims = GEOMETRIES[geometry]
    results = []
    for backend in ("direct", "amg", "cg"):
        compiled = _build([PDESpec(field="u", diffusion=2.0, cells=[{"uptake": 0.5}, {"production": 0.2}],
                                   solver="steady", solver_options={"backend": backend, "rtol": 1e-10})],
                          geometry=geometry, dims=dims, density=1.0, restchannels=1, seed=2)
        results.append(_field(compiled))
    np.testing.assert_allclose(results[1], results[0], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(results[2], results[0], rtol=1e-7, atol=1e-10)


def test_steady_point_source_matches_the_fft_solution():
    """Periodic square lattice: D Δc - k c + P δ = 0, solved in Fourier space with the same stencil."""
    size, D, k = 32, 1.5, 0.02
    source = np.zeros((size, size))
    source[5, 9] = 2.0
    compiled = _build([PDESpec(field="u", diffusion=D, decay=k, production="source", solver="steady",
                               solver_options={"rtol": 1e-12})],
                      dims=(size, size), boundary="periodic", fields={"u": 0.0, "source": source})
    q = 2 * np.pi * np.fft.fftfreq(size)
    symbol = D * (2 * np.cos(q)[:, None] + 2 * np.cos(q)[None, :] - 4) - k
    exact = np.real(np.fft.ifft2(np.fft.fft2(-source) / symbol))
    np.testing.assert_allclose(_field(compiled), exact, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("solver", ["implicit", "explicit"])
def test_steady_is_the_long_time_limit(solver):
    def model(solver, **options):
        return _build([PDESpec(field="u", diffusion=1.0, decay=0.05, cells=[{"uptake": 0.2}],
                               boundary={"value": 1.0}, solver=solver, solver_options=options)],
                      dims=(16, 16), nodes=_disc(16), restchannels=1, fields={"u": 1.0})
    steady = model("steady", rtol=1e-12)
    transient = model(solver, **({"substeps": 1, "rtol": 1e-12} if solver == "implicit" else
                                 {"rtol": 1e-8, "atol": 1e-12}))
    for _ in range(300):
        transient.step()
    np.testing.assert_allclose(_field(transient), _field(steady), rtol=1e-8)


def test_steady_reuses_the_multigrid_hierarchy():
    turnover = {"name": "birth_death", "parameters": {"birth_rate": 0.1, "death_rate": 0.1}}
    compiled = _build([turnover,  # the cells change every step, and the pde sees them as they are after it
                       PDESpec(field="u", diffusion=1.0, decay=0.001, cells=[{"uptake": 0.05}],
                               boundary={"value": 1.0}, solver="steady", solver_options={"backend": "amg"})],
                      dims=(60, 60), nodes=_disc(60), restchannels=1, fields={"u": 1.0})
    for _ in range(20):
        compiled.step()
        residual = _steady_residual(compiled, 1.0, 0.001, 0.05)
        assert np.abs(residual).max() < 1e-5
    assert 0.01 < _field(compiled).min() < 0.5
    statistics = compiled.metadata["fields"]["u"]
    assert statistics["amg_setups"] < 10  # the matrix changes every step; the hierarchy is kept most of the time
    assert statistics["calls"] == 20


def test_steady_without_pyamg(monkeypatch):
    import lgca.fields
    monkeypatch.setattr(lgca.fields, "_pyamg", lambda: None)
    with pytest.raises(ValueError, match="needs pyamg"):
        _build([PDESpec(field="u", decay=1.0, solver="steady", solver_options={"backend": "amg"})])
    small = _build([PDESpec(field="u", diffusion=1.0, decay=0.1, cells=[{"uptake": 0.2}], solver="steady")],
                   density=1.0, restchannels=1)
    assert small.metadata["fields"]["u"]["backend"] == "direct"
    monkeypatch.setattr(lgca.fields, "_DIRECT_LIMIT", 10)
    large = _build([PDESpec(field="u", diffusion=1.0, decay=0.1, cells=[{"uptake": 0.2}], solver="steady")],
                   density=1.0, restchannels=1)
    assert large.metadata["fields"]["u"]["backend"] == "cg"
    np.testing.assert_allclose(_field(large), _field(small), rtol=1e-5)


def test_steady_needs_something_that_removes_the_field():
    with pytest.raises(ValueError, match="no unique steady state"):
        _build([PDESpec(field="u", diffusion=1.0, production=1.0, solver="steady")])
    with pytest.raises(ValueError, match="without diffusion needs decay"):
        _build([PDESpec(field="u", cells=[{"uptake": 1.0}], solver="steady")], density=1.0)
    with pytest.raises(RuntimeError, match="no cells take up the field"):
        _build([PDESpec(field="u", diffusion=1.0, cells=[{"uptake": 1.0}], solver="steady")])  # no cells
    _build([PDESpec(field="u", diffusion=1.0, solver="steady", boundary={"value": 1.0})])  # a fixed value


# ------------------------------------------------------------------ boundaries and ghost nodes

@pytest.mark.parametrize("boundary,geometry,lattice", [
    ("no_flux", "square", "reflecting"), ({"value": 3.0}, "square", "reflecting"),
    ("periodic", "square", "periodic"), ({"value": 3.0}, "hex", "absorbing"), ("no_flux", "moore", "reflecting")])
def test_ghost_nodes_follow_the_boundary(boundary, geometry, lattice):
    dims = (6, 5, 4) if geometry == "moore" else (6, 4)
    initial = np.random.default_rng(1).random(dims)
    compiled = _build([PDESpec(field="u", diffusion=0.2, boundary=boundary)], geometry=geometry, dims=dims,
                      boundary=lattice, fields={"u": initial})
    for _ in range(2):  # at build time and after steps
        padded = compiled.lgca.u
        interior = padded[compiled.lgca.nonborder]
        if boundary == "periodic":
            expected = np.pad(interior, 1, mode="wrap")
        elif boundary == "no_flux":
            expected = np.pad(interior, 1, mode="edge")
        else:
            expected = np.pad(interior, 1, mode="constant", constant_values=3.0)
        np.testing.assert_array_equal(padded, expected)
        compiled.step()


def test_linear_profile_between_fixed_values_gives_an_even_gradient():
    compiled = _build([PDESpec(field="u", diffusion=50.0, boundary={"x-": {"value": 1.0}, "x+": {"value": 0.0}})],
                      dims=(10, 4))
    for _ in range(50):
        compiled.step()
    from lgca.lattice_state import LatticeState
    gradient = LatticeState(compiled.lgca).gradient("u")
    np.testing.assert_allclose(gradient[..., 0], -1 / 11, rtol=1e-6)  # also at the edges
    np.testing.assert_allclose(gradient[..., 1], 0, atol=1e-10)


# ------------------------------------------------------------------ coupling to cells

def _counts(compiled):
    from lgca.lattice_state import LatticeState
    return LatticeState(compiled.lgca).counts


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("geometry", ["lin", "hex", "cubic"])
def test_uptake_and_production_per_cell(family, geometry):
    dims = GEOMETRIES[geometry]
    uptake = _build([PDESpec(field="u", cells=[{"uptake": 0.2}])], geometry=geometry, dims=dims,
                    fields={"u": 1.0}, density=1.5, restchannels=1, **FAMILIES[family])
    n = _counts(uptake).sum(axis=(-2, -1))
    uptake.step()
    np.testing.assert_allclose(_field(uptake), 1 / (1 + 0.2 * n), rtol=1e-12)
    explicit = _build([PDESpec(field="u", cells=[{"production": 0.5}], solver="explicit")], geometry=geometry,
                      dims=dims, density=1.5, restchannels=1, **FAMILIES[family])
    n = _counts(explicit).sum(axis=(-2, -1))
    explicit.step()
    np.testing.assert_allclose(_field(explicit), 0.5 * n, rtol=1e-12)


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_species_and_channels_select_the_cells(volume_exclusion):
    rng = np.random.default_rng(2)
    nodes = rng.integers(0, 2 if volume_exclusion else 3, size=(12, 10, 2, 5))
    nodes = nodes.astype(bool) if volume_exclusion else nodes
    terms = [{"production": 1.0, "species": 1}, {"production": 10.0, "channels": "rest"},
             {"production": 100.0, "species": [0], "channels": "velocity"}]
    compiled = _build([PDESpec(field="u", cells=terms)], dims=(12, 10), nodes=nodes, restchannels=1,
                      n_species=2, volume_exclusion=volume_exclusion, capacity=None if volume_exclusion else 20)
    compiled.step()
    counts = nodes.astype(int)
    expected = counts[:, :, 1].sum(-1) + 10 * counts[..., 4:].sum((-2, -1)) + 100 * counts[:, :, 0, :4].sum(-1)
    np.testing.assert_allclose(_field(compiled), expected)


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_rates_can_be_cell_traits(volume_exclusion):
    compiled = _build([PDESpec(field="u", cells=[{"production": "secretion"}])], dims=(8, 8), density=2.0,
                      restchannels=1, identity_based=True, volume_exclusion=volume_exclusion,
                      capacity=None if volume_exclusion else 10, traits={"secretion": 0.0})
    lgca = compiled.lgca
    rates = np.random.default_rng(4).random(len(lgca.props["secretion"]))
    lgca.props["secretion"][:] = rates
    from lgca.lattice_state import LatticeState
    cells = LatticeState(lgca).cells
    expected = np.bincount(cells.index, weights=rates[cells.label], minlength=64).reshape(8, 8)
    assert expected.sum() > 0
    compiled.step()
    np.testing.assert_allclose(_field(compiled), expected, rtol=1e-12)


@pytest.mark.parametrize("solver", ["implicit", "explicit"])
def test_saturating_uptake(solver):
    """Michaelis-Menten uptake at one node: dc/dt = -r n c / (K + c)."""
    nodes = np.zeros((4, 3), dtype=bool)
    nodes[0, :] = True  # three cells at the first node
    options = {"rtol": 1e-10, "atol": 1e-12} if solver == "explicit" else {"rtol": 1e-12, "max_iterations": 50}
    compiled = _build([PDESpec(field="u", cells=[{"uptake": 0.4, "saturation": 0.5}], solver=solver,
                               solver_options=options)],
                      geometry="lin", dims=4, fields={"u": 2.0}, nodes=nodes, restchannels=1)
    compiled.step()
    c = _field(compiled)
    if solver == "explicit":
        reference = solve_ivp(lambda t, y: -1.2 * y / (0.5 + y), (0, 1), [2.0], rtol=1e-12, atol=1e-14).y[0, -1]
        assert c[0] == pytest.approx(reference, rel=1e-8)
    else:  # backward Euler: c = c_old - r n c / (K + c)
        assert c[0] == pytest.approx(2.0 - 1.2 * c[0] / (0.5 + c[0]), rel=1e-10)
    np.testing.assert_allclose(c[1:], 2.0)


def test_the_field_sees_the_cells_as_the_previous_operator_left_them():
    death = {"name": "birth_death", "parameters": {"birth_rate": 0.0, "death_rate": 1.0}}
    secretion = PDESpec(field="u", cells=[{"production": 1.0}])
    before = _build([secretion, death], density=2.0, restchannels=1)
    after = _build([death, secretion], density=2.0, restchannels=1)
    n = _counts(before).sum(axis=(-2, -1))
    before.step()
    after.step()
    np.testing.assert_allclose(_field(before), n)
    np.testing.assert_allclose(_field(after), 0)


def test_chemotaxis_toward_a_secreted_signal_aggregates_the_cells():
    def run(beta):
        compiled = _build([PDESpec(field="signal", diffusion=1.0, decay=0.05, cells=[{"production": 0.1}]),
                           ReorientationSpec(terms=[ReorientationTermSpec(name="chemotaxis", beta=beta,
                                                                          parameters={"field": "signal"})])],
                          dims=(30, 30), fields={"signal": 0.0}, density=0.5, restchannels=1, propagation=True,
                          seed=5)
        for _ in range(60):
            compiled.step()
        density = _counts(compiled).sum(axis=(-2, -1))
        return density.var() / density.mean()  # 1 for independent cells, larger for clusters
    assert run(beta=6.0) > 2 * run(beta=0.0)


def test_production_from_a_map_and_decay():
    source = np.zeros((6, 4))
    source[2, 1] = 3.0
    compiled = _build([PDESpec(field="u", production="source")], dims=(6, 4),
                      fields={"u": 0.0, "source": source})
    compiled.step()
    np.testing.assert_allclose(_field(compiled), source)


# ------------------------------------------------------------------ recording, model files, CLI

def test_field_recorder_records_the_history():
    compiled = _build([PDESpec(field="u", production=1.0)], fields={"u": 0.0, "other": 2.0}, steps=6,
                      observers=[FieldRecorder(["u", "other"], schedule=Schedule(every=3))])
    result = compiled.run(showprogress=False)
    assert result.data["u"].shape == (3, 10, 8)
    np.testing.assert_array_equal(result.data.steps("u"), [0, 3, 6])
    np.testing.assert_allclose(result.data["u"][:, 0, 0], [0, 3, 6])
    np.testing.assert_allclose(result.data["other"], 2.0)
    assert estimate_recording_bytes(compiled.lgca, 6, [FieldRecorder(["u", "other"], Schedule(every=3))]) \
        >= 3 * 2 * 80 * 8


def test_field_recorder_rejects_clashing_names():
    with pytest.raises(ValueError, match="name of a recording"):
        _build([], fields={"population": 1.0}, observers=[FieldRecorder("population")]).run(showprogress=False)
    with pytest.raises(ValueError, match="several FieldRecorders"):
        _build([], observers=[FieldRecorder("u"), FieldRecorder("u")]).run(showprogress=False)


def _secretion_model(tmp_path=None):
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(12, 12), boundary="reflecting"),
        state=StateSpec(density=0.5, restchannels=1, fields={"signal": 0.0}),
        time=TimeSpec(steps=4, seed=9),
        dynamics=InteractionPipelineSpec(operators=[
            PDESpec(field="signal", diffusion=1.0, decay=0.1, cells=[{"production": 0.2}],
                    boundary={"x-": {"value": 1.0}, "default": "no_flux"}),
            ReorientationSpec(terms=[ReorientationTermSpec(name="chemotaxis", beta=3.0,
                                                           parameters={"field": "signal"})])]),
        analysis=AnalysisSpec(observers=[FieldRecorder(["signal"], schedule=Schedule(every=2))]))


def test_model_file_round_trip(tmp_path):
    spec = _secretion_model()
    path = save_model_spec(spec, tmp_path / "model.json")
    data = json.loads(path.read_text())
    jsonschema.Draft202012Validator(load_model_spec_schema()).validate(data)
    assert data["model"]["dynamics"]["operators"][0]["name"] == "pde"
    original = run_model(spec, showprogress=False)
    loaded = run_model(load_model_spec(path), showprogress=False)
    np.testing.assert_array_equal(loaded.data["signal"], original.data["signal"])
    np.testing.assert_array_equal(loaded.lgca.nodes, original.lgca.nodes)


def test_schema_rejects_an_invalid_pde():
    spec = _secretion_model()
    validator = jsonschema.Draft202012Validator(load_model_spec_schema())
    from lgca.model import model_spec_to_dict
    data = model_spec_to_dict(spec)
    data["model"]["dynamics"]["operators"][0]["parameters"]["solver"] = "steady-ish"
    assert not validator.is_valid(data)


def test_model_graph_shows_the_written_field():
    graph = describe_model_graph(_secretion_model())
    edges = {(edge["source"], edge["target"]) for edge in graph["edges"]}
    assert ("operator:0:pde", "field:signal") in edges
    assert ("operator:0:pde", "output:nodes") not in edges
    assert ("field:signal", "operator:1:reorientation.boltzmann") in edges


def test_cli_writes_the_field_history(tmp_path):
    path = save_model_spec(_secretion_model(), tmp_path / "model.json")
    output = tmp_path / "run"
    result = subprocess.run([sys.executable, "-m", "lgca.cli", "run", str(path), "--output", str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    with np.load(output / "measurements.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["field_signal_steps"], [0, 2, 4])
        assert data["field_signal"].shape == (3, 12, 12)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["fields"]["signal"]["calls"] == 4


# ------------------------------------------------------------------ plots

@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")
def test_field_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lgca.plots import LatticeAnimation
    from lgca.plotting import animate

    for geometry in ("square", "hex"):
        compiled = _build([PDESpec(field="u", diffusion=0.5, production=0.1)], geometry=geometry, steps=3,
                          observers=[FieldRecorder("u")])
        result = compiled.run(showprogress=False)
        animation = result.lgca.animate_scalarfield(result.data["u"], steps=result.data.steps("u"))
        assert isinstance(animation, LatticeAnimation)
        animation._func(2)
        assert animation._fig.axes[0].get_title() == "Time $k =$2"
        dispatched = animate(result.lgca, "scalarfield", data=result.data["u"])
        assert isinstance(dispatched, LatticeAnimation)
        dispatched._func(1)
        plt.close("all")
    compiled = _build([PDESpec(field="u", diffusion=0.5, production=0.1)], geometry="lin", dims=15, steps=4,
                      observers=[FieldRecorder("u", schedule=Schedule(every=2))])
    result = compiled.run(showprogress=False)
    image = result.lgca.plot_scalarfield(result.data["u"], steps=result.data.steps("u"))
    assert image.get_array().shape == (3, 15)
    line = result.lgca.plot_scalarfield(result.lgca.u)
    assert len(line.get_xdata()) == 15
    plt.close("all")
    del animation, dispatched
    gc.collect()
