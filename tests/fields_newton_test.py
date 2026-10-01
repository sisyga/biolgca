"""Newton's method for fields with saturating uptake: accuracy against independent solutions, steady states
that exist and that do not, conservation, the backends, and the errors of the field solver."""

import importlib.util
import pickle

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from lgca.fields import FieldSolverError, PDESpec, reaction
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec

HAVE_PYAMG = importlib.util.find_spec("pyamg") is not None
AMG = pytest.param("amg", marks=pytest.mark.skipif(not HAVE_PYAMG, reason="pyamg has no wheels for Python 3.14 yet"))
BACKENDS = ["direct", "cg", AMG]


def _model(pde, cells, *, u0=1.0, boundary="periodic", operators=(), fields=None):
    """A model of resting cells, ``cells`` per node (on a 1D, square or cubic lattice), with ``operators``
    and then ``pde``."""
    cells = np.asarray(cells)
    nodes = np.zeros(cells.shape + (2 * cells.ndim + 1,), dtype=int)
    nodes[..., -1] = cells
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=("lin", "square", "cubic")[cells.ndim - 1],
                        dims=cells.shape[0] if cells.ndim == 1 else cells.shape, boundary=boundary),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=False, fields={"u": u0, **(fields or {})}),
        time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[*operators, pde])))


def _u(model):
    return model.lgca.u[model.lgca.nonborder]


def _hill(u, K, n):
    """u^n / (K^n + u^n), also where the powers overflow."""
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(u > 0, 1 / (1 + (K / np.maximum(u, 1e-300)) ** n), 0.0)


def _uptake(rate, K, n, **term):
    return [{"uptake": rate, "saturation": K, "n": n, **term}]


# ------------------------------------------------------------------ the reproductions of the code review

@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
@pytest.mark.parametrize("u0", [0.0, 1e-12, 0.5, 5.0])
def test_a_steady_field_that_only_the_cells_remove_is_found_from_any_start(boundary, u0):
    # production 1 against the uptake 2 u² / (1 + u²) at every node: u = 1. From 0 Picard's first matrix is
    # singular, and from 1e-12 its iteration did not converge.
    pde = PDESpec(field="u", diffusion=1.0, production=1.0, cells=_uptake(2.0, 1.0, 2), solver="steady")
    model = _model(pde, [1, 1, 1], u0=u0, boundary=boundary)
    np.testing.assert_allclose(_u(model), 1.0, rtol=0, atol=1e-8)
    model.step()
    np.testing.assert_allclose(_u(model), 1.0, rtol=0, atol=1e-8)
    assert model.metadata["fields"]["u"]["nonlinear"] == "newton"


@pytest.mark.parametrize("backend", BACKENDS)
def test_steep_uptake_that_cycles_picard_iteration_converges(backend):
    # one backward Euler step of du/dt = -4 u^64 / (1 + u^64) from 2
    exact = brentq(lambda u: u + 4 * _hill(u, 1.0, 64) - 2.0, 0.5, 1.5, xtol=1e-15, rtol=1e-15)
    assert exact == pytest.approx(0.98332043697, abs=1e-11)
    model = _model(PDESpec(field="u", cells=_uptake(4.0, 1.0, 64), solver_options={"backend": backend}),
                   [1, 1, 1], u0=2.0)
    model.step()
    np.testing.assert_allclose(_u(model), exact, rtol=0, atol=1e-6)
    statistics = model.metadata["fields"]["u"]
    assert statistics["max_iterations_used"] <= 10 and statistics["backtracks"] >= 1


# ------------------------------------------------------------------ against brentq

SWEEP = [(n, K, rate) for n in (1, 2, 3, 4, 8, 64) for K in (0.05, 1.0) for rate in (0.5, 20.0)]


@pytest.mark.parametrize("n, K, rate", SWEEP)
def test_implicit_uptake_matches_brentq(n, K, rate):
    # u + w h(u) = 1 at every node: without diffusion with 0 to 3 cells per node, each node on its own, and with
    # diffusion where every node has the same cells
    for diffusion, cells in ((0.0, np.arange(4)), (1.0, np.full(4, 2))):
        model = _model(PDESpec(field="u", diffusion=diffusion, cells=_uptake(rate, K, n)), cells)
        model.step()
        exact = np.array([brentq(lambda u, w=w: u + w * _hill(u, K, n) - 1.0, 0.0, 1.0, xtol=1e-15)
                          for w in rate * cells])
        # the error is estimated to be at most rtol relative to the largest value: allow twice that
        np.testing.assert_allclose(_u(model), exact, rtol=0, atol=2e-6 * exact.max())


@pytest.mark.parametrize("n, K, rate", SWEEP)
def test_steady_uptake_matches_brentq(n, K, rate):
    # without diffusion, with decay: 0.1 u + w h(u) = 1 at every node
    cells = np.arange(4)
    model = _model(PDESpec(field="u", decay=0.1, production=1.0, cells=_uptake(rate, K, n), solver="steady"), cells)
    exact = np.array([brentq(lambda u, w=w: 0.1 * u + w * _hill(u, K, n) - 1.0, 0.0, 10.0, xtol=1e-14)
                      for w in rate * cells])
    np.testing.assert_allclose(_u(model), exact, rtol=0, atol=2e-6 * exact.max())
    # with diffusion and nothing else that removes the field, the uptake balances the production: w h(u) = 0.4
    model = _model(PDESpec(field="u", diffusion=1.0, production=0.4, cells=_uptake(rate, K, n), solver="steady"),
                   np.ones(4, dtype=int))
    np.testing.assert_allclose(_u(model), K * (0.4 / (rate - 0.4)) ** (1 / n), rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [2, 4])
def test_a_spheroid_that_takes_up_a_field_supplied_at_the_boundary(n, backend):
    # strong uptake where the cells are: the field falls from 1 at the boundary to about K inside
    cells = np.zeros((24, 24), dtype=int)
    cells[6:18, 6:18] = 4
    K, rate, diffusion = 0.05, 20.0, 100.0
    pde = PDESpec(field="u", diffusion=diffusion, cells=_uptake(rate, K, n), boundary={"value": 1.0},
                  solver="steady", solver_options={"backend": backend})
    model = _model(pde, cells, boundary="reflecting")
    u = _u(model).ravel()
    # the reference: Newton's method with exact solves, written out with the five-point stencil
    size = 24
    laplacian = sp.diags([-4.0, 1.0, 1.0, 1.0, 1.0], [0, 1, -1, size, -size], shape=(size**2, size**2)).tolil()
    for i in range(size):  # no neighbours across the rows of the lattice; a fixed value beyond its edge
        for j in (i * size, i * size + size - 1):
            neighbour = j - 1 if j % size == 0 else j + 1
            if 0 <= neighbour < size**2:
                laplacian[j, neighbour] = 0.0
    source = np.zeros((size, size))
    source[0, :] += 1
    source[-1, :] += 1
    source[:, 0] += 1
    source[:, -1] += 1
    w = rate * cells.ravel()
    reference = u.copy()
    for _ in range(20):
        hill = _hill(reference, K, n)
        derivative = n * hill * (1 - hill) / np.maximum(reference, 1e-300)
        residual = -diffusion * (laplacian @ reference + source.ravel()) + w * hill
        jacobian = (-diffusion * laplacian.tocsr() + sp.diags(w * derivative)).tocsc()
        reference = reference - spla.spsolve(jacobian, residual)
    # the error is estimated to be at most rtol relative to the largest value: allow twice that
    assert reference.min() < 2 * K
    np.testing.assert_allclose(u, reference, rtol=0, atol=2e-6 * reference.max())


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_a_steady_state_near_the_capacity_of_the_cells(n):
    # production 1.9 per node against at most 2 taken up: 2 u^n / (1 + u^n) = 1.9, so u = 19^(1/n)
    pde = PDESpec(field="u", diffusion=1.0, production=1.9, cells=_uptake(2.0, 1.0, n), solver="steady")
    model = _model(pde, np.ones(10, dtype=int), u0=0.0)
    np.testing.assert_allclose(_u(model), 19 ** (1 / n), rtol=1e-8)


@pytest.mark.parametrize("nonlinear", ["newton", "picard"])
def test_production_beyond_what_the_cells_can_take_up_has_no_steady_state(nonlinear):
    pde = PDESpec(field="u", diffusion=1.0, production=3.0, cells=_uptake(2.0, 1.0, 2), solver="steady",
                  solver_options={"nonlinear": nonlinear})
    with pytest.raises(FieldSolverError, match="no steady state exists: the field is produced at 30 per step "
                                               "in total, and the cells take up at most 20") as error:
        _model(pde, np.ones(10, dtype=int))
    assert error.value.kind == "no_steady_state"
    # a little decay, or a fixed value at the boundary, removes the rest
    for removal in ({"decay": 0.01}, {"boundary": {"value": 0.0}}):
        pde = PDESpec(field="u", diffusion=1.0, production=3.0, cells=_uptake(2.0, 1.0, 2), solver="steady",
                      **removal)
        assert np.all(np.isfinite(_u(_model(pde, np.ones(10, dtype=int), boundary="reflecting"))))


def test_implicit_saturating_uptake_converges_at_first_order_in_the_substeps():
    # du/dt = -2 u² / (0.25 + u²) at every node, from 1
    exact = solve_ivp(lambda t, u: -2 * u**2 / (0.25 + u**2), (0, 1), [1.0], rtol=1e-12, atol=1e-14).y[0, -1]
    errors = []
    for substeps in (2, 4, 8, 16):
        pde = PDESpec(field="u", cells=_uptake(2.0, 0.5, 2), solver_options={"substeps": substeps, "rtol": 1e-12})
        model = _model(pde, [1, 1], u0=1.0)
        model.step()
        errors.append(abs(_u(model)[0] - exact))
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 2.0, rtol=0.1)


# ------------------------------------------------------------------ conservation and the backends

@pytest.mark.parametrize("solver", ["steady", "implicit"])
def test_saturating_uptake_with_advection_and_no_flux_conserves_the_field(solver):
    rng = np.random.default_rng(3)
    cells, production = rng.integers(0, 3, (12, 10)), rng.uniform(0, 1, (12, 10))
    pde = PDESpec(field="u", diffusion=0.5, advection=[0.3, -0.2], production="production",
                  cells=_uptake(2.0, 0.5, 2), solver=solver, solver_options={"rtol": 1e-10})
    model = _model(pde, cells, boundary="reflecting", fields={"production": production})
    before = _u(model).copy()
    model.step()
    after = _u(model)
    uptake = 2.0 * cells * _hill(after, 0.5, 2)
    gain = 0.0 if solver == "steady" else after.sum() - before.sum()
    assert abs(gain - (production.sum() - uptake.sum())) <= 1e-8 * production.sum()


@pytest.mark.parametrize("solver", ["steady", "implicit"])
def test_the_backends_agree(solver):
    cells = np.random.default_rng(5).integers(0, 4, (20, 20))
    fields = {}
    for backend in ("direct", "cg") + (("amg",) if HAVE_PYAMG else ()):
        pde = PDESpec(field="u", diffusion=5.0, cells=_uptake(1.0, 0.1, 2), boundary={"value": 1.0}, solver=solver,
                      solver_options={"backend": backend, "rtol": 1e-10})
        model = _model(pde, cells, boundary="reflecting")
        model.step()
        fields[backend] = _u(model)
    for field in fields.values():
        np.testing.assert_allclose(field, fields["direct"], rtol=0, atol=2e-10 * fields["direct"].max())


@pytest.mark.parametrize("shape, backend", [((16, 16), "direct"), ((8, 8, 8), "cg")])
def test_steady_saturating_uptake_without_pyamg(monkeypatch, shape, backend):
    import lgca.fields

    monkeypatch.setattr(lgca.fields, "_pyamg", lambda: None)  # Python 3.14: no wheels yet
    cells = np.random.default_rng(7).integers(0, 4, shape)
    fields = []
    for options in ({}, {"backend": "direct", "rtol": 1e-12}):
        pde = PDESpec(field="u", diffusion=10.0, cells=_uptake(5.0, 0.1, 2), boundary={"value": 1.0},
                      solver="steady", solver_options=options)
        model = _model(pde, cells, boundary="reflecting")
        model.step()
        fields.append(_u(model))
        if not options:
            assert model.metadata["fields"]["u"]["backend"] == backend
    assert fields[1].min() < 0.5
    np.testing.assert_allclose(fields[0], fields[1], rtol=0, atol=2e-6 * fields[1].max())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("solver", ["steady", "implicit"])
def test_problems_linear_in_the_field_are_solved_as_before(solver, backend):
    # linear uptake and secretion, and saturating uptake by cells that are not there (on velocity channels):
    # one linear solve, the same with either iteration
    cells = np.random.default_rng(6).integers(0, 3, (10, 8))
    terms = [{"uptake": 0.3}, {"production": 0.2}, *_uptake(2.0, 0.1, 2, channels="velocity")]
    fields = []
    for nonlinear in ("newton", "picard"):
        pde = PDESpec(field="u", diffusion=2.0, decay=0.01, cells=terms, boundary={"value": 1.0}, solver=solver,
                      solver_options={"backend": backend, "nonlinear": nonlinear})
        model = _model(pde, cells, boundary="reflecting")
        model.step()
        fields.append(_u(model).copy())
        assert "max_iterations_used" not in model.metadata["fields"]["u"]
    np.testing.assert_array_equal(*fields)


# ------------------------------------------------------------------ reactions

@reaction(name="test_newton_logistic")
def _logistic_reaction(state, c, r=1.0):
    return r * c, r * c  # dc/dt = r c (1 - c)


@reaction(name="test_newton_autocatalysis")
def _autocatalysis_reaction(state, c, k=0.5):
    # with decay 0.1, 0.1 c = k c² / (1 + c²): equilibria 0 and (5 + √21) / 2 (stable), (5 - √21) / 2 (unstable)
    return k * c**2 / (1 + c**2), 0.0


@reaction(name="test_newton_switches")
def _switches_reaction(state, c):
    # with decay 0.1 three stable equilibria: 0, 3.0245 and 8.7106, and unstable ones at 0.8994 and 6.0000
    return 0.3 * c**8 / (1 + c**8) + 0.6 * c**8 / (6.0**8 + c**8), 0.0


@reaction(name="test_newton_consumption")
def _consumption_reaction(state, c, rate=1.0, K=1.0, n=1):
    # saturating uptake written as a reaction: rate c^n / (K^n + c^n) per step
    from lgca.fields import _hill_rate

    return 0.0, rate * state.density * _hill_rate(c, K, n)


BISTABLE = {"decay": 0.1, "reactions": [{"name": "test_newton_autocatalysis"}]}
LOW, HIGH = 0.0, (5 + np.sqrt(21)) / 2  # the stable equilibria of BISTABLE; the unstable one is (5 - √21) / 2


def _switches(u):
    return 0.1 * u - 0.3 * _hill(u, 1.0, 8) - 0.6 * _hill(u, 6.0, 8)


@pytest.mark.parametrize("u0", [0.1, 1.5, 3.0, "random"])
def test_logistic_growth_of_a_steady_field_reaches_its_capacity_from_below_and_above(u0):
    # 0 is an equilibrium too, an unstable one: Newton's method with the derivative of the reaction heads for it
    # from below 1/2, and the balanced start of saturating uptake would shift the field to it
    start = np.random.default_rng(4).uniform(0.01, 3.0, 12) if u0 == "random" else u0
    pde = PDESpec(field="u", diffusion=1.0, reactions=[{"name": "test_newton_logistic", "r": 0.8}], solver="steady",
                  solver_options={"rtol": 1e-10})
    model = _model(pde, np.zeros(12, dtype=int), u0=start)
    np.testing.assert_allclose(_u(model), 1.0, rtol=0, atol=1e-8)
    assert model.metadata["fields"]["u"]["nonlinear"] == "newton"


@pytest.mark.parametrize("r", [0.8, 3.0, 10.0])
@pytest.mark.parametrize("u0", [0.1, 1.5, 3.0])
def test_an_implicit_step_of_logistic_growth_is_the_root_of_backward_euler(r, u0):
    # c - u0 = r c (1 - c) has one positive root
    exact = (r - 1 + np.sqrt((1 - r) ** 2 + 4 * r * u0)) / (2 * r)
    pde = PDESpec(field="u", diffusion=1.0, reactions=[{"name": "test_newton_logistic", "r": r}],
                  solver_options={"rtol": 1e-10, "max_iterations": 100})
    model = _model(pde, np.zeros(6, dtype=int), u0=u0)
    model.step()
    np.testing.assert_allclose(_u(model), exact, rtol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("u0, expected", [(0.2, LOW), (0.15, LOW), (0.25, HIGH), (0.5, HIGH), (3.0, HIGH),
                                          (10.0, HIGH)])
def test_bistable_autocatalysis_settles_in_the_equilibrium_on_its_side_of_the_threshold(u0, expected, backend):
    # without the floor of the reaction's derivative Newton's method converged to the unstable equilibrium from
    # 0.15 to 0.5; on the iterative backends a field that falls to 0 approaches it only geometrically
    pde = PDESpec(field="u", diffusion=1.0, **BISTABLE, solver="steady",
                  solver_options={"rtol": 1e-10, "backend": backend})
    model = _model(pde, np.zeros(10, dtype=int), u0=u0)
    np.testing.assert_allclose(_u(model), expected, rtol=0, atol=1e-8 * HIGH)


@pytest.mark.parametrize("diffusion", [0.0, 1.0])
@pytest.mark.parametrize("u0", [0.85, 0.95, 4.0, 5.0, 5.9, 6.1, 12.0])
def test_a_field_with_three_stable_equilibria_reaches_the_one_that_its_dynamics_reach(u0, diffusion):
    # the equilibrium that dc/dt = -F(c) reaches from u0: the nearest root in the direction of -F(u0). A floor
    # that allowed steps longer than Picard iteration's where the production rises with c (the earlier design
    # kept a tenth of the diagonal dominance) jumped from 5.0, 5.5 and 5.9 over 3.0245 and 0.8994 to 0.
    grid = np.linspace(1e-6, 20.0, 200001)
    signs = np.sign(_switches(grid))
    roots = [0.0] + [brentq(_switches, grid[i], grid[i + 1], xtol=1e-14)
                     for i in np.flatnonzero(signs[:-1] * signs[1:] < 0)]
    np.testing.assert_allclose(roots, [0.0, 0.8994, 3.0245, 6.0, 8.7106], atol=1e-4)
    expected = max(r for r in roots if r < u0) if _switches(u0) > 0 else min(r for r in roots if r > u0)
    # (next to the fold at 6, linearized as in Picard iteration, the field converges slowly from 6.1)
    pde = PDESpec(field="u", diffusion=diffusion, decay=0.1, reactions=[{"name": "test_newton_switches"}],
                  solver="steady", solver_options={"rtol": 1e-10, "max_iterations": 50})
    model = _model(pde, np.zeros(6, dtype=int), u0=u0)
    np.testing.assert_allclose(_u(model), expected, rtol=0, atol=1e-8 * max(expected, 1.0))


@pytest.mark.parametrize("rate", [0.5, 2.0])
def test_steep_uptake_and_a_reaction_together_match_brentq(rate):
    # u + 0.05 u + 4 h(u) - rate u² / (1 + u²) = 2 on the nodes with a cell, without the h term on the others,
    # with h = u^64 / (1 + u^64); Picard iteration cycles here
    cells = np.tile([1, 0], 5)
    pde = PDESpec(field="u", decay=0.05, cells=_uptake(4.0, 1.0, 64),
                  reactions=[{"name": "test_newton_autocatalysis", "k": rate}], solver_options={"rtol": 1e-10})
    model = _model(pde, cells, u0=2.0)
    model.step()
    exact = [brentq(lambda u, w=4.0 * n: 1.05 * u + w * _hill(u, 1.0, 64) - rate * u**2 / (1 + u**2) - 2.0, 0.0, 10.0,
                    xtol=1e-14) for n in cells]
    np.testing.assert_allclose(_u(model), exact, rtol=0, atol=1e-8)
    with pytest.raises(FieldSolverError, match="did not converge") as error:
        _model(PDESpec(field="u", decay=0.05, cells=_uptake(4.0, 1.0, 64), solver_options={"nonlinear": "picard"},
                       reactions=[{"name": "test_newton_autocatalysis", "k": rate}]), cells, u0=2.0).step()
    assert error.value.kind == "nonlinear"


def test_a_steady_field_with_a_reaction_and_saturating_uptake_matches_brentq():
    # without diffusion every node on its own: 0.1 u + uptake by n cells, u^4 / (0.5^4 + u^4) each, and a consumption
    # u² / (1 + u²) per cell written as a reaction, against a production of 1
    cells = np.arange(5)
    pde = PDESpec(field="u", decay=0.1, production=1.0, cells=_uptake(0.3, 0.5, 4), solver="steady",
                  reactions=[{"name": "test_newton_consumption", "n": 2}], solver_options={"rtol": 1e-10})
    model = _model(pde, cells, u0=1.0)
    exact = [brentq(lambda u, n=n: 0.1 * u + n * (0.3 * _hill(u, 0.5, 4) + _hill(u, 1.0, 2)) - 1.0, 0.0, 10.0,
                    xtol=1e-14) for n in cells]
    np.testing.assert_allclose(_u(model), exact, rtol=0, atol=1e-8)


def test_saturating_uptake_written_as_a_reaction_is_solved_like_the_cell_term():
    # the reproduction of the code review, implicit with n = 64: Picard iteration cycles, Newton's method
    # converges as for the cell term (test_steep_uptake_that_cycles_picard_iteration_converges)
    exact = brentq(lambda u: u + 4 * _hill(u, 1.0, 64) - 2.0, 0.5, 1.5, xtol=1e-15, rtol=1e-15)
    model = _model(PDESpec(field="u", reactions=[{"name": "test_newton_consumption", "rate": 4.0, "n": 64}]),
                   [1, 1, 1], u0=2.0)
    model.step()
    np.testing.assert_allclose(_u(model), exact, rtol=0, atol=1e-6)
    assert model.metadata["fields"]["u"]["max_iterations_used"] <= 10
    assert "floored_iterations" not in model.metadata["fields"]["u"]  # the loss rises with c: no floor
    # steady, only the cells remove the field: 2 u / (1 + u) = 1.9 at every node; Picard iteration converges too
    # slowly for max_iterations
    pde = PDESpec(field="u", diffusion=1.0, production=1.9, reactions=[{"name": "test_newton_consumption",
                                                                         "rate": 2.0}], solver="steady")
    np.testing.assert_allclose(_u(_model(pde, np.ones(10, dtype=int))), 19.0, rtol=1e-6)


def test_the_balanced_start_of_saturating_uptake_is_not_applied_to_reactions():
    # a steady field that only the cells remove, from 0: with uptake by the cells the start is shifted to the
    # uniform field that balances the production; a reaction gets no such start (it may have several such
    # fields, as logistic growth has 0 and 1), and from 0 its loss rate makes the first matrix singular
    reactions = [{"name": "test_newton_consumption", "rate": 2.0, "n": 2}]
    for u0 in (0.5, 5.0):
        pde = PDESpec(field="u", diffusion=1.0, production=1.0, reactions=reactions, solver="steady")
        np.testing.assert_allclose(_u(_model(pde, [1, 1, 1], u0=u0)), 1.0, rtol=1e-6)
    with pytest.raises(FieldSolverError, match="no unique solution") as error:
        _model(PDESpec(field="u", diffusion=1.0, production=1.0, reactions=reactions, solver="steady"), [1, 1, 1],
               u0=0.0)
    assert error.value.kind == "singular"


@pytest.mark.parametrize("solver", ["steady", "implicit"])
def test_reactions_with_advection_and_no_flux_conserve_the_field(solver):
    rng = np.random.default_rng(8)
    cells, production = rng.integers(0, 3, (12, 10)), rng.uniform(0, 0.5, (12, 10))
    pde = PDESpec(field="u", diffusion=0.5, advection=[0.3, -0.2], production="production", decay=0.1,
                  reactions=[{"name": "test_newton_consumption", "n": 2}, {"name": "test_newton_autocatalysis"}],
                  solver=solver, solver_options={"rtol": 1e-10, "max_iterations": 100})
    model = _model(pde, cells, u0=1.0, boundary="reflecting", fields={"production": production})
    before = _u(model).copy()
    model.step()
    after = _u(model)
    made = 0.5 * after**2 / (1 + after**2)
    lost = 0.1 * after + cells * _hill(after, 1.0, 2)
    gain = 0.0 if solver == "steady" else after.sum() - before.sum()
    assert abs(gain - (production.sum() + made.sum() - lost.sum())) <= 1e-8 * production.sum()


@pytest.mark.parametrize("solver", ["steady", "implicit"])
def test_the_backends_agree_on_reactions(solver):
    cells = np.random.default_rng(9).integers(0, 4, (20, 20))
    fields = {}
    for backend in ("direct", "cg") + (("amg",) if HAVE_PYAMG else ()):
        pde = PDESpec(field="u", diffusion=5.0, decay=0.1, boundary={"value": 1.0}, solver=solver,
                      reactions=[{"name": "test_newton_consumption", "rate": 0.5, "K": 0.2, "n": 2},
                                 {"name": "test_newton_logistic", "r": 0.3}],
                      solver_options={"backend": backend, "rtol": 1e-10, "max_iterations": 50})
        model = _model(pde, cells, boundary="reflecting")
        model.step()
        fields[backend] = _u(model)
    assert fields["direct"].min() < 0.5
    for field in fields.values():
        np.testing.assert_allclose(field, fields["direct"], rtol=0, atol=2e-10 * fields["direct"].max())


def test_an_implicit_reaction_converges_at_first_order_in_the_substeps():
    # du/dt = u (1 - u) from 0.1: the logistic curve
    exact = 0.1 * np.e / (1 - 0.1 + 0.1 * np.e)
    errors = []
    for substeps in (4, 8, 16, 32):
        pde = PDESpec(field="u", reactions=[{"name": "test_newton_logistic"}],
                      solver_options={"substeps": substeps, "rtol": 1e-12, "max_iterations": 100})
        model = _model(pde, [0, 0], u0=0.1)
        model.step()
        errors.append(abs(_u(model)[0] - exact))
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 2.0, rtol=0.1)


@pytest.mark.parametrize("shape, backend", [((16, 16), "direct"), ((8, 8, 8), "cg")])
def test_steady_reactions_without_pyamg(monkeypatch, shape, backend):
    import lgca.fields

    monkeypatch.setattr(lgca.fields, "_pyamg", lambda: None)  # Python 3.14: no wheels yet
    cells = np.random.default_rng(10).integers(0, 4, shape)
    fields = []
    for options in ({}, {"backend": "direct", "rtol": 1e-12}):
        pde = PDESpec(field="u", diffusion=10.0, boundary={"value": 1.0}, solver="steady", solver_options=options,
                      reactions=[{"name": "test_newton_consumption", "rate": 5.0, "K": 0.1, "n": 2}])
        model = _model(pde, cells, boundary="reflecting")
        model.step()
        fields.append(_u(model))
        if not options:
            assert model.metadata["fields"]["u"]["backend"] == backend
    assert fields[1].min() < 0.5
    np.testing.assert_allclose(fields[0], fields[1], rtol=0, atol=2e-6 * fields[1].max())


@pytest.mark.parametrize("pde, u0, floored", [
    ({"reactions": [{"name": "test_newton_consumption", "rate": 4.0, "K": 0.1}]}, 1.0, False),  # 6 against 19
    ({"diffusion": 1.0, "production": 1.9, "solver": "steady",  # 8; Picard iteration stalls
      "reactions": [{"name": "test_newton_consumption", "rate": 2.0}]}, 1.0, False),
    ({"reactions": [{"name": "test_newton_logistic", "r": 3.0}]}, 3.0, False),  # above the capacity: 5 against 17
    ({"reactions": [{"name": "test_newton_logistic", "r": 3.0}]}, 0.1, True),  # below it: 19 against 20
    ({**BISTABLE, "solver": "steady", "diffusion": 1.0}, 0.5, True),  # 12 against 13
    ({**BISTABLE, "solver": "steady", "diffusion": 1.0}, 10.0, True),  # 9 against 10
])
def test_iterations_of_newtons_method_against_picard_iteration_on_reactions(pde, u0, floored):
    # where a reaction's production rises with the field it is linearized as in Picard iteration (the floor),
    # and Newton's method converges at Picard's rate; elsewhere it needs a third of the iterations or fewer
    used = {}
    for nonlinear in ("newton", "picard"):
        spec = PDESpec(field="u", **pde, solver_options={"nonlinear": nonlinear, "rtol": 1e-10, "max_iterations": 50})
        try:
            model = _model(spec, np.ones(8, dtype=int), u0=u0)
            if spec.solver != "steady":
                model.step()
        except FieldSolverError:
            assert nonlinear == "picard"
            used[nonlinear] = 51
            continue
        statistics = model.metadata["fields"]["u"]
        used[nonlinear] = statistics["max_iterations_used"]
        if nonlinear == "newton":
            assert ("floored_iterations" in statistics) == floored
    assert used["newton"] <= (used["picard"] if floored else used["picard"] / 3), used


# ------------------------------------------------------------------ the option, the record and the errors

@reaction(name="test_newton_decay")
def _decay_reaction(state, c, rate=0.1):
    return 0.0, rate


@pytest.mark.parametrize("solver, terms, expected", [
    ("steady", {"cells": _uptake(1.0, 0.5, 2)}, "newton"),
    ("implicit", {"cells": _uptake(1.0, 0.5, 2), "solver_options": {"nonlinear": "picard"}}, "picard"),
    ("implicit", {"cells": _uptake(1.0, 0.5, 2), "reactions": [{"name": "test_newton_decay"}]}, "newton"),
    ("steady", {"reactions": [{"name": "test_newton_decay"}]}, "newton"),
    ("steady", {"reactions": [{"name": "test_newton_decay"}], "solver_options": {"nonlinear": "picard"}}, "picard"),
    ("implicit", {"cells": [{"uptake": 1.0}]}, None),
    ("explicit", {"cells": _uptake(1.0, 0.5, 2)}, None),
    ("explicit", {"reactions": [{"name": "test_newton_decay"}]}, None),
])
def test_the_metadata_say_which_iteration_a_run_used(solver, terms, expected):
    pde = PDESpec(field="u", diffusion=1.0, decay=0.1, solver=solver, **terms)
    model = _model(pde, [1, 0, 2])
    model.step()
    assert model.metadata["fields"]["u"].get("nonlinear") == expected


@pytest.mark.parametrize("solver, options, message", [
    ("steady", {"nonlinear": "anderson"}, "nonlinear must be one of newton, picard, got 'anderson'"),
    ("implicit", {"nonlinear": None}, "nonlinear must be one of newton, picard, got None"),
    ("explicit", {"nonlinear": "newton"}, r"\['nonlinear'\] do not apply to solver 'explicit'"),
])
def test_the_nonlinear_option_is_validated(solver, options, message):
    with pytest.raises(ValueError, match=message):
        _model(PDESpec(field="u", decay=1.0, solver=solver, solver_options=options), [1, 1])


def test_failures_are_counted_by_kind():
    death = {"name": "birth_death", "parameters": {"birth_rate": 0.0, "death_rate": 1.0}}
    pde = PDESpec(field="u", diffusion=1.0, production=1.0, cells=_uptake(2.0, 1.0, 2), solver="steady")
    model = _model(pde, [1, 1, 1], operators=[death])
    with pytest.raises(FieldSolverError, match="no cells take up the field") as error:
        model.step()  # the cells have died, and nothing else removes the field
    assert error.value.kind == "no_steady_state"
    statistics = model.metadata["fields"]["u"]
    assert statistics["failures"] == 1 and statistics["failure_kinds"] == {"no_steady_state": 1}


def test_field_solver_errors_are_runtime_errors_that_keep_their_kind_through_pickling():
    error = pickle.loads(pickle.dumps(FieldSolverError("pde 'u': the iterative solver did not converge", "linear")))
    assert isinstance(error, RuntimeError)
    assert (error.kind, str(error)) == ("linear", "pde 'u': the iterative solver did not converge")
    with pytest.raises(ValueError, match="kind must be one of"):
        FieldSolverError("pde 'u': failed", "unknown")
