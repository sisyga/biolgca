"""
Tests for lgca/nove_ib_interactions.py

Covers: tanh_switch, random_walk, birth, birthdeath, go_or_grow, go_or_grow_kappa,
birthdeath_cancerdfe.

Usage note: ``nove_ib_interactions`` is designed for NoVE (no volume exclusion) +
identity-based LGCA. The correct ``get_lgca`` flags are ``ve=False, ib=True``
(NOT ``nove=True, ib=True`` — that was incorrect; nove is not an accepted kwarg).
go_or_grow_kappa and birthdeath_cancerdfe are only dispatched via
``NoVE_IBLGCA_base.set_interaction``; older stubs tried ``nove=True`` which silently
fell through to vanilla ``IBLGCA_1D``.
"""
import numpy as np
import pytest
import warnings


def _lgca(interaction, dims=30, density=0.3, restchannels=0, seed=0, **kw):
    """Helper: build a 1-D NoVE IB LGCA (ve=False, ib=True)."""
    from lgca import get_lgca
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_lgca(
            ve=False, ib=True, geometry="lin",
            interaction=interaction,
            density=density, dims=dims,
            restchannels=restchannels,
            seed=seed,
            **kw
        )


# ---------------------------------------------------------------------------
# tanh_switch
# ---------------------------------------------------------------------------

class TestTanhSwitch:
    """Unit tests for nove_ib_interactions.tanh_switch (sigmoid helper)."""

    def test_scalar_midpoint_equals_half(self):
        from lgca.nove_ib_interactions import tanh_switch
        # At rho == theta the function must return exactly 0.5
        assert abs(tanh_switch(0.8, kappa=5.0, theta=0.8) - 0.5) < 1e-10

    def test_below_threshold_less_than_half(self):
        from lgca.nove_ib_interactions import tanh_switch
        assert tanh_switch(0.0) < 0.5

    def test_above_threshold_greater_than_half(self):
        from lgca.nove_ib_interactions import tanh_switch
        assert tanh_switch(1.0) > 0.5

    def test_output_in_unit_interval(self):
        from lgca.nove_ib_interactions import tanh_switch
        for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = tanh_switch(rho)
            assert 0.0 <= val <= 1.0, f"tanh_switch({rho}) = {val} outside [0,1]"

    def test_array_input_shape_preserved(self):
        from lgca.nove_ib_interactions import tanh_switch
        rho = np.array([0.0, 0.4, 0.8, 1.0])
        assert tanh_switch(rho).shape == rho.shape

    def test_monotonically_increasing(self):
        from lgca.nove_ib_interactions import tanh_switch
        rho = np.linspace(0.0, 1.0, 50)
        vals = tanh_switch(rho)
        assert np.all(np.diff(vals) > 0)

    def test_higher_kappa_sharpens_transition(self):
        from lgca.nove_ib_interactions import tanh_switch
        val_low = tanh_switch(0.9, kappa=1.0, theta=0.8)
        val_high = tanh_switch(0.9, kappa=20.0, theta=0.8)
        assert val_high > val_low

    def test_custom_theta_shifts_midpoint(self):
        from lgca.nove_ib_interactions import tanh_switch
        assert abs(tanh_switch(0.3, kappa=5.0, theta=0.3) - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# random_walk
# ---------------------------------------------------------------------------

class TestRandomWalk:
    """Tests for nove_ib_interactions.random_walk."""

    def test_nonborder_cell_count_conserved(self):
        """random_walk must not create or destroy cells in the non-border region."""
        lgca = _lgca("random_walk", density=0.4, seed=42)
        before = lgca.cell_density[lgca.nonborder].sum()
        lgca.timeevo(timesteps=10)
        after = lgca.cell_density[lgca.nonborder].sum()
        assert before == after

    def test_cells_can_move(self):
        """After several timesteps spatial distribution should differ."""
        lgca = _lgca("random_walk", density=0.4, seed=42)
        density_before = lgca.cell_density.copy()
        lgca.timeevo(timesteps=20)
        assert not np.array_equal(density_before, lgca.cell_density)

    def test_density_nonnegative(self):
        lgca = _lgca("random_walk", density=0.4, seed=0)
        lgca.timeevo(timesteps=5)
        assert np.all(lgca.cell_density >= 0)


# ---------------------------------------------------------------------------
# birth
# ---------------------------------------------------------------------------

class TestBirth:
    """Tests for nove_ib_interactions.birth."""

    def test_population_grows_over_time(self):
        lgca = _lgca("birth", density=0.1, seed=1)
        before = lgca.cell_density.sum()
        lgca.timeevo(timesteps=10)
        after = lgca.cell_density.sum()
        assert after >= before, "Population should not decrease under birth-only interaction"

    def test_density_nonnegative(self):
        lgca = _lgca("birth", density=0.1, seed=0)
        lgca.timeevo(timesteps=5)
        assert np.all(lgca.cell_density >= 0)

    def test_population_respects_capacity(self):
        lgca = _lgca("birth", density=0.1, seed=0)
        capacity = lgca.interaction_params.get("capacity", 8)
        lgca.timeevo(timesteps=30)
        max_possible = lgca.cell_density[lgca.nonborder].size * capacity
        assert lgca.cell_density.sum() <= max_possible + capacity  # small tolerance for border cells


# ---------------------------------------------------------------------------
# birthdeath
# ---------------------------------------------------------------------------

class TestBirthdeath:
    """Tests for nove_ib_interactions.birthdeath."""

    def test_timeevo_completes_without_error(self):
        lgca = _lgca("birthdeath", density=0.5, seed=0)
        lgca.timeevo(timesteps=10)

    def test_density_nonnegative(self):
        lgca = _lgca("birthdeath", density=0.5, seed=0)
        lgca.timeevo(timesteps=5)
        assert np.all(lgca.cell_density >= 0)

    def test_interaction_params_set(self):
        lgca = _lgca("birthdeath", density=0.5, seed=0)
        assert "r_b" in lgca.interaction_params
        assert "r_d" in lgca.interaction_params
        assert 0 < lgca.interaction_params["r_b"] <= 1.0
        assert 0 < lgca.interaction_params["r_d"] <= 1.0


# ---------------------------------------------------------------------------
# go_or_grow
# ---------------------------------------------------------------------------

class TestGoOrGrow:
    """Tests for nove_ib_interactions.go_or_grow (requires restchannels>=1)."""

    def test_timeevo_completes(self):
        lgca = _lgca("go_or_grow", density=0.4, restchannels=1, seed=7)
        lgca.timeevo(timesteps=10)

    def test_density_nonnegative(self):
        lgca = _lgca("go_or_grow", density=0.4, restchannels=1, seed=7)
        lgca.timeevo(timesteps=5)
        assert np.all(lgca.cell_density >= 0)

    def test_interaction_params_have_kappa_theta(self):
        lgca = _lgca("go_or_grow", density=0.4, restchannels=1, seed=7)
        assert "kappa" in lgca.interaction_params
        assert "theta" in lgca.interaction_params


# ---------------------------------------------------------------------------
# go_or_grow_kappa
# ---------------------------------------------------------------------------

class TestGoOrGrowKappa:
    """Tests for nove_ib_interactions.go_or_grow_kappa (NoVE_IBLGCA_1D)."""

    def test_dispatched_correctly(self):
        """Ensure go_or_grow_kappa is properly wired in NoVE_IBLGCA_1D."""
        from lgca.nove_ib_interactions import go_or_grow_kappa
        lgca = _lgca("go_or_grow_kappa", density=0.4, restchannels=1, seed=3)
        assert lgca.interaction is go_or_grow_kappa

    def test_single_step_does_not_raise(self):
        lgca = _lgca("go_or_grow_kappa", density=0.4, restchannels=1, seed=3)
        lgca.interaction(lgca)

    def test_density_nonnegative_after_timeevo(self):
        lgca = _lgca("go_or_grow_kappa", density=0.4, restchannels=1, seed=3)
        lgca.timeevo(timesteps=5)
        assert np.all(lgca.cell_density >= 0)

    def test_interaction_params_include_kappa_std(self):
        lgca = _lgca("go_or_grow_kappa", density=0.4, restchannels=1, seed=0)
        assert "kappa_std" in lgca.interaction_params


# ---------------------------------------------------------------------------
# birthdeath_cancerdfe
# ---------------------------------------------------------------------------

class TestBirthdeath_CancerDFE:
    """Tests for nove_ib_interactions.birthdeath_cancerdfe."""

    def test_dispatched_correctly(self):
        from lgca.nove_ib_interactions import birthdeath_cancerdfe
        lgca = _lgca("birthdeath_cancerdfe", density=0.4, seed=1)
        assert lgca.interaction is birthdeath_cancerdfe

    def test_timeevo_completes(self):
        lgca = _lgca("birthdeath_cancerdfe", density=0.4, seed=1)
        lgca.timeevo(timesteps=5)

    def test_density_nonnegative(self):
        lgca = _lgca("birthdeath_cancerdfe", density=0.4, seed=1)
        lgca.timeevo(timesteps=5)
        assert np.all(lgca.cell_density >= 0)

    def test_interaction_params_set(self):
        lgca = _lgca("birthdeath_cancerdfe", density=0.4, seed=0)
        params = lgca.interaction_params
        for key in ("r_b", "r_d", "p_d", "p_p", "s_d"):
            assert key in params, f"Expected '{key}' in interaction_params"
