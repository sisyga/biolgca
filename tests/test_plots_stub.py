"""
Stub tests for lgca/plots.py and lgca/square_plotting.py — both at 6% coverage.

All plotting functions are untested. Strategy: use `matplotlib.use('Agg')` 
backend to avoid display requirements.

Functions to test in plots.py:
  - plot_density
  - plot_flux
  - plot_flow
  - animate_density
  - live_animate_density

Functions to test in square_plotting.py:
  - plot_density (square geometry)
  - plot_flux (square)
  - plot_flow (square)
  - plot_prop_spatial

Run baseline: 6% coverage each
"""
import pytest


@pytest.fixture(autouse=True)
def use_agg_backend():
    """Switch to non-interactive Agg backend for all plotting tests."""
    import matplotlib
    matplotlib.use('Agg')


# TODO: test plot_density (square)
@pytest.mark.skip(reason="TODO: implement basic plot smoke test")
def test_plot_density_square(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from lgca import get_lgca
    lgca = get_lgca(geometry='square', interaction='random_walk', dims=(10, 10))
    lgca.timeevo(timesteps=5)
    # Should not raise
    lgca.plot_density()


# TODO: test plot_flux (square)
@pytest.mark.skip(reason="TODO: implement flux plot smoke test")
def test_plot_flux_square():
    import matplotlib
    matplotlib.use('Agg')
    from lgca import get_lgca
    lgca = get_lgca(geometry='square', interaction='alignment', dims=(10, 10))
    lgca.timeevo(timesteps=5, record=True)
    lgca.plot_flux()


# TODO: test plot_density (hex)
@pytest.mark.skip(reason="TODO: implement hex density plot smoke test")
def test_plot_density_hex():
    import matplotlib
    matplotlib.use('Agg')
    from lgca import get_lgca
    lgca = get_lgca(geometry='hex', interaction='random_walk')
    lgca.timeevo(timesteps=5)
    lgca.plot_density()


# TODO: test plot_density (1D)
@pytest.mark.skip(reason="TODO: implement 1D density plot smoke test")
def test_plot_density_1d():
    import matplotlib
    matplotlib.use('Agg')
    from lgca import get_lgca
    lgca = get_lgca(geometry='lin', interaction='random_walk', dims=(30,))
    lgca.timeevo(timesteps=5, record=True)
    lgca.plot_density()
