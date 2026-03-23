"""
Stub tests for lgca/base_extensions.py — top under-tested module.

TODO: Add tests for the following coverage gaps:
  - Identity tracking methods (lines ~200–400)
  - Spatial analysis helpers (density_sum, etc.)
  - set_interaction dispatch for ib-specific interactions (lines ~1366–1483)
  - Recording/replay methods
  - NoVE extension class init and propagation overrides (lines ~1726–2035)
  - 2D extension plotting hooks (lines ~2049–2232)

Run baseline: 35% coverage (669/1034 lines uncovered)
"""
import pytest


# TODO: test identity-based LGCA class initialisation
@pytest.mark.skip(reason="TODO: implement identity-based extension tests")
def test_ib_lgca_init():
    from lgca import get_lgca
    lgca = get_lgca(ib=True, geometry='square', interaction='random_walk')
    assert lgca is not None


# TODO: test NoVE extension class initialisation
@pytest.mark.skip(reason="TODO: implement NoVE extension tests")
def test_nove_ext_init():
    from lgca import get_lgca
    lgca = get_lgca(nove=True, geometry='square', interaction='random_walk')
    assert lgca is not None


# TODO: test interaction dispatch for 'ib_alignment'
@pytest.mark.skip(reason="TODO: implement ib interaction dispatch tests")
def test_ib_interaction_dispatch():
    from lgca import get_lgca
    lgca = get_lgca(ib=True, geometry='square', interaction='ib_alignment')
    lgca.timeevo(timesteps=5)
    assert lgca.t == 5


# TODO: test spatial density helpers
@pytest.mark.skip(reason="TODO: implement spatial analysis tests")
def test_density_helpers():
    from lgca import get_lgca
    lgca = get_lgca(geometry='square', interaction='random_walk', dims=(10, 10))
    lgca.timeevo(timesteps=5)
    # check that density sums are non-negative
    assert lgca.cell_density.min() >= 0
