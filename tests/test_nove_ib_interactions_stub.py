"""
Stub tests for lgca/nove_ib_interactions.py — 6% coverage (208/222 lines uncovered).

Functions to test (all currently untested):
  - random_walk (NoVE + IB)
  - evo_steric
  - birth
  - birthdeath
  - birthdeath_cancerdfe
  - go_or_grow
  - go_or_grow_kappa
  - tanh_switch
  - go_or_grow_kappa_chemo

Run baseline: 6% coverage
"""
import pytest


# TODO: test go_or_grow with NoVE + IB LGCA
@pytest.mark.skip(reason="TODO: implement nove_ib go_or_grow test")
def test_nove_ib_go_or_grow():
    from lgca import get_lgca
    lgca = get_lgca(nove=True, ib=True, geometry='square', interaction='go_or_grow')
    lgca.timeevo(timesteps=5)
    assert lgca.t == 5


# TODO: test birth interaction
@pytest.mark.skip(reason="TODO: implement nove_ib birth test")
def test_nove_ib_birth():
    from lgca import get_lgca
    lgca = get_lgca(nove=True, ib=True, geometry='lin', interaction='birth')
    lgca.timeevo(timesteps=5)
    assert lgca.t == 5


# TODO: test tanh_switch helper directly
@pytest.mark.skip(reason="TODO: implement tanh_switch unit test")
def test_tanh_switch():
    from lgca.nove_ib_interactions import tanh_switch
    val = tanh_switch(0.5, kappa=5.0, theta=0.8)
    assert 0.0 <= val <= 1.0
    # Below threshold → should approach 0
    assert tanh_switch(0.0) < 0.5
    # Above threshold → should approach 1
    assert tanh_switch(1.0) > 0.5


# TODO: test go_or_grow_kappa_chemo
@pytest.mark.skip(reason="TODO: implement go_or_grow_kappa_chemo test")
def test_nove_ib_go_or_grow_kappa_chemo():
    from lgca import get_lgca
    lgca = get_lgca(nove=True, ib=True, geometry='square', interaction='go_or_grow_kappa_chemo')
    lgca.timeevo(timesteps=3)
    assert lgca.t == 3
