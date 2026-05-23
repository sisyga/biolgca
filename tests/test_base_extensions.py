import lgca.lgca_square
import lgca.base

import pytest
import numpy as np
from lgca import get_lgca

def test_ib_lgca_init():
    lgca = get_lgca(ib=True, geometry='square', interaction='random_walk', dims=10, density=0.5)
    assert lgca is not None
    assert lgca.cell_density.shape == (12, 12)
    assert hasattr(lgca, 'props')
    assert hasattr(lgca, 'maxlabel')

def test_nove_ext_init():
    lgca = get_lgca(ve=False, geometry='square', interaction='random_walk', dims=10, restchannels=1, capacity=5)
    assert lgca is not None
    assert lgca.cell_density.shape == (12, 12)
    assert hasattr(lgca, 'capacity')
    assert lgca.capacity == 5

def test_nove_ib_ext_init():
    lgca = get_lgca(ve=False, ib=True, geometry='square', interaction='random_walk', dims=10, density=0.2, capacity=10)
    assert lgca is not None
    assert lgca.cell_density.shape == (12, 12)
    assert hasattr(lgca, 'props')
    assert hasattr(lgca, 'capacity')

def test_ib_interaction_dispatch():
    lgca = get_lgca(ib=True, geometry='square', interaction='birth', dims=10, density=0.1, r_b=0.5)
    lgca.timeevo(timesteps=5, showprogress=False)
    # Check that it actually ran
    assert hasattr(lgca, 'props')
    assert 'r_b' in lgca.props

def test_nove_interaction_dispatch():
    lgca = get_lgca(ve=False, geometry='square', interaction='dd_alignment', dims=10, restchannels=0, beta=1.0)
    lgca.timeevo(timesteps=5, showprogress=False)
    assert lgca.interaction_params.get('beta') == 1.0

def test_density_helpers():
    lgca = get_lgca(geometry='square', interaction='random_walk', dims=(10, 10), density=0.2)
    lgca.timeevo(timesteps=5, recorddens=True, showprogress=False)
    # check that density sums are non-negative
    assert lgca.cell_density.min() >= 0
    assert hasattr(lgca, 'dens_t')
    assert lgca.dens_t.shape == (6, 10, 10)

def test_ib_properties():
    lgca = get_lgca(ib=True, geometry='square', interaction='birth', dims=10, density=0.2, r_b=0.5)
    prop = lgca.get_prop(propname='r_b')
    assert prop.shape == lgca.nodes[lgca.nonborder].shape
    mean_prop = lgca.calc_prop_mean(propname='r_b')
    assert mean_prop.shape == (10, 10)

def test_families():
    lgca = get_lgca(ib=True, geometry='square', interaction='go_and_grow', dims=10, density=0.2, track_inheritance=True)
    assert 'family' in lgca.props
    
    # Try recording family pop
    lgca = get_lgca(ib=True, geometry='square', interaction='go_and_grow_mutations', dims=10, density=0.2, track_inheritance=True)
    lgca.timeevo(timesteps=2, recordfampop=True, showprogress=False)
    assert hasattr(lgca, 'fam_pop_t')

