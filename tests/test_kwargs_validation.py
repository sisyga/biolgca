"""Tests for kwargs validation in LGCA constructors."""
import pytest
from lgca import get_lgca


class TestKwargsValidation:
    """Test that LGCA constructors validate kwargs and catch typos."""

    def test_unknown_kwarg_raises(self):
        """Test that unknown kwargs raise TypeError."""
        with pytest.raises(TypeError, match='densty'):
            get_lgca(densty=0.5)

    def test_valid_kwargs_accepted(self):
        """Test that valid kwargs are accepted without error."""
        # Test common valid kwargs
        lgca = get_lgca(density=0.5)
        assert lgca is not None
        
        lgca = get_lgca(dims=(5, 5), restchannels=2)
        assert lgca is not None
        
        lgca = get_lgca(bc='periodic', seed=42)
        assert lgca is not None

    def test_typo_suggestion(self):
        """Test that error message suggests the closest valid parameter."""
        with pytest.raises(TypeError, match='density'):
            get_lgca(densty=0.5)
        
        with pytest.raises(TypeError, match='restchannels'):
            get_lgca(restchannel=2)
        
        with pytest.raises(TypeError, match='interaction'):
            get_lgca(interacton='birth')

    def test_multiple_unknown_kwargs_raises(self):
        """Test that multiple unknown kwargs are caught."""
        with pytest.raises(TypeError) as exc_info:
            get_lgca(densty=0.5, bcx='periodic')
        # Should mention at least one of the typos
        assert 'densty' in str(exc_info.value) or 'bcx' in str(exc_info.value)

    def test_interaction_params_accepted(self):
        """Test that interaction-specific params are accepted."""
        lgca = get_lgca(interaction='birth', r_b=0.3)
        assert lgca is not None
        
        lgca = get_lgca(interaction='go_or_grow', r_b=0.2, r_d=0.01, kappa=5.0, theta=0.75)
        assert lgca is not None
        
        lgca = get_lgca(interaction='chemotaxis', beta=5.0)
        assert lgca is not None

    def test_interaction_param_typo_suggestion(self):
        """Test that typos in interaction params are caught with suggestions."""
        with pytest.raises(TypeError, match='r_b'):
            get_lgca(interaction='birth', r_birth=0.3)
        
        with pytest.raises(TypeError, match='beta'):
            get_lgca(interaction='chemotaxis', betta=5.0)

    def test_nove_specific_kwargs_accepted(self):
        """Test that NoVE-specific kwargs are accepted."""
        lgca = get_lgca(ve=False, capacity=10)
        assert lgca is not None
        
        lgca = get_lgca(ve=False, hom='homogeneous')
        assert lgca is not None

    def test_nove_kwarg_typo_caught(self):
        """Test that typos in NoVE-specific kwargs are caught."""
        with pytest.raises(TypeError, match='capacity'):
            get_lgca(ve=False, capasity=10)

    def test_ib_kwargs_accepted(self):
        """Test that IB-specific interaction params are accepted."""
        lgca = get_lgca(ib=True, interaction='birth', r_b=0.2, std=0.01, a_max=1.0)
        assert lgca is not None

    def test_geometry_specific_init(self):
        """Test that different geometries accept valid kwargs."""
        for geom in ['hex', 'square', '1d']:
            lgca = get_lgca(geometry=geom, density=0.3, dims=(5, 5) if geom != '1d' else 10)
            assert lgca is not None

    def test_nodes_param_accepted(self):
        """Test that 'nodes' parameter is accepted."""
        import numpy as np
        nodes = np.zeros((5, 5, 8))
        lgca = get_lgca(geometry='hex', nodes=nodes)
        assert lgca is not None
