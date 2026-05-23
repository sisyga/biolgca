import pytest
import numpy as np

from lgca import get_lgca
from lgca.lgca_1d import LGCA_1D, IBLGCA_1D, NoVE_LGCA_1D, NoVE_IBLGCA_1D
from lgca.lgca_square import LGCA_Square
from lgca.square_ext import IBLGCA_Square, NoVE_LGCA_Square, NoVE_IBLGCA_Square
from lgca.lgca_hex import LGCA_Hex, IBLGCA_Hex, NoVE_LGCA_Hex, NoVE_IBLGCA_Hex
from lgca.lgca_cubic import LGCA_Cubic
from lgca.cubic_ext import (
    IBLGCA_Cubic,
    NoVE_LGCA_Cubic,
    NoVE_IBLGCA_Cubic,
)
from lgca.lgca_3dmoore import (
    LGCA_3dMoore,
    IBLGCA_Moore,
    NoVE_LGCA_Moore,
    NoVE_IBLGCA_Moore,
)

EXPECTED = {
    'lin': {
        (False, True): LGCA_1D,
        (True, True): IBLGCA_1D,
        (False, False): NoVE_LGCA_1D,
        (True, False): NoVE_IBLGCA_1D,
    },
    'square': {
        (False, True): LGCA_Square,
        (True, True): IBLGCA_Square,
        (False, False): NoVE_LGCA_Square,
        (True, False): NoVE_IBLGCA_Square,
    },
    'hex': {
        (False, True): LGCA_Hex,
        (True, True): IBLGCA_Hex,
        (False, False): NoVE_LGCA_Hex,
        (True, False): NoVE_IBLGCA_Hex,
    },
    'cubic': {
        (False, True): LGCA_Cubic,
        (True, True): IBLGCA_Cubic,
        (False, False): NoVE_LGCA_Cubic,
        (True, False): NoVE_IBLGCA_Cubic,
    },
    'moore': {
        (False, True): LGCA_3dMoore,
        (True, True): IBLGCA_Moore,
        (False, False): NoVE_LGCA_Moore,
        (True, False): NoVE_IBLGCA_Moore,
    },
}

geometries = ['lin', 'square', 'hex', 'cubic', 'moore']

PARAMS = [
    (g, ib, ve)
    for g in geometries
    for ib, ve in [
        (False, True),  # classical LGCA
        (True, True),   # identity based
        (False, False), # NoVE classical
        (True, False),  # NoVE identity based
    ]
]

@pytest.mark.parametrize("geom, ib, ve", PARAMS)
def test_get_lgca_returns_correct_subclass(geom, ib, ve):
    lgca = get_lgca(
        geometry=geom,
        ib=ib,
        ve=ve,
        density=0,
        dims=2,
        restchannels=1,
        interaction="only_propagation",
    )
    assert isinstance(lgca, EXPECTED[geom][(ib, ve)])


def test_warning_on_mismatched_dims():
    nodes = np.zeros((3, 4))
    with pytest.warns(UserWarning):
        get_lgca(
            geometry='lin',
            nodes=nodes,
            dims=10,
            interaction='only_propagation',
        )


def test_warning_on_mismatched_restchannels():
    nodes = np.zeros((3, 4))
    with pytest.warns(UserWarning):
        get_lgca(
            geometry='lin',
            nodes=nodes,
            restchannels=1,
            interaction='only_propagation',
        )


def test_warning_when_nodes_override_density():
    nodes = np.zeros((3, 2), dtype=bool)
    with pytest.warns(UserWarning, match="density"):
        get_lgca(
            geometry='lin',
            nodes=nodes,
            density=0.9,
            interaction='only_propagation',
        )


def test_warning_on_nonboolean_nodes():
    nodes = np.array([[2, 0], [3, 1]])
    with pytest.warns(UserWarning):
        lgca = get_lgca(
            geometry='lin',
            ib=False,
            ve=True,
            nodes=nodes,
            interaction='only_propagation',
        )
    assert set(np.unique(lgca.nodes[lgca.nonborder])) <= {0, 1}


