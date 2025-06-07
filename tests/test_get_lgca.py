import pytest

from lgca import get_lgca
from lgca.lgca_1d import LGCA_1D, IBLGCA_1D, NoVE_LGCA_1D, NoVE_IBLGCA_1D
from lgca.lgca_square import LGCA_Square, IBLGCA_Square, NoVE_LGCA_Square, NoVE_IBLGCA_Square
from lgca.lgca_hex import LGCA_Hex, IBLGCA_Hex, NoVE_LGCA_Hex, NoVE_IBLGCA_Hex

try:
    from lgca.lgca_cubic import (
        LGCA_Cubic,
        IBLGCA_Cubic,
        NoVE_LGCA_Cubic,
        NoVE_IBLGCA_Cubic,
    )
    HAS_CUBIC = True
except Exception:
    HAS_CUBIC = False

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
}

if HAS_CUBIC:
    EXPECTED['cubic'] = {
        (False, True): LGCA_Cubic,
        (True, True): IBLGCA_Cubic,
        (False, False): NoVE_LGCA_Cubic,
        (True, False): NoVE_IBLGCA_Cubic,
    }

geometries = ['lin', 'square', 'hex']
if HAS_CUBIC:
    geometries.append('cubic')

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
        restchannels=0,
        interaction="only_propagation",
    )
    assert isinstance(lgca, EXPECTED[geom][(ib, ve)])

