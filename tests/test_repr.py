import pytest
from lgca import get_lgca

try:
    from lgca.lgca_cubic import LGCA_Cubic  # noqa: F401
except Exception:
    HAS_CUBIC = False
else:
    HAS_CUBIC = True

geometries = ['lin', 'square', 'hex']
if HAS_CUBIC:
    geometries.append('cubic')

@pytest.mark.parametrize('geom', geometries)
def test_repr_and_str(geom):
    lgca = get_lgca(geometry=geom, dims=2, interaction='only_propagation')
    r = repr(lgca)
    assert lgca.__class__.__name__ in r
    assert f"geometry={lgca.geometry}" in r
    assert f"dims={lgca.dims}" in r
    assert "bc=periodic" in r
    assert "interaction=only_propagation" in r

    s = str(lgca)
    assert "Model:" in s
    assert f"Geometry: {lgca.geometry}" in s
    assert "Interaction:" in s
