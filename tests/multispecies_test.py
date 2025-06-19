import numpy as np
import pytest

import pytest

from lgca import get_lgca
from lgca.ms_square import MSLGCA_Square, MSLGCA_NoVE_Square
from lgca.ms_1d import MSLGCA_1D, MSLGCA_NoVE_1D
from lgca.ms_hex import MSLGCA_Hex, MSLGCA_NoVE_Hex
from lgca.ms_cubic import MSLGCA_Cubic, MSLGCA_NoVE_Cubic
from lgca.ms_moore import MSLGCA_Moore, MSLGCA_NoVE_Moore

GEOMS = {
    "square": (MSLGCA_Square, MSLGCA_NoVE_Square),
    "lin": (MSLGCA_1D, MSLGCA_NoVE_1D),
    "hex": (MSLGCA_Hex, MSLGCA_NoVE_Hex),
    "cubic": (MSLGCA_Cubic, MSLGCA_NoVE_Cubic),
    "moore": (MSLGCA_Moore, MSLGCA_NoVE_Moore),
}


@pytest.mark.parametrize("geom", GEOMS)
def test_get_lgca_multispecies(geom):
    cls, _ = GEOMS[geom]
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        density=0,
        restchannels=1,
        interaction="only_propagation",
        dims=3,
    )
    assert isinstance(lgca, cls)
    expected_shape = tuple(d + 2 * lgca.r_int for d in lgca.dims) + (2, lgca.K)
    assert lgca.nodes.shape == expected_shape
    assert lgca.species_density.shape == lgca.nodes.shape[:-1]
    assert lgca.cell_density.shape == lgca.nodes.shape[:-2]


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_propagation(geom):
    cls, _ = GEOMS[geom]
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        density=0,
        restchannels=1,
        interaction="only_propagation",
        dims=3,
    )
    lgca.nodes.fill(False)
    center = (lgca.r_int,) * len(lgca.dims)
    channel = 0 if geom != "moore" else cls._vels.index((1, 0, 0))
    lgca.nodes[center + (1, channel)] = True
    lgca.timestep()
    if geom == "lin":
        target = (lgca.r_int + 1,)
    elif geom in {"square", "hex"}:
        target = (lgca.r_int + 1, lgca.r_int)
    else:
        target = (lgca.r_int + 1, lgca.r_int, lgca.r_int)
    assert lgca.nodes[target + (1, channel)]
    assert not lgca.nodes[center + (1, channel)]


@pytest.mark.parametrize("geom", GEOMS)
def test_get_lgca_multispecies_nove(geom):
    _, cls = GEOMS[geom]
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        restchannels=1,
        density=0,
        interaction="only_propagation",
        dims=3,
    )
    assert isinstance(lgca, cls)
    assert lgca.nodes.dtype.kind in "iu"
    expected_shape = tuple(d + 2 * lgca.r_int for d in lgca.dims) + (2, lgca.K)
    assert lgca.nodes.shape == expected_shape


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_nove_propagation(geom):
    _, cls = GEOMS[geom]
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        restchannels=1,
        density=0,
        interaction="only_propagation",
        dims=3,
    )
    lgca.nodes.fill(0)
    center = (lgca.r_int,) * len(lgca.dims)
    channel = 0 if geom != "moore" else cls._vels.index((1, 0, 0))
    lgca.nodes[center + (1, channel)] = 2
    lgca.timestep()
    if geom == "lin":
        target = (lgca.r_int + 1,)
    elif geom in {"square", "hex"}:
        target = (lgca.r_int + 1, lgca.r_int)
    else:
        target = (lgca.r_int + 1, lgca.r_int, lgca.r_int)
    assert lgca.nodes[target + (1, channel)] == 2
    assert lgca.nodes[center + (1, channel)] == 0


def test_excitable_medium_ms_resting_species():
    lgca = get_lgca(
        geometry="square",
        n_species=2,
        restchannels=1,
        density=0,
        interaction="excitable_medium_ms",
        propagation=False,
        seed=1,
    )

    center = (lgca.r_int, lgca.r_int)
    lgca.nodes.fill(False)
    lgca.nodes[center + (0, lgca.velocitychannels)] = True
    lgca.nodes[center + (1, 0)] = True
    lgca.timestep()

    assert lgca.nodes[..., 0, : lgca.velocitychannels].sum() == 0
    assert lgca.nodes[..., 1, lgca.velocitychannels :].sum() == 0
