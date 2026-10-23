"""Multi-species 3D Moore lattice LGCA implementations."""

from __future__ import annotations

from .cubic_ext import NoVE_LGCA_Cubic
from .lgca_3dmoore import LGCA_3dMoore
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_Moore(MultiSpeciesLGCA_base, LGCA_3dMoore):
    """Classical multi-species LGCA on a 3D Moore lattice."""

    geometry = "moore"


class MSLGCA_NoVE_Moore(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Cubic, LGCA_3dMoore):
    """No-volume-exclusion multi-species LGCA on a 3D Moore lattice."""

    geometry = "moore"
