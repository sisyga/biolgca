"""Multi-species cubic lattice LGCA implementations."""

from __future__ import annotations

from .cubic_ext import NoVE_LGCA_Cubic
from .lgca_cubic import LGCA_Cubic
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_Cubic(MultiSpeciesLGCA_base, LGCA_Cubic):
    """Classical multi-species LGCA on a cubic lattice."""

    geometry = "cubic"


class MSLGCA_NoVE_Cubic(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Cubic):
    """No-volume-exclusion multi-species LGCA on a cubic lattice."""

    geometry = "cubic"
