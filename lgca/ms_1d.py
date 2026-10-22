"""Multi-species 1D LGCA implementations."""

from __future__ import annotations

from .lgca_1d import LGCA_1D, NoVE_LGCA_1D
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_1D(MultiSpeciesLGCA_base, LGCA_1D):
    """Classical multi-species LGCA on a 1D lattice."""

    geometry = "lin"


class MSLGCA_NoVE_1D(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_1D):
    """No-volume-exclusion multi-species LGCA on a 1D lattice."""

    geometry = "lin"
