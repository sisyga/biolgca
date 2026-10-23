"""Multi-species hexagonal LGCA implementations."""

from __future__ import annotations

from .lgca_hex import LGCA_Hex, NoVE_LGCA_Hex
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_Hex(MultiSpeciesLGCA_base, LGCA_Hex):
    """Classical multi-species LGCA on a hexagonal lattice."""

    geometry = "hex"


class MSLGCA_NoVE_Hex(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Hex):
    """No-volume-exclusion multi-species LGCA on a hexagonal lattice."""

    geometry = "hex"
