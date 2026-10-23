"""Multi-species square lattice LGCA classes."""

from __future__ import annotations

from .lgca_square import LGCA_Square
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base
from .square_ext import NoVE_LGCA_Square


class MSLGCA_Square(MultiSpeciesLGCA_base, LGCA_Square):
    """Classical multi-species LGCA on a square lattice."""

    geometry = "square"


class MSLGCA_NoVE_Square(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Square):
    """No-volume-exclusion multi-species LGCA on a square lattice."""

    geometry = "square"
