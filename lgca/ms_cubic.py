"""Multi-species cubic lattice LGCA implementations."""

from __future__ import annotations

import numpy as np

from .lgca_cubic import LGCA_Cubic
from .cubic_ext import NoVE_LGCA_Cubic
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_Cubic(MultiSpeciesLGCA_base, LGCA_Cubic):
    """Classical multi-species LGCA on a cubic lattice."""

    geometry = "cubic"
    interactions = LGCA_Cubic.interactions + ["excitable_medium_ms"]

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.n_species,
                self.K,
            ),
            dtype=bool,
        )
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = self._ensure_bool_nodes(nodes)
            self.apply_boundaries()



class MSLGCA_NoVE_Cubic(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Cubic):
    """No-volume-exclusion multi-species LGCA on a cubic lattice."""

    geometry = "cubic"

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.n_species,
                self.K,
            ),
            dtype=np.uint,
        )
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.apply_boundaries()

