"""Multi-species hexagonal LGCA implementations."""

from __future__ import annotations

import numpy as np

from .lgca_hex import LGCA_Hex, NoVE_LGCA_Hex
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_Hex(MultiSpeciesLGCA_base, LGCA_Hex):
    """Classical multi-species LGCA on a hexagonal lattice."""

    geometry = "hex"
    interactions = LGCA_Hex.interactions + ["excitable_medium_ms"]

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.n_species, self.K), dtype=bool)
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = self._ensure_bool_nodes(nodes)
            self.apply_boundaries()



class MSLGCA_NoVE_Hex(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Hex):
    """No-volume-exclusion multi-species LGCA on a hexagonal lattice."""

    geometry = "hex"

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.n_species, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.apply_boundaries()

