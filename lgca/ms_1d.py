"""Multi-species 1D LGCA implementations."""

from __future__ import annotations

import numpy as np

from .lgca_1d import LGCA_1D, NoVE_LGCA_1D
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_1D(MultiSpeciesLGCA_base, LGCA_1D):
    """Classical multi-species LGCA on a 1D lattice."""

    geometry = "lin"
    interactions = LGCA_1D.interactions + ["excitable_medium_ms"]

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.n_species, self.K), dtype=bool)
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = self._ensure_bool_nodes(nodes)
            self.apply_boundaries()



class MSLGCA_NoVE_1D(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_1D):
    """No-volume-exclusion multi-species LGCA on a 1D lattice."""

    geometry = "lin"

    def init_nodes(self, density: float = 0.1, nodes: np.ndarray | None = None, **kwargs) -> None:
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.n_species, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.apply_boundaries()
