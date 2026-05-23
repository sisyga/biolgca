"""Multi-species 3D Moore lattice LGCA implementations."""

from __future__ import annotations

import numpy as np

from .lgca_3dmoore import LGCA_3dMoore
from .cubic_ext import NoVE_LGCA_Cubic
from .multispecies_base import MultiSpeciesLGCA_base, MultiSpeciesNoVE_LGCA_base


class MSLGCA_Moore(MultiSpeciesLGCA_base, LGCA_3dMoore):
    """Classical multi-species LGCA on a 3D Moore lattice."""

    geometry = "moore"
    interactions = LGCA_3dMoore.interactions + ["excitable_medium_ms"]

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



class MSLGCA_NoVE_Moore(MultiSpeciesNoVE_LGCA_base, NoVE_LGCA_Cubic, LGCA_3dMoore):
    """No-volume-exclusion multi-species LGCA on a 3D Moore lattice."""

    geometry = "moore"

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

    def propagation(self) -> None:
        newnodes = np.zeros_like(self.nodes)
        newnodes[..., self.velocitychannels :] = self.nodes[..., self.velocitychannels :]
        for k, (dx, dy, dz) in enumerate(self._vels):
            src_x = slice(max(-dx, 0), self.nodes.shape[0] - max(dx, 0))
            dst_x = slice(max(dx, 0), self.nodes.shape[0] - max(-dx, 0))
            src_y = slice(max(-dy, 0), self.nodes.shape[1] - max(dy, 0))
            dst_y = slice(max(dy, 0), self.nodes.shape[1] - max(-dy, 0))
            src_z = slice(max(-dz, 0), self.nodes.shape[2] - max(dz, 0))
            dst_z = slice(max(dz, 0), self.nodes.shape[2] - max(-dz, 0))
            newnodes[dst_x, dst_y, dst_z, :, k] = self.nodes[src_x, src_y, src_z, :, k]
        self.nodes = newnodes

    def _apply_rbc_x(self) -> None:
        for i, (dx, _, _) in enumerate(self._vels):
            j = self._ref_x[i]
            if dx == -1:
                self.nodes[self.r_int, :, :, :, j] += self.nodes[self.r_int - 1, :, :, :, i]
            elif dx == 1:
                self.nodes[-self.r_int - 1, :, :, :, j] += self.nodes[-self.r_int, :, :, :, i]
        self._apply_abc_x()

    def _apply_rbc_y(self) -> None:
        for i, (_, dy, _) in enumerate(self._vels):
            j = self._ref_y[i]
            if dy == -1:
                self.nodes[:, self.r_int, :, :, j] += self.nodes[:, self.r_int - 1, :, :, i]
            elif dy == 1:
                self.nodes[:, -self.r_int - 1, :, :, j] += self.nodes[:, -self.r_int, :, :, i]
        self._apply_abc_y()

    def _apply_rbc_z(self) -> None:
        for i, (_, _, dz) in enumerate(self._vels):
            j = self._ref_z[i]
            if dz == -1:
                self.nodes[:, :, self.r_int, :, j] += self.nodes[:, :, self.r_int - 1, :, i]
            elif dz == 1:
                self.nodes[:, :, -self.r_int - 1, :, j] += self.nodes[:, :, -self.r_int, :, i]
        self._apply_abc_z()

    def apply_rbc(self) -> None:
        self._apply_rbc_x()
        self._apply_rbc_y()
        self._apply_rbc_z()
        self.apply_abc()

    def _apply_pbc_x(self) -> None:
        self.nodes[: self.r_int, ...] = self.nodes[-2 * self.r_int : -self.r_int, ...]
        self.nodes[-self.r_int :, ...] = self.nodes[self.r_int : 2 * self.r_int, ...]

    def _apply_pbc_y(self) -> None:
        self.nodes[:, : self.r_int, ...] = self.nodes[:, -2 * self.r_int : -self.r_int, ...]
        self.nodes[:, -self.r_int :, ...] = self.nodes[:, self.r_int : 2 * self.r_int, ...]

    def _apply_pbc_z(self) -> None:
        self.nodes[:, :, : self.r_int, ...] = self.nodes[:, :, -2 * self.r_int : -self.r_int, ...]
        self.nodes[:, :, -self.r_int :, ...] = self.nodes[:, :, self.r_int : 2 * self.r_int, ...]

    def apply_pbc(self) -> None:
        self._apply_pbc_x()
        self._apply_pbc_y()
        self._apply_pbc_z()

    def _apply_abc_x(self) -> None:
        self.nodes[: self.r_int, ...] = 0
        self.nodes[-self.r_int :, ...] = 0

    def _apply_abc_y(self) -> None:
        self.nodes[:, : self.r_int, ...] = 0
        self.nodes[:, -self.r_int :, ...] = 0

    def _apply_abc_z(self) -> None:
        self.nodes[:, :, : self.r_int, ...] = 0
        self.nodes[:, :, -self.r_int :, ...] = 0

    def apply_abc(self) -> None:
        self._apply_abc_x()
        self._apply_abc_y()
        self._apply_abc_z()
