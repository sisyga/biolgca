# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""3D Moore lattice LGCA implementations."""

import numpy as np
from lgca.lgca_cubic import LGCA_Cubic
from lgca.cubic_ext import (
    IBLGCA_Cubic,
    NoVE_LGCA_Cubic,
    NoVE_IBLGCA_Cubic,
)


_empty_list_array = np.frompyfunc(list, 0, 1)


def _get_arr_of_empty_lists(shape):
    return _empty_list_array(np.empty(shape, dtype=object))


class LGCA_3dMoore(LGCA_Cubic):
    """Classical LGCA on a 3D Moore lattice."""

    geometry = "moore"
    velocitychannels = 26
    velocities = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    cix = np.array([v[0] for v in velocities], dtype=float)
    ciy = np.array([v[1] for v in velocities], dtype=float)
    ciz = np.array([v[2] for v in velocities], dtype=float)
    c = np.array([cix, ciy, ciz])

    _vels = velocities
    _ref_x = {}
    _ref_y = {}
    _ref_z = {}
    for i, (dx, dy, dz) in enumerate(velocities):
        _ref_x[i] = velocities.index((-dx, dy, dz))
        _ref_y[i] = velocities.index((dx, -dy, dz))
        _ref_z[i] = velocities.index((dx, dy, -dz))

    def propagation(self):
        """Move particles according to their velocity channels."""
        newnodes = np.zeros_like(self.nodes)
        newnodes[..., self.velocitychannels :] = self.nodes[..., self.velocitychannels :]
        for k, (dx, dy, dz) in enumerate(self._vels):
            src_x = slice(max(-dx, 0), self.nodes.shape[0] - max(dx, 0))
            dst_x = slice(max(dx, 0), self.nodes.shape[0] - max(-dx, 0))
            src_y = slice(max(-dy, 0), self.nodes.shape[1] - max(dy, 0))
            dst_y = slice(max(dy, 0), self.nodes.shape[1] - max(-dy, 0))
            src_z = slice(max(-dz, 0), self.nodes.shape[2] - max(dz, 0))
            dst_z = slice(max(dz, 0), self.nodes.shape[2] - max(-dz, 0))
            newnodes[dst_x, dst_y, dst_z, ..., k] = self.nodes[src_x, src_y, src_z, ..., k]
        self.nodes = newnodes

    def nb_sum(self, qty):
        """Sum values of ``qty`` in the 26-neighbourhood of each node."""
        s = np.zeros_like(qty)
        for dx, dy, dz in self._vels:
            src_x = slice(max(-dx, 0), qty.shape[0] - max(dx, 0))
            dst_x = slice(max(dx, 0), qty.shape[0] - max(-dx, 0))
            src_y = slice(max(-dy, 0), qty.shape[1] - max(dy, 0))
            dst_y = slice(max(dy, 0), qty.shape[1] - max(-dy, 0))
            src_z = slice(max(-dz, 0), qty.shape[2] - max(dz, 0))
            dst_z = slice(max(dz, 0), qty.shape[2] - max(-dz, 0))
            s[dst_x, dst_y, dst_z, ...] += qty[src_x, src_y, src_z, ...]
        return s

    def gradient(self, qty):
        """Return the spatial gradient of ``qty``."""
        gx, gy, gz = np.gradient(qty, 0.5)
        return np.stack((gx, gy, gz), axis=-1)

    def channel_weight(self, qty):
        """Compute neighbour weights for interaction fields."""
        w = np.zeros(qty.shape + (self.velocitychannels,))
        for k, (dx, dy, dz) in enumerate(self._vels):
            src_x = slice(max(-dx, 0), qty.shape[0] - max(dx, 0))
            dst_x = slice(max(dx, 0), qty.shape[0] - max(-dx, 0))
            src_y = slice(max(-dy, 0), qty.shape[1] - max(dy, 0))
            dst_y = slice(max(dy, 0), qty.shape[1] - max(-dy, 0))
            src_z = slice(max(-dz, 0), qty.shape[2] - max(dz, 0))
            dst_z = slice(max(dz, 0), qty.shape[2] - max(-dz, 0))
            w[dst_x, dst_y, dst_z, k] = qty[src_x, src_y, src_z]
        return w

    def _apply_rbc_x(self):
        for i, (dx, _, _) in enumerate(self._vels):
            j = self._ref_x[i]
            if dx == -1:
                self.nodes[self.r_int, :, :, ..., j] += self.nodes[self.r_int - 1, :, :, ..., i]
            elif dx == 1:
                self.nodes[-self.r_int - 1, :, :, ..., j] += self.nodes[-self.r_int, :, :, ..., i]

    def _apply_rbc_y(self):
        for i, (_, dy, _) in enumerate(self._vels):
            j = self._ref_y[i]
            if dy == -1:
                self.nodes[:, self.r_int, :, ..., j] += self.nodes[:, self.r_int - 1, :, ..., i]
            elif dy == 1:
                self.nodes[:, -self.r_int - 1, :, ..., j] += self.nodes[:, -self.r_int, :, ..., i]

    def _apply_rbc_z(self):
        for i, (_, _, dz) in enumerate(self._vels):
            j = self._ref_z[i]
            if dz == -1:
                self.nodes[:, :, self.r_int, ..., j] += self.nodes[:, :, self.r_int - 1, ..., i]
            elif dz == 1:
                self.nodes[:, :, -self.r_int - 1, ..., j] += self.nodes[:, :, -self.r_int, ..., i]

    def apply_rbc(self):
        self._apply_rbc_x()
        self._apply_rbc_y()
        self._apply_rbc_z()
        self.apply_abc()


class IBLGCA_Moore(IBLGCA_Cubic, LGCA_3dMoore):
    """Identity-based LGCA on a 3D Moore lattice."""


class NoVE_LGCA_Moore(NoVE_LGCA_Cubic, LGCA_3dMoore):
    """3D Moore LGCA without volume exclusion."""

    def nb_sum(self, qty):
        return LGCA_3dMoore.nb_sum(self, qty)


class NoVE_IBLGCA_Moore(NoVE_IBLGCA_Cubic, LGCA_3dMoore):
    """Identity-based 3D Moore LGCA without volume exclusion."""

    def propagation(self):
        """Move object-list particles through all 26 Moore velocity channels."""
        newnodes = _get_arr_of_empty_lists(self.nodes.shape)
        newnodes[..., self.velocitychannels :] = self.nodes[..., self.velocitychannels :]
        for k, (dx, dy, dz) in enumerate(self._vels):
            src_x = slice(max(-dx, 0), self.nodes.shape[0] - max(dx, 0))
            dst_x = slice(max(dx, 0), self.nodes.shape[0] - max(-dx, 0))
            src_y = slice(max(-dy, 0), self.nodes.shape[1] - max(dy, 0))
            dst_y = slice(max(dy, 0), self.nodes.shape[1] - max(-dy, 0))
            src_z = slice(max(-dz, 0), self.nodes.shape[2] - max(dz, 0))
            dst_z = slice(max(dz, 0), self.nodes.shape[2] - max(-dz, 0))
            newnodes[dst_x, dst_y, dst_z, ..., k] = self.nodes[src_x, src_y, src_z, ..., k]
        self.nodes = newnodes

    def _apply_rbc_x(self):
        for i, (dx, _, _) in enumerate(self._vels):
            j = self._ref_x[i]
            if dx == -1:
                self.nodes[self.r_int, :, :, ..., j] = (
                    self.nodes[self.r_int, :, :, ..., j] + self.nodes[self.r_int - 1, :, :, ..., i]
                )
            elif dx == 1:
                self.nodes[-self.r_int - 1, :, :, ..., j] = (
                    self.nodes[-self.r_int - 1, :, :, ..., j] + self.nodes[-self.r_int, :, :, ..., i]
                )

    def _apply_rbc_y(self):
        for i, (_, dy, _) in enumerate(self._vels):
            j = self._ref_y[i]
            if dy == -1:
                self.nodes[:, self.r_int, :, ..., j] = (
                    self.nodes[:, self.r_int, :, ..., j] + self.nodes[:, self.r_int - 1, :, ..., i]
                )
            elif dy == 1:
                self.nodes[:, -self.r_int - 1, :, ..., j] = (
                    self.nodes[:, -self.r_int - 1, :, ..., j] + self.nodes[:, -self.r_int, :, ..., i]
                )

    def _apply_rbc_z(self):
        for i, (_, _, dz) in enumerate(self._vels):
            j = self._ref_z[i]
            if dz == -1:
                self.nodes[:, :, self.r_int, ..., j] = (
                    self.nodes[:, :, self.r_int, ..., j] + self.nodes[:, :, self.r_int - 1, ..., i]
                )
            elif dz == 1:
                self.nodes[:, :, -self.r_int - 1, ..., j] = (
                    self.nodes[:, :, -self.r_int - 1, ..., j] + self.nodes[:, :, -self.r_int, ..., i]
                )

