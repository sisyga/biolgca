# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2026 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Extended 3D cubic LGCA models.

Adds identity-based and no-volume-exclusion variants for cubic geometries.
"""


from __future__ import annotations

import numpy as np

from lgca.base import normalize_dims
from lgca.ib_base import IBLGCA_base
from lgca.list_utils import get_arr_of_empty_lists
from lgca.nove_base import NoVE_LGCA_base
from lgca.nove_ib_base import NoVE_IBLGCA_base

from .lgca_cubic import LGCA_Cubic


class IBLGCA_Cubic(IBLGCA_base, LGCA_Cubic):
    """
    Identity-based LGCA simulator class for a 3D cubic lattice.
    """


    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize the lattice for IBLGCA.
        """
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
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

    def plot_prop_spatial(self, nodes=None, props=None, propname=None, **kwargs):
        """
        Plot the mean value of a cell property in every occupied node with Mayavi.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`, optional
            Lattice configuration with particle labels. Default: the current state.
        props : dict, optional
            Property dictionary. Default: ``self.props``.
        propname : str, optional
            Property to plot. Default: the first property in `props`.
        **kwargs
            Passed to :py:meth:`lgca.lgca_cubic.LGCA_Cubic.plot_scalarfield`.

        Returns
        -------
        tuple
            As returned by :py:meth:`lgca.lgca_cubic.LGCA_Cubic.plot_scalarfield`.
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propname is None:
            propname = next(iter(props))

        meanprop = np.ma.masked_array(self.calc_prop_mean(propname=propname, props=props, nodes=nodes),
                                      mask=~np.any(self._channel_counts(nodes), axis=-1))
        kwargs.setdefault("cbarlabel", str(propname))
        return self.plot_scalarfield(meanprop, **kwargs)



def _set_dims_3d(lgca, dims, nodes, restchannels, capacity):
    """Set the size, the channels and the capacity of a 3D model without volume exclusion."""
    if nodes is not None:
        try:
            lgca.lx, lgca.ly, lgca.lz, lgca.K = nodes.shape
        except ValueError as e:
            raise ValueError("Node shape does not match the 3D geometry! Shape must be (x,y,z,channels)") from e
    else:
        lgca.lx, lgca.ly, lgca.lz = normalize_dims(dims, 3, default=(10, 10, 10))
    lgca.dims = (lgca.lx, lgca.ly, lgca.lz)
    lgca._set_channels(restchannels, capacity, from_nodes=nodes is not None)


class NoVE_LGCA_Cubic(LGCA_Cubic, NoVE_LGCA_base):
    """
    3D cubic version of an LGCA without volume exclusion.
    """


    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None):
        """
        Set the dimensions of the instance according to given values.
        """
        _set_dims_3d(self, dims, nodes, restchannels, capacity)

    def init_nodes(self, density=4, nodes=None):
        """
        Initialize lattice nodes.
        """
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.K,
            ),
            dtype=np.uint,
        )
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[
                self.r_int : -self.r_int,
                self.r_int : -self.r_int,
                self.r_int : -self.r_int,
                :,
            ] = nodes.astype(np.uint)
            self.apply_boundaries()


class NoVE_IBLGCA_Cubic(NoVE_IBLGCA_base, LGCA_Cubic):
    """
    Identity-based LGCA without volume exclusion for a 3D cubic lattice.
    """

    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None):
        """
        Set the dimensions of the instance according to given values.
        """
        _set_dims_3d(self, dims, nodes, restchannels, capacity)

    def init_nodes(self, density=0.1, nodes=None):
        """
        Initialize the lattice nodes for NoVE_IBLGCA.
        """
        self.nodes = get_arr_of_empty_lists(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.K,
            )
        )
        if nodes is None:
            self.random_reset(density)
        elif nodes.dtype == object:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes
        else:
            self._warn_nodes_shape(nodes)
            occ = nodes.astype(int)
            self.nodes[self.nonborder] = self.convert_int_to_ib(occ)
        self.calc_max_label()

    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.
        """
        newnodes = get_arr_of_empty_lists(self.nodes.shape)
        newnodes[..., self.velocitychannels :] = self.nodes[
            ..., self.velocitychannels :
        ]

        newnodes[1:, :, :, 0] = self.nodes[:-1, :, :, 0]
        newnodes[:-1, :, :, 1] = self.nodes[1:, :, :, 1]
        newnodes[:, 1:, :, 2] = self.nodes[:, :-1, :, 2]
        newnodes[:, :-1, :, 3] = self.nodes[:, 1:, :, 3]
        newnodes[:, :, 1:, 4] = self.nodes[:, :, :-1, 4]
        newnodes[:, :, :-1, 5] = self.nodes[:, :, 1:, 5]

        self.nodes = newnodes

    def _apply_abc_x(self):
        """
        Apply absorbing boundary conditions in x-direction.
        """
        self.nodes[: self.r_int, :, :, :] = get_arr_of_empty_lists(
            self.nodes[: self.r_int, :, :, :].shape
        )
        self.nodes[-self.r_int :, :, :, :] = get_arr_of_empty_lists(
            self.nodes[-self.r_int :, :, :, :].shape
        )

    def _apply_abc_y(self):
        """
        Apply absorbing boundary conditions in y-direction.
        """
        self.nodes[:, : self.r_int, :, :] = get_arr_of_empty_lists(
            self.nodes[:, : self.r_int, :, :].shape
        )
        self.nodes[:, -self.r_int :, :, :] = get_arr_of_empty_lists(
            self.nodes[:, -self.r_int :, :, :].shape
        )

    def _apply_abc_z(self):
        """
        Apply absorbing boundary conditions in z-direction.
        """
        self.nodes[:, :, : self.r_int, :] = get_arr_of_empty_lists(
            self.nodes[:, :, : self.r_int, :].shape
        )
        self.nodes[:, :, -self.r_int :, :] = get_arr_of_empty_lists(
            self.nodes[:, :, -self.r_int :, :].shape
        )

    def plot_prop_spatial(self, nodes=None, props=None, propname=None, **kwargs):
        """
        Plot the mean value of a cell property in every occupied node with Mayavi.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`, optional
            Lattice configuration with particle labels. Default: the current state.
        props : dict, optional
            Property dictionary. Default: ``self.props``.
        propname : str, optional
            Property to plot. Default: the first property in `props`.
        **kwargs
            Passed to :py:meth:`lgca.lgca_cubic.LGCA_Cubic.plot_scalarfield`.

        Returns
        -------
        tuple
            As returned by :py:meth:`lgca.lgca_cubic.LGCA_Cubic.plot_scalarfield`.
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propname is None:
            propname = next(iter(props))

        meanprop = np.ma.masked_array(self.calc_prop_mean(propname=propname, props=props, nodes=nodes),
                                      mask=~np.any(self._channel_counts(nodes), axis=-1))
        kwargs.setdefault("cbarlabel", str(propname))
        return self.plot_scalarfield(meanprop, **kwargs)
