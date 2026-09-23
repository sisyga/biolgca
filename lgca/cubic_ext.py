# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Extended 3D cubic LGCA models.

Adds identity-based and no-volume-exclusion variants for cubic geometries.
"""


from __future__ import annotations

import numpy as np

from lgca.ib_base import IBLGCA_base
from lgca.list_utils import get_arr_of_empty_lists
from lgca.nove_base import NoVE_LGCA_base
from lgca.nove_ib_base import NoVE_IBLGCA_base

from .lgca_cubic import LGCA_Cubic


class IBLGCA_Cubic(IBLGCA_base, LGCA_Cubic):
    """
    Identity-based LGCA simulator class for a 3D cubic lattice.
    """

    interactions = [
        "go_or_grow",
        "go_and_grow",
        "random_walk",
        "birth",
        "birthdeath",
        "birthdeath_discrete",
        "only_propagation",
        "go_and_grow_mutations",
    ]

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



class NoVE_LGCA_Cubic(LGCA_Cubic, NoVE_LGCA_base):
    """
    3D cubic version of an LGCA without volume exclusion.
    """

    interactions = ["dd_alignment", "di_alignment", "go_or_grow", "go_or_rest"]

    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None):
        """
        Set the dimensions of the instance according to given values.
        """
        if nodes is not None:
            try:
                self.lx, self.ly, self.lz, self.K = nodes.shape
            except ValueError as e:
                raise ValueError(
                    "Node shape does not match the 3D geometry! Shape must be (x,y,z,channels)"
                ) from e

            if self.K < self.velocitychannels:
                raise RuntimeError(
                    f"Not enough channels specified! Required: {self.velocitychannels}, provided: {self.K}"
                )
            self.restchannels = self.K - self.velocitychannels
            if capacity is not None:
                self.capacity = capacity
            elif restchannels is not None:
                self.capacity = self.velocitychannels + restchannels
            else:
                self.capacity = self.K
            self.dims = (self.lx, self.ly, self.lz)
            return
        elif dims is not None:
            if isinstance(dims, tuple) and len(dims) == 3:
                self.lx, self.ly, self.lz = dims
            elif isinstance(dims, int):
                self.lx = self.ly = self.lz = dims
            else:
                raise TypeError(
                    "Keyword 'dims' must be a tuple of three integers or an int!"
                )
        else:
            self.lx = self.ly = self.lz = 50

        self.dims = (self.lx, self.ly, self.lz)
        self.restchannels = restchannels or 0
        self.K = self.velocitychannels + self.restchannels
        self.capacity = capacity if capacity is not None else self.K

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

    def _apply_rbc_x(self):
        self.nodes[self.r_int, :, :, 0] = (
            self.nodes[self.r_int, :, :, 0] + self.nodes[self.r_int - 1, :, :, 1]
        )
        self.nodes[-self.r_int - 1, :, :, 1] = (
            self.nodes[-self.r_int - 1, :, :, 1] + self.nodes[-self.r_int, :, :, 0]
        )

    def _apply_rbc_y(self):
        self.nodes[:, self.r_int, :, 2] = (
            self.nodes[:, self.r_int, :, 2] + self.nodes[:, self.r_int - 1, :, 3]
        )
        self.nodes[:, -self.r_int - 1, :, 3] = (
            self.nodes[:, -self.r_int - 1, :, 3] + self.nodes[:, -self.r_int, :, 2]
        )

    def _apply_rbc_z(self):
        self.nodes[:, :, self.r_int, 4] = (
            self.nodes[:, :, self.r_int, 4] + self.nodes[:, :, self.r_int - 1, 5]
        )
        self.nodes[:, :, -self.r_int - 1, 5] = (
            self.nodes[:, :, -self.r_int - 1, 5] + self.nodes[:, :, -self.r_int, 4]
        )

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
