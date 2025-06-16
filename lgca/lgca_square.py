# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""
Classes for two-dimensional LGCA on a square lattice. They specify
geometry-dependent LGCA behavior and inherit properties and structure from the
respective abstract base classes.
Objects of these classes can be used to simulate.

Supported LGCA types:

- classical LGCA (:py:class:`LGCA_Square`)
- identity-based LGCA (:py:class:`IBLGCA_Square`)
- classical LGCA without volume exclusion (:py:class:`NoVE_LGCA_Square`)
- identity-based LGCA without volume exclusion (:py:class:`NoVE_IBLGCA_Square`)
"""

import numpy as np
try:  # optional plotting dependencies
    import matplotlib.animation as animation
    import matplotlib.colors as colors
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:  # pragma: no cover - handled at runtime
    from lgca.base import _MissingPlotLib  # reuse stub

    animation = colors = mticker = FuncFormatter = PatchCollection = Normalize = (
        RegularPolygon
    ) = Circle = FancyArrowPatch = cm = make_axes_locatable = _MissingPlotLib(
        "matplotlib"
    )
import warnings
from copy import copy

from lgca.base import *

from .square_plotting import SquarePlotMixin
class LGCA_Square(SquarePlotMixin, LGCA_base):
    """
    Classical LGCA with volume exclusion on a 2D square lattice.

    It holds all methods and attributes that are specific for a square geometry.

    Attributes
    ----------
    cix, ciy : np.ndarray
        Elements of :py:attr:`lgca.c`.
    coord_pairs : list of tuple
        Indices of non-border nodes in the :py:attr:`lgca.nodes` array, linearized, each tuple is (x-index, y-index).
    dy : float
        Scaling factor for the y axis. 1 for square geometry.
    lx, ly : int
        Lattice dimensions in x and y direction.
    orientation : float
        Attribute for drawing polygons that represent the nodes. Orientation of the polygon in rad.
        This is passed to :py:func:`matplotlib.patches.RegularPolygon()`. Pi/4 for square geometry.
    r_poly : float
        Attribute for drawing polygons that represent the nodes. Distance between polygon center and vertices.
    xcoords, ycoords : np.ndarray
        Logical coordinates of non-border nodes starting with 0. Dimensions: ``(lgca.lx, lgca.ly)``.

    See Also
    --------
    lgca.base.LGCA_base : Base class with geometry-independent methods and attributes.

    """
    # set class attributes
    geometry = 'square'
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation',
                    'random_walk', 'excitable_medium', 'nematic', 'persistent_motion', 'chemotaxis', 'contact_guidance',
                    'only_propagation']
    velocitychannels = 4
    # build velocity channel vectors
    cix = np.array([1, 0, -1, 0], dtype=float)
    ciy = np.array([0, 1, 0, -1], dtype=float)
    c = np.array([cix, ciy])
    dy = np.sin(2 * np.pi / velocitychannels)
    # attributes to draw polygons representing the nodes
    r_poly = 0.5 / np.cos(np.pi / velocitychannels)
    orientation = np.pi / velocitychannels

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        """
        Set LGCA dimensions.

        Initializes :py:attr:`self.K`, :py:attr:`self.restchannels`, :py:attr:`self.dims`,
        :py:attr:`self.lx` and :py:attr:`self.ly`.

        Parameters
        ----------
        dims : int or tuple, default=(50,50)
            Lattice dimensions. Must match with specified geometry, an integer is interpreted as
            ``(dims, dims)``.
        nodes : np.ndarray
            Custom initial lattice configuration.
        restchannels : int, default=0
            Number of resting channels.

        """
        # set dimensions according to provided initial condition
        if nodes is not None:
            self.lx, self.ly, self.K = nodes.shape
            if self.K < self.velocitychannels:
                raise RuntimeError(
                    'Not enough channels specified for the chosen geometry! '
                    f'Required: {self.velocitychannels}, provided: {self.K}'
                )
            self.restchannels = self.K - self.velocitychannels
            self.dims = self.lx, self.ly
            return

        # default
        elif dims is None:
            dims = (50, 50)

        # set dimensions to keyword value
        if isinstance(dims, tuple):
            try:
                self.lx, self.ly = dims
            except ValueError:
                self.lx, self.ly = dims[0], dims[1]
        elif isinstance(dims, int):
            self.lx, self.ly = dims, dims
        else:
            raise TypeError("Keyword 'dims' must be int or tuple!")
        self.dims = self.lx, self.ly
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize LGCA lattice configuration. Create the lattice and then assign particles to
        channels in the nodes.

        Initializes :py:attr:`self.nodes`. If `nodes` is not provided, the lattice is initialized randomly so that
        each node contains on average ``density`` particles. For the random initialization there is a choice between
        a fixed or random number of particles per node.

        Parameters
        ----------
        density : float, default=0.1
            If `nodes` is None, initialize lattice randomly with this average number of particles per node.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration. Dimensions: ``(self.dims[0], self.dims[1], self.K)``.

        See Also
        --------
        base.LGCA_base.random_reset : Initialize lattice nodes with average density `density`.
        set_dims : Set LGCA dimensions.
        init_coords : Initialize LGCA coordinates.

        """
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=bool)
        # random initialization
        if nodes is None:
            self.random_reset(density)
        # initialization with provided initial condition
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = self._ensure_bool_nodes(nodes)
            self.apply_boundaries()

    def init_coords(self):
        """
        Initialize LGCA coordinates.

        These are used to index the lattice nodes logically and programmatically (see below).
        Initializes :py:attr:`self.nonborder`, :py:attr:`self.xcoords`, :py:attr:`self.ycoords` and
        :py:attr:`self.coord_pairs`.

        See Also
        --------
        set_dims : Set LGCA dimensions.
        init_nodes : Initialize LGCA lattice configuration.
        set_r_int : Change the interaction radius.

        Notes
        -----
        :py:attr:`self.xcoords` and :py:attr:`self.ycoords` hold the logical coordinates of non-border nodes in x- and
        y-direction starting with 0. Non-border nodes belong to the lattice in the mathematical definition of the LGCA,
        while border nodes (=shadow nodes) are only included in order to implement boundary conditions. Note that since
        the lattice is two-dimensional, so are the coordinates.

        >>> lgca = get_lgca(geometry='square', dims=3)
        >>> lgca.xcoords
        array([[0., 0., 0.],
               [1., 1., 1.],
               [2., 2., 2.]])
        >>> lgca.ycoords
        array([[0., 1., 2.],
               [0., 1., 2.],
               [0., 1., 2.]])

        A column in the printout is a row in the LGCA lattice.
        :py:attr:`self.nonborder` holds the programmatical coordinates of non-border nodes, i.e. the indices of the
        :py:attr:`self.nodes` array where non-border nodes are stored. This is why it is a tuple: Because it
        is used to index a numpy array. All non-border lattice nodes can be called as ``self.nodes[self.nonborder]``.

        >>> lgca = get_lgca(geometry='square', dims=2)  # default: periodic boundary conditions
        >>> lgca.r_int
        1
        >>> lgca.nodes.sum(-1)  # show contents of the lattice
        array([[1, 0, 1, 0],
               [0, 0, 0, 0],
               [1, 0, 1, 0],
               [0, 0, 0, 0]])
        >>> lgca.nodes[lgca.nonborder].sum(-1)
        array([[0, 0],
               [0, 1]])

        Summing along the last axis means summing over all channels of a node since we are interested in the geometry.
        The first and the last row and column in the output of ``lgca.nodes.sum(-1)`` are the contents of the border
        (=shadow) nodes, which reflects the interaction radius of 1. The innermost four elements are the contents of
        the non-border nodes. Accordingly we find their indices to be:

        >>> lgca.nonborder
        (array([[1, 1],
                [2, 2]]),
         array([[1, 2],
                [1, 2]]))

        The first element of the tuple is the index in x-direction, the second element the index in y-direction.
        Changing the interaction radius updates the shape of :py:attr:`self.nodes` by including more border (=shadow)
        nodes. This also changes the coordinates. With an interaction radius of 2, there is 2 border nodes on each side
        enveloping the non-border nodes whose contents remain the same. Therefore the first non-border node has the
        index 2 in each direction.

        >>> lgca.set_r_int(2)  # change the interaction radius
        >>> lgca.r_int
        2
        >>> lgca.nodes.sum(-1)  # show contents of the lattice
        array([[0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1]])
        >>> lgca.nonborder
        (array([[2, 2],
                [3, 3]]),
         array([[2, 3],
                [2, 3]]))

        :py:attr:`self.coord_pairs` is a list of programmatical (x,y) coordinate tuples for iterating through nodes one
        by one.

        >>> lgca.set_r_int(1)
        >>> lgca.coord_pairs
        [(1, 1), (1, 2), (2, 1), (2, 2)]

        """
        # dimension of logical coordinates in x and y, shifted by interaction radius to yield programmatical coordinates
        x = np.arange(self.lx) + self.r_int
        y = np.arange(self.ly) + self.r_int
        # create coordinate matrices from coordinate vectors
        xx, yy = np.meshgrid(x, y, indexing='ij')
        # create tuple of coordinate meshes
        self.nonborder = (xx, yy)

        # create iterable list of coordinate pair tuples
        self.coord_pairs = list(zip(xx.flat, yy.flat))
        # create coordinate matrices for logical coordinates including interaction radius
        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords[self.nonborder].astype(float)
        self.ycoords = self.ycoords[self.nonborder].astype(float)

    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.

        Updates :py:attr:`self.nodes` such that resting particles (the contents of ``self.nodes[:, 4:]``) stay in their
        position and particles in velocity channels (the contents of ``self.nodes[:, :4]``) are relocated according to
        the direction of the channel they reside in. Boundary conditions are enforced later by
        :py:meth:`apply_boundaries`.

        See Also
        --------
        base.LGCA_base.nodes : State of the lattice showing the structure of the ``lgca.nodes`` array.

        Notes
        -----
        >>> # set up the node configuration
        >>> nodes = np.zeros((3,3,5)).astype(bool)
        >>> nodes[1,1,:] = True
        >>> lgca = get_lgca(geometry='square', nodes=nodes)
        >>> lgca.cell_density[lgca.nonborder]
        array([[0, 0, 0],
               [0, 5, 0],
               [0, 0, 0]])
        >>> lgca.nodes[lgca.nonborder]
               # left column of the lattice
        array([[[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
               # central column
               [[False, False, False, False, False],
                [ True,  True,  True,  True,  True], # node (1,1): all channels are filled
                [False, False, False, False, False]],
               # right column
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]]])

        Before propagation, five particles occupy the central node. One resides in the velocity channel to the right,
        one in the velocity channel upwards, one in the velocity channel to the left, one in the velocity channel
        downwards and one in the resting channel.

        >>> lgca.propagation()
        >>> lgca.update_dynamic_fields()  # to update lgca.cell_density
        >>> lgca.cell_density[lgca.nonborder]
        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])
        >>> lgca.nodes[lgca.nonborder]
               # left column of the lattice
        array([[[False, False, False, False, False],
                [False, False,  True, False, False], # node (0,1): particle moving to the left
                [False, False, False, False, False]],
               # central column
               [[False, False, False,  True, False], # node (1,0): particle moving downwards
                [False, False, False, False,  True], # node (1,1): resting particle
                [False,  True, False, False, False]], # node (1,2): particle moving upwards
               # right column
               [[False, False, False, False, False],
                [ True, False, False, False, False], # node (2,1): particle moving to the right
                [False, False, False, False, False]]])

        The particle with velocity to the right has moved to the right velocity channel in the central node of the
        right side of the lattice (second to last line of the output). The particles in the other velocity channels
        have also moved according to their direction (see output annotation). The resting particle stayed
        in its channel in the very center.

        """
        newnodes = np.zeros(self.nodes.shape, dtype=self.nodes.dtype)
        # resting particles stay
        newnodes[..., 4:] = self.nodes[..., 4:]

        # prop. to the right
        newnodes[1:, :, 0] = self.nodes[:-1, :, 0]

        # prop. to the left
        newnodes[:-1, :, 2] = self.nodes[1:, :, 2]

        # prop. upwards
        newnodes[:, 1:, 1] = self.nodes[:, :-1, 1]

        # prop. downwards
        newnodes[:, :-1, 3] = self.nodes[:, 1:, 3]

        self.nodes = newnodes

    def _apply_pbcx(self):
        """
        Apply periodic boundary conditions in x-direction.

        Written for :py:meth:`self.apply_pbc`.
        """
        self.nodes[:self.r_int, ...] = self.nodes[-2 * self.r_int:-self.r_int, ...]  # left boundary
        self.nodes[-self.r_int:, ...] = self.nodes[self.r_int:2 * self.r_int, ...]  # right boundary

    def _apply_pbcy(self):
        """
        Apply periodic boundary conditions in y-direction.

        Written for :py:meth:`self.apply_pbc` and :py:meth:`self.apply_inflowbc`.
        """
        self.nodes[:, :self.r_int, :] = self.nodes[:, -2 * self.r_int:-self.r_int, :]  # upper boundary
        self.nodes[:, -self.r_int:, :] = self.nodes[:, self.r_int:2 * self.r_int, :]  # lower boundary

    def apply_pbc(self):
        # documented in parent class
        self._apply_pbcx()
        self._apply_pbcy()

    def _apply_rbcx(self):
        """
        Apply reflecting boundary conditions in x-direction.

        Written for :py:meth:`self.apply_rbc` and :py:meth:`self.apply_inflowbc`.
        """
        self.nodes[self.r_int, :, 0] += self.nodes[self.r_int - 1, :, 2]
        self.nodes[-self.r_int - 1, :, 2] += self.nodes[-self.r_int, :, 0]
        self._apply_abcx()

    def _apply_rbcy(self):
        """
        Apply reflecting boundary conditions in y-direction.

        Written for :py:meth:`self.apply_rbc`.
        """
        self.nodes[:, self.r_int, 1] += self.nodes[:, self.r_int - 1, 3]
        self.nodes[:, -self.r_int - 1, 3] += self.nodes[:, -self.r_int, 1]
        self._apply_abcy()

    def apply_rbc(self):
        # documented in parent class
        self._apply_rbcx()
        self._apply_rbcy()

    def _apply_abcx(self):
        """
        Apply absorbing boundary conditions in x-direction.

        Written for :py:meth:`self.apply_abc` and :py:meth:`self._apply_rbcx`.
        """
        self.nodes[:self.r_int, ...] = 0
        self.nodes[-self.r_int:, ...] = 0

    def _apply_abcy(self):
        """
        Apply absorbing boundary conditions in y-direction.

        Written for :py:meth:`self.apply_abc` and :py:meth:`self._apply_rbcy`.
        """
        self.nodes[:, :self.r_int, :] = 0
        self.nodes[:, -self.r_int:, :] = 0

    def apply_abc(self):
        # documented in parent class
        self._apply_abcx()
        self._apply_abcy()

    def apply_inflowbc(self):
        """
        Apply inflow boundary conditions.

        Update :py:attr:`self.nodes`, using the shadow border nodes and respecting the geometry.

        Boundary condition for an inflow from x=0, y=:, with reflecting boundary conditions along the y axis and
        periodic boundaries along the x axis. Nodes at (x=0, y) are set to a homogeneous state with a constant average
        density given by the attribute ``0 <= self.inflow <= 1``.

        If there is no such attribute, the nodes are filled with the maximum density.

        """
        self._apply_rbcx()

        if hasattr(self, 'inflow'):
            self.nodes[self.r_int, ...] = npr.random(self.nodes[0].shape) < self.inflow
        else:
            self.nodes[self.r_int, ...] = 1

        self._apply_pbcy()

    def nb_sum(self, qty):
        """
        For each node, sum up the contents of `qty` for the 4 nodes in the von Neumann neughborhood, excluding the
        center.

        `qty` is assumed to contain the value of a calculated quantity for each node in the lattice. `nb_sum` calculates
        the "neighborhood sum" of this quantity for each node, excluding the value for the node's own position.

        Parameters
        ----------
        qty : :py:class:`numpy.ndarray`
            Array holding some quantity of the LGCA, e.g. a flux. Of shape ``self.dims + x``, where ``x`` is the shape
            of the quantity for one node, e.g. ``(2,)`` if it is a vector with 2 elements. ``self.dims`` ensures that
            lattice positions can be indexed the same way as in ``self.nodes``.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Sum of the content of `qty` in each node's neighborhood, shape: ``qty.shape``. Lattice positions can be
            indexed the same way as in ``self.nodes``.

        Examples
        --------
        >>> lgca = get_lgca(geometry='square', dims=3) # periodic boundary conditions
        >>> lgca.cell_density[lgca.nonborder]
        array([[0, 1, 0],
               [0, 0, 0],
               [0, 1, 2]])
        >>> lgca.nb_sum(lgca.cell_density).astype(int)[lgca.nonborder]
        array([[1, 1, 3],
               [0, 2, 2],
               [3, 3, 1]])

        ``lgca.cell_density`` is used as the argument `qty`. The value at each position in the resulting array is the
        sum of the values at the neighboring positions in the source array. Note that the reduction to the non-border
        nodes can only be done after the sum calculation in order to preserve boundary conditions.

        """
        sum = np.zeros_like(qty)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        sum[:, :-1, ...] += qty[:, 1:, ...]
        sum[:, 1:, ...] += qty[:, :-1, ...]
        return sum

    def gradient(self, qty):
        # documented in parent class
        return np.moveaxis(np.asarray(np.gradient(qty, 0.5)), 0, -1)

    def channel_weight(self, qty):
        """
        Calculate weights for the velocity channels in interactions depending on a field `qty`.

        The weight for the right/upwards/left/downwards velocity channel is given by the value of `qty` of the
        right/upwards/left/downwards neighboring node.

        Parameters
        ----------
        qty : :py:class:`numpy.ndarray`
            Scalar field with the same shape as ``self.cell_density``.

        Returns
        -------
        :py:class:`numpy.ndarray` of `float`
            Weights for the velocity channels of shape ``self.dims + (self.velocitychannels,)``.

        """
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, :, 0] = qty[1:, ...]
        weights[1:, :, 2] = qty[:-1, ...]
        weights[:, :-1, 1] = qty[:, 1:, ...]
        weights[:, 1:, 3] = qty[:, :-1, ...]

        return weights

    def calc_vorticity(self, nodes=None):
        """
        Calculate the vorticity of the flow field corresponding to the lgca state 'nodes'. The vorticity is used to
        characterize rotations in a flow field. For more, see https://en.wikipedia.org/wiki/Vorticity
        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`

        Returns
        -------
        :py:class:`numpy.ndarray` of `float`
            Scalar field with the same shape as ``self.cell_density``.

        """
        if nodes is None:
            nodes = self.nodes
        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

        flux = self.calc_flux(nodes)
        # dens = nodes.sum(-1)
        # flux = np.divide(flux, dens[..., None], where=dens[..., None] > 0, out=np.zeros_like(flux))
        fx, fy = flux[..., 0], flux[..., 1]
        dfx = self.gradient(fx)
        dfy = self.gradient(fy)
        dfxdy = dfx[..., 1]
        dfydx = dfy[..., 0]
        vorticity = dfydx - dfxdy
        return vorticity

    def calc_velocity_correlation(self, nodes=None):
        """
        Calculate the correlation between the node fluxes and the mean node flux in the neighborhood. Used to quantify
        correlated movement.
        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`

        Returns
        -------
        :py:class:`numpy.ndarray` of `float`
            Scalar field with the same shape as ``self.cell_density``.

        """
        if nodes is None:
            nodes = self.nodes
        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

        flux = self.calc_flux(nodes)
        flux_norm = np.linalg.norm(flux, axis=-1)
        nb_flux = self.nb_sum(flux)
        nb_flux_norm = np.linalg.norm(nb_flux, axis=-1)
        corr = np.einsum('...i, ...i', flux, nb_flux)
        corr = np.divide(corr, flux_norm, where=flux_norm > 1e-6, out=np.zeros_like(corr))
        corr = np.divide(corr, nb_flux_norm, where=nb_flux_norm > 1e-6, out=np.zeros_like(corr))
        return corr

from .square_ext import IBLGCA_Square, NoVE_LGCA_Square, NoVE_IBLGCA_Square

