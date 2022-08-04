# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
Classes for one-dimensional LGCA. They specify geometry-dependent LGCA behavior
and inherit properties and structure from the respective abstract base classes.
Objects of these classes can be used to simulate.

Supported LGCA types:

- classical LGCA (:py:class:`LGCA_1D`)
- identity-based LGCA (:py:class:`IBLGCA_1D`)
- LGCA without volume exclusion (:py:class:`NoVE_LGCA_1D`)
"""

import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lgca.base import *


class LGCA_1D(LGCA_base):
    """
    Classical LGCA with volume exclusion on a 1D lattice.

    It holds all methods and attributes that are specific for a linear geometry. See :py:class:`lgca.base.LGCA_base` for
    the documentation of inherited attributes.

    Attributes
    ----------
    l : int
        Lattice dimension.
    xcoords : :py:class:`numpy.ndarray`
        Logical coordinates of non-border nodes starting with 0. Dimensions: ``(lgca.l,)``.

    See Also
    --------
    lgca.base.LGCA_base : Base class with geometry-independent methods and attributes.

    """
    # set class attributes
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk', 'persistent_motion', 'birthdeath', 'only_propagation']
    velocitychannels = 2
    c = np.array([1., -1.])[None, ...]  # directions of velocity channels; shape: (1,2)

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        """
        Set LGCA dimensions.

        Initializes :py:attr:`self.K`, :py:attr:`self.restchannels`, :py:attr:`self.dims` and
        :py:attr:`self.l`.

        Parameters
        ----------
        dims : int or tuple, default=100
            Lattice dimensions. Must match with specified geometry, everything except the first tuple element
            is ignored.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration.
        restchannels : int, default=0
            Number of resting channels.

        See Also
        --------
        init_nodes : Initialize LGCA lattice configuration.
        init_coords : Initialize LGCA coordinates.

        """
        # set dimensions according to provided initial condition
        if nodes is not None:
            self.l, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            self.dims = self.l,
            return

        # default
        elif dims is None:
            dims = 100

        # set dimensions to keyword value
        if isinstance(dims, int):
            self.l = dims
        elif isinstance(dims, tuple):
            self.l = dims[0]
        else:
            raise TypeError("Keyword 'dims' must be int or tuple!")
        self.dims = self.l,
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize LGCA lattice configuration. Create the lattice and then assign particles to
        channels in the nodes.

        Initializes :py:attr:`self.nodes`. If `nodes` is not provided, the lattice is initialized with particles
        randomly so that the averge lattice density is `density`. For the random initialization there is a choice
        between a fixed or random number of particles per node.

        Parameters
        ----------
        density : float, default=0.1
            If `nodes` is None, initialize lattice randomly with this particle density.
        hom : float, default=False
            Fill channels randomly with particle density `density`, but with an equal number of particles for each node.
            Note that depending on :py:attr:`self.K` not all densities can be realized.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration. Dimensions: ``(self.dims[0], self.K)``.

        See Also
        --------
        base.LGCA_base.random_reset : Initialize lattice nodes with average density `density`.
        base.LGCA_base.homogeneous_random_reset : Initialize lattice nodes with average density `density` and a fixed number
            of particles per node.
        set_dims : Set LGCA dimensions.
        init_coords : Initialize LGCA coordinates.

        """
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=bool)

        # random initialization
        if 'hom' in kwargs:
            hom = kwargs['hom']
        else:
            hom = None
        if nodes is None and hom:
            self.homogeneous_random_reset(density)
        elif nodes is None:
            self.random_reset(density)
        # initialization with provided initial condition
        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(bool)
            self.apply_boundaries()

    def init_coords(self):
        """
        Initialize LGCA coordinates.

        These are used to index the lattice nodes logically and programmatically (see below).
        Initializes :py:attr:`self.nonborder` and :py:attr:`self.xcoords`.

        See Also
        --------
        set_dims : Set LGCA dimensions.
        init_nodes : Initialize LGCA lattice configuration.
        set_r_int : Change the interaction radius.

        Notes
        --------
        :py:attr:`self.xcoords` holds the logical coordinates of non-border nodes starting with 0. Non-border nodes
        belong to the lattice in the mathematical definition of the LGCA, while border nodes (=shadow nodes) are only
        included in order to implement boundary conditions.

        >>> lgca = get_lgca(geometry='lin', dims=3)
        >>> lgca.xcoords
        array([0., 1., 2.])

        :py:attr:`self.nonborder` holds the programmatical coordinates of non-border nodes, i.e. the indices of the
        :py:attr:`self.nodes` array where non-border nodes are stored. This is why it is a tuple: Because it
        is used to index a numpy array. All non-border lattice nodes can be called as ``self.nodes[self.nonborder]``.

        >>> lgca = get_lgca(geometry='lin', dims=3)  # default: periodic boundary conditions
        >>> lgca.r_int
        1
        >>> lgca.nodes.sum(-1)  # show contents of the lattice
        array([0, 0, 1, 0, 0])
        >>> lgca.nodes[lgca.nonborder].sum(-1)
        array([0, 1, 0])

        Summing along the last axis means summing over all channels of a node since we are interested in the geometry.
        The first and the last element in the output of ``lgca.nodes.sum(-1)`` are the contents of the border (=shadow)
        nodes, which reflects the interaction radius of 1. The innermost three elements are the contents of the
        non-border nodes. Accordingly we find their indices to be:

        >>> lgca.nonborder
        (array([1, 2, 3]),)

        In one dimension the y component of the tuple is empty.
        Changing the interaction radius updates the shape of :py:attr:`self.nodes` by including more border (=shadow)
        nodes. This also changes the coordinates. With an interaction radius of 3, there is 3 border nodes on each side
        enveloping the non-border nodes whose contents remain the same. Therefore the first non-border node has the
        index 3.

        >>> lgca.set_r_int(3)  # change the interaction radius
        >>> lgca.r_int
        3
        >>> lgca.nodes.sum(-1)  # show contents of the lattice
        array([0, 1, 0, 0, 1, 0, 0, 1, 0])
        >>> lgca.nonborder
        (array([3, 4, 5]),)

        """
        self.nonborder = (np.arange(self.l) + self.r_int,)
        self.xcoords = np.arange(self.l + 2 * self.r_int) - self.r_int
        self.xcoords = self.xcoords[self.nonborder].astype(float)

    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.

        Updates :py:attr:`self.nodes` such that resting particles (the contents of ``self.nodes[:, 2:]``) stay in their
        position and particles in velocity channels (the contents of ``self.nodes[:, :2]``) are relocated according to
        the direction of the channel they reside in. Boundary conditions are enforced later by
        :py:meth:`apply_boundaries`.

        See Also
        --------
        base.LGCA_base.nodes : State of the lattice showing the structure of the ``lgca.nodes`` array.

        Notes
        --------
        >>> lgca = get_lgca(geometry='lin', density=0.1, dims=5, restchannels=1)
        >>> lgca.cell_density[lgca.nonborder]
        array([0, 0, 0, 3, 0])
        >>> lgca.nodes[lgca.nonborder]
        array([[False, False, False],
               [False, False, False],
               [False, False, False],
               [ True,  True, True],
               [False, False, False]])

        Before propagation, three particles occupy the fourth node. One resides in the velocity channel to the right,
        one in the velocity channel to the left and one in the resting channel.

        >>> lgca.propagation()
        >>> lgca.update_dynamic_fields()  # to update lgca.cell_density
        >>> lgca.cell_density[lgca.nonborder]
        array([0, 0, 1, 1, 1])
        >>> lgca.nodes[lgca.nonborder]
        array([[False, False, False],
               [False, False, False],
               [False,  True, False],
               [False, False, True],
               [ True, False, False]])

        The particle with velocity 1 has moved to the right velocity channel in the fifth node. The particle in the
        velocity channel to the left has moved to the respective channel in the third node. The resting particle stayed
        in its channel in the fourth node.

        """
        newnodes = np.zeros_like(self.nodes)
        # resting particles stay
        newnodes[:, 2:] = self.nodes[:, 2:]

        # propagation to the right
        newnodes[1:, 0] = self.nodes[:-1, 0]

        # propagation to the left
        newnodes[:-1, 1] = self.nodes[1:, 1]

        self.nodes = newnodes

    def apply_pbc(self):
        # documented in parent class
        self.nodes[:self.r_int, :] = self.nodes[-2 * self.r_int:-self.r_int, :]
        self.nodes[-self.r_int:, :] = self.nodes[self.r_int:2 * self.r_int, :]

    def apply_rbc(self):
        # documented in parent class
        # left boundary cell inside domain: right channel gets added left channel from the left
        self.nodes[self.r_int, 0] += self.nodes[self.r_int - 1, 1]
        # right boundary cell inside domain: left channel gets added right channel from the right
        self.nodes[-self.r_int - 1, 1] += self.nodes[-self.r_int, 0]
        self.apply_abc()

    def apply_abc(self):
        # documented in parent class
        self.nodes[:self.r_int, :] = 0
        self.nodes[-self.r_int:, :] = 0

    def nb_sum(self, qty):
        """
        For each node, sum up the contents of `qty` for the left and right neighboring nodes, excluding the center.

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
        >>> lgca = get_lgca(geometry='lin', density=0.5, dims=5) # periodic boundary conditions
        >>> lgca.cell_density[lgca.nonborder]
        array([1, 0, 2, 2, 0])
        >>> lgca.nb_sum(lgca.cell_density).astype(int)[lgca.nonborder]
        array([0, 3, 2, 2, 3])

        ``lgca.cell_density`` is used as the argument `qty`. The value at each position in the resulting array is the
        sum of the values at the neighboring positions in the source array. Note that the reduction to the non-border
        nodes can only be done after the sum calculation in order to preserve boundary conditions.

        """
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        # shift to left without padding and add to shift to the right without padding
        # sums up configurations (in qty) of neighboring particles
        return sum

    def gradient(self, qty):
        # documented in parent class
        return np.gradient(qty, 0.5)[..., None]
        # None adds a new axis to the ndarray and keeps the remaining array unchanged

    def channel_weight(self, qty):
        """
        Calculate weights for the velocity channels in interactions depending on a field `qty`.

        The weight for the right rsp. left velocity channel is given by the value of `qty` of the right rsp. left
        neighboring node.

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
        weights[:-1, ..., 0] = qty[1:, ...]
        weights[1:, ..., 1] = qty[:-1, ...]
        return weights

    def setup_figure(self, tmax, figindex=None, figsize=(8, 8), tight_layout=True):
        """
        Create a :py:module:`matplotlib` figure and manage basic layout.

        Used by the class' plotting functions.

        Parameters
        ----------
        figindex : int or str, optional
            An identifier for the figure (passed to :py:function:`matplotlib.pyplot.figure`). If it is a string, the
            figure label and the window title is set to this value.
        figsize : tuple of int or tuple of float with 2 elements, default=(8,8)
            Desired figure size in inches ``(x, y)``.
        tight_layout : bool, default=True
            If :py:method:`matplotlib.figure.Figure.tight_layout` is called for padding between and around subplots.
        tmax : int or float
            Maximum simulation time to plot in order to scale the y axis.

        Returns
        -------
        fig : :py:class:`matplotlib.figure.Figure`
            New customized figure.
        ax : :py:class:`matplotlib.axes.Axes`
            Drawing axis associated with `fig`.

        See Also
        --------
        plot_density : Plot particle density over time.
        plot_flux : Plot flux over time.

        """
        # create or retrieve figure, set size and layout
        if figindex is None:
            fig = plt.gcf()
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        else:
            fig = plt.figure(num=figindex)
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        # retrieve drawing axis and scale
        ax = plt.gca()
        xmax = self.xcoords.max() + 0.5 * self.r_int
        xmin = self.xcoords.min() - 0.5 * self.r_int
        ymax = tmax - 0.5
        ymin = -0.5
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)
        ax.set_aspect('equal')

        # label axes, set tick positions and adjust their appearance
        plt.xlabel('Lattice node $r \\, (\\varepsilon)$')
        plt.ylabel('Time $k'
                   '\\, (\\tau)$')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_autoscale_on(False)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        return fig, ax

    def plot_density(self, density_t=None, cmap='hot_r', vmax='auto', colorbarwidth=0.03, **kwargs):
        """
        Plot particle density over time. X axis: 1D lattice, y axis: time. A color bar on the right side shows the
        color coding of density values. Empty nodes are white.

        Parameters
        ----------
        cmap : str or :py:class:`matplotlib.colors.Colormap`
            Color map for the density values. Passed on to :py:function:`lgca.base.cmap_discretize`.
        colorbarwidth : float
            Width of the additional axis for the color bar, passed to
            :py:method:`mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        density_t : :py:class:`numpy.ndarray`, optional
            Particle density values for a lattice over time to plot. If set to None and a simulation has been performed
            before, the result of the simulation is plotted. Dimensions: ``(timesteps + 1,) + self.dims``.
        vmax : int or 'auto', default='auto'
            Maximum density value for the color scaling. The minimum value is zero. All density values higher than
            `vmax` are drawn in the color at the end of the color bar. If None, `vmax` is set to the number of channels
            ``self.K``. 'auto' sets it to the maximum value found in `density_t`.
        **kwargs
            Arguments to be passed on to :py:method:`setup_figure`.

        Returns
        -------
        :py:class:`matplotlib.image.AxesImage`
            Density plot over time.

        See Also
        --------
        setup_figure : Manage basic layout.

        """

        # set image content
        if density_t is None:
            if hasattr(self, 'dens_t'):
                density_t = self.dens_t
            else:
                raise RuntimeError("Node-wise state of the lattice required for density plotting but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword recorddens=True")

        # prepare plot
        tmax = density_t.shape[0]
        fig, ax = self.setup_figure(tmax, **kwargs)

        # prepare axis for colour bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=colorbarwidth, pad=0.1)

        # colour scaling
        if vmax is None:
            vmax = self.K
        elif vmax == 'auto':
            vmax = int(density_t.max())
        cmap = cmap_discretize(cmap, 1 + vmax)

        # create plot
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=vmax, cmap=cmap)
        cbar = colorbar_index(ncolors=1 + vmax, cmap=cmap, use_gridspec=True, cax=cax)
        cbar.set_label('Particle number $n$')
        plt.sca(ax)
        return plot

    def plot_flux(self, nodes_t=None, **kwargs):
        """
        Plot flux in each node over time. X axis: 1D lattice, y axis: time.

        A flux vector to the left is indicated by a blue color of the node, a flux vector to the right by red. If the
        velocities of all particles cancel out, the node is colored in black. Empty nodes are white.

        Parameters
        ----------
        nodes_t : :py:class:`numpy.ndarray`, optional
            Node configurations for a lattice over time, used to calculate the flux and plot it. If set to None and a
            simulation has been performed before with ``record=True``, the result of the simulation is plotted.
            Dimensions: ``(timesteps + 1,) + self.dims + (self.K,)``.
        **kwargs
            Arguments to be passed on to :py:method:`setup_figure`.

        Returns
        -------
        :py:class:`matplotlib.image.AxesImage`
            Density plot over time.

        See Also
        --------
        setup_figure : Manage basic layout.

        """

        # set image content
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call lgca.timeevo() with keyword record=True")
        dens_t = nodes_t.sum(-1)
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)

        # color code flux
        rgba = np.zeros((tmax, l, 4)) #  4: RGBA A=alpha: transparency
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t > 0, 0] = 1.
        rgba[flux_t < 0, 2] = 1.
        rgba[flux_t == 0, :-1] = 0.  # unpopulated lattice sites are white

        # create plot
        fig, ax = self.setup_figure(tmax, **kwargs)
        plot = ax.imshow(rgba, interpolation='None', origin='upper')
        plt.xlabel(r'Lattice node $r \, (\varepsilon)$', )
        plt.ylabel(r'Time step $k \, (\tau)$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        # color bar option is missing here
        return plot


class IBLGCA_1D(IBLGCA_base, LGCA_1D):
    """
    Identity-based LGCA with volume exclusion on a 1D lattice.

    It holds all methods and attributes that are specific for a linear geometry. See :py:class:`lgca.base.LGCA_base`
    and :py:class:`lgca.base.IBLGCA_base` for the documentation of inherited attributes.

    Attributes
    ----------
    l : int
        Lattice dimension.
    xcoords : :py:class:`numpy.ndarray`
        Logical coordinates of non-border nodes starting with 0. Dimensions: ``(lgca.l,)``.

    See Also
    --------
    lgca.base.LGCA_base : Base class for LGCA with volume exclusion with geometry-independent methods and attributes.
    lgca.base.IBLGCA_base : Base class for IBLGCA with volume exclusion with geometry-independent methods and attributes.

    """
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath', 'birthdeath_discrete', 'only_propagation', 'go_and_grow_mutations']

    def init_nodes(self, density, nodes=None, **kwargs):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)

        else:
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.apply_boundaries()

    def plot_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t.astype('bool')
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call lgca.timeevo() with keyword record=True")

        if nodes_t.dtype != 'bool':
            nodes_t = nodes_t.astype('bool')
        LGCA_1D.plot_flux(self, nodes_t, **kwargs)

    def plot_prop_spatial(self, nodes_t=None, props=None, propname=None, cmap='cividis', **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(props))

        tmax, l, _ = nodes_t.shape
        fig, ax = self.setup_figure(tmax, **kwargs)
        mean_prop_t = self.calc_prop_mean(propname=propname, props=props, nodes=nodes_t)

        plot = plt.imshow(mean_prop_t, interpolation='none', aspect='equal', cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        cbar = fig.colorbar(plot, use_gridspec=True, cax=cax)
        cbar.set_label(r'Property ${}$'.format(propname))
        plt.sca(ax)
        return plot


class NoVE_LGCA_1D(LGCA_1D, NoVE_LGCA_base):
    """
    1D version of an LGCA without volume exclusion.
    """
    interactions = ['dd_alignment', 'di_alignment', 'go_or_grow', 'go_or_rest']

    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None):
        """
        Set the dimensions of the instance according to given values. Sets self.l, self.K, self.dims and self.restchannels
        :param dims: desired lattice size (int or array-like)
        :param nodes: existing lattice to use (ndarray)
        :param restchannels: desired number of resting channels, will be capped to 1 if >1 because of no volume exclusion
        """
        # set instance dimensions according to passed lattice
        if nodes is not None:
            try:
                self.l, self.K = nodes.shape
            except ValueError as e:
                raise ValueError("Node shape does not match the 1D geometry! Shape must be (x,channels)") from e
            # set number of rest channels to <= 1 because >1 cells are allowed per channel
            # for now, raise Exception if format of nodes does no fit
            # (To Do: just sum the cells in surplus rest channels in init_nodes and print a warning)
            if self.K - self.velocitychannels > 1:
                raise RuntimeError('Only one resting channel allowed, '+
                 'but {} resting channels specified!'.format(self.K - self.velocitychannels))
            elif self.K < self.velocitychannels:
                raise RuntimeError('Not enough channels specified for the chosen geometry! Required: {}, provided: {}'.format(
                    self.velocitychannels, self.K))
            else:
                self.restchannels = self.K - self.velocitychannels
        # set instance dimensions according to required dimensions
        elif dims is not None:
            if isinstance(dims, int):
                self.l = dims
            else:
                self.l = dims[0]
        # set default value for dimension
        else:
            self.l = 100
        self.dims = self.l,

        # set number of rest channels to <= 1 because >1 cells are allowed per channel
        if nodes is None and restchannels is not None:
            if restchannels > 1:
                self.restchannels = 1
            elif 0 <= restchannels <= 1:
                self.restchannels = restchannels
        elif nodes is None:
            self.restchannels = 0
        self.K = self.velocitychannels + self.restchannels

        # set capacity according to keyword or specified resting channels
        if capacity is not None:
            self.capacity = capacity
        elif restchannels is not None and restchannels > 1:
            self.capacity = self.velocitychannels + restchannels
        else:
            self.capacity = self.K

    def init_nodes(self, density, nodes=None, hom=None):
        """
        Initialize nodes for the instance.
        :param density: desired particle density in the lattice: number of particles/(dimensions*number of channels)
        :param nodes: existing lattice to use, optionally containing particles (ndarray)
        """
        # create lattice according to size specified earlier
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.uint)
        # if no lattice given, populate randomly
        if nodes is None:
            if hom:
                self.homogeneous_random_reset(density)
            else:
                self.random_reset(density)
        # if lattice given, populate lattice with given particles. Virtual lattice sites for boundary conditions not included
        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.uint)
            self.apply_boundaries()

    def plot_density(self, density_t=None, figindex=None, figsize=None, cmap='hot_r', relative_max=None,
                     absolute_max=None, offset_t=0, offset_x=0):
        """
        Create a plot showing the number of particles per lattice site.
        :param density_t: particle number per lattice site (ndarray of dimension (timesteps + 1,) + self.dims)
        :param figindex: number of the figure to create/activate
        :param figsize: desired figure size
        :param cmap: matplotlib color map for encoding the number of particles
        :return: plot as a matplotlib.image
        """
        # construct conditions
        x_has_offset = offset_x != 0 and isinstance(offset_x, int)
        t_has_offset = offset_t != 0 and isinstance(offset_t, int)
        # set values for unused arguments
        if density_t is None:
            if hasattr(self, 'dens_t'):
                density_t = self.dens_t
            else:
                raise RuntimeError("Node-wise state of the lattice required for density plotting but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword recorddens=True")
            if x_has_offset:
                density_t = density_t[:, offset_x:]
            if t_has_offset:
                density_t = density_t[offset_t:, :]
        if figsize is None:
            figsize = estimate_figsize(density_t.T, cbar=True)

        # set up figure
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        # set up color scaling
        if relative_max is not None:
            scale = relative_max
        else:
            scale = 1.0
        max_part_per_cell = int(scale * density_t.max())
        if absolute_max is not None:
            max_part_per_cell = int(absolute_max)
        cmap = cmap_discretize(cmap, max_part_per_cell + 1)
        # create plot with color bar, axis labels, title and layout
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=max_part_per_cell, cmap=cmap,
                            extent =[offset_x-0.5, density_t.shape[1] + offset_x-0.5, density_t.shape[0] + offset_t - 0.5, offset_t-0.5])
        loc = mticker.MaxNLocator(nbins='auto', steps=[1,2,5,10], integer=True)
        ax.xaxis.set_major_locator(loc)
        loc = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
        ax.yaxis.set_major_locator(loc)
        cbar = colorbar_index(ncolors=max_part_per_cell + 1, cmap=cmap, use_gridspec=True)
        cbar.set_label(r'Particle number $n$')
        plt.xlabel(r'Lattice node $r \, (\varepsilon)$', )
        plt.ylabel(r'Time step $k \, (\tau)$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return plot

    def nb_sum(self, qty, addCenter=False):
        """
        Calculate sum of values in neighboring lattice sites of each lattice site.
        :param qty: ndarray in which neighboring values have to be added
                  first dimension indexes lattice sites
        :param addCenter: toggle adding central value
        :return: sum as ndarray
        """
        sum = np.zeros(qty.shape)
        # shift to left padding 0 and add to shift to the right padding 0
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        # add central value
        if addCenter:
           sum += qty
        return sum


class NoVE_IBLGCA_1D(NoVE_IBLGCA_base, NoVE_LGCA_1D):
    def propagation(self):
        """
        :return:
        """
        newnodes = get_arr_of_empty_lists(self.nodes.shape)

        # prop. to the left
        newnodes[1:, 0] = self.nodes[:-1, 1]

        # prop. to the right
        newnodes[:-1, 1] = self.nodes[1:, 0]

        # resting cells stay
        newnodes[:, -1] = self.nodes[:, -1]

        self.nodes = newnodes

    def apply_rbc(self):
        self.nodes[self.r_int, 0] = self.nodes[self.r_int, 0] + self.nodes[self.r_int - 1, 1]
        self.nodes[-self.r_int - 1, 1] = self.nodes[-self.r_int - 1, 1] + self.nodes[-self.r_int, 0]
        self.apply_abc()

    def apply_abc(self):
        self.nodes[self.border] = get_arr_of_empty_lists(self.nodes[self.border].shape)
        # for channel in self.nodes[self.border].flat:
        #     channel.clear()

    def init_nodes(self, density, nodes=None):
        """
        initialize the nodes. there are three options:
        1) you provide only the argument "density", which should be a positive float that indicates the average number
        of cells in each channel
        2) you provide an array "nodes" with nodes.dtype == int,
            where each integer determines the number of cells in each channel
        3) you provide an array "nodes" with nodes.dtype == object, where each element is a list of unique cell labels
        """
        self.nodes = get_arr_of_empty_lists(((self.l + 2 * self.r_int, self.K)))
        if nodes is None:
            self.random_reset(density)

        elif nodes.dtype == object:
            self.nodes[self.nonborder] = nodes

        else:
            occ = nodes.astype(int)
            self.nodes[self.nonborder] = self.convert_int_to_ib(occ)

        self.calc_max_label()

    def plot_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.length_checker(self.nodes_t)
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call lgca.timeevo() with keyword record=True")

        if nodes_t.dtype != 'int':
            nodes_t = self.length_checker(self.nodes_t)
        LGCA_1D.plot_flux(self, nodes_t, **kwargs)

    def plot_prop_spatial(self, nodes_t=None, props=None, propname=None, cmap='cividis', cbarlabel=None, **kwargs):
        """
        Plot the spatial distribution of a cell property 'propname'. At each node, for each time step the mean value in
        the node is shown. Empty nodes are masked.
        :param nodes_t:
        :param props:
        :param propname:
        :param cmap:
        :param cbarlabel:
        :param kwargs:
        :return:
        """
        if nodes_t is None:
            nodes_t = self.nodes_t
        if props is None:
            props = self.props
        if propname is None:
            propname = next(iter(props))

        if self.mean_prop_t == {}:
            self.calc_prop_mean_spatiotemp()

        tmax, l, _ = nodes_t.shape
        fig, ax = self.setup_figure(tmax, **kwargs)
        mean_prop_t = self.mean_prop_t[propname]

        plot = plt.imshow(mean_prop_t, interpolation='none', cmap=cmap, aspect='equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        cbar = fig.colorbar(plot, use_gridspec=True, cax=cax)
        if cbarlabel is None:
            cbar.set_label(r'Property ${}$'.format(propname))
        else:
            cbar.set_label(cbarlabel)
        plt.sca(ax)
        return plot

