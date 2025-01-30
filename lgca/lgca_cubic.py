import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lgca.base import *



class LGCA_Cubic(LGCA_base):
    """
    Classical LGCA with volume exclusion on a 3D cubic lattice.

    It holds all methods and attributes that are specific for a cubic geometry. See :py:class:`lgca.base.LGCA_base` for
    the documentation of inherited attributes.

    Attributes
    ----------
    lx, ly, lz : int
        Lattice dimensions.
    xcoords, ycoords, zcoords : :py:class:`numpy.ndarray`
        Logical coordinates of non-border nodes starting with 0. Dimensions: ``(lgca.lx, lgca.ly, lgca.lz)``.

    See Also
    --------
    lgca.base.LGCA_base : Base class with geometry-independent methods and attributes.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk', 'persistent_motion', 'birthdeath', 'only_propagation']
    velocitychannels = 6
    c = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])  # directions of velocity channels

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        """
        Set LGCA dimensions.

        Initializes :py:attr:`self.K`, :py:attr:`self.restchannels`, :py:attr:`self.dims`, :py:attr:`self.lx`, :py:attr:`self.ly`, and :py:attr:`self.lz`.

        Parameters
        ----------
        dims : tuple of int, default=(10, 10, 10)
            Lattice dimensions. Must match with specified geometry.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration.
        restchannels : int, default=0
            Number of resting channels.

        See Also
        --------
        init_nodes : Initialize LGCA lattice configuration.
        init_coords : Initialize LGCA coordinates.
        """
        if nodes is not None:
            self.lx, self.ly, self.lz, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            self.dims = (self.lx, self.ly, self.lz)
            return

        if dims is None:
            dims = (10, 10, 10)

        if isinstance(dims, tuple) and len(dims) == 3:
            self.lx, self.ly, self.lz = dims

        elif isinstance(dims, int):
            self.lx = self.ly = self.lz = dims
        else:
            raise TypeError("Keyword 'dims' must be a tuple of three integers or int!")

        self.dims = (self.lx, self.ly, self.lz)
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize LGCA lattice configuration. Create the lattice and then assign particles to channels in the nodes.

        Initializes :py:attr:`self.nodes`. If `nodes` is not provided, the lattice is initialized with particles
        randomly so that the average lattice density is `density`.

        Parameters
        ----------
        density : float, default=0.1
            If `nodes` is None, initialize lattice randomly with this particle density.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration. Dimensions: ``(self.lx, self.ly, self.lz, self.K)``.

        See Also
        --------
        base.LGCA_base.random_reset : Initialize lattice nodes with average density `density`.
        set_dims : Set LGCA dimensions.
        init_coords : Initialize LGCA coordinates.
        """
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.lz + 2 * self.r_int, self.K),
                              dtype=bool)

        if nodes is None:
            self.random_reset(density)
        else:
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(bool)
            self.apply_boundaries()

    def init_coords(self):
        """
        Initialize LGCA coordinates.

        These are used to index the lattice nodes logically and programmatically.

        Initializes :py:attr:`self.nonborder`, :py:attr:`self.xcoords`, :py:attr:`self.ycoords`, and :py:attr:`self.zcoords`.

        See Also
        --------
        set_dims : Set LGCA dimensions.
        init_nodes : Initialize LGCA lattice configuration.
        set_r_int : Change the interaction radius.
        """
        self.nonborder = (
        np.arange(self.lx) + self.r_int, np.arange(self.ly) + self.r_int, np.arange(self.lz) + self.r_int)
        self.xcoords, self.ycoords, self.zcoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                               np.arange(self.ly + 2 * self.r_int) - self.r_int,
                                                               np.arange(self.lz + 2 * self.r_int) - self.r_int,
                                                               indexing='ij')
        self.xcoords = self.xcoords[self.nonborder].astype(float)
        self.ycoords = self.ycoords[self.nonborder].astype(float)
        self.zcoords = self.zcoords[self.nonborder].astype(float)

    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.

        Updates :py:attr:`self.nodes` such that resting particles stay in their position and particles in velocity channels
        are relocated according to the direction of the channel they reside in. Boundary conditions are enforced later by
        :py:meth:`apply_boundaries`.

        See Also
        --------
        base.LGCA_base.nodes : State of the lattice showing the structure of the ``lgca.nodes`` array.
        """
        newnodes = np.zeros_like(self.nodes)
        newnodes[..., self.velocitychannels:] = self.nodes[..., self.velocitychannels:]

        # propagation in each direction
        newnodes[1:, :, :, 0] = self.nodes[:-1, :, :, 0]
        newnodes[:-1, :, :, 1] = self.nodes[1:, :, :, 1]
        newnodes[:, 1:, :, 2] = self.nodes[:, :-1, :, 2]
        newnodes[:, :-1, :, 3] = self.nodes[:, 1:, :, 3]
        newnodes[:, :, 1:, 4] = self.nodes[:, :, :-1, 4]
        newnodes[:, :, :-1, 5] = self.nodes[:, :, 1:, 5]

        self.nodes = newnodes

    def apply_pbc(self):
        # Apply periodic boundary conditions
        self.nodes[:self.r_int, :, :, :] = self.nodes[-2 * self.r_int:-self.r_int, :, :, :]
        self.nodes[-self.r_int:, :, :, :] = self.nodes[self.r_int:2 * self.r_int, :, :, :]
        self.nodes[:, :self.r_int, :, :] = self.nodes[:, -2 * self.r_int:-self.r_int, :, :]
        self.nodes[:, -self.r_int:, :, :] = self.nodes[:, self.r_int:2 * self.r_int, :, :]
        self.nodes[:, :, :self.r_int, :] = self.nodes[:, :, -2 * self.r_int:-self.r_int, :]
        self.nodes[:, :, -self.r_int:, :] = self.nodes[:, :, self.r_int:2 * self.r_int, :]

    def apply_rbc(self):
        # Apply reflective boundary conditions
        self.nodes[self.r_int, :, :, 0] += self.nodes[self.r_int - 1, :, :, 1]
        self.nodes[-self.r_int - 1, :, :, 1] += self.nodes[-self.r_int, :, :, 0]
        self.nodes[:, self.r_int, :, 2] += self.nodes[:, self.r_int - 1, :, 3]
        self.nodes[:, -self.r_int - 1, :, 3] += self.nodes[:, -self.r_int, :, 2]
        self.nodes[:, :, self.r_int, 4] += self.nodes[:, :, self.r_int - 1, 5]
        self.nodes[:, :, -self.r_int - 1, 5] += self.nodes[:, :, -self.r_int, 4]
        self.apply_abc()

    def apply_abc(self):
        # Apply absorbing boundary conditions
        self.nodes[:self.r_int, :, :, :] = 0
        self.nodes[-self.r_int:, :, :, :] = 0
        self.nodes[:, :self.r_int, :, :] = 0
        self.nodes[:, -self.r_int:, :, :] = 0
        self.nodes[:, :, :self.r_int, :] = 0
        self.nodes[:, :, -self.r_int:, :] = 0

    def nb_sum(self, qty):
        """
        For each node, sum up the contents of `qty` for the 6 neighboring nodes, excluding the center.

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
        """
        sum = np.zeros(qty.shape)
        sum[:-1, :, :, ...] += qty[1:, :, :, ...]
        sum[1:, :, :, ...] += qty[:-1, :, :, ...]
        sum[:, :-1, :, ...] += qty[:, 1:, :, ...]
        sum[:, 1:, :, ...] += qty[:, :-1, :, ...]
        sum[:, :, :-1, ...] += qty[:, :, 1:, ...]
        sum[:, :, 1:, ...] += qty[:, :, :-1, ...]
        return sum

    def gradient(self, qty):
        # Calculate the gradient of a quantity in the lattice
        gx, gy, gz = np.gradient(qty, 0.5)
        return np.stack((gx, gy, gz), axis=-1)

    def channel_weight(self, qty):
        """
        Calculate weights for the velocity channels in interactions depending on a field `qty`.

        The weight for each velocity channel is given by the value of `qty` of the respective neighboring node.

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
        weights[:-1, :, :, 0] = qty[1:, :, :, ...]
        weights[1:, :, :, 1] = qty[:-1, :, :, ...]
        weights[:, :-1, :, 2] = qty[:, 1:, :, ...]
        weights[:, 1:, :, 3] = qty[:, :-1, :, ...]
        weights[:, :, :-1, 4] = qty[:, :, 1:, ...]
        weights[:, :, 1:, 5] = qty[:, :, :-1, ...]
        return weights

    def setup_figure(self, figindex=None, figsize=(8, 8), tight_layout=True):
        # Create a 3D figure for plotting
        fig = plt.figure(figindex, figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        if tight_layout:
            plt.tight_layout()
        return fig, ax

    def plot_density(self, density_t=None, cmap='hot_r', vmax='auto', colorbarwidth=0.03, cbar=True, **kwargs):
        """
        Plot particle density over time in 3D. X, Y, Z axes: 3D lattice.

        Parameters
        ----------
        density_t : :py:class:`numpy.ndarray`, optional
            Particle density values for a lattice over time to plot. If set to None and a simulation has been performed
            before, the result of the simulation is plotted. Dimensions: ``(timesteps + 1,) + self.dims``.
        cmap : str or :py:class:`matplotlib.colors.Colormap`
            Color map for the density values. Passed on to :py:func:`lgca.base.cmap_discretize`.
        colorbarwidth : float
            Width of the additional axis for the color bar, passed to
            :py:meth:`mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        vmax : int or 'auto', default='auto'
            Maximum density value for the color scaling. The minimum value is zero. All density values higher than
            `vmax` are drawn in the color at the end of the color bar. If None, `vmax` is set to the number of channels
            ``self.K``. 'auto' sets it to the maximum value found in `density_t`.
        cbar : bool, default=True
            If True, a color bar is added to the plot.
        **kwargs
            Arguments to be passed on to :py:meth:`setup_figure`.

        Returns
        -------
        :py:class:`matplotlib.image.AxesImage`
            Density plot over time.
        """
        if density_t is None:
            if hasattr(self, 'dens_t'):
                density_t = self.dens_t
            else:
                raise RuntimeError("Node-wise state of the lattice required for density plotting but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword recorddens=True")

        fig, ax = self.setup_figure(**kwargs)

        if vmax is None:
            vmax = self.K
        elif vmax == 'auto':
            vmax = int(density_t.max())

        cmap = cmap_discretize(cmap, 1 + vmax)

        # Create a scatter plot for 3D density
        x, y, z = self.xcoords.flat, self.ycoords.flat, self.zcoords.flat
        c = density_t.flat
        sc = ax.scatter(x, y, z, c=c, cmap=cmap, vmin=0, vmax=vmax)

        if cbar:
            fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

        plt.show()
        return sc