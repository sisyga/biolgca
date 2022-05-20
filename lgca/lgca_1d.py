# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lgca.base import *


class LGCA_1D(LGCA_base):
    """
    1D version of an LGCA.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk', 'persistent_motion', 'birthdeath', 'only_propagation']
    velocitychannels = 2
    c = np.array([1., -1.])[None, ...] #directions of velocity channels; shape: (1,2)

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        if nodes is not None:
            self.l, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            self.dims = self.l,
            return

        elif dims is None:
            dims = 100

        if isinstance(dims, int):
            self.l = dims
        else:
            self.l = dims[0]

        self.dims = self.l,
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density, nodes=None, **kwargs):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=bool)
        if 'hom' in kwargs:
            hom = kwargs['hom']
        else:
            hom = None
        if nodes is None and hom:
            self.homogeneous_random_reset(density)
        elif nodes is None:
            self.random_reset(density)
        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(bool)
            self.apply_boundaries()

    def init_coords(self):
        self.nonborder = (np.arange(self.l) + self.r_int,) # tuple s.t. lattice sites can be called as: nodes[nonborder]
        # self.nonborder is a tuple of indices along the lattice dimensions to correctly index xcoords
        self.xcoords = np.arange(self.l + 2 * self.r_int) - self.r_int # x-coordinates starting at -r_int to l+r_int

    def propagation(self):
        """
        :return:
        """
        newnodes = np.zeros_like(self.nodes)
        # resting particles stay
        newnodes[:, 2:] = self.nodes[:, 2:]

        # prop. to the right
        newnodes[1:, 0] = self.nodes[:-1, 0]

        # prop. to the left
        newnodes[:-1, 1] = self.nodes[1:, 1]

        self.nodes = newnodes

    def apply_pbc(self):
        self.nodes[:self.r_int, :] = self.nodes[-2 * self.r_int:-self.r_int, :]
        self.nodes[-self.r_int:, :] = self.nodes[self.r_int:2 * self.r_int, :]

    def apply_rbc(self):
        # left boundary cell inside domain: right channel gets added left channel from the left
        self.nodes[self.r_int, 0] += self.nodes[self.r_int - 1, 1]
        # right boundary cell inside domain: left channel gets added right channel from the right
        self.nodes[-self.r_int - 1, 1] += self.nodes[-self.r_int, 0]
        self.apply_abc()

    def apply_abc(self):
        self.nodes[:self.r_int, :] = 0
        self.nodes[-self.r_int:, :] = 0

    def nb_sum(self, qty):
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        # shift to left without padding and add to shift to the right without padding
        # sums up fluxes (in qty) of neighboring particles
        return sum

    def gradient(self, qty):
        return np.gradient(qty, 0.5)[..., None]
        # None adds a new axis to the ndarray and keeps the remaining array unchanged

    def channel_weight(self, qty):
        """
        Calculate weights for channels in interactions depending on a field qty
        :param qty: scalar field with the same shape as self.cell_density
        :return: weights, shaped like self.nodes, type float
        """
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, ..., 0] = qty[1:, ...]
        weights[1:, ..., 1] = qty[:-1, ...]
        return weights

    def setup_figure(self, tmax, figindex=None, figsize=(8, 8), tight_layout=True):
        if figindex is None:
            fig = plt.gcf()
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        else:
            fig = plt.figure(num=figindex)
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        ax = plt.gca()
        xmax = self.xcoords.max() - 0.5 * self.r_int
        xmin = self.xcoords.min() + 0.5 * self.r_int
        ymax = tmax - 0.5
        ymin = -0.5
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)
        ax.set_aspect('equal')

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
        if density_t is None:
            if hasattr(self, 'dens_t'):
                density_t = self.dens_t
            else:
                raise RuntimeError("Node-wise state of the lattice required for density plotting but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword recorddens=True")

        tmax = density_t.shape[0]
        fig, ax = self.setup_figure(tmax, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=colorbarwidth, pad=0.1)
        if vmax is None:
            vmax = self.K

        elif vmax == 'auto':
            vmax = int(density_t.max())

        cmap = cmap_discretize(cmap, 1 + vmax)
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=vmax, cmap=cmap)
        cbar = colorbar_index(ncolors=1 + vmax, cmap=cmap, use_gridspec=True, cax=cax)

        cbar.set_label('Particle number $n$')
        plt.sca(ax)
        return plot

    def plot_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call lgca.timeevo() with keyword record=True")

        dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)

        rgba = np.zeros((tmax, l, 4)) #4: RGBA A=alpha: transparency
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t > 0, 0] = 1.
        rgba[flux_t < 0, 2] = 1.
        rgba[flux_t == 0, :-1] = 0.  # unpopulated lattice sites are white
        fix, ax = self.setup_figure(tmax, **kwargs)
        plot = ax.imshow(rgba, interpolation='None', origin='upper')
        plt.xlabel(r'Lattice node $r \, (\varepsilon)$', )
        plt.ylabel(r'Time step $k \, (\tau)$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return plot


class IBLGCA_1D(IBLGCA_base, LGCA_1D):
    """
    1D version of an identity-based LGCA.
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
            nodes_t = nodes.astype('bool')
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

    def plot_density(self, density_t=None, figindex=None, figsize=None, cmap='viridis_r', relative_max=None, absolute_max=None, offset_t=0, offset_x=0):
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

    # def plot_flux(self, nodes_t=None, figindex=None, figsize=None):
    #     """
    #     Create a plot showing the main direction of particles per lattice site.
    #     :param nodes_t: particles per velocity channel at lattice site
    #                     (ndarray of dimension (timesteps + 1,) + self.dims + (self.K,))
    #     :param figindex: number of the figure to create/activate
    #     :param figsize: desired figure size
    #     :return: plot as a matplotlib.image
    #     """
    #     # set values for unused arguments
    #     if nodes_t is None:
    #         if hasattr(self, 'nodes_t'):
    #             nodes_t = self.nodes_t
    #         else:
    #             raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
    #                                "in past LGCA run, call lgca.timeevo() with keyword record=True")
    #     # calculate particle density, max time and dimension values, flux
    #     dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
    #     tmax, l = dens_t.shape
    #     flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)
    #     if figsize is None:
    #         figsize = estimate_figsize(dens_t.T)
    #
    #     # encode flux as RGBA values
    #     # 4: RGBA A=alpha: transparency
    #     rgba = np.zeros((tmax, l, 4))
    #     rgba[dens_t > 0, -1] = 1.
    #     rgba[flux_t > 0, 0] = 1.  # can I do this in ve, too?
    #     rgba[flux_t < 0, 2] = 1.
    #     rgba[flux_t == 0, :-1] = 0.  # unpopulated lattice sites are white
    #     # set up figure
    #     fig = plt.figure(num=figindex, figsize=figsize)
    #     ax = fig.add_subplot(111)
    #     # create plot with axis labels, title and layout
    #     plot = ax.imshow(rgba, interpolation='None', origin='upper')
    #     plt.xlabel(r'Lattice node $r \, [\varepsilon]$', )
    #     plt.ylabel(r'Time step $k \, [\tau]$')
    #     ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
    #     ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
    #     ax.xaxis.set_label_position('top')
    #     ax.xaxis.set_ticks_position('top')
    #     ax.xaxis.tick_top()
    #     plt.tight_layout()
    #     return plot

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
