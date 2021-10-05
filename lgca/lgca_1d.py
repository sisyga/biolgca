import matplotlib.ticker as mticker

try:
    from base import *
except ModuleNotFoundError:
    from .base import *


class LGCA_1D(LGCA_base):
    """
    1D version of an LGCA.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk', 'persistent_motion', 'birthdeath']
    velocitychannels = 2
    c = np.array([1., -1.])[None, ...] #directions of velocity channels; shape: (1,2)

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        if nodes is not None:
            self.l, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            self.dims = self.l,
            print(type(self.dims))
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
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.bool)
        if 'hom' in kwargs:
            hom = kwargs['hom']
        else:
            hom = None
        if nodes is None and hom:
            self.homogeneous_random_reset(density)
        elif nodes is None:
            self.random_reset(density)
        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.bool) #what if that doesn't fit?! it always
            # does because set_dims has a case for that where it just copies the stuff

    def init_coords(self):
        self.nonborder = (np.arange(self.l) + self.r_int,) #tuple s.t. I can call my lattice sites "nodes[nonborder]" like this
        # self.nonborder ist ein tuple von indices entlang der dimensionen des Gitters, deswegen das Komma. Im 1D Fall ist das Tuple dann nur von der Länge 1.
        # Zelldichte aller Nicht-Randzellen celldens = self.cell_density[self.nonborder]
        # Die "echten Werte" von xcoords sind 0 bis 4, aber die dazugehörigen Indizes, um die Koordinaten abzurufen sind 1 bis 5, dh xcoords[nonborder] gibt dir 0 bis 4 aus.
        self.xcoords = np.arange(self.l + 2 * self.r_int) - self.r_int #indexing of x-coordinates starting at -r_int to l+r_int?

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
        #qty: array with some function
        #2: spacing between samples in qty TODO why 2?
        return np.gradient(qty, 2)[..., None]
        # The ellipsis is used to slice higher-dimensional data structures.
        # It's designed to mean at this point, insert as many full slices (:) to extend the multi-dimensional slice to all dimensions.
        # None adds a new axis to the ndarray and keeps the whole rest unchanged

    def channel_weight(self, qty): #whatever this does
        weights = np.zeros(qty.shape + (self.velocitychannels,)) #velocity channels added as a dimension, ',' to make it a tuple to add it to the shape tuple
        # adding tuples: a = (0,1) b = (2,3) a+b=(0, 1, 2, 3) -> indexed in the same fashion as arrays
        weights[:-1, ..., 0] = qty[1:, ...] #shift first dimension of qty left to put in right velocity channel
        weights[1:, ..., 1] = qty[:-1, ...] #shift second dimension of qty right to put in left velocity channel
        return weights

    #
    # def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True):
    #     self.update_dynamic_fields()
    #     if record:
    #         self.nodes_t = np.zeros((timesteps + 1, self.l, 2 + self.restchannels), dtype=self.nodes.dtype)
    #         self.nodes_t[0, ...] = self.nodes[self.r_int:-self.r_int, ...]
    #     if recordN:
    #         self.n_t = np.zeros(timesteps + 1, dtype=np.int)
    #         self.n_t[0] = self.nodes.sum()
    #     if recorddens:
    #         self.dens_t = np.zeros((timesteps + 1, self.l))
    #         self.dens_t[0, ...] = self.cell_density[self.r_int:-self.r_int]
    #     for t in range(1, timesteps + 1):
    #         self.timestep()
    #         if record:
    #             self.nodes_t[t, ...] = self.nodes[self.r_int:-self.r_int]
    #         if recordN:
    #             self.n_t[t] = self.cell_density.sum()
    #         if recorddens:
    #             self.dens_t[t, ...] = self.cell_density[self.r_int:-self.r_int]
    #         if showprogress:
    #             update_progress(1.0 * t / timesteps)

    def plot_density(self, density_t=None, figindex=None, figsize=None, cmap='viridis_r'): #cmap='hot_r')
        if density_t is None:
            density_t = self.dens_t
        if figsize is None:
            figsize = estimate_figsize(density_t.T, cbar=True)

        # print("Density:")
        # print(density_t)
        # print(density_t.max())
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        cmap = cmap_discretize(cmap, 3 + self.restchannels)
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=2 + self.restchannels, cmap=cmap)
        cbar = colorbar_index(ncolors=3 + self.restchannels, cmap=cmap, use_gridspec=True)
        cbar.set_label(r'Particle number $n$')
        plt.xlabel(r'Lattice node $r \, (\varepsilon)$', )
        plt.ylabel(r'Time step $k \, (\tau)$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return plot

    def plot_flux(self, nodes_t=None, figindex=None, figsize=None):
        if nodes_t is None:
            nodes_t = self.nodes_t

        dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)
        if figsize is None:
            figsize = estimate_figsize(dens_t.T)

        rgba = np.zeros((tmax, l, 4)) #4: RGBA A=alpha: transparency
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t == 1, 0] = 1.
        rgba[flux_t == -1, 2] = 1.
        rgba[flux_t == 0, :-1] = 0.
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        plot = ax.imshow(rgba, interpolation='None', origin='upper')
        plt.xlabel(r'Lattice node $r \, [\varepsilon]$', )
        plt.ylabel(r'Time step $k \, [\tau]$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return plot


class IBLGCA_1D(IBLGCA_base, LGCA_1D):
    """
    1D version of an identity-based LGCA.
    """
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath']

    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)
            self.maxlabel = self.nodes.max()

        else:
            occ = nodes > 0
            self.nodes[self.r_int:-self.r_int] = self.convert_bool_to_ib(occ)
            self.maxlabel = self.nodes.max()

    def plot_prop_spatial(self, nodes_t=None, props_t=None, figindex=None, figsize=None, propname=None, cmap='cividis'):
        if nodes_t is None:
            nodes_t = self.nodes_t
        if figsize is None:
            figsize = estimate_figsize(nodes_t.sum(-1).T, cbar=True)

        if props_t is None:
            props_t = self.props_t

        if propname is None:
            propname = list(props_t[0])[0]

        tmax, l, _ = nodes_t.shape
        mean_prop_t = np.zeros((tmax, l))
        for t in range(tmax):
            meanprop = self.calc_prop_mean(propname=propname, props=props_t[t], nodes=nodes_t[t])
            mean_prop_t[t] = meanprop

        dens_t = nodes_t.astype(bool).sum(-1)
        vmax = np.max(mean_prop_t)
        vmin = np.min(mean_prop_t)
        rgba = plt.get_cmap(cmap)
        rgba = rgba((mean_prop_t - vmin) / (vmax - vmin))
        rgba[dens_t == 0, :] = 0.
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        plot = ax.imshow(rgba, interpolation='none', aspect='equal')
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([vmin, vmax])
        cbar = plt.colorbar(sm, use_gridspec=True)
        cbar.set_label(r'Property ${}$'.format(propname))

        plt.xlabel(r'Lattice node $r \, [\varepsilon]$')
        plt.ylabel(r'Time step $k \, [\tau]$')
        ax.xaxis.set_label_position('top')
        ax.title.set_y(1.05)
        ax.xaxis.tick_top()
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        return plot


class LGCA_noVE_1D(LGCA_1D, LGCA_noVE_base):
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
            self.l, self.K = nodes.shape
            # set number of rest channels to <= 1 because >1 cells are allowed per channel
            if restchannels > 1:
                self.restchannels = 1
            elif 0 <= restchannels <= 1:
                self.restchannels = restchannels
            # for now, raise Exception if format of nodes does no fit
            # (To Do: just sum the cells in surplus rest channels in init_nodes and print a warning)
            if self.K - self.restchannels > self.velocitychannels:
                raise Exception('Only one resting channel allowed, \
                 but {} resting channels specified!'.format(self.K - self.velocitychannels))
            self.dims = self.l,
            return
        # default value for dimension
        elif dims is None:
            dims = 100,
        # set instance dimensions according to desired size
        if isinstance(dims, int):
            self.l = dims
        else:
            self.l = dims[0]
        self.dims = self.l,

        # set number of rest channels to <= 1 because >1 cells are allowed per channel
        if restchannels is not None:
            if restchannels > 1:
                self.restchannels = 1
            elif 0 <= restchannels <= 1:
                self.restchannels = restchannels
        else:
            self.restchannels = 0
        self.K = self.velocitychannels + self.restchannels

        if capacity is not None:
            self.capacity = capacity
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


    def plot_density(self, density_t=None, figindex=None, figsize=None, cmap='viridis_r', scaling=None, absolute_max=None, offset_t=None, offset_x=None):
        """
        Create a plot showing the number of particles per lattice site.
        :param density_t: particle number per lattice site (ndarray of dimension (timesteps + 1,) + self.dims)
        :param figindex: number of the figure to create/activate
        :param figsize: desired figure size
        :param cmap: matplotlib color map for encoding the number of particles
        :return: plot as a matplotlib.image
        """
        # set values for unused arguments
        if density_t is None:
            density_t = self.dens_t
        if figsize is None:
            figsize = estimate_figsize(density_t.T, cbar=True)

        # set up figure
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        # set up color scaling
        if scaling is not None:
            scale = scaling
        else:
            scale = 1.0
        max_part_per_cell = int(scale*density_t.max())
        if absolute_max is not None:
            max_part_per_cell = int(absolute_max)
        cmap = cmap_discretize(cmap, max_part_per_cell + 1)
        # create plot with color bar, axis labels, title and layout
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=max_part_per_cell, cmap=cmap)
        # plot slice with an x-offset
        if offset_x is not None and isinstance(offset_x, int):
            fig.canvas.draw() # otherwise the list will be empty
            labels = [item.get_text() for item in ax.get_xticklabels()]
            for i in range(len(labels)):
                labels[i] = str(int(float(labels[i].strip().replace("−", "-")))+offset_x) #replace long dash by minus
            ax.set_xticklabels(labels)
        # plot slice with a y-offset
        if offset_t is not None and isinstance(offset_t, int):
            fig.canvas.draw() # otherwise the list will be empty
            labels = [item.get_text() for item in ax.get_yticklabels()]
            for i in range(len(labels)):
                labels[i] = str(int(float(labels[i].strip().replace("−", "-")))+offset_t) #replace long dash by minus
            ax.set_yticklabels(labels)
        cbar = colorbar_index(ncolors=max_part_per_cell + 1, cmap=cmap, use_gridspec=True)
        cbar.set_label(r'Particle number $n$')
        plt.xlabel(r'Lattice node $r \, (\varepsilon)$', )
        plt.ylabel(r'Time step $k \, (\tau)$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return plot


    def plot_flux(self, nodes_t=None, figindex=None, figsize=None): # TODO: is this different and needed?
        """
        Create a plot showing the main direction of particles per lattice site.
        :param nodes_t: particles per velocity channel at lattice site
                        (ndarray of dimension (timesteps + 1,) + self.dims + (self.K,))
        :param figindex: number of the figure to create/activate
        :param figsize: desired figure size
        :return: plot as a matplotlib.image
        """
        # set values for unused arguments
        if nodes_t is None:
            nodes_t = self.nodes_t
        # calculate particle density, max time and dimension values, flux
        dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)
        if figsize is None:
            figsize = estimate_figsize(dens_t.T)

        # encode flux as RGBA values
        # 4: RGBA A=alpha: transparency
        rgba = np.zeros((tmax, l, 4))
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t > 0, 0] = 1.
        rgba[flux_t < 0, 2] = 1.
        rgba[flux_t == 0, :-1] = 0. # unpopulated lattice sites are white
        # set up figure
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        # create plot with axis labels, title and layout
        plot = ax.imshow(rgba, interpolation='None', origin='upper')
        plt.xlabel(r'Lattice node $r \, [\varepsilon]$', )
        plt.ylabel(r'Time step $k \, [\tau]$')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
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
         # used for summing up fluxes
         return sum

    def calc_entropy(self):
        """
        Calculate entropy of the lattice.
        :return: entropy according to information theory as scalar
        """
        # calculate relative frequencies
        rel_freq = self.nodes[self.nonborder].sum(-1)/self.nodes[self.nonborder].sum()
        # empty lattice sites are not considered in the sum
        a = np.where(rel_freq > 0, np.log(rel_freq), 0)
        return -np.multiply(rel_freq, a).sum()

    def calc_normalized_entropy(self):
        """
        Calculate entropy of the lattice normalized to maximal possible entropy.
        :return: normalized entropy as scalar
        """
        smax = np.log(self.l)
        self.smax = smax
        return 1 - self.calc_entropy()/smax

    def calc_polar_alignment_parameter(self):
        """
        Calculate the polar alignment parameter.
        The polar alignment parameter is a measure for global agreement of particle orientation in the lattice.
        It is calculated as the magnitude of the sum of the velocities of all particles normalized by the number of particles.
        :return: polar alignment parameter as scalar
        """
        return np.abs(self.calc_flux(self.nodes)[self.nonborder].sum()/self.nodes[self.nonborder].sum())

    def calc_mean_alignment(self):
        """
        Calculate the mean alignment measure.
        The mean alignment is a measure for local alignment of particle orientation in the lattice.
        It is calculated as the agreement in direction between the ﬂux of a lattice site and the ﬂux of the director ﬁeld
        summed up and normalized over all lattice sites.
        :return: mean alignment as scalar
        """
        # retrieve particle numbers and fluxes from instance
        no_neighbors = self.nb_sum(np.ones(self.cell_density[self.nonborder].shape))
        f = self.calc_flux(self.nodes[self.nonborder])
        d = self.cell_density[self.nonborder]
        d_div = np.where(d > 0, d, 1)
        # calculate flux director field and normalize by number of neighbors
        f_norm = f.flatten()/d_div
        f_norm = self.nb_sum(f_norm)
        f_norm = f_norm/no_neighbors
        # calculate agreement between flux and director field flux, take mean over lattice
        return (np.dot(f_norm, f)).sum() / d.sum() #first sum probably unnecessary


    def update_dynamic_fields(self):
        """
        Update "fields" that store important variables to compute other dynamic steps.
        """
        self.cell_density = self.nodes.sum(-1)
        self.eff_dens = self.nodes[self.nonborder].sum()/(self.capacity * self.l)


if __name__ == '__main__':
    l = 100
    restchannels = 2
    n_channels = restchannels + 2
    nodes = 1 + np.arange(l * n_channels, dtype=np.uint).reshape((l, n_channels))
    nodes[1:, :] = 0

    system = IBLGCA_1D(bc='reflect', dims=100, interaction='birthdeath', density=0.1, restchannels=2, r_b=0.1, std=0.05,
                       nodes=nodes)
    system.timeevo(timesteps=200, record=True)
    # system.plot_prop()
    # system.plot_density(figindex=1)
    # props = np.array(system.props['kappa'])[system.nodes[system.nodes > 0]]
    # print(np.mean(props))
    # system.plot_prop_timecourse()
    # plt.ylabel('$\kappa$')
    # system.plot_density()
    system.plot_prop_spatial()
    # system.plot_prop_timecourse()
    plt.show()
