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

    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.bool)
        if nodes is None:
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

    def plot_density(self, density_t=None, figindex=None, figsize=None, cmap='hot_r'):
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
    interactions = ['dd_alignment', 'di_alignment']

    def set_dims(self, dims=None, nodes=None, restchannels=0): # changed to not allow resting channels
        # works with the current default values in the _init_() method/here
        if nodes is not None:
            self.l, self.K = nodes.shape
            if self.K - self.velocitychannels != 0:
                raise Exception('No resting channels allowed, but {} resting channels specified!'.format(self.K - self.velocitychannels))
            self.restchannels = 0
            self.dims = self.l,
            return

        elif dims is None:
            dims = 100

        if restchannels != 0:
            raise Exception('No resting channels allowed, but {} resting channels specified!'.format(restchannels))

        if isinstance(dims, int):
            self.l = dims

        else:
            self.l = dims[0]

        self.dims = self.l,
        self.restchannels = 0
        self.K = self.velocitychannels


    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)  # TODO!

        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.uint)


    def plot_density(self, density_t=None, figindex=None, figsize=None, cmap='hot_r'):
        if density_t is None:
            density_t = self.dens_t
        if figsize is None:
            figsize = estimate_figsize(density_t.T, cbar=True)

        max_part_per_cell = int(density_t.max()) #alternatively plot using the expected density - number of particles in total / lattice sites
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        cmap = cmap_discretize(cmap, max_part_per_cell + 1) #todo adjust number of colours
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=max_part_per_cell, cmap=cmap) #TODO adjust vmax
        cbar = colorbar_index(ncolors=max_part_per_cell + 1, cmap=cmap, use_gridspec=True) #todo adjust ncolors
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

        rgba = np.zeros((tmax, l, 4))
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t > 0, 0] = 1.
        rgba[flux_t < 0, 2] = 1.
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
