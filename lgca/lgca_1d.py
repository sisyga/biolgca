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
    c = np.array([1., -1.])[None, ...]

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

    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.bool)
        if nodes is None:
            self.random_reset(density)

        else:
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.bool)

    def init_coords(self):
        self.nonborder = (np.arange(self.l) + self.r_int,)
        self.xcoords = np.arange(self.l + 2 * self.r_int) - self.r_int

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
        self.nodes[self.r_int, 0] += self.nodes[self.r_int - 1, 1]
        self.nodes[-self.r_int - 1, 1] += self.nodes[-self.r_int, 0]
        self.apply_abc()

    def apply_abc(self):
        self.nodes[:self.r_int, :] = 0
        self.nodes[-self.r_int:, :] = 0

    def nb_sum(self, qty):
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        return sum

    def gradient(self, qty):
        return np.gradient(qty, 2)[..., None]

    def channel_weight(self, qty):
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, :, 0] = qty[1:, ...]
        weights[1:, :, 2] = qty[:-1, ...]
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

        rgba = np.zeros((tmax, l, 4))
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
