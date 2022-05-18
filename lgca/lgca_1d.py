import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
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
        self.nodes[self.r_int, 0] += self.nodes[self.r_int-1, 1]
        self.nodes[-self.r_int-1, 1] += self.nodes[-self.r_int, 0]
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
        ax.yaxis.set_ticks_position('left')
        ax.set_autoscale_on(False)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        return fig, ax

    def plot_density(self, density_t=None, cmap='hot_r', vmax='auto', colorbarwidth=0.03,
                     cbarlabel='Particle number $n$', **kwargs):
        if density_t is None:
            density_t = self.dens_t

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

        cbar.set_label(cbarlabel)
        plt.sca(ax)
        return plot

    def plot_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t

        dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)

        rgba = np.zeros((tmax, l, 4))
        rgba[dens_t > 0, -1] = 1.
        rgba[flux_t == 1, 0] = 1.
        rgba[flux_t == -1, 2] = 1.
        rgba[flux_t == 0, :-1] = 0.
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

    def plot_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t.astype('bool')
        if nodes_t.dtype != 'bool':
            nodes_t = nodes.astype('bool')
        LGCA_1D.plot_flux(self, nodes_t, **kwargs)

    def plot_prop_spatial(self, nodes_t=None, props=None, propname=None, cmap='cividis', cbarlabel=None, **kwargs):
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
        if cbarlabel is None:
            cbar.set_label(r'Property ${}$'.format(propname))
        else:
            cbar.set_label(cbarlabel)
        plt.sca(ax)
        return plot

class BOSON_IBLGCA_1D(BOSON_IBLGCA_base, IBLGCA_1D):
    interactions = ['go_or_grow', 'birthdeath']
    
    def propagation(self):
        """
        :return:
        """
        newnodes = deepcopy(self.nodes)

        # prop. to the left
        newnodes[1:, 0] = self.nodes[:-1, 1]

        # prop. to the right
        newnodes[:-1, 1] = self.nodes[1:, 0]

        self.nodes = deepcopy(newnodes)

    def apply_pbc(self):
        self.nodes[:self.r_int, :] = self.nodes[-2 * self.r_int:-self.r_int, :]
        self.nodes[-self.r_int:, :] = self.nodes[self.r_int:2 * self.r_int, :]

    def apply_rbc(self):
        self.nodes[self.r_int, 0] = self.nodes[self.r_int, 0] + self.nodes[self.r_int - 1, 1]
        self.nodes[-self.r_int - 1, 1] = self.nodes[-self.r_int - 1, 1] + self.nodes[-self.r_int, 0]
        self.nodes[self.r_int - 1, 1] = []
        self.nodes[-self.r_int, 0] = []

    def apply_abc(self):
        for channel in self.nodes[:self.r_int].flat:
            channel.clear()

        for channel in self.nodes[-self.r_int:].flat:
            channel.clear()

    
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
        self.K = self.velocitychannels + restchannels

    def init_nodes(self, density, nodes=None):
        """
        initialize the nodes. there are three options:
        1) you provide only the argument "density", which should be a positive float that indicates the average number
        of cells in each channel
        2) you provide an array "nodes" with nodes.dtype == int,
            where each integer determines the number of cells in each channel
        3) you provide an array "nodes" with nodes.dtype == object, where each element is a list of unique cell labels
        """
        temp = np.empty((self.l + 2 * self.r_int, self.K), dtype=object)
        temp = temp.flatten()
        temp[:] = [[] for _ in range(len(temp))]
        self.nodes = temp.reshape((self.l + 2 * self.r_int, self.K))
        if nodes is None:
            self.random_reset(density)

        elif nodes.dtype == object:
            self.nodes[self.nonborder] = nodes

        else:
            occ = nodes.astype(int)
            self.nodes[self.nonborder] = self.convert_int_to_ib(occ)

        self.calc_max_label()

    def plot_density(self, density_t=None, cmap='hot_r', channel_type='all', **kwargs):
        if channel_type == 'all':
            if density_t is None:
                density_t = self.dens_t
        # other channel types are not supported yet
        elif channel_type == 'rest':
            if density_t is None:
                density_t = self.resting_density_t
        elif channel_type == 'velocity':
            if density_t is None:
                density_t = self.moving_density_t
        tmax = density_t.shape[0]
        dmax = density_t.max().astype(int)
        fig, ax = self.setup_figure(tmax, **kwargs)
        cmap = cmap_discretize(cmap, 1+dmax)
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=dmax, cmap=cmap, aspect='equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=.3, pad=0.1)
        cbar = colorbar_index(ncolors=1+dmax, cmap=cmap, use_gridspec=True, cax=cax)
        cbar.set_label('Particle number $n$')
        plt.sca(ax)
        return plot
    
    def plot_prop_spatial(self, nodes_t=None, props=None, propname=None, cmap='cividis', channeltype='all', **kwargs):
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
        if channeltype == 'all':
            mean_prop_t = self.mean_prop_t[propname]
        #elif(channeltype == 'velocity'):
        #    mean_prop_t = self.mean_prop_vel_t[propname]
        #elif(channeltype == 'rest'):   
        #    mean_prop_t = self.mean_prop_rest_t[propname]

        plot = plt.imshow(mean_prop_t, interpolation='none', cmap=cmap, aspect='equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        cbar = fig.colorbar(plot, use_gridspec=True, cax=cax)
        cbar.set_label(r'Property ${}$'.format(propname))
        plt.sca(ax)
        return plot
    
if __name__ == '__main__':
    l = 200
    restchannels = 1
    n_channels = restchannels + 2
    nodes = np.zeros((l, n_channels))
    nodes[...] = 2

    system = BOSON_IBLGCA_1D(bc='rbc', dims=l, interaction='go_or_grow', density=1.5, restchannels=1, theta=.25, kappa=0,
                             r_d=0.05, r_b=0.25, theta_std=1e-6, nodes=nodes)
    print(system.cell_density[system.nonborder].sum(), system.maxlabel, max(system.nodes.sum()), len(system.props['theta']), len(system.props['kappa']))
    system.timeevo(timesteps=100, record=True, showprogress=1)
    # system.timestep()
    print(system.cell_density[system.nonborder].sum(), system.maxlabel, max(system.nodes.sum()), len(system.props['theta']), len(system.props['kappa']))
    # system.plot_density(figindex=0)
    # system.plot_prop_spatial(figindex=1)
    system.plot_prop_timecourse(figindex=2)
    system.plot_prop_2dhist(figindex=3, extent=(min(system.get_prop(propname='kappa')), max(system.get_prop(propname='kappa')), 0, 1),
                            gridsize=round(system.cell_density[system.nonborder].sum() ** 0.5))
    print(system.get_prop(propname='kappa'), system.get_prop(propname='theta'))
    plt.show()
