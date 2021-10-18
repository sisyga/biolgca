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

    def plot_density(self, density_t=None, cmap='hot_r', **kwargs):
        if density_t is None:
            density_t = self.dens_t

        tmax = density_t.shape[0]
        fig, ax = self.setup_figure(tmax, **kwargs)
        cmap = cmap_discretize(cmap, 3 + self.restchannels)
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=self.K, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=.3, pad=0.1)
        cbar = colorbar_index(ncolors=3 + self.restchannels, cmap=cmap, use_gridspec=True, cax=cax)
        cbar.set_label('Particle number $n$')
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

class BOSON_IBLGCA_1D(BOSON_IBLGCA_base, IBLGCA_1D):
    interactions = ['go_or_grow', 'memory_go_or_grow']
    
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
        # same like the classical LGCA version without Boson and IB
        self.nodes[:self.r_int, :] = self.nodes[-2 * self.r_int:-self.r_int, :]
        self.nodes[-self.r_int:, :] = self.nodes[self.r_int:2 * self.r_int, :]

    def apply_rbc(self):
        # same like the classical LGCA version without Boson and IB
        self.nodes[self.r_int, 0] = self.nodes[self.r_int - 1, 1] + self.nodes[self.r_int, 0]
        self.nodes[-self.r_int-1, 1] = self.nodes[-self.r_int, 0] + self.nodes[-self.r_int-1, 1]
        self.apply_abc()

    def apply_abc(self):
        # make the empty boundary nodes permanent after first call of this function
        # notably not very elegant
        if not hasattr(self, "boundary_nodes"):
            # construct empty boundary nodes and save
            boundary_nodes = np.empty(self.r_int * self.K, dtype=object)
            for n in range(self.r_int * self.K):
                boundary_nodes[n] = []
            self.boundary_nodes = boundary_nodes.reshape((self.r_int, self.K))
        # enforce abc
        self.nodes[:self.r_int] = self.boundary_nodes
        self.nodes[-self.r_int:] = self.boundary_nodes

    def set_dims(self, dims=None, nodes=None):
        if nodes is not None:
            self.l, self.K = nodes.shape
            self.restchannels = 1
            self.dims = self.l,
            return

        elif dims is None:
            dims = 100,

        if isinstance(dims, int):
            self.l = dims

        else:
            self.l = dims[0]
           
        self.dims = self.l,
        self.restchannels = 1
        self.K = 3
        

    def init_nodes(self, ini_channel_pop=None, nodes=None, nodes_filled=None, capacity=4, **kwargs):
        if nodes_filled is not None and ini_channel_pop is not None: #nodes_filled is number of nodes to fill,
            # ini_channel_pop is number of particles to put into each node
            oldnodes = np.empty((self.l+2*self.r_int)*self.K, dtype=object)
            #oldnodes[0:self.K] = [],[],[]
            #oldnodes[-self.K:] = [],[],[]
            for k in range((self.l+2*self.r_int)*self.K):
                oldnodes[k] = []
            for n in range(nodes_filled):
                for c in range(self.K):
                    oldnodes[self.K+n*self.K+c] = [ini_channel_pop*(n*self.K+c+1)+j-ini_channel_pop+1 for j in range(ini_channel_pop)]
            #for n in range(self.l*self.K 
            self.nodes = oldnodes.reshape((self.l+2*self.r_int,self.K))
            self.maxlabel = nodes_filled*self.K*ini_channel_pop
            # self.maxlabel = nodes_filled*capacity*ini_channel_pop #why capacity?
        if nodes is not None:
            # set boundary nodes to be empty - they have to be lists to ensure function of other methods
            boundary_nodes = np.empty(self.r_int*self.K, dtype=object)
            for n in range(self.r_int*self.K):
                    boundary_nodes[n] = []
            boundary_nodes = boundary_nodes.reshape((self.r_int, self.K))
            # store empty boundary nodes for use in absorbing boundary conditions
            self.boundary_nodes = boundary_nodes
            # set non-boundary nodes to be the passed nodes
            self.nodes = np.concatenate((boundary_nodes,nodes,boundary_nodes), axis=0)
            # set current maximum ID for birth process
            if not 'maxlabel' in kwargs:
                raise ValueError("Maximum ID from provided nodes must be in kwarg 'maxlabel'. Too expensive to compute it myself.")
            maxlabel = kwargs['maxlabel']
            self.maxlabel=maxlabel
            
    def plot_density(self, density_t=None, cmap='hot_r', channel_type='all', scaling=1, **kwargs):
        if(channel_type == 'all'):
            if density_t is None:
                density_t = self.density_t
        elif(channel_type == 'rest'):
            if density_t is None:
                density_t = self.resting_density_t
        elif(channel_type == 'velocity'):
            if density_t is None:
                density_t = self.moving_density_t
        tmax = density_t.shape[0]
        fig, ax = self.setup_figure(tmax, **kwargs)
        # allow plotting densities > self.capacity as color for highest occupancy
        if np.abs(scaling)>1:
            maxval = self.capacity*scaling
        else:
            maxval = self.capacity
        cmap = cmap_discretize(cmap, 4 + maxval)
        plot = ax.imshow(density_t, interpolation='None', vmin=0, vmax=maxval+4, cmap=cmap, aspect = 'auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=.3, pad=0.1)
        cbar = colorbar_index(ncolors=4+maxval, cmap=cmap, use_gridspec=True, cax=cax)
        cbar.set_label('Particle number $n$')
        plt.sca(ax)
        return plot
    
    def plot_prop_spatial(self, nodes_t=None, props=None, propname=None, cmap='cividis', channeltype='all', **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t
        """
        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(props))
        """
        if(self.mean_prop_t=={}):
            self.calc_prop_mean_spatiotemp()
        #make some changes to the next few lines to remove redundancy    
        tmax, l, _ = nodes_t.shape
        fig, ax = self.setup_figure(tmax, **kwargs)
        #mean_prop_t = np.zeros([tmax, l])
        if(channeltype == 'all'):
            mean_prop_t = self.mean_prop_t[propname]
        #elif(channeltype == 'velocity'):
        #    mean_prop_t = self.mean_prop_vel_t[propname]
        #elif(channeltype == 'rest'):   
        #    mean_prop_t = self.mean_prop_rest_t[propname]

        masked_mean_prop_t = np.ma.masked_where(mean_prop_t==-1000, mean_prop_t)
        plot = plt.imshow(masked_mean_prop_t, interpolation='none', cmap=cmap, aspect = 'auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        cbar = fig.colorbar(plot, use_gridspec=True, cax=cax)
        cbar.set_label(r'Property ${}$'.format(propname))
        plt.sca(ax)
        return plot
    
if __name__ == '__main__':
    l = 100
    restchannels = 2
    n_channels = restchannels + 2
    nodes = np.zeros((l, n_channels))
    nodes[0] = 1

    system = IBLGCA_1D(bc='reflect', dims=100, interaction='birthdeath', density=0.1, restchannels=2, r_b=0.1, std=0.005,
                       nodes=nodes)
    system.timeevo(timesteps=100, record=True)
    print(system.nodes_t[0].shape)
    print(system.get_prop(system.nodes_t[0]).shape)
    # system.plot_prop()
    # system.plot_density(figindex=1)
    # props = np.array(system.props['kappa'])[system.nodes[system.nodes > 0]]
    # print(np.mean(props))
    # system.plot_prop_timecourse()
    # plt.ylabel('$\kappa$')
    system.plot_density()
    #system.plot_prop_spatial()
    # system.plot_prop_timecourse()
    plt.show()
