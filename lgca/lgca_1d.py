from copy import deepcopy as copy

import matplotlib.ticker as mticker
from datetime import datetime
import pathlib


try:
    from base import *
except ModuleNotFoundError:
    from .base import *


class LGCA_1D(LGCA_base):
    """
    1D version of an LGCA.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk', 'persistent_motion']
    velocitychannels = 2
    c = np.array([1., -1.])[None, ...]

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        if nodes is not None:
            self.l, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            return

        elif dims is None:
            dims = 100

        if isinstance(dims, int):
            self.l = dims

        else:
            self.l = dims[0]

        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels
        # print('capacity = ', self.K)

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
        chronicle = False
        m = self.r_int
        if chronicle:
            print('nodes', self.nodes)
        newnodes = np.zeros_like(self.nodes)
        # resting particles stay
        newnodes[:, 2:] = self.nodes[:, 2:]

        # prop. to the right
        ###newnodes[1:, 0] = self.nodes[:-1, 0]
        newnodes[m:, 0] = self.nodes[:-m, 0]
        if chronicle:
            print('to right', newnodes)
        # prop. to the left
        ###newnodes[:-1, 1] = self.nodes[1:, 1]
        newnodes[:-m, 1] = self.nodes[m:, 1]
        if chronicle:
            print('to left', newnodes)
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

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros((timesteps + 1, self.l, 2 + self.restchannels), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[self.r_int:-self.r_int, ...]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.int)
            self.n_t[0] = self.nodes.sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1, self.l))
            self.dens_t[0, ...] = self.cell_density[self.r_int:-self.r_int]
        for t in range(1, timesteps + 1):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.r_int:-self.r_int]
            if recordN:
                self.n_t[t] = self.cell_density.sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.r_int:-self.r_int]
            if showprogress:
                update_progress(1.0 * t / timesteps)

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
        plt.xlabel(r'Lattice node $r \, [\varepsilon]$', )
        plt.ylabel(r'Time step $k \, [\tau]$')
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
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath', 'inheritance']

    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.l + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            occupied = npr.random((self.l, self.K)) < density
            self.nodes[self.r_int:-self.r_int] = self.convert_bool_to_ib(occupied)
            self.apply_boundaries()
            self.maxlabel = self.nodes.max()

        else:
            occ = nodes > 0
            self.nodes[self.r_int:-self.r_int] = self.convert_bool_to_ib(occ)
            self.maxlabel = self.nodes.max()

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True, recordLast=False):
        self.update_dynamic_fields()
        # si = True
        self.sim_ind = [self.simpsonindex()]
        if record:
            self.nodes_t = np.zeros((timesteps + 1, self.l, 2 + self.restchannels), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[self.r_int:-self.r_int, ...]
            self.props_t = [copy(self.props)]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.nodes.sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1, self.l))
            self.dens_t[0, ...] = self.cell_density[self.r_int:-self.r_int]
        if recordLast:
            self.props_t = [copy(self.props)]
        for t in range(1, timesteps + 1):
            self.timestep()

            if record:
                self.nodes_t[t, ...] = self.nodes[self.r_int:-self.r_int]
                self.props_t.append(copy(self.props))
            if recordN:
                self.n_t[t] = self.cell_density.sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.r_int:-self.r_int]
            if recordLast and t == (timesteps + 1):
                self.props_t.append(copy(self.props))
            if showprogress:
                update_progress(1.0 * t / timesteps)
            # if si:
            #     if self.simpsonindex() == 0:
            #         print('Homogeneity since k = ', t)
            #         si = False
            self.sim_ind.append(self.simpsonindex())
            # if self.simpsonindex() == 0:
            #     break
            # print('index=', self.sim_ind)
            # print('t=', t)
            # print('props in timeevo:', self.props['num_off'])
            # print('props_t in timeevo', self.props_t[t]['num_off'][:])

    def plot_prop_spatial(self, nodes_t=None, props_t=None, figindex=None, figsize=None, prop=None, cmap='cividis'):
        if nodes_t is None:
            nodes_t = self.nodes_t
        if figsize is None:
            figsize = estimate_figsize(nodes_t.sum(-1).T, cbar=True)

        if props_t is None:
            props_t = self.props_t
        if prop is None:
            prop = list(props_t[0].keys())[0]

        tmax, l, _ = nodes_t.shape
        # print('tmax=%d, l=%d' %(tmax, l))
        mean_prop = np.zeros((tmax, l))

        for t in range(tmax):
            for x in range(l):
                node = nodes_t[t, x]
                occ = node.astype(np.bool)
                if occ.sum() == 0:
                    continue
                mean_prop[t, x] = np.mean(np.array(props_t[t][prop])[node[node > 0]])

        dens_t = nodes_t.astype(bool).sum(-1)
        vmax = np.max(mean_prop)
        vmin = np.min(mean_prop)
        rgba = plt.get_cmap(cmap)
        rgba = rgba((mean_prop - vmin) / (vmax - vmin))
        rgba[dens_t == 0, :] = 0.
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        plot = ax.imshow(rgba, interpolation='none', aspect='equal')
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([vmin, vmax])
        cbar = plt.colorbar(sm, use_gridspec=True)
        cbar.set_label(r'Property ${}$'.format(prop))

        plt.xlabel(r'Lattice node $r \, [\varepsilon]$')
        plt.ylabel(r'Time step $k \, [\tau]$')
        # matshow style
        ax.xaxis.set_label_position('top')
        ax.title.set_y(1.05)
        ax.xaxis.tick_top()
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        return plot

    def spatial_plot(self, nodes_t=None, props_t=None, figindex = None, figsize=None, prop='lab_m',\
                     cmap='nipy_spectral', tbeg=None, tend=None, save=False, id=0):
        if nodes_t is None:
            nodes_t = self.nodes_t
        # if figsize is None:
        #     figsize = estimate_figsize(nodes_t.sum(-1).T, cbar=True)
        # figsize = None
        if props_t is None:
            props_t = self.props_t
        # if prop is None:
        #   prop = list(props_t[0].keys())[0]

        tmax, l, _ = nodes_t.shape
        if tbeg is None:
            tbeg = 0
        if tend is None:
            tend = tmax
        k = self.restchannels + self.velocitychannels
        ltotal = l * k
        val = np.zeros((tmax, ltotal))

        for t in range(0, tmax):
            for x in range(l):
                node = nodes_t[t, x]
                occ = node.astype(np.bool)
                if occ.sum() == 0:
                    i = 0
                    while i < k:
                        val[t, x * k + i] = None
                        i = i + 1
                    continue
                for pos in range(len(node)):
                    lab = node[pos]
                    if pos == 0 or pos == 1:
                        if lab == 0:
                            val[t, x*k + pos * (k - 1)] = None
                        else:
                            val[t, x*k + pos * (k-1)] = props_t[t][prop][lab]
                    else:
                        if lab == 0:
                            val[t, x*k + pos - 1] = None
                        else:
                            val[t, x*k + pos - 1] = props_t[t][prop][lab]
        # print('val', val)

        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        plot = ax.matshow(val, cmap=cmap)
        # fig.colorbar(plot, shrink = 0.5)

        plt.ylabel('timesteps')
        plt.xlabel('lattice site')
        # nur "Knotenanfang"
        plt.xlim(0, ltotal)
        plt.xticks((np.arange(0, ltotal, k)))

        plt.ylim(tend, tbeg)
        if tend - tbeg > 700:
            plt.yticks(np.arange(tbeg, tend, 100))
        elif tend - tbeg > 100:
            plt.yticks(np.arange(tbeg, tend, 50))
        elif tend - tbeg <= 100:
            plt.yticks(np.arange(tbeg, tend, 10))

        if save == True:
            filename = str(self.r_b) + ', ' + str(id) + ', ' + ' frequency ' +\
                       str(tbeg) + ' to ' + str(tend) + '.jpg'

            plt.savefig(pathlib.Path('pictures').resolve() / filename)

        # plt.tight_layout()
        plt.show()

    def bar_stacked_relative(self, save=False, id=0):
        tmax, l, _ = self.nodes_t.shape
        # print('tmax', tmax)
        ancs = np.arange(1, self.maxlabel_init.astype(int) + 1)
        val = np.zeros((tmax, self.maxlabel_init.astype(int) + 1))

        for t in range(0, tmax):
            c_sum = 0
            for c in ancs:
                val[t, c] = self.props_t[t]['num_off'][c]
                c_sum = c_sum + self.props_t[t]['num_off'][c]
            if c_sum != 0:
                val[t] = val[t] / c_sum
        # print('val', val)

        plt.figure(num=None)
        ind = np.arange(0, tmax, 1)
        width = 1
        for c in ancs:
            if c > 1:
                b = np.zeros(tmax)
                for i in range(1, c):
                    b = b + val[:, i]
                plt.bar(ind, val[:, c], width, bottom=b, label=c)
            else:
                plt.bar(ind, val[:, c], width, color=['red'], label=c)

        ###plot settings

        plt.ylabel('relative frequency of families')
        plt.xlabel('timesteps')
        # plt.title('Ratio of offsprings')
        if len(ind) <= 15:
            plt.xticks(ind)
        else:
            plt.xticks(np.arange(0, len(ind) - 1, 5))

        if tmax >= 700:
            plt.xticks(np.arange(0, tmax, 100))
        elif tmax >= 100:
            plt.xticks(np.arange(0, tmax, 50))

        # plt.subplots_adjust(right=0.85)
        # plt.legend(bbox_to_anchor=(1.04, 1))
        plt.tight_layout()
        plt.show()
        if save == True:
            # plt.savefig('pictures/' + str(id) + '  frequency' + str(datetime.now()) +'.jpg')
            # plt.savefig('probe_bar.jpg')
            # filename = str(lgca.r_b) + ', ' + str(id) + ', ' + str(t) + '  frequency' + '.jpg'
            filename = str(self.r_b) + ', dens' + str(self.maxlabel_init / (self.K * self.l)) + ', ' \
                       + str(id) + ', ' + str(t) + '  rel_frequency' + '.jpg'

            plt.savefig(pathlib.Path('pictures').resolve() / filename)

    def mullerplot(self, nodes_t=None, props_t=None, figindex=None, figsize=None):
        if nodes_t is None:
            nodes_t = self.nodes_t
        # if figsize is None:
        #     figsize = estimate_figsize(nodes_t.sum(-1).T, cbar=True)
        if props_t is None:
            props_t = self.props_t

        #create values
        tmax, l, _ = nodes_t.shape
        ancs = np.arange(1, self.maxlabel_init.astype(int) + 1)
        # if len(ancs) != lgca.maxlabel_init:
        #     print('FEHLER: len(ancs) != maxlabel_init!')
        val = np.zeros((tmax, self.maxlabel_init.astype(int) + 1))
        for t in range(0, tmax):
            for c in ancs:
                val[t, c] = props_t[t]['num_off'][c]

        #write in .txt
        file = str(datetime.now()) + '.txt'
        np.savetxt(file, val, fmt="%d")


    def plot_prop_timecourse(self, nodes_t=None, props_t=None, propname=None, figindex=None, figsize=None):
        if nodes_t is None:
            nodes_t = self.nodes_t

        if props_t is None:
            props_t = self.props_t

        if propname is None:
            propname = list(props_t[0].keys())[0]

        plt.figure(num=figindex, figsize=figsize)
        tmax = len(props_t)
        mean_prop = np.zeros(tmax)
        std_mean_prop = np.zeros(mean_prop.shape)
        for t in range(tmax):
            props = props_t[t]
            nodes = nodes_t[t]
            prop = np.array(props[propname])[nodes[nodes > 0]]
            mean_prop[t] = np.mean(prop)
            std_mean_prop[t] = np.std(prop, ddof=1) / np.sqrt(len(prop))

        yerr = std_mean_prop
        x = np.arange(tmax)
        y = mean_prop

        plt.xlabel('$t$')
        plt.ylabel('${}$'.format(propname))
        plt.title('Time course of the cell property')
        plt.plot(x, y)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, antialiased=True, interpolate=True)
        return


if __name__ == '__main__':
    l = 100
    restchannels = 2
    n_channels = restchannels + 2
    nodes = 1 + np.arange(l * n_channels, dtype=np.uint).reshape((l, n_channels))
    # nodes[1:, :] = 0
    # nodes[0, 1:] = 0

    system = IBLGCA_1D(bc='reflect', dims=1, interaction='birthdeath', density=1, restchannels=5000, r_b=0.2)
    system.timeevo(timesteps=2000, record=True)
    # system.plot_prop()
    # system.plot_density(figindex=1)
    # props = np.array(system.props['kappa'])[system.nodes[system.nodes > 0]]
    # print(np.mean(props))
    # system.plot_prop_timecourse()
    # plt.ylabel('$\kappa$')
    # system.plot_density()
    # system.plot_prop_spatial()
    system.plot_prop_timecourse()
    plt.show()
