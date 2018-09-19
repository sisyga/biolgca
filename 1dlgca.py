from __future__ import division

import numpy.random as npr
from sympy.utilities.iterables import multiset_permutations

from tools import *


class LGCA_1D:
    """
    1D version of an LGCA. Mainly used to compare simulations with analytic results.
    """

    def __init__(self, nodes=None, l=100, restchannels=1, beta=1., density=0.1, bc='periodic', r_int=1):
        """
        Initialize class instance.
        :param nodes:
        :param l:
        :param restchannels:
        :param alpha:
        :param beta:
        :param gamma:
        :param mcs:
        :param tau:
        :param density:
        """
        self.dens_t, self.nodes_t = np.empty(2)
        self.beta = beta
        self.r_int = r_int  # interaction range; must be at least 1, to handle propagation.
        if nodes is None:
            self.l = l
            self.restchannels = restchannels
            self.nodes = np.zeros((l + 2 * self.r_int, 2 + self.restchannels), dtype=np.bool)
            self.nodes = npr.random(self.nodes.shape) < density
        if nodes is not None:
            assert len(nodes.shape) == 2
            self.l, self.restchannels = nodes.shape
            self.nodes = np.zeros((self.l + 2 * self.r_int, self.restchannels))
            self.nodes[self.r_int:-self.r_int, :] = nodes
            self.restchannels -= 2
        self.x = np.arange(self.l)

        if bc is 'absorbing':
            self.apply_boundaries = self.apply_abc
        elif bc is 'reflecting':
            self.apply_boundaries = self.apply_rbc
        else:
            self.apply_boundaries = self.apply_pbc
        self.phen_change = lambda: None  # functions that return the argument. Can be replaced by meaningful things later
        self.birth_death = lambda: None
        self.config_energy = lambda p, x: 1.  # constant weight == random walk

        # self.occupiedchannels = self.nodes > 0
        self.cell_density = self.nodes.sum(-1)  # not needed for boolean lgca
        # self.occupiednodes = self.cell_density > 0
        # self.nbs = self.calc_nbs(self.cell_density)

    def propagation(self, nodes):
        """

        :param nodes:
        :return:
        """
        newnodes = np.empty(nodes.shape, dtype=nodes.dtype)
        # resting particles stay
        newnodes[:, 2:] = nodes[:, 2:]

        # prop. to the right
        newnodes[1:, 0] = nodes[:-1, 0]

        # prop. to the left
        newnodes[:-1, 1] = nodes[1:, 1]

        self.apply_boundaries(newnodes)
        self.reset_boundaries(newnodes)
        self.nodes = newnodes

    def reset_boundaries(self, nodes):
        nodes[0, :] = 0
        nodes[-1, :] = 0

    def apply_pbc(self, newnodes):
        newnodes[1, 0] = newnodes[-1, 0]
        newnodes[-2, 1] = newnodes[0, 1]

    def apply_rbc(self, newnodes):
        newnodes[1, 0] = newnodes[0, 1]
        newnodes[-2, 1] = newnodes[-1, 0]

    def apply_abc(self, newnodes):
        self.reset_boundaries(newnodes)

    def reset_config(self, density):
        """

        :param density:
        :return:
        """
        self.nodes = npr.random(self.nodes.shape) < density
        self.update_dynamic_fields()

    def calc_nbs(self, cell_density):
        """

        :param density:
        :return:
        """
        nbs = np.zeros(self.l)

        # right neighbor
        nbs[:-1] += cell_density[1:]
        nbs[-1] += cell_density[0]

        # left neighbor
        nbs[1:] += cell_density[:-1]
        nbs[0] += cell_density[-1]

        nbs += cell_density - 1
        nbs[cell_density == 0] = 0
        return nbs

    def update_dynamic_fields(self):
        """Update "fields" that store important variables to compute other dynamic steps

        :return:
        """
        self.cell_density = self.nodes[self.r_int:-self.r_int].sum(-1)
        # self.occupiednodes = self.cell_density > 0
        # self.nbs = self.calc_nbs(self.cell_density)

    def rearrange(self):
        """
        :return:
        """
        # newnodes = self.nodes.copy()
        newnodes = np.empty(self.nodes.shape,
                            dtype=self.nodes.dtype)  # this is faster, if the interaction is applied to all nodes
        for x in self.r_int + self.x:
            node = self.nodes[x, :]
            weights = []
            permutations = list(multiset_permutations(node))
            for p in permutations:
                weights.append(self.config_energy(p, x))

            weights = np.asarray(weights)
            ind = npr.choice(len(weights), p=weights / weights.sum())
            newnodes[x, :] = permutations[ind]

        self.nodes = newnodes

    def timestep(self):
        self.birth_death()
        self.phen_change()
        self.rearrange()
        self.propagation(self.nodes)
        self.update_dynamic_fields()

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros((timesteps + 1, self.l, 2 + self.restchannels), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[1:-1, ...]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.int)
            self.n_t[0] = self.nodes.sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1, self.l))
            self.dens_t[0, ...] = self.cell_density
        for t in range(1, timesteps + 1):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[1:-1]
            if recordN:
                self.n_t[t] = self.cell_density.sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density
            if showprogress:
                update_progress(1.0 * t / timesteps)

    def plot_timeevo(self, density_t, figindex=0, figsize=(16, 9)):
        import seaborn as sns
        sns.set_style('white')
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        cmap = cmap_discretize('hot_r', 3 + self.restchannels)
        plot = ax.matshow(density_t, interpolation='None', vmin=0, vmax=2 + self.restchannels, cmap=cmap)
        cbar = colorbar_index(ncolors=3 + self.restchannels, cmap=cmap, use_gridspec=True)
        cbar.set_label(r'Particle number $n$')
        plt.xlabel(r'Lattice node $r \, [\varepsilon]$', )
        plt.ylabel(r'Time step $k \, [\tau]$')
        ax.xaxis.set_label_position('top')
        return plot


if __name__ == '__main__':
    nodes = np.zeros((10, 2), dtype=np.bool)
    nodes[0, 0] = 1
    system = LGCA_1D(l=100, restchannels=0, nodes=nodes, bc='reflecting')
    system.timeevo(timesteps=100, recorddens=True, record=True)
    system.plot_timeevo(system.dens_t)
    plt.show()
