from __future__ import division

import numpy.random as npr
from sympy.utilities.iterables import multiset_permutations

from tools import *


def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


class LGCA_1D:
    """
    1D version of an LGCA. Mainly used to compare simulations with analytic results.
    """

    def __init__(self, nodes=None, l=101, restchannels=2, density=0.1, bc='periodic', r_int=1, **kwargs):
        """
        Initialize class instance.
        :param nodes:
        :param l:
        :param restchannels:
        :param density:
        :param bc:
        :param r_int:
        :param kwargs:
        """
        self.dens_t, self.nodes_t, self.n_t = np.empty(3)  # placeholders for record of dynamics
        self.r_int = r_int  # interaction range; must be at least 1 to handle propagation.
        if nodes is None:
            self.l = l
            self.restchannels = restchannels
            self.nodes = np.zeros((l + 2 * self.r_int, 2 + self.restchannels), dtype=np.bool)
            self.nodes = npr.random(self.nodes.shape) < density
        if nodes is not None:
            assert len(nodes.shape) == 2
            self.l, self.restchannels = nodes.shape
            self.nodes = np.zeros((self.l + 2 * self.r_int, self.restchannels), dtype=np.bool)
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.bool)
            self.restchannels -= 2
        self.x = np.arange(self.l)
        self.K = 2 + self.restchannels

        if bc in ['absorbing', 'absorb', 'abs']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl']:
            self.apply_boundaries = self.apply_rbc
        else:
            self.apply_boundaries = self.apply_pbc

        # set phenotypic change function
        if 'phen_change' in kwargs:
            if kwargs['phen_change'] is 'none':
                def phen_change():
                    pass

                self.phen_change = phen_change
            else:
                print 'keyword', kwargs['phen_change'], 'is not defined!'

                def phen_change():
                    pass

                self.phen_change = phen_change

        else:
            def phen_change():
                pass

            self.phen_change = phen_change
        # set birth_death process
        if 'birthdeath' in kwargs:
            if kwargs['birthdeath'] is 'birth':
                self.birth_death = self.birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
            elif kwargs['birthdeath'] is 'none':
                def birth_death():
                    pass

                self.birth_death = birth_death
            else:
                print 'keyword', kwargs['birthdeath'], 'is not defined!'
        else:
            def birth_death():
                pass

            self.birth_death = birth_death

        if 'interaction' in kwargs:
            if kwargs['interaction'] is 'go_or_grow':
                self.interaction = self.go_or_grow_interaction
                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.01
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                if 'kappa' in kwargs:
                    self.kappa = kwargs['kappa']
                else:
                    self.kappa = 5.
                if 'theta' in kwargs:
                    self.theta = kwargs['theta']
                else:
                    self.theta = 0.75

            elif kwargs['interaction'] is 'go_and_grow':
                self.birth_death = self.birth
                self.interaction = self.random_walk
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2

            elif kwargs['interaction'] is 'random_walk':
                self.interaction = self.random_walk
            else:
                print 'keyword', kwargs['interaction'], 'is not defined!'
        else:
            self.interaction = self.rearrange

        # add same procedure for config_energy

        # set rearrangement energy function
        self.config_energy = lambda p, x: 1.  # constant weight == random walk

        # self.occupiedchannels = self.nodes > 0
        self.cell_density = self.nodes[self.r_int:-self.r_int].sum(-1)
        # self.occupiednodes = self.cell_density > 0
        # self.nbs = self.calc_nbs(self.cell_density)

    def propagation(self):
        """

        :param nodes:
        :return:
        """
        newnodes = np.empty(self.nodes.shape, dtype=nodes.dtype)
        # resting particles stay
        newnodes[:, 2:] = self.nodes[:, 2:]

        # prop. to the right
        newnodes[1:, 0] = self.nodes[:-1, 0]

        # prop. to the left
        newnodes[:-1, 1] = self.nodes[1:, 1]

        self.apply_boundaries(newnodes)
        self.reset_boundaries(newnodes)
        self.nodes = newnodes

    def reset_boundaries(self, newnodes):
        newnodes[0, :] = 0
        newnodes[-1, :] = 0

    def apply_pbc(self, newnodes):
        newnodes[1, 0] = newnodes[-1, 0]
        newnodes[-2, 1] = newnodes[0, 1]

    def apply_rbc(self, newnodes):
        newnodes[1, 0] = newnodes[0, 1]
        newnodes[-2, 1] = newnodes[-1, 0]

    def apply_abc(self, newnodes):
        newnodes[0, :] = 0
        newnodes[-1:, :] = 0

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

    def go_or_grow_interaction(self):
        """
        interactions of the go-or-grow model. formulation too complex for 1d, but to be generalized.
        :return:
        """
        n_m = self.nodes[:, :2].sum(-1)
        n_r = self.nodes[:, 2:].sum(-1)
        M1 = np.minimum(n_m, self.restchannels - n_r)
        M2 = np.minimum(n_r, 2 - n_m)
        for x in 1 + self.x:
            node = self.nodes[x, :]
            n = node.sum()
            if n == 0:
                continue

            rho = n / self.K
            j_1 = npr.binomial(M1[x], tanh_switch(rho, kappa=self.kappa, theta=self.theta))
            j_2 = npr.binomial(M2[x], 1 - tanh_switch(rho, kappa=self.kappa, theta=self.theta))
            n_m[x] += j_2 - j_1
            n_r[x] += j_1 - j_2
            n_m[x] -= npr.binomial(n_m[x] * np.heaviside(n_m[x], 0), self.r_d)
            n_r[x] -= npr.binomial(n_r[x] * np.heaviside(n_r[x], 0), self.r_d)
            M = min([n_r[x], self.restchannels - n_r[x]])
            n_r[x] += npr.binomial(M * np.heaviside(M, 0), self.r_b)

            v_channels = [1] * n_m[x] + [0] * (2 - n_m[x])
            v_channels = npr.permutation(v_channels)
            r_channels = np.zeros(self.restchannels)
            r_channels[:n_r[x]] = 1
            node = np.hstack((v_channels, r_channels))
            self.nodes[x, :] = node

    def birth(self):
        """
        Simple birth process
        :return:
        """
        # newnodes = np.empty(self.nodes.shape,
        #                     dtype=self.nodes.dtype)  # this is faster, if the interaction is applied to all nodes
        inds = np.arange(self.K)
        for x in 1 + self.x:
            node = self.nodes[x, :]
            n = node.sum()
            if n == 0 or n == self.K:  # no growth on full or empty nodes
                continue

            dn = npr.binomial(n, self.r_b)
            n = min([n + dn, self.K])
            if n == self.K:
                self.nodes[x, :] = np.ones(len(self.nodes[x, :]))
                continue

            while dn > 0:
                p = 1. - node
                Z = p.sum()
                p /= Z
                ind = npr.choice(inds, p=p)
                node[ind] = 1
                dn -= 1

            self.nodes[x, :] = node

    def random_walk(self):
        """
        Perform a random walk. Giant speed-up by use of numpy function shuffle.
        :return:
        """
        self.nodes = self.nodes.T
        npr.shuffle(self.nodes)
        self.nodes = self.nodes.T

    def rearrange(self):
        """ Rearrangement step, depends on the "configuration energy"
        :return:
        """
        # newnodes = self.nodes.copy()
        newnodes = np.zeros(self.nodes.shape,
                            dtype=self.nodes.dtype)  # this is faster, if the interaction is applied to all nodes
        for x in self.r_int + self.x:
            node = self.nodes[x, :]
            n = node.sum()
            if n == 0 or n == 2 + self.restchannels:  # full or empty nodes cannot be rearranged!
                newnodes[x, :] = node
                continue

            weights = []
            permutations = list(multiset_permutations(node))
            for p in permutations:  # this can sped up by vectorizing the function config_energy()
                weights.append(self.config_energy(p, x))

            weights = np.asarray(weights)
            ind = npr.choice(len(weights), p=weights / weights.sum())
            newnodes[x, :] = permutations[ind]

        self.nodes = newnodes

    def timestep(self):
        self.birth_death()
        self.phen_change()
        self.interaction()
        self.propagation()
        self.update_dynamic_fields()

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
            self.dens_t[0, ...] = self.cell_density
        for t in range(1, timesteps + 1):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.r_int:-self.r_int]
            if recordN:
                self.n_t[t] = self.cell_density.sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density
            if showprogress:
                update_progress(1.0 * t / timesteps)

    def plot_timeevo(self, density_t, figindex=0, figsize=None, cmap='hot_r'):
        import seaborn as sns
        sns.set_style('white')
        if figsize is None:
            tmax, l = density_t.shape
            x = 10.
            y = min([x * tmax / l, 15.])
            figsize = (x, y)
        fig = plt.figure(num=figindex, figsize=figsize)
        ax = fig.add_subplot(111)
        cmap = cmap_discretize(cmap, 3 + self.restchannels)
        plot = ax.matshow(density_t, interpolation='None', vmin=0, vmax=2 + self.restchannels, cmap=cmap)
        cbar = colorbar_index(ncolors=3 + self.restchannels, cmap=cmap, use_gridspec=True)
        cbar.set_label(r'Particle number $n$')
        plt.xlabel(r'Lattice node $r \, [\varepsilon]$', )
        plt.ylabel(r'Time step $k \, [\tau]$')
        ax.xaxis.set_label_position('top')
        plt.tight_layout()
        return plot


if __name__ == '__main__':
    l = 100
    nodes = np.zeros((l, 4), dtype=np.bool)
    nodes[0, :] = 1
    system = LGCA_1D(bc='reflect', interaction='go_or_grow')
    system.timeevo(timesteps=200, recorddens=True)
    system.plot_timeevo(system.dens_t)
    plt.show()
