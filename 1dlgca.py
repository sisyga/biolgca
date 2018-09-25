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

    def __init__(self, nodes=None, l=100, restchannels=2, density=0.1, bc='periodic', r_int=1, **kwargs):
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
        self.dens_t, self.nodes_t, self.n_t = np.empty(3)  # placeholders to record dynamics
        assert r_int > 0
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
        self.x = np.arange(self.l) + self.r_int
        self.K = 2 + self.restchannels

        if bc in ['absorbing', 'absorb', 'abs']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl']:
            self.apply_boundaries = self.apply_rbc
        else:
            self.apply_boundaries = self.apply_pbc
            self.apply_boundaries()

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
                    print 'birth rate set to r_b = ', self.r_b
            elif kwargs['birthdeath'] is 'none':
                def birth_death():
                    pass

                self.birth_death = birth_death
            else:
                print 'keyword', kwargs['birthdeath'], 'is not defined!'

                def birth_death():
                    pass

                self.birth_death = birth_death
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
                    print 'death rate set to r_d = ', self.r_d
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print 'birth rate set to r_b = ', self.r_b
                if 'kappa' in kwargs:
                    self.kappa = kwargs['kappa']
                else:
                    self.kappa = 5.
                    print 'switch rate set to kappa = ', self.kappa
                if 'theta' in kwargs:
                    self.theta = kwargs['theta']
                else:
                    self.theta = 0.75
                    print 'switch threshold set to theta = ', self.theta

            elif kwargs['interaction'] is 'go_and_grow':
                self.birth_death = self.birth
                self.interaction = self.random_walk
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print 'birth rate set to r_b = ', self.r_b

            elif kwargs['interaction'] is 'alignment':
                self.interaction = self.alignment
                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print 'sensitivity set to beta = ', self.beta

            elif kwargs['interaction'] is 'aggregation':
                self.interaction = self.aggregation
                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print 'sensitivity set to beta = ', self.beta

            elif kwargs['interaction'] is 'parameter_diffusion':
                self.interaction = self.parameter_diffusion
                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print 'sensitivity set to beta = ', self.beta

            elif kwargs['interaction'] is 'random_walk':
                self.interaction = self.random_walk

            else:
                print 'keyword', kwargs['interaction'], 'is not defined! Random walk used instead.'
                self.interaction = self.random_walk

        else:
            self.interaction = self.random_walk

        self.cell_density = self.nodes[self.r_int:-self.r_int].sum(-1)


    def propagation(self):
        """

        :param nodes:
        :return:
        """
        newnodes = np.zeros(self.nodes.shape, dtype=nodes.dtype)
        # self.clean_boundaries(newnodes)
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

    def reset_config(self, density):
        """

        :param density:
        :return:
        """
        self.nodes = npr.random(self.nodes.shape) < density
        self.update_dynamic_fields()

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
        for x in self.x:
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
        for x in self.x:
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
        Perform a random walk. Giant speed-up by use of numpy function shuffle, which performs a permutation along
        the first axis (therefore we need the .T on the nodes)
        :return:
        """
        self.nodes = self.nodes.T
        npr.shuffle(self.nodes)
        self.nodes = self.nodes.T

    def alignment(self):
        """
        Rearrangement step for alignment interaction
        :return:
        """
        newnodes = np.zeros(self.nodes.shape,
                            dtype=self.nodes.dtype)  # this is faster, if the interaction is applied to all nodes
        for x in self.x:
            node = self.nodes[x, :]
            n = node.sum()
            if n == 0 or n == 2 + self.restchannels:  # full or empty nodes cannot be rearranged!
                newnodes[x, :] = node
                continue

            G = int(self.nodes[x + 1, 0]) + int(self.nodes[x - 1, 0]) - int(self.nodes[x + 1, 1]) - int(
                self.nodes[x - 1, 1])
            permutations = np.array(list(multiset_permutations(node)), dtype=int)
            Js = permutations[:, 0] - permutations[:, 1]
            weights = np.exp(self.beta * G * Js)
            ind = npr.choice(len(weights), p=weights / weights.sum())
            newnodes[x, :] = permutations[ind]

        self.nodes = newnodes

    def aggregation(self):
        """
        Rearrangement step for alignment interaction
        :return:
        """
        newnodes = np.zeros(self.nodes.shape,
                            dtype=self.nodes.dtype)  # this is faster, if the interaction is applied to all nodes
        for x in self.x:
            node = self.nodes[x, :]
            n = node.sum()
            if n == 0 or n == 2 + self.restchannels:  # full or empty nodes cannot be rearranged!
                newnodes[x, :] = node
                continue

            G = self.nodes[x + 1].sum() - self.nodes[x - 1].sum()
            permutations = np.array(list(multiset_permutations(node)), dtype=int)
            Js = permutations[:, 0] - permutations[:, 1]
            weights = np.exp(self.beta * G * Js)
            ind = npr.choice(len(weights), p=weights / weights.sum())
            newnodes[x, :] = permutations[ind]

        self.nodes = newnodes

    def parameter_diffusion(self):
        """
        Rearrangement step for alignment interaction
        :return:
        """
        newnodes = np.zeros(self.nodes.shape,
                            dtype=self.nodes.dtype)  # this is faster, if the interaction is applied to all nodes
        for x in self.x:
            node = self.nodes[x, :]
            n = node.sum()
            if n == 0 or n == 2 + self.restchannels:  # full or empty nodes cannot be rearranged!
                newnodes[x, :] = node
                continue

            permutations = np.array(list(multiset_permutations(node)), dtype=int)
            weights = np.exp(self.beta * permutations[:, 2:].sum(-1))
            ind = npr.choice(len(weights), p=weights / weights.sum())
            newnodes[x, :] = permutations[ind]

        self.nodes = newnodes

    def timestep(self):
        self.birth_death()
        self.apply_boundaries()
        # self.phen_change()
        self.interaction()
        self.apply_boundaries()
        self.propagation()
        self.apply_boundaries()
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
    l = 10
    nodes = np.zeros((l, 3), dtype=np.bool)
    nodes[0, 0] = 1
    system = LGCA_1D(nodes=nodes, bc='reflect', interaction='parameter_diffusion', beta=.4)
    system.timeevo(timesteps=100, recorddens=True)
    system.plot_timeevo(system.dens_t, cmap='viridis')
    plt.show()
