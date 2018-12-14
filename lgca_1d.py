from __future__ import division

from tools import *


class LGCA_1D(LGCA):
    """
    1D version of an LGCA.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation', 'parameter_controlled_diffusion',
                    'random_walk']
    velocitychannels = 2
    c = np.array([1., -1.])

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
                if self.restchannels < 2:
                    print 'WARNING: not enough rest channels - system will die out!!!'

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

            elif kwargs['interaction'] is 'parameter_controlled_diffusion':
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
        newnodes = np.zeros(self.nodes.shape, dtype=self.nodes.dtype)
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
            n = self.cell_density[x]
            if n == 0 or n == self.K:  # no growth on full or empty nodes
                continue

            N = min([n, self.K - n])
            dn = npr.binomial(N, self.r_b)
            n += dn
            if n == self.K:
                self.nodes[x, :] = 1
                continue

            while dn > 0:
                p = 1. - node
                Z = p.sum()
                p /= Z
                ind = npr.choice(inds, p=p)
                node[ind] = 1
                dn -= 1

            self.nodes[x, :] = node

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
            n = self.cell_density[x]
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
            n = self.cell_density[x]
            if n == 0 or n == 2 + self.restchannels:  # full or empty nodes cannot be rearranged!
                newnodes[x, :] = node
                continue

            permutations = np.array(list(multiset_permutations(node)), dtype=int)
            weights = np.exp(self.beta * (permutations[:, 2:] > 0).sum(-1))
            ind = npr.choice(len(weights), p=weights / weights.sum())
            newnodes[x, :] = permutations[ind]

        self.nodes = newnodes

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

    def plot_density(self, density_t=None, figindex=0, figsize=None, cmap='hot_r'):
        # import seaborn as sns
        # sns.set_style('white')
        if density_t is None:
            density_t = self.dens_t
        if figsize is None:
            tmax, l = density_t.shape
            x = 8.
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
        return plot

    def plot_flux(self, nodes_t=None, figindex=0, figsize=None):
        # import seaborn as sns
        #sns.set_style('white')
        if nodes_t is None:
            nodes_t = self.nodes_t

        dens_t = nodes_t.sum(-1) / nodes_t.shape[-1]
        tmax, l = dens_t.shape
        flux_t = nodes_t[..., 0].astype(int) - nodes_t[..., 1].astype(int)
        if figsize is None:
            figsize = estimate_figsize(nodes_t)

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
        plt.tight_layout()
        return plot


class IBLGCA_1D(LGCA_1D):
    """
    1D version of an identity-based LGCA.
    """

    def __init__(self, nodes=None, properties=None, l=100, restchannels=2, density=0.1, bc='periodic', r_int=1,
                 **kwargs):
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
        self.maxlabel = 0  # maximum cell label
        if nodes is None:
            self.l = l
            self.restchannels = restchannels
            self.nodes = np.zeros((l + 2 * self.r_int, 2 + self.restchannels), dtype=np.uint)
            occupied = npr.random(self.nodes.shape) < density
            self.nodes[occupied] = 1 + np.arange(len(occupied.flat))
            self.maxlabel = np.amax(self.nodes)
        else:
            assert len(nodes.shape) == 2
            self.l, self.restchannels = nodes.shape
            self.nodes = np.zeros((self.l + 2 * self.r_int, self.restchannels), dtype=np.uint)
            self.nodes[self.r_int:-self.r_int, :] = nodes.astype(np.uint)
            self.restchannels -= 2
            self.maxlabel = np.amax(self.nodes)

        # "properties is a dict of cell properties, e.g. props['r_b'] returns a list of all birth rates
        if properties is None:
            self.props = {}

        else:
            self.props = properties

        self.x = np.arange(self.l) + self.r_int
        self.K = 2 + self.restchannels

        if bc in ['absorbing', 'absorb', 'abs']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl']:
            self.apply_boundaries = self.apply_rbc
        else:
            self.apply_boundaries = self.apply_pbc
            self.apply_boundaries()

        # set birth_death process
        if 'birthdeath' in kwargs:
            if kwargs['birthdeath'] is 'birth':
                self.birth_death = self.birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print 'birth rate set to r_b = ', self.r_b
                self.props.update(r_b=[0.] + [self.r_b] * self.maxlabel)

            if kwargs['birthdeath'] is 'birthdeath':
                self.birth_death = self.birthdeath
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print 'birth rate set to r_b = ', self.r_b
                self.props.update(r_b=[0.] + [self.r_b] * self.maxlabel)
                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_b']
                else:
                    self.r_d = 0.02
                    print 'death rate set to r_d = ', self.r_d

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

        self.interactions = ['go_or_grow', 'go_and_grow', 'random_walk']
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
                self.props.update(kappa=[0.] + [self.kappa] * self.maxlabel)
                if 'theta' in kwargs:
                    self.theta = kwargs['theta']
                else:
                    self.theta = 0.75
                    print 'switch threshold set to theta = ', self.theta
                if self.restchannels < 2:
                    print 'WARNING: not enough rest channels - system will die out!!!'

            elif kwargs['interaction'] is 'go_and_grow':
                self.birth_death = self.birth
                self.interaction = self.random_walk
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print 'birth rate set to r_b = ', self.r_b

                self.props.update(r_b=[0.] + [self.r_b] * self.maxlabel)

            elif kwargs['interaction'] is 'random_walk':
                self.interaction = self.random_walk

            else:
                print 'keyword', kwargs['interaction'], 'is not defined! Random walk used instead.'
                self.interaction = self.random_walk

        else:
            self.interaction = self.random_walk

        self.occupied = self.nodes.astype(np.bool)
        self.cell_density = self.occupied.sum(-1)

    def update_dynamic_fields(self):
        """Update "fields" that store important variables to compute other dynamic steps

        :return:
        """
        self.occupied = self.nodes.astype(np.bool)
        self.cell_density = self.occupied.sum(-1)

    def birth(self):
        """
        Simple birth process
        :return:
        """
        inds = np.arange(self.K)
        for x in self.x:
            n = self.cell_density[x]
            node = self.nodes[x]
            if n == 0 or n == self.K:  # no growth on full or empty nodes
                continue

            # choose cells that proliferate
            r_bs = [self.props['r_b'][i] for i in node]
            proliferating = npr.random(self.K) < r_bs
            dn = proliferating.sum()
            n += dn
            # assure that there are not too many cells. if there are, randomly kick enough of them
            while n > self.K:
                p = proliferating.astype(float)
                Z = p.sum()
                p /= Z
                ind = npr.choice(inds, p=p)
                proliferating[ind] = 0
                n -= 1

            # distribute daughter cells randomly in channels
            dn = proliferating.sum()
            for label in node[proliferating]:
                p = 1. - self.occupied[x]
                Z = p.sum()
                p /= Z
                ind = npr.choice(inds, p=p)
                self.maxlabel += 1
                node[ind] = self.maxlabel
                r_b = self.props['r_b'][label]
                self.props['r_b'].append(npr.normal(loc=r_b, scale=0.2 * r_b))
                dn -= 1

            self.nodes[x, :] = node

    def birthdeath(self):
        """
        Simple birth-death process with evolutionary dynamics towards a higher proliferation rate
        :return:
        """
        inds = np.arange(self.K)
        # death process
        dying = npr.random(self.nodes.shape) < self.r_d
        self.nodes[dying] = 0
        # birth
        self.update_dynamic_fields()
        for x in self.x:
            n = self.cell_density[x]
            node = self.nodes[x]
            if n == 0:  # no growth on empty nodes
                continue

            # choose cells that proliferate
            r_bs = [self.props['r_b'][i] for i in node]
            proliferating = npr.random(self.K) < r_bs
            dn = proliferating.sum()
            n += dn
            # assure that there are not too many cells. if there are, randomly kick enough of them
            while n > self.K:
                p = proliferating.astype(float)
                Z = p.sum()
                p /= Z
                ind = npr.choice(inds, p=p)
                proliferating[ind] = 0
                n -= 1

            # distribute daughter cells randomly in channels
            dn = proliferating.sum()
            for label in node[proliferating]:
                p = 1. - self.occupied[x]
                Z = p.sum()
                p /= Z
                ind = npr.choice(inds, p=p)
                self.maxlabel += 1
                node[ind] = self.maxlabel
                r_b = self.props['r_b'][label]
                self.props['r_b'].append(npr.normal(loc=r_b, scale=0.2 * r_b))
                dn -= 1

            self.nodes[x, :] = node

    def go_or_grow_interaction(self):
        """
        interactions of the go-or-grow model. formulation too complex for 1d, but to be generalized.
        :return:
        """
        # death
        dying = npr.random(self.nodes.shape) < self.r_d
        self.nodes[dying] = 0
        # birth
        self.update_dynamic_fields()
        n_m = self.occupied[:, :2].sum(-1)
        n_r = self.occupied[:, 2:].sum(-1)
        for x in self.x:
            node = self.nodes[x]
            vel = node[:2]
            rest = node[2:]
            n = self.cell_density[x]
            if n == 0 or n == self.K:
                continue

            rho = n / self.K
            # determine cells to switch to rest channels and cells that switch to moving state
            # kappas = np.array([self.props['kappa'][i] for i in node])
            # r_s = tanh_switch(rho, kappa=kappas, theta=self.theta)

            free_rest = self.restchannels - n_r[x]
            free_vel = 2 - n_m[x]
            # choose a number of cells that try to switch. the cell number must fit to the number of free channels
            can_switch_to_rest = npr.permutation(vel[vel > 0])[:free_rest]
            can_switch_to_vel = npr.permutation(rest[rest > 0])[:free_vel]

            for cell in can_switch_to_rest:
                if npr.random() < tanh_switch(rho, kappa=self.props['kappa'][cell], theta=self.theta):
                    # print 'switch to rest', cell
                    rest[np.where(rest == 0)[0][0]] = cell
                    vel[np.where(vel == cell)[0][0]] = 0

            for cell in can_switch_to_vel:
                if npr.random() < 1 - tanh_switch(rho, kappa=self.props['kappa'][cell], theta=self.theta):
                    # print 'switch to vel', cell
                    vel[np.where(vel == 0)[0][0]] = cell
                    rest[np.where(rest == cell)[0][0]] = 0

            # cells in rest channels can proliferate
            can_proliferate = npr.permutation(rest[rest > 0])[:(rest == 0).sum()]
            for cell in can_proliferate:
                if npr.random() < self.r_b:
                    self.maxlabel += 1
                    rest[np.where(rest == 0)[0][0]] = self.maxlabel
                    kappa = self.props['kappa'][cell]
                    self.props['kappa'].append(npr.normal(loc=kappa))

            v_channels = npr.permutation(vel)
            r_channels = npr.permutation(rest)
            node = np.hstack((v_channels, r_channels))
            self.nodes[x, :] = node

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros((timesteps + 1, self.l, 2 + self.restchannels), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[self.r_int:-self.r_int, ...]
            self.props_t = [self.props]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.nodes.sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1, self.l))
            self.dens_t[0, ...] = self.cell_density[self.r_int:-self.r_int]
        for t in range(1, timesteps + 1):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.r_int:-self.r_int]
                self.props_t.append(self.props)
            if recordN:
                self.n_t[t] = self.cell_density.sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.r_int:-self.r_int]
            if showprogress:
                update_progress(1.0 * t / timesteps)

    def plot_prop_spatial(self, nodes_t=None, props_t=None, figindex=0, figsize=None, prop=None, cmap='cividis'):
        import seaborn as sns
        sns.set_style('white')
        import matplotlib.ticker as mticker
        if nodes_t is None:
            nodes_t = self.nodes_t
        if figsize is None:
            figsize = estimate_figsize(nodes_t.sum(-1))

        if props_t is None:
            props_t = self.props_t

        if prop is None:
            prop = props_t[0].keys()[0]

        tmax, l, _ = nodes_t.shape
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
        fig = plt.figure(num=figindex, figsize=figsize, tight_layout=True)
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

    def plot_prop_timecourse(self, nodes_t=None, props_t=None, propname=None):
        import seaborn as sns
        sns.set_style('white')
        if nodes_t is None:
            nodes_t = self.nodes_t

        if props_t is None:
            props_t = self.props_t

        if propname is None:
            propname = props_t[0].keys()[0]

        tmax = len(props_t)
        fig = plt.figure()
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
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, antialiased=True)
        sns.despine()
        return



if __name__ == '__main__':
    l = 100
    restchannels = 2
    n_channels = restchannels + 2
    nodes = 1 + np.arange(l * n_channels, dtype=np.uint).reshape((l, n_channels))
    # nodes[1:, :] = 0
    # nodes[0, 1:] = 0

    system = IBLGCA_1D(nodes=nodes, bc='reflect', interaction='go_or_grow', kappa=4., r_d=0.01, r_b=.2, theta=0.5)
    system.timeevo(timesteps=100, record=True)
    #system.plot_prop()
    system.plot_density(figindex=1)
    props = np.array(system.props['kappa'])[system.nodes[system.nodes > 0]]
    print np.mean(props)
    system.plot_prop_timecourse()
    plt.ylabel('$\kappa$')
    plt.show()
