import sys

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from numpy import random as npr
from sympy.utilities.iterables import multiset_permutations

pi2 = 2 * np.pi
plt.style.use('default')


def update_progress(progress):
    """
    Simple progress bar update.
    :param progress: float. Fraction of the work done, to update bar.
    :return:
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rProgress: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 1),
                                               status)
    sys.stdout.write(text)
    sys.stdout.flush()


def colorbar_index(ncolors, cmap, use_gridspec=False):
    """Return a discrete colormap with n colors from the continuous colormap cmap and add correct tick labels

    :param ncolors: number of colors of the colormap
    :param cmap: continuous colormap to create discrete colormap from
    :param use_gridspec: optional, use option for colorbar
    :return: colormap instance
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    colorbar = plt.colorbar(mappable, use_gridspec=use_gridspec)
    colorbar.set_ticks(np.linspace(-0.5, ncolors + 0.5, 2 * ncolors + 1)[1::2])
    colorbar.set_ticklabels(list(range(ncolors)))
    return colorbar


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

        Example
            x = resize(arange(100), (5,100))
            djet = cmap_discretize(cm.jet, 5)
           imshow(x, cmap=djet)
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]

    return plt.matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def estimate_figsize(array, x=8., cbar=False, dy=1.):
    lx, ly = array.shape
    if cbar:
        y = min([x * ly / lx - 1, 15.])
    else:
        y = min([x * ly / lx, 15.])  #
    y *= dy
    figsize = (x, y)
    return figsize


def calc_nematic_tensor(v):
    return np.einsum('...i,...j->...ij', v, v) - 0.5 * np.diag(np.ones(2))[None, ...]


class LGCA_base():
    """
    Base class for a lattice-gas. Not meant to be used alone!
    """
    interactions = [
        'This is only a helper class, it cannot simulate! Use one the following classes: \n LGCA_1D, LGCA_SQUARE, LGCA_HEX']

    def __init__(self, nodes=None, dims=None, restchannels=0, density=0.1, bc='periodic', **kwargs):
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
        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        self.set_bc(bc)
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        self.init_nodes(density, nodes=nodes)
        self.init_coords()
        self.set_interaction(**kwargs)
        self.cell_density = self.nodes.sum(-1)
        self.apply_boundaries()

    def set_r_int(self, r):
        self.r_int = r
        self.init_nodes(nodes=self.nodes[self.nonborder])
        self.init_coords()
        self.update_dynamic_fields()

    def set_interaction(self, **kwargs):
        from .interactions import go_or_grow, birth, alignment, persistent_walk, chemotaxis, \
            contact_guidance, nematic, aggregation, wetting, random_walk, birth_death, excitable_medium
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction == 'go_or_grow':
                self.interaction = go_or_grow
                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.01
                    print('death rate set to r_d = ', self.r_d)
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)
                if 'kappa' in kwargs:
                    self.kappa = kwargs['kappa']
                else:
                    self.kappa = 5.
                    print('switch rate set to kappa = ', self.kappa)
                if 'theta' in kwargs:
                    self.theta = kwargs['theta']
                else:
                    self.theta = 0.75
                    print('switch threshold set to theta = ', self.theta)
                if self.restchannels < 2:
                    print('WARNING: not enough rest channels - system will die out!!!')

            elif interaction == 'go_and_grow':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

            elif interaction == 'alignment':
                self.interaction = alignment
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'persistent_motion':
                self.interaction = persistent_walk
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'chemotaxis':
                self.interaction = chemotaxis
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 5.
                    print('sensitivity set to beta = ', self.beta)

                if 'gradient' in kwargs:
                    self.g = kwargs['gradient']
                else:
                    if self.velocitychannels > 2:
                        x_source = npr.normal(self.xcoords.mean(), 1)
                        y_source = npr.normal(self.ycoords.mean(), 1)
                        rx = self.xcoords - x_source
                        ry = self.ycoords - y_source
                        r = np.sqrt(rx ** 2 + ry ** 2)
                        self.concentration = np.exp(-2 * r / self.ly)
                        self.g = self.gradient(np.pad(self.concentration, 1, 'constant'))
                    else:
                        source = npr.normal(self.l / 2, 1)
                        r = abs(self.xcoords - source)
                        self.concentration = np.exp(-2 * r / self.l)
                        self.g = self.gradient(np.pad(self.concentration, 1, 'constant'))
                        self.g /= self.g.max()

            elif interaction == 'contact_guidance':
                self.interaction = contact_guidance
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

                if 'director' in kwargs:
                    self.g = kwargs['director']
                else:
                    self.g = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, 2))
                    self.g[..., 0] = 1
                    self.guiding_tensor = calc_nematic_tensor(self.g)
                if self.velocitychannels < 4:
                    print('WARNING: NEMATIC INTERACTION UNDEFINED IN 1D!')

            elif interaction == 'nematic':
                self.interaction = nematic
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'aggregation':
                self.interaction = aggregation
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'wetting':
                self.interaction = wetting
                self.calc_permutations()
                self.set_r_int(2)

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('adhesion sensitivity set to beta = ', self.beta)

                if 'gamma' in kwargs:
                    self.gamma = kwargs['gamma']
                else:
                    self.gamma = 2.
                    print('alignment sensitivity set to gamma = ', self.gamma)

                if 'alpha' in kwargs:
                    self.alpha = kwargs['alpha']
                else:
                    self.alpha = 2.
                    print('substrate sensitivity set to alpha = ', self.alpha)

            elif interaction == 'random_walk':
                self.interaction = random_walk

            elif interaction == 'birth':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

            elif interaction == 'birth_death':
                self.interaction = birth_death
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.05
                    print('death rate set to r_d = ', self.r_d)

            elif interaction == 'excitable_medium':
                self.interaction = excitable_medium
                if 'beta' in kwargs:
                    self.beta = kwargs['beta']

                else:
                    self.beta = .05
                    print('alignment sensitivity set to beta = ', self.beta)

                if 'alpha' in kwargs:
                    self.alpha = kwargs['alpha']
                else:
                    self.alpha = 1.
                    print('aggregation sensitivity set to alpha = ', self.alpha)

                if 'N' in kwargs:
                    self.N = kwargs['N']
                else:
                    self.N = 50
                    print('repetition of fast reaction set to N = ', self.N)

            else:
                print('interaction', kwargs['interaction'], 'is not defined! Random walk used instead.')
                print('Implemented interactions:', self.interactions)
                self.interaction = random_walk

        else:
            print('Random walk interaction is used.')
            self.interaction = random_walk

    def set_bc(self, bc):
        if bc in ['absorbing', 'absorb', 'abs']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl']:
            self.apply_boundaries = self.apply_rbc
        elif bc in ['periodic', 'pbc']:
            self.apply_boundaries = self.apply_pbc
        else:
            print(bc, 'not defined, using periodic boundaries')
            self.apply_boundaries = self.apply_pbc

    def calc_flux(self, nodes):
        return np.einsum('ij,...j', self.c, nodes[..., :self.velocitychannels])

    def get_interactions(self):
        print(self.interactions)

    def random_reset(self, density):
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
        self.cell_density = self.nodes.sum(-1)

    def timestep(self):
        """
        Update the state of the LGCA from time k to k+1.
        :return:
        """
        self.interaction(self)
        self.apply_boundaries()
        self.propagation()
        self.apply_boundaries()
        self.update_dynamic_fields()

    def calc_permutations(self):
        self.permutations = [np.array(list(multiset_permutations([1] * n + [0] * (self.K - n))), dtype=np.int8)
                             for n in range(self.K + 1)]
        self.j = [np.dot(self.c, self.permutations[n][:, :self.velocitychannels].T) for n in range(self.K + 1)]
        self.cij = np.einsum('ij,kj->jik', self.c, self.c) - 0.5 * np.diag(np.ones(2))[None, ...]
        self.si = [np.einsum('ij,jkl', self.permutations[n][:, :self.velocitychannels], self.cij) for n in
                   range(self.K + 1)]


class IBLGCA_base(LGCA_base):
    """
    Base class for identity-based LGCA.
    """
    props = {}

    def set_interaction(self, **kwargs):
        from .ib_interactions import birth, birthdeath, go_or_grow_interaction
        from .interactions import random_walk
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction is 'birth':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)
                self.props.update(r_b=[0.] + [self.r_b] * self.maxlabel)

            if interaction is 'birthdeath':
                self.interaction = birthdeath
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)
                self.props.update(r_b=[0.] + [self.r_b] * self.maxlabel)
                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.02
                    print('death rate set to r_d = ', self.r_d)

                if 'std' in kwargs:
                    self.std = kwargs['std']
                else:
                    self.std = 0.1
                    print('standard deviation set to = ', self.std)

            if interaction is 'go_or_grow':
                self.interaction = go_or_grow_interaction
                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.01
                    print('death rate set to r_d = ', self.r_d)
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)
                if 'kappa' in kwargs:
                    self.kappa = kwargs['kappa']
                else:
                    self.kappa = 5.
                    print('switch rate set to kappa = ', self.kappa)
                self.props.update(kappa=[0.] + [self.kappa] * self.maxlabel)
                if 'theta' in kwargs:
                    self.theta = kwargs['theta']
                else:
                    self.theta = 0.75
                    print('switch threshold set to theta = ', self.theta)
                if self.restchannels < 2:
                    print('WARNING: not enough rest channels - system will die out!!!')

            elif interaction is 'go_and_grow':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

                self.props.update(r_b=[0.] + [self.r_b] * self.maxlabel)

            elif interaction is 'random_walk':
                self.interaction = random_walk

            else:
                print('keyword', interaction, 'is not defined! Random walk used instead.')
                self.interaction = random_walk

        else:
            self.interaction = random_walk

    def update_dynamic_fields(self):
        """Update "fields" that store important variables to compute other dynamic steps

        :return:
        """
        self.occupied = self.nodes > 0
        self.cell_density = self.occupied.sum(-1)

    def convert_bool_to_ib(self, occ):
        occ = occ.astype(np.uint)
        occ[occ > 0] = 1 + np.arange((occ.sum()), dtype=np.uint)
        return occ

    def random_reset(self, density):
        """

        :param density:
        :return:
        """
        occupied = npr.random(self.nonborder.shape) < density
        self.nodes[self.nonborder] = self.convert_bool_to_ib(occupied)
        self.apply_boundaries()
        self.maxlabel = self.nodes.max()
        self.update_dynamic_fields()