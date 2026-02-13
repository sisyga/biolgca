# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2024 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
LGCA without volume exclusion base class module.
"""

from abc import ABC, abstractmethod
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
from matplotlib import cm
from numpy import random as npr
from sympy.utilities.iterables import multiset_permutations
from copy import copy, deepcopy
from lgca.plots import muller_plot
import warnings
from tqdm.auto import tqdm
from lgca.base import LGCA_base

# configure matplotlib style
plt.style.use('default')


class NoVE_LGCA_base(LGCA_base, ABC):
    """
    Base class for LGCA without volume exclusion.
    """
    def __init__(self, nodes=None, dims=None, restchannels=None, density=0.1, hom=None, bc='periodic', seed=None, capacity=None,
                 **kwargs):
        """
        Initialize class instance.
        :param nodes: :py:class:`numpy.ndarray` initial configuration set manually
        :param dims: tuple determining lattice dimensions
        :param restchannels: number of resting channels
        :param density: float, if nodes is None, initialize lattice randomly with this particle density
        :param bc: boundary conditions
        :param r_int: interaction range
        """

        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        self.rng = npr.default_rng(seed=seed)
        self.set_bc(bc)
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes, capacity=capacity)
        self.init_coords()
        self.init_nodes(density=density, nodes=nodes, hom=hom)
        self.update_dynamic_fields()
        self.interaction_params = {}
        self.set_interaction(**kwargs)

    def set_interaction(self, **kwargs):
        from lgca.nove_interactions import dd_alignment, di_alignment, go_or_grow, go_or_rest, random_walk
        from lgca.interactions import only_propagation
        # configure interaction
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction == 'random_walk':
                self.interaction = random_walk
            # density-dependent interaction rule
            elif interaction == 'dd_alignment':
                if self.restchannels > 0:
                    raise RuntimeError("Rest channels ({:d}) defined, interaction will crash! Set number of"
                                       " rest channels to 0 with restchannels keyword.".format(self.restchannels))
                self.interaction = dd_alignment

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])
                if 'include_center' in kwargs:
                    self.interaction_params['nb_include_center'] = kwargs['include_center']
                else:
                    self.interaction_params['nb_include_center'] = False
                    print('neighbourhood set to exclude the central node')
            # density-independent alignment rule
            elif interaction == 'di_alignment':
                if self.restchannels > 0:
                    raise RuntimeError("Rest channels ({:d}) defined, interaction will crash! Set number of"
                                       " rest channels to 0 with restchannels keyword.".format(self.restchannels))
                self.interaction = di_alignment
                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])
                if 'include_center' in kwargs:
                    self.interaction_params['nb_include_center'] = kwargs['include_center']
                else:
                    self.interaction_params['nb_include_center'] = False
                    print('neighbourhood set to exclude the central node')
            elif interaction == 'go_or_grow':
                if self.restchannels < 1:
                    raise RuntimeError("No rest channels ({:d}) defined, interaction cannot be performed! Set number of"
                                       " rest channels with restchannels keyword.".format(self.restchannels))
                self.interaction = go_or_grow
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.01
                    print('death rate set to r_d = ', self.interaction_params['r_d'])
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                if 'kappa' in kwargs:
                    self.interaction_params['kappa'] = kwargs['kappa']
                else:
                    self.interaction_params['kappa'] = 5.
                    print('switch rate set to kappa = ', self.interaction_params['kappa'])
                if 'theta' in kwargs:
                    self.interaction_params['theta'] = kwargs['theta']
                else:
                    self.interaction_params['theta'] = 0.75
                    print('switch threshold set to theta = ', self.interaction_params['theta'])
            elif interaction == 'go_or_rest':
                if self.restchannels < 1:
                    raise RuntimeError(
                        "No rest channels ({:d}) defined, interaction cannot be performed! Set number of rest "
                        "channels with restchannels keyword.".format(
                            self.restchannels))

                self.interaction = go_or_rest
                if 'kappa' in kwargs:
                    self.interaction_params['kappa'] = kwargs['kappa']
                else:
                    self.interaction_params['kappa'] = 5.
                    print('switch rate set to kappa = ', self.interaction_params['kappa'])
                if 'theta' in kwargs:
                    self.interaction_params['theta'] = kwargs['theta']
                else:
                    self.interaction_params['theta'] = 0.75
                    print('switch threshold set to theta = ', self.interaction_params['theta'])

            elif interaction == 'only_propagation':
                self.interaction = only_propagation

            else:
                print('interaction', kwargs['interaction'], 'is not defined! Density-dependent alignment '
                                                            'interaction used instead.')
                print('Implemented interactions:', self.interactions)
                self.interaction = dd_alignment

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])
        # if nothing is specified, use density-dependent interaction rule
        else:
            print('Density-dependent alignment interaction is used.')
            self.interaction = dd_alignment

            if 'beta' in kwargs:
                self.interaction_params['beta'] = kwargs['beta']
            else:
                self.interaction_params['beta'] = 2.
                print('sensitivity set to beta = ', self.interaction_params['beta'])

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True,
                recordorderparams=False, recordpertype=False):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros((timesteps + 1,) + self.dims + (self.K,), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[self.nonborder]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.cell_density[self.nonborder].sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1,) + self.dims)
            self.dens_t[0, ...] = self.cell_density[self.nonborder]
        if recordorderparams:
            self.ent_t = np.zeros(timesteps + 1, dtype=float)
            self.ent_t[0, ...] = self.calc_entropy()
            self.normEnt_t = np.zeros(timesteps + 1, dtype=float)
            self.normEnt_t[0, ...] = self.calc_normalized_entropy()
            self.polAlParam_t = np.zeros(timesteps + 1, dtype=float)
            self.polAlParam_t[0, ...] = self.calc_polar_alignment_parameter()
            self.meanAlign_t = np.zeros(timesteps + 1, dtype=float)
            self.meanAlign_t[0, ...] = self.calc_mean_alignment()
        if recordpertype:
            self.velcells_t = np.zeros((timesteps + 1,) + self.dims)
            self.velcells_t[0, ...] = self.nodes[self.nonborder][..., :self.velocitychannels].sum(-1)
            self.restcells_t = np.zeros((timesteps + 1,) + self.dims)
            self.restcells_t[0, ...] = self.nodes[self.nonborder][..., self.velocitychannels:].sum(-1)
        for t in tqdm(iterable=range(1, timesteps + 1), disable=1-showprogress):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.nonborder]
            if recordN:
                self.n_t[t] = self.cell_density[self.nonborder].sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.nonborder]
            if recordorderparams:
                self.ent_t[t, ...] = self.calc_entropy()
                self.normEnt_t[t, ...] = self.calc_normalized_entropy()
                self.polAlParam_t[t, ...] = self.calc_polar_alignment_parameter()
                self.meanAlign_t[t, ...] = self.calc_mean_alignment()
            if recordpertype:
                self.velcells_t[t, ...] = self.nodes[self.nonborder][..., :self.velocitychannels].sum(-1)
                self.restcells_t[t, ...] = self.nodes[self.nonborder][..., self.velocitychannels:].sum(-1)

    def random_reset(self, density):
        """
        Distribute particles in the lattice according to a given density; can yield different cell numbers per
        lattice site
        :param density: particle density in the lattice: average number of particles per channel
        """

        # sample from a Poisson distribution with mean=density
        density = abs(density)
        draw1 = npr.poisson(lam=density, size=self.nodes.shape)
        if self.capacity > self.K:
            draw2 = npr.poisson(lam=density, size=self.nodes.shape[:-1]+((self.capacity-self.K),))
            draw1[..., -1] += draw2.sum(-1)
        self.nodes = draw1
        self.apply_boundaries()
        self.update_dynamic_fields()
        eff_dens = self.nodes[self.nonborder].sum()/(self.capacity * self.cell_density[self.nonborder].size)
        print("Required density: {:.3f}, Achieved density: {:.3f}".format(density, eff_dens))

    def homogeneous_random_reset(self, density):
        """
        Distribute particles in the lattice homogeneously according to a given density: each lattice site has the same
            number of particles, randomly distributed among the channels
        :param density: particle density in the lattice: number of particles/(dimensions*capacity)
        """
        # find the number of particles per lattice site which is closest to the desired density
        if int(density * self.capacity) == density * self.capacity:
            initcells = int(density * self.capacity)
        else:
            initcells = int(density * self.capacity) + 1
        # distribute calculated number of particles among channels in the lattice
        self.nodes = npr.multinomial(initcells, [1 / self.K] * self.K, size=self.nodes.shape[:-1])
        self.apply_boundaries()
        self.update_dynamic_fields()
        # check result
        eff_dens = self.nodes[self.nonborder].sum() / (self.capacity * self.cell_density[self.nonborder].size)
        print("Required density: {:.3f}, Achieved density: {:.3f}".format(density, eff_dens))

    def calc_entropy(self, base=None):
        """
        Calculate entropy of the lattice.
        :param base: base of the logarithm, defaults to 2
        :return: entropy according to information theory as scalar
        """
        if base is None:
            base = 2
        # calculate relative frequencies, self.cell_density[self.nonborder].size = number of nodes
        _, freq = np.unique(self.cell_density[self.nonborder], return_counts=True)
        freq = freq / self.cell_density[self.nonborder].size
        log_val = np.divide(np.log(freq), np.log(base))
        return -np.multiply(freq, log_val).sum()

    def calc_normalized_entropy(self, base=None):
        """
        Calculate entropy of the lattice normalized to maximal possible entropy.
        :param base: base of the logarithm, defaults to 2
        :return: normalized entropy as scalar
        """
        if base is None:
            base = 2
        # calculate maximal entropy, self.cell_density[self.nonborder].size = number of nodes
        smax = - np.divide(np.log(1/self.cell_density[self.nonborder].size), np.log(base))
        return 1 - self.calc_entropy(base=base)/smax

    def calc_polar_alignment_parameter(self):
        """
        Calculate the polar alignment parameter.
        The polar alignment parameter is a measure for global agreement of particle orientation in the lattice.
        It is calculated as the magnitude of the sum of the velocities of all particles normalized by the number
        of particles.
        :return: Polar alignment parameter of the lattice from 0 (no alignment) to 1 (complete alignment)
        """
        # calculate flux only for non-boundary nodes, result is a flux vector at each node position
        flux = self.calc_flux(self.nodes[self.nonborder])
        # calculate along which axes the lattice needs to be summed up, e.g. axes=(0) for 1D, axes=(0,1) for 2D
        axes = tuple(np.arange(self.c.shape[0]))
        # sum fluxes up accordingly
        flux = np.sum(flux, axis=axes)
        # take Euclidean norm and normalise by number of particles
        return np.linalg.norm(flux, ord=None)/self.cell_density[self.nonborder].sum()

    def calc_mean_alignment(self):
        """
        Calculate the mean alignment measure.
        The mean alignment is a measure for local alignment of particle orientation in the lattice.
        It is calculated as the agreement in direction between the ﬂux of a lattice site and the ﬂux of the director
        field
        summed up and normalized over all lattice sites.
        .. note:: This is buggy!
        :return: Local alignment parameter: ranging from -1 (antiparallel alignment) through 0 (no alignment)
        to 1 (parallel alignment)
        """
        # Calculate the director field
        flux = self.calc_flux(self.nodes)
        # # retrieve number of particles and reshape to combine with flux
        norm_factor = np.where(self.cell_density > 0, self.cell_density, 1)
        norm_factor = 1 / norm_factor
        norm_factor = norm_factor.reshape(norm_factor.shape + (1,))
        norm_factor = np.broadcast_to(norm_factor, flux.shape)
        # # normalise flux at each node with number of cells in the node
        dir_field = np.multiply(flux, norm_factor)  # max element value: 1
        # # apply boundary conditions -
        # #  (not clean, but this is the only application of applying bc to anything but nodes so far)
        temp = self.nodes
        self.nodes = dir_field
        self.apply_boundaries()
        dir_field = self.nodes
        self.nodes = temp
        # # sum fluxes over neighbours
        dir_field = self.nb_sum(dir_field)  # max element value: no. of neighbours

        # Calculate agreement between node flux and director field flux
        alignment = np.einsum('...j,...j', dir_field, flux)

        # Average over lattice
        # # also normalise director field by no. of neighbours retrospectively -
        # #  (computation on less elements if done here)
        no_neighbours = self.c.shape[-1]
        N = self.cell_density[self.nonborder].sum()
        return alignment[self.nonborder].sum() / (no_neighbours * N)

# create a numpy universal function (ufunc) of the python function 'list'. Can be used to create an numpy array of
# empty lists if applied to an empty array
ufunclist = np.frompyfunc(list, 0, 1)

def get_arr_of_empty_lists(dims):
    """
    Create a numpy array of dimensions 'dims' that is filled with empty lists.

    Parameters
    ----------
    dims: tuple, corresponding to shape of the array.

    Returns
    -------
    :py:class:`numpy.ndarray`

    """
    return ufunclist(np.empty(dims, dtype=object))


