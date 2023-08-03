# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
Abstract base classes. These classes define properties and structure of the LGCA
types/subclasses and specify geometry-independent LGCA behavior.
They cannot be used to simulate.

Supported LGCA types:

- classical LGCA (:py:class:`LGCA_base`)
- identity-based LGCA (:py:class:`IBLGCA_base`)
- LGCA without volume exclusion (:py:class:`NoVE_LGCA_base`)
- identity-based LGCA without volume exclusion (:py:class:`NoVE_IBLGCA_base`)
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

# configure matplotlib style
plt.style.use('default')



def colorbar_index(ncolors: int, cmap, use_gridspec: bool=False, cax=None):
    """
    Create a colorbar with `ncolors` colors.

    Builds a discrete colormap with `ncolors` colors from the near-continuous colormap `cmap`,
    adds it to the axis `cax` and draws tick labels in the center of each color. If
    ncolors is high, some labels are omitted to avoid cluttering.

    .. note:: To Do: Implement the label stride with Locator and Formatter instead.

    Parameters
    ----------
    ncolors : int
        Desired number of colors for the discretized colormap.
    cmap : str or :py:class:`matplotlib.colors.Colormap`
        Near-continuous colormap to create discrete colormap from, e.g. ``matplotlib.cm.jet`` or ``'jet'``.
    use_gridspec : bool, optional
        Passed on to :py:func:`matplotlib.pyplot.colorbar`.
    cax : :py:class:`matplotlib.axes.Axes` object, optional
        Axis into which the colorbar will be drawn.

    Returns
    -------
    colorbar : :py:class:`matplotlib.colorbar.Colorbar`
        Colorbar instance.
    """
    # discretize the colormap
    cmap = cmap_discretize(cmap, ncolors)
    # stride the colorbar labels to avoid cluttering for many colors
    if ncolors > 100:
        stride = 10
    elif ncolors > 50:
        stride = 5
    elif ncolors > 31:
        stride = 2
    else:
        stride = 1
    # map colors to values
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    # create colorbar
    colorbar = plt.colorbar(mappable, use_gridspec=use_gridspec, cax=cax)
    # set ticklabels to the center of respective color and support label stride
    ticks = np.linspace(-0.5, ncolors + 0.5, 2 * ncolors + 1)[1::2]
    labels = list(range(ncolors))
    # if last strided label is the maximum label, plot all strided labels
    if ticks[-1] == ticks[0::stride][-1]:
        colorbar.set_ticks(ticks[0::stride])
        colorbar.set_ticklabels(labels[0::stride])
    # if last strided label is different from the maximum label by less than half the stride:
    # only plot strided labels up to the second last and the maximum label
    elif stride > 1 and ticks[-1] != ticks[0::stride][-1] and ticks[-1] - ticks[0::stride][-1] < stride/2:
        colorbar.set_ticks(list(ticks[0::stride][:-1]) + [ticks[-1]])
        colorbar.set_ticklabels(labels[0::stride][:-1] + [labels[-1]])
    # otherwise plot all strided labels and the maximum label
    else:
        colorbar.set_ticks(list(ticks[0::stride]) + [ticks[-1]])
        colorbar.set_ticklabels(labels[0::stride] + [labels[-1]])
    return colorbar


def cmap_discretize(cmap, N: int):
    """
    Downsample the near-continuous colormap `cmap` to the number of colors `N`.

    Parameters
    ----------
    cmap : str or :py:class:`matplotlib.colors.Colormap`
        Colormap to be discretized, e.g. ``matplotlib.cm.jet`` or ``'jet'``.
    N : int
        Number of colors of the new colormap.

    Returns
    -------
    :py:class:`matplotlib.colors.LinearSegmentedColormap`
        Discretized colormap with `N` colors.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> x = np.resize(np.arange(100), (5,20))
    >>> # discretize jet colormap
    >>> djet = cmap_discretize(cm.jet, 5)
    >>> # show color limits
    >>> plt.imshow(x, cmap=djet)

    The name of the colormap is updated to ``cmap.name + '_N'``:

    >>> djet.name
    'jet_5'

    """
    # see https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#creating-linear-segmented-colormaps
    # for details
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    # create anchor points and fill them with colors from the colormap
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    # index rgba values according to discretization
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # create new linear segmented colormap
    return plt.matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def estimate_figsize(array, x: float=8., cbar: bool=False, dy: float=1.):
    """
    .. deprecated:: 1.0
        :py:func:`estimate_figsize` will be removed in biolgca 1.0, it is replaced
        by the default value for the figure size in :py:func:`setup_figure` of the
        respective LGCA object.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`
        Array holding the data to be plotted.
    x : float, default=8.0
        Desired x dimension of the figure. Used to scale the y dimension.
    cbar : bool, optional
        If the figure will contain a colorbar.
    dy : float, default=1.0
        Scale of a unit in the y direction as compared to the x direction.

    Returns
    -------
    figsize : tuple(float, float)
        Optimal figure size.

    """
    lx, ly = array.shape
    if cbar:
        y = min([abs(x * ly /lx - 1), 10.])
    else:
        y = min([x * ly / lx, 10.])
    y *= dy
    figsize = (x, y)
    return figsize


def get_cmap(density, vmax=None, cmap='viridis', cbar=True, cbarlabel=''):
    if vmax is None:
        K = int(density.max())
    else:
        K = vmax

    cmap = copy(cm.get_cmap(cmap))  # do not modify a globally registered colormap in matplotlib > 3.3.2
    cmap.set_under(alpha=0.0)
    cmap_scaled = False

    if 1 < K <= cmap.N:
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
    elif K > 1:
        cmap_scaled = True
        scaling_factor = K / cmap.N
        nbins = cmap.N
        density = density / scaling_factor
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(cmap.N + 1), cmap.N))
    else:
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=1e-6, vmax=1))
    cmap.set_array(density)

    if not cbar:
        return cmap

    if K <= 1:
        # requires extra treatment because there is only one colour
        cbar = plt.colorbar(cmap, extend='min', use_gridspec=True, boundaries=[0, 0.5, 1], values=[0, 1])
    else:
        cbar = plt.colorbar(cmap, extend='min', use_gridspec=True)
    cbar.set_label(cbarlabel)
    if cmap_scaled:
        ncolors = nbins
    else:
        ncolors = max(1, K)
    # set a numbering interval for high densities (e.g. 5-10-15-20)
    if ncolors > 101:
        stride = 10
    elif ncolors > 51:
        stride = 5
    elif ncolors > 31:
        stride = 2
    else:
        stride = 1
    if K <= 1:
        # requires extra treatment because there is only one colour
        ticks = np.array([0.75])
    else:
        ticks = np.arange(1, ncolors + 1, 1) + 0.5
    if cmap_scaled:
        indices = np.arange(1, nbins + 2)
        low_label = np.ceil(indices * scaling_factor - 1e-6)
        low_label = np.roll(low_label, 1)
        low_label = np.delete(low_label, 0).astype(int)
        labels = list(low_label)
    else:
        labels = list(np.arange(1, ncolors + 1, 1, dtype=int))
    # if max label comes up automatically, leave it as it is
    if ticks[-1] == ticks[0::stride][-1]:
        cbar.set_ticks(ticks[0::stride])
        cbar.set_ticklabels(labels[0::stride])
    # if max label is too close to last strided label, leave the latter out
    elif stride > 1 and ticks[-1] != ticks[0::stride][-1] and ticks[-1] - ticks[0::stride][-1] < stride / 2:
        cbar.set_ticks(list(ticks[0::stride][:-1]) + [ticks[-1]])
        cbar.set_ticklabels(labels[0::stride][:-1] + [labels[-1]])
    # if there is enough space, just add the max label
    else:
        cbar.set_ticks(list(ticks[0::stride]) + [ticks[-1]])
        cbar.set_ticklabels(labels[0::stride] + [labels[-1]])
    return cmap


def calc_nematic_tensor(v):
    return np.einsum('...i,...j->...ij', v, v) - 0.5 * np.diag(np.ones(2))[None, ...]


class LGCA_base(ABC):
    """
    Base class for a lattice-gas. Not meant to be used alone!
    """
    rng = npr.default_rng()  # random number generator. would be better to initialize with init and keywords

    @property
    @abstractmethod
    def interactions(self) -> list:
        """List of interactions for this LGCA class."""
        # This is only a helper class, it cannot simulate! Use one the following classes:
        # LGCA_1D, LGCA_Square, LGCA_Hex

        # ... notation as of https://stackoverflow.com/a/58321197
        ...

    @property
    @abstractmethod
    def velocitychannels(self) -> int:
        """Number of velocity channels."""
        ...

    @property
    @abstractmethod
    def c(self) -> np.ndarray:
        """Array of the velocity channel vectors."""
        ...

    @abstractmethod
    def set_dims(self, dims=None, nodes=None, restchannels=0):
        ...

    @abstractmethod
    def init_coords(self):
        ...

    @abstractmethod
    def init_nodes(self, density, nodes=None, **kwargs):
        ...

    @abstractmethod
    def gradient(qty):
        ...

    def apply_pbc(self):
        raise NotImplementedError("Periodic boundary conditions not yet implemented for class "+str(self.__class__)+".")

    def apply_rbc(self):
        raise NotImplementedError("Reflecting boundary conditions not yet implemented for class "+str(self.__class__)+".")

    def apply_abc(self):
        raise NotImplementedError("Absorbing boundary conditions not yet implemented for class "+str(self.__class__)+".")

    def apply_inflowbc(self):
        raise NotImplementedError("Inflow boundary conditions not yet implemented for class "+str(self.__class__)+".")

    def __init__(self, nodes=None, dims=None, restchannels=0, density=0.1, bc='periodic', **kwargs):
        """
        Initialize class instance.
        :param nodes: np.ndarray initial configuration set manually
        :param dims: tuple determining lattice dimensions
        :param restchannels: number of resting channels
        :param density: float, if nodes is None, initialize lattice randomly with this particle density
        :param bc: boundary conditions
        :param r_int: interaction range
        """
        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        self.set_bc(bc)
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        self.init_coords()
        self.init_nodes(density=density, nodes=nodes, **kwargs)
        self.update_dynamic_fields()
        self.interaction_params = {}
        self.set_interaction(**kwargs)
        print(kwargs)

    def set_r_int(self, r):
        self.r_int = r
        self.init_nodes(nodes=self.nodes[self.nonborder])
        self.init_coords()
        self.update_dynamic_fields()

    def set_interaction(self, **kwargs):
        from lgca.interactions import go_or_grow, go_or_rest, birth, alignment, persistent_walk, chemotaxis, \
                contact_guidance, nematic, aggregation, wetting, random_walk, birthdeath, excitable_medium, \
                only_propagation
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction == 'go_or_grow':
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
                if self.restchannels < 2:
                    print('WARNING: not enough rest channels - system will die out!')

            elif interaction == 'go_or_rest':
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
                if self.restchannels < 2:
                    print('WARNING: not enough rest channels - system will die out!!!')

            elif interaction == 'go_and_grow':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])

            elif interaction == 'alignment':
                self.interaction = alignment
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

            elif interaction == 'persistent_motion':
                self.interaction = persistent_walk
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

            elif interaction == 'chemotaxis':
                self.interaction = chemotaxis
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 5.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

                if 'gradient' in kwargs:
                    self.interaction_params['gradient_field'] = kwargs['gradient']
                else:
                    if self.velocitychannels > 2:
                        x_source = npr.normal(self.xcoords.mean(), 1)
                        y_source = npr.normal(self.ycoords.mean(), 1)
                        rx = self.xcoords - x_source
                        ry = self.ycoords - y_source
                        r = np.sqrt(rx ** 2 + ry ** 2)
                        self.concentration = np.exp(-2 * r / self.ly)
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1, 'reflect'))
                    else:
                        source = npr.normal(self.l / 2, 1)
                        r = abs(self.xcoords - source)
                        self.concentration = np.exp(-2 * r / self.l)
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1, 'reflect'))
                        self.interaction_params['gradient_field'] /= self.interaction_params['gradient_field'].max()

            elif interaction == 'contact_guidance':
                self.interaction = contact_guidance
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

                if 'director' in kwargs:
                    self.interaction_params['gradient_field'] = kwargs['director']
                else:
                    self.interaction_params['gradient_field'] = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, 2))
                    self.interaction_params['gradient_field'][..., 0] = 1
                    self.guiding_tensor = calc_nematic_tensor(self.interaction_params['gradient_field'])
                if self.velocitychannels < 4:
                    print('WARNING: NEMATIC INTERACTION UNDEFINED IN 1D!')

            elif interaction == 'nematic':
                self.interaction = nematic
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

            elif interaction == 'aggregation':
                self.interaction = aggregation
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

            elif interaction == 'wetting':
                self.interaction = wetting
                self.calc_permutations()
                self.set_r_int(2)

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('adhesion sensitivity set to beta = ', self.interaction_params['beta'])

                if 'alpha' in kwargs:
                    self.interaction_params['alpha'] = kwargs['alpha']
                else:
                    self.interaction_params['alpha'] = 2.
                    print('substrate sensitivity set to alpha = ', self.interaction_params['alpha'])

                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 2.
                    print('pressure sensitivity set to gamma = ', self.interaction_params['gamma'])

                if 'rho_0' in kwargs:
                    self.interaction_params['rho_0'] = kwargs['rho_0']
                else:
                    self.interaction_params['rho_0'] = self.restchannels // 2
                self.n_crit = (self.velocitychannels + 1) * self.interaction_params['rho_0']

            elif interaction == 'random_walk':
                self.interaction = random_walk

            elif interaction == 'birth':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])

            elif interaction == 'birthdeath':
                self.interaction = birthdeath
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.05
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

            elif interaction == 'excitable_medium':
                self.interaction = excitable_medium
                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']

                else:
                    self.interaction_params['beta'] = .05
                    print('alignment sensitivity set to beta = ', self.interaction_params['beta'])

                if 'alpha' in kwargs:
                    self.interaction_params['alpha'] = kwargs['alpha']
                else:
                    self.interaction_params['alpha'] = 1.
                    print('aggregation sensitivity set to alpha = ', self.interaction_params['alpha'])

                if 'N' in kwargs:
                    self.interaction_params['N'] = kwargs['N']
                else:
                    self.interaction_params['N'] = 50
                    print('repetition of fast reaction set to N = ', self.interaction_params['N'])

            elif interaction == 'only_propagation':
                self.interaction = only_propagation

            else:
                print('interaction', kwargs['interaction'], 'is not defined! Random walk used instead.')
                print('Implemented interactions:', self.interactions)
                self.interaction = random_walk

        else:
            print('Random walk interaction is used.')
            self.interaction = random_walk

    def set_bc(self, bc):
        if bc in ['absorbing', 'absorb', 'abs', 'abc']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl', 'rbc']:
            self.apply_boundaries = self.apply_rbc
        elif bc in ['periodic', 'pbc']:
            self.apply_boundaries = self.apply_pbc
        elif bc in ['inflow']:
            self.apply_boundaries = self.apply_inflowbc
        else:
            print(bc, 'not defined, using periodic boundaries')
            self.apply_boundaries = self.apply_pbc

    def calc_flux(self, nodes):
        # 1st+ dim: lattice sites, last dim: channels
        # dot product between c vectors and actual configuration of site
        return np.einsum('ij,...j', self.c, nodes[..., :self.velocitychannels])

    def get_interactions(self):
        print(self.interactions)

    def print_nodes(self):
        print(self.nodes.astype(int))

    def random_reset(self, density):
        """
        Distribute particles in the lattice according to a given density; can yield different cell numbers per lattice site
        :param density: particle density in the lattice: average number of particles per channel
        """
        self.nodes = npr.random(self.nodes.shape) < density
        self.apply_boundaries()
        self.update_dynamic_fields()

    def homogeneous_random_reset(self, density):
        """
        Distribute particles in the lattice homogeneously according to a given density: each lattice site has the same
            number of particles, randomly distributed among channels
        :param density: particle density in the lattice: number of particles/(dimensions * number of channels)
        """
        # find the number of particles per lattice site which is closest to the desired density
        if int(density * self.K) == density * self.K:
            initcells = int(density * self.K)
        else:
            initcells = min(int(density * self.K) + 1, self.K)
        # create a configuration for one node with the calculated number of particles
        channels = [1] * initcells + [0] * (self.K - initcells)
        # permutate it to fill the lattice
        n_nodes = self.nodes[..., 0].size
        channels = np.array([npr.permutation(channels) for i in range(n_nodes)])
        self.nodes = channels.reshape(self.nodes.shape)

        self.apply_boundaries()
        self.update_dynamic_fields()
        eff_dens = self.nodes[self.nonborder].sum() / (self.K * self.cell_density[self.nonborder].size)
        print("Required density: {:.3f}, Achieved density: {:.3f}".format(density, eff_dens))

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

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True, recordpertype=False):
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
            if recordpertype:
                self.velcells_t[t, ...] = self.nodes[self.nonborder][..., :self.velocitychannels].sum(-1)
                self.restcells_t[t, ...] = self.nodes[self.nonborder][..., self.velocitychannels:].sum(-1)

    def calc_permutations(self):
        self.permutations = [np.array(list(multiset_permutations([1] * n + [0] * (self.K - n))), dtype=np.int8)
                                                                # builds list with all possible configurations of
                                                                # particles in an array of size self.K (velocity + resting)
                                      # builds nested list with all possible permutations for this array
                             # first dim: number of particles
                             # second dim: all permutations for this n
                             for n in range(self.K + 1)]
        # -> list of all possible configurations for a lattice site

        self.j = [np.dot(self.c, self.permutations[n][:, :self.velocitychannels].T) for n in range(self.K + 1)]
        # dot product between the directions and the particles in the velocity channels for each possible number of particles
        # array of flux for each permutation for each number of particles
        # first dim: number of particles
        # second dim: flux vector for each permutation (directions as specified in c)

        self.cij = np.einsum('ij,kj->jik', self.c, self.c) - 0.5 * np.diag(np.ones(2))[None, ...]
        self.si = [np.einsum('ij,jkl', self.permutations[n][:, :self.velocitychannels], self.cij) for n in
                   range(self.K + 1)]

    def total_population(self):
        """
        Calculate the total population of cells in the lattice.
        :returns: numpy.int32 total population size
        """
        return self.cell_density[self.nonborder].sum()


class IBLGCA_base(LGCA_base, ABC):
    """
    Base class for identity-based LGCA.
    """

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
        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        self.props = {}
        self.set_bc(bc)
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        self.init_coords()
        self.init_nodes(density=density, nodes=nodes)
        self.maxlabel = self.nodes.max()
        self.update_dynamic_fields()
        self.interaction_params = {}
        self.set_interaction(**kwargs)

    def set_interaction(self, **kwargs):
        from lgca.ib_interactions import random_walk, birth, birthdeath, birthdeath_discrete, go_or_grow, \
            go_and_grow_mutations
        from lgca.interactions import only_propagation
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction == 'birth':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('Birth rate set to r_b =', self.interaction_params['r_b'])
                self.props.update(r_b=[0.] + [self.interaction_params['r_b']] * self.maxlabel)
                if 'std' in kwargs:
                    self.interaction_params['std'] = kwargs['std']
                else:
                    self.interaction_params['std'] = 0.01
                    print('Standard deviation set to std =', self.interaction_params['std'])
                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])

            elif interaction == 'birthdeath' or interaction == 'go_and_grow':
                self.interaction = birthdeath
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                self.props.update(r_b=[0.] + [self.interaction_params['r_b']] * self.maxlabel)
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'std' in kwargs:
                    self.interaction_params['std'] = kwargs['std']
                else:
                    self.interaction_params['std'] = 0.01
                    print('standard deviation set to = ', self.interaction_params['std'])
                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])

                if 'track_inheritance' in kwargs:
                    self.interaction_params['track_inheritance'] = kwargs['track_inheritance']
                else:
                    self.interaction_params['track_inheritance'] = False
                    print('Family relationships not tracked.')
                if self.interaction_params['track_inheritance']:
                    self.init_families(type='heterogeneous', mutation=False)

            elif interaction == 'birthdeath_discrete':
                self.interaction = birthdeath_discrete
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('Birth rate set to r_b = ', self.interaction_params['r_b'])

                self.props.update(r_b=[0.] + [self.interaction_params['r_b']] * self.maxlabel)
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    print('Death rate set to r_d = ', self.interaction_params['r_d'])

                if 'drb' in kwargs:
                    self.interaction_params['drb'] = kwargs['drb']
                else:
                    self.interaction_params['drb'] = 0.01
                    print('Delta r_b set to = ', self.interaction_params['drb'])
                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])

                if 'pmut' in kwargs:
                    self.interaction_params['pmut'] = kwargs['pmut']
                else:
                    self.interaction_params['pmut'] = 0.1
                    print('Mutation probability set to p_mut =', self.interaction_params['pmut'])

            elif interaction == 'go_or_grow':
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
                    kappa = kwargs['kappa']
                    try:
                        self.interaction_params['kappa'] = list(kappa)
                    except TypeError:
                        self.interaction_params['kappa'] = [kappa] * self.maxlabel
                else:
                    self.interaction_params['kappa'] = [5.] * self.maxlabel
                    print('switch rate set to kappa = ', self.interaction_params['kappa'][0])
                self.props.update(kappa=[0.] + self.interaction_params['kappa'])
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    try:
                        self.interaction_params['theta'] = list(theta)
                    except TypeError:
                        self.interaction_params['theta'] = [theta] * self.maxlabel
                else:
                    self.interaction_params['theta'] = [0.75] * self.maxlabel
                    print('switch threshold set to theta = ', self.interaction_params['theta'][0])
                self.props.update(theta=[0.] + self.interaction_params['theta'])
                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    print('Standard deviation for kappa mutation set to ', self.interaction_params['kappa_std'])
                if 'theta_std' in kwargs:
                    self.interaction_params['theta_std'] = kwargs['theta_std']
                else:
                    self.interaction_params['theta_std'] = 0.05
                    print('Standard deviation for theta mutation set to ', self.interaction_params['theta_std'])

                if self.restchannels < 2:
                    print('WARNING: not enough rest channels - system will die out!!!')

            elif interaction == 'random_walk':
                self.interaction = random_walk

            elif interaction == 'only_propagation':
                self.interaction = only_propagation

            elif interaction == 'go_and_grow_mutations':
                self.interaction = go_and_grow_mutations
                if 'effect' in kwargs:
                    self.interaction_params['effect'] = kwargs['effect']
                else:
                    self.interaction_params['effect'] = 'passenger_mutation'
                    print('fitness effect set to passenger mutation, rb=const.')
                if 'r_int' in kwargs:
                    self.set_r_int(kwargs['r_int'])
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.5
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                if 'r_m' in kwargs:
                    self.interaction_params['r_m'] = kwargs['r_m']
                else:
                    self.interaction_params['r_m'] = 0.001
                    print('mutation rate set to r_m = ', self.interaction_params['r_m'])
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    print('death rate set to r_d = ', self.interaction_params['r_d'])
                self.init_families(type='homogeneous', mutation=True)
                if self.interaction_params['effect'] == 'driver_mutation':
                    self.family_props.update(r_b=[0] + [self.interaction_params['r_b']] * self.maxfamily)
                    if 'fitness_increase' in kwargs:
                        self.interaction_params['fitness_increase'] = kwargs['fitness_increase']
                    else:
                        self.interaction_params['fitness_increase'] = 1.1
                        print('fitness increase for driver mutations set to ',
                              self.interaction_params['fitness_increase'])
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
        occ[occ > 0] = np.arange(1, 1+occ.sum(), dtype=np.uint)
        return occ

    def random_reset(self, density):
        """

        :param density:
        :return:
        """
        occupied = npr.random(self.dims + (self.K,)) < density
        self.nodes[self.nonborder] = self.convert_bool_to_ib(occupied)
        self.apply_boundaries()

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True, recordfampop=False):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros((timesteps + 1,) + self.dims + (self.K,), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[self.nonborder]
            # self.props_t = [copy(self.props)]  # this is mostly useless, just use self.props of the last time step
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.cell_density[self.nonborder].sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1,) + self.dims)
            self.dens_t[0, ...] = self.cell_density[self.nonborder]
        if recordfampop:
            from lgca.ib_interactions import go_and_grow_mutations
            # this needs to include all interactions that can increase the number of recorded families!
            if self.interaction == go_and_grow_mutations:
                # if mutations are allowed, this is a list because it will be ragged due to increasing family numbers
                self.fam_pop_t = [self.calc_family_pop_alive()]
                is_mutating = True
            else:
                if 'family' not in self.props:
                    raise RuntimeError("Interaction does not deal with families, "
                                       "family population can therefore not be recorded.")
                # otherwise standard procedure
                self.fam_pop_t = np.zeros((timesteps + 1, self.maxfamily+1))
                self.fam_pop_t[0, ...] = self.calc_family_pop_alive()
                is_mutating = False
        for t in tqdm(iterable=range(1, timesteps + 1), disable=1-showprogress):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.nonborder]
                # self.props_t.append(copy(self.props))
            if recordN:
                self.n_t[t] = self.cell_density[self.nonborder].sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.nonborder]
            if recordfampop:
                if is_mutating:
                    # append to the ragged nested list
                    self.fam_pop_t.append(self.calc_family_pop_alive())
                else:
                    # standard procedure
                    try:
                        self.fam_pop_t[t, ...] = self.calc_family_pop_alive()
                    except ValueError as e:
                        raise ValueError("Number of families has increased, interaction must be included in the case " +
                                         "distinction for the recordfampop keyword in the IBLGCA base timeevo function!") from e
        if recordfampop and is_mutating:
            self.straighten_family_populations()

    def calc_flux(self, nodes):
        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

        return super().calc_flux(nodes)

    def get_prop(self, nodes=None, props=None, propname=None):
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(self.props))

        prop = np.array(props[propname])
        proparray = prop[nodes]
        return proparray

    def calc_prop_mean(self, nodes=None, props=None, propname=None):
        prop = self.get_prop(nodes=nodes, props=props, propname=propname)
        occupied = nodes.astype(bool)
        mask = ~occupied
        prop = np.ma.array(prop, mask=mask)
        mean_prop = prop.mean(-1)
        return mean_prop

    def plot_prop_timecourse(self, nodes_t=None, props=None, propname=None, figindex=None, figsize=None, **kwargs):
        """
        Plot the time evolution of the cell property 'propname'
        :param nodes_t:
        :param props:
        :param propname:
        :param figindex:
        :param figsize:
        :param kwargs: keyword arguments for the matplotlib.plot command
        :return:
        """
        if nodes_t is None:
            nodes_t = self.nodes_t

        prop = self.get_prop(nodes=nodes_t, props=props, propname=propname)
        occupied = nodes_t.astype(bool)
        mask = 1 - occupied
        prop = np.ma.array(prop, mask=mask)
        tocollapse = tuple(range(1, prop.ndim))
        mean_prop_t = np.mean(prop, axis=tocollapse)
        std_mean_prop_t = np.std(prop, axis=tocollapse, ddof=1) / np.sqrt(np.sum(occupied, axis=tocollapse))
        if figindex is None:
            fig = plt.gcf()

        else:
            fig = plt.figure(num=figindex)

        if figsize is not None:
            fig.set_size_inches(figsize)

        tmax = nodes_t.shape[0]

        yerr = std_mean_prop_t
        x = np.arange(tmax)
        y = mean_prop_t

        plt.xlabel('$t$')
        plt.ylabel('${}$'.format(propname))
        plt.title('Time course of the cell property')
        line = plt.plot(x, y, **kwargs)
        errors = plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, antialiased=True, interpolate=True)
        return line, errors

    def init_families(self, type='homogeneous', mutation=True):
        if type == 'homogeneous':
            families = [0] + [1] * self.maxlabel
            self.props.update(family=families)
            if mutation:
                self.family_props = {}  # properties of families
                self.family_props.update(ancestor=[0, 0])
                self.family_props.update(descendants=[[1], []])
                self.maxfamily = 1
        elif type == 'heterogeneous':
            families = list(np.arange(self.maxlabel+1))
            self.props.update(family=families)
            if mutation:
                self.family_props = {}  # properties of families
                self.family_props.update(ancestor=[0] * (self.maxlabel + 1))
                self.family_props.update(descendants=[families[1:]])
                for i in range(self.maxlabel):
                    self.family_props['descendants'].append([])
                self.maxfamily = self.maxlabel

    def calc_family_pop_alive(self):
        """
        Calculate how many cells of each family are alive.
        :returns: np.ndarray fam_pop_array - array of family population counts indexed by family ID
        """
        if 'family' not in self.props:
            raise RuntimeError("Family properties are not recorded by the LGCA, choose suitable interaction.")

        cells_alive = self.nodes[self.nonborder][
            np.where(self.nodes[self.nonborder] > 0)]  # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
        fam_alive, fam_pop = np.unique(cell_fam_alive, return_counts=True)  # count number of cells for each family
        # transform into array with population entry for all families that ever existed
        fam_pop_array = np.zeros(self.maxfamily+1, dtype=int)
        fam_pop_array[fam_alive] = fam_pop
        return fam_pop_array
        # alternative: look up family for each live cell, then do unique on those altered nodes

    def straighten_family_populations(self):
        """
        Utility for the timeevo method. Straighten out the ragged family population record over time
        in the presence of mutations.
        """
        for pop_arr in self.fam_pop_t:
            if len(pop_arr) < self.maxfamily + 1:
                pop_arr.resize(self.maxfamily + 1, refcheck=False)
        self.fam_pop_t = np.array(self.fam_pop_t)

    @staticmethod
    def propagate_pop_to_parents(pop_t, ancestor: list):
        """
        Propagate family population to all ancestors.
        :param pop_t: np.ndarray of shape (timesteps, families) that holds the population of each family for all timesteps
        :param ancestor: list of length families + 1 with each family's parent (initially existing families have 0)
        :returns: np.ndarray cum_pop_t - family populations over time summed up to all ancestors recursively, cum_pop_t[:, 0]
                  holds total population over time
        """
        cum_pop_t = deepcopy(pop_t)
        # turn list around to traverse tree in reverse, leverage that parents always have lower indices than children
        ancestor_reverse = np.flip(ancestor)
        indices_reverse = np.flip(np.arange(len(ancestor)))
        for ind, el in zip(indices_reverse, ancestor_reverse):
            # don't sum elements with themselves (relevant for 0)
            if ind != el:
                cum_pop_t[:, el] += cum_pop_t[:, ind]
        return cum_pop_t

    def muller_plot(self, t_start_index=None, t_slice=slice(None), prop=None, cutoff_abs=None, cutoff_rel=None, **kwargs):
        """
        Draw a Muller plot from the LGCA simulation data (runtime, family tree, recorded family populations).
        :param t_start_index first number to appear on the x axis
        :param t_slice slice, which parts of the simulation time to consider for the plot
        :param prop: key of the dictionary lgca.family_props - colour the area for each family according to
                     this family property. If None, colour area according to family identity.
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :param kwargs: keyword arguments to further style the plot drawn by lgca.plots.muller_plot
        :returns: (fig, ax, ret) fig = matplotlib figure handle, ax = Muller plot axis handle,
                                 ret = handle of legend, handle of colourbar or None. The separate colourbar axis handle
                                 can be retrieved as ret.ax
        """
        # prepare content arguments for plot function from LGCA data
        root_ID = 0
        parent_list = self.family_props['ancestor']
        children_nlist = self.family_props['descendants']
        # slice and adjust time axis
        if type(t_slice) != slice:
            raise TypeError("'t_slice' must be slice object to index the time dimension")
        if not hasattr(self, 'fam_pop_t'):
            raise AttributeError(
                "LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before to draw a Muller plot.")
        if cutoff_rel or cutoff_abs:
            fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)
        else:
            fam_pop_t = self.fam_pop_t
        cum_pop_t = self.propagate_pop_to_parents(fam_pop_t[t_slice], parent_list)
        if t_start_index is None:
            if t_slice.start is not None:
                t_start_index = t_slice.start
            else:
                t_start_index = 0
        timeline = np.arange(cum_pop_t.shape[0]) + t_start_index

        # choose drawing mode depending on keyword arguments
        if 'facecolour' not in kwargs:
            # colour by family identity
            if prop is None:
                kwargs['facecolour'] = 'identity'

            # colour by family property value
            else:
                prop_values = self.family_props[prop]
                if 'facecolour_map' in kwargs:
                    print("If the families should be coloured according to "+str(prop) +
                          ", facecolour mapping is done by the LGCA - 'facecolour_map' ignored")
                kwargs['facecolour'] = 'property'
                kwargs['facecolour_map'] = prop_values
                if 'legend_title' not in kwargs:
                    kwargs['legend_title'] = prop
        # else: drawing mode determined by user

        # plot
        return muller_plot(root_ID, cum_pop_t, children_nlist, parent_list, timeline, **kwargs)

    def total_population_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate the total population of cells at each simulation step.  Requires a previous timeevo with recordfampop=True.
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :returns: np.ndarray of shape timesteps+1, number of cells at each timestep
        """
        if not hasattr(self, 'fam_pop_t'):
            raise AttributeError("LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before "
                                 "to calculate this.")
        # filter families below cutoff
        fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)
        # fam_pop_t is of shape (timesteps + 1, number of families)
        # summing of population values along the family dimension yields total population
        return fam_pop_t.sum(-1)

    def num_families_total(self):
        """
        Calculate how many families there are in total, including extinct ones.
        :returns: int - number of families
        """
        return len(self.family_props['ancestor'])-1  # subtract placeholder 0 family

    def num_families_total_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate how many families there are in total, including extinct ones.
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :returns: np.ndarray of shape timesteps+1, number of families at each timestep
        """
        # filter families below cutoff
        fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)
        # remove all-zero families
        empty_family_ind = np.where(np.all(fam_pop_t == 0, axis=0))[0]
        fam_filtered = np.delete(fam_pop_t, empty_family_ind, axis=1)
        # highest index of live family = total families that existed until this point - extinct families
        # obtain family indices for each timestep
        fam_indices = np.arange(fam_filtered.shape[1]).reshape(1, fam_filtered.shape[1])
        fam_indices_t = np.repeat(fam_indices, fam_filtered.shape[0], axis=0)
        # helper array to indicate existence of a family at a timestep
        fam_indicator_t = copy(fam_filtered)
        fam_indicator_t[fam_filtered > 0] = 1
        # find family alive at each timestep with the highest ID and add 1 for number of families
        max_fam_alive_t = np.amax(np.multiply(fam_indices_t+1, fam_indicator_t), axis=1)
        # propagate maximum value forward in time to account for died out families with high indices
        fam_total_t = [max_fam_alive_t[0]]
        for i in range(len(max_fam_alive_t) - 1):
            if max_fam_alive_t[i + 1] >= fam_total_t[i]:
                fam_total_t.append(max_fam_alive_t[i+1])
            else:
                fam_total_t.append(fam_total_t[i])
        return np.array(fam_total_t)

    def filter_family_population_t(self, fam_pop_t=None, cutoff_abs=None, cutoff_rel=None):
        """
        Mask out families from fam_pop_t that never exceeded the given threshold.
        :param fam_pop_t np.ndarray of shape (timesteps + 1, families + 1) holding each family's population
                         size over time. Defaults to self.fam_pop_t
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :returns np.ndarray of shape (timesteps + 1, families + 1) with all population values for
                 too small families set to 0. Original fam_pop_t if both cutoff parameters are None.
        """
        if fam_pop_t is None:
            if not hasattr(self, 'fam_pop_t'):
                raise AttributeError("LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before "
                                     "in order to calculate this with keyword arg fam_pop_t=None.")
            fam_pop_t = self.fam_pop_t

        if cutoff_rel:
            fam_pop_t_filtered = copy(fam_pop_t)
            # calculate fraction of the total population for each family at all timesteps
            total_pop = fam_pop_t_filtered.sum(-1)
            rel_fam_pop = np.divide(fam_pop_t_filtered.transpose(), total_pop)  # transpose to align time axis
            rel_fam_pop = rel_fam_pop.transpose()  # transpose back to match with fam_pop_t again
            # obtain positions of families in the array that have never exceeded the threshold
            small_family_pos = ~ np.any(rel_fam_pop >= cutoff_rel, axis=0)

        elif cutoff_abs:
            fam_pop_t_filtered = copy(fam_pop_t)
            # obtain positions of families in the array that have never exceeded the threshold
            small_family_pos = ~ np.any(fam_pop_t_filtered >= cutoff_abs, axis=0)

        else:
            return fam_pop_t

        # filter them out
        fam_pop_t_filtered[:, small_family_pos] = 0

        return fam_pop_t_filtered

    def num_families_alive(self):
        """
        Calculate how many families are alive.
        :returns: int - number of families
        """
        cell_fam_alive = self.list_families_alive()  # list all live family IDs
        return len(cell_fam_alive)

    def num_families_alive_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate how many families have been alive at each simulation step. Requires a previous timeevo with recordfampop=True.
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :returns: np.ndarray of shape timesteps+1, number of families at each timestep
        """
        if not hasattr(self, 'fam_pop_t'):
            raise AttributeError("LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before "
                                 "to calculate this.")

        # filter families below cutoff
        fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)

        # fam_pop_t is of shape (timesteps + 1, number of families)
        # filter which families are alive at which timestep
        # result is an array of Booleans
        # summing along the family dimension yields the sum of living families
        return (fam_pop_t > 0).sum(-1)

    def list_families_alive(self):
        """
        Calculate which families are alive.
        :returns: np.ndarray - array of family IDs in ascending order
        """
        cells_alive = self.nodes[self.nonborder][
            np.where(self.nodes[self.nonborder] > 0)]  # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
        return np.unique(cell_fam_alive) # remove duplicate entries

    def calc_family_generations(self):
        """
        Calculate which generation each family belongs to, i.e. how many mutations have occurred. The list is
        returned and stored in lgca.family_props['generation'].
        :returns: list - family generations indexed by family ID, helper root family = generation 0,
                  families at init = generation 1
        """
        # check if it has been calculated and is up to date
        num_families = self.num_families_total()
        if 'generation' in self.family_props:
            if len(self.family_props['generation']) == num_families:
                return self.family_props['generation']
        # if not, initialise the property and fill it
        self.family_props['generation'] = list(np.zeros(num_families+1))
        self.recurse_generation(0, 0)
        return self.family_props['generation']

    def recurse_generation(self, fam_ID, gen):
        """
        Utility for calculating the generation of each family. Recursively updates the entries in
        self.family_props['generation'].
        """
        # update generation for this family
        self.family_props['generation'][fam_ID] = gen
        # update children generations with increased counter
        gen += 1
        for desc in self.family_props['descendants'][fam_ID]:
            self.recurse_generation(desc, gen)

    def calc_generation_pop(self):
        """
        Calculate the current population per family generation.
        :returns: np.ndarray of cell numbers per generation, indexed by generation (helper root family is generation 0,
                  families present at initialisation are generation 1)
        """
        # if it has not been done or is not up to date, calculate generation of each family
        self.calc_family_generations()

        # initialise result array
        max_generation = np.array(self.family_props['generation']).max()
        generations_pop = np.zeros(max_generation+1, dtype=int)

        # filter living families by generation to add their population to the right array element
        fam_pop = self.calc_family_pop_alive()
        fam_generation = np.array([self.family_props['generation'][fam_ID] for fam_ID in range(len(fam_pop))])
        for gen in range(len(generations_pop)):
            generations_pop[gen] = fam_pop[fam_generation == gen].sum()
        return generations_pop

    def calc_generation_pop_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate the population per family generation at each simulation step. Requires a previous timeevo with recordfampop=True.
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :returns: np.ndarray of shape (timesteps+1,generations+1) cell numbers per generation at each timestep
                  (helper root family is generation 0, families present at initialisation are generation 1)
        """
        # check existence of self.fam_pop_t
        if not hasattr(self, 'fam_pop_t'):
            raise AttributeError("LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before "
                                 "to calculate this.")

        # if it has not been done or is not up to date, calculate generation of each family
        self.calc_family_generations()

        # initialise result array of shape (timesteps+1,generations+1)
        max_generation = np.array(self.family_props['generation']).max()
        generations_pop_t = np.zeros((self.fam_pop_t.shape[0], max_generation + 1), dtype=int)

        # filter families with population cutoff
        fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)
        num_families = fam_pop_t.shape[1]
        # filter families by generation to add their population to the right array element
        fam_generation = np.array([self.family_props['generation'][fam_ID] for fam_ID in range(num_families)])
        for gen in range(generations_pop_t.shape[1]):
            generations_pop_t[:, gen] += fam_pop_t[:, fam_generation == gen].sum(-1)
        return generations_pop_t

    def propagate_ancestor_to_descendants(self):
        """
        Traverse family tree to find the families that were initialised in the beginning of the simulation
        ("initial ancestors") and which families originally descend from them, including the 2nd, 3rd, ... generations
        :returns: list of initial ancestors indexed by family ID. The helper root family and initial ancestors will have
                  0, all others their ancestor's family ID
        """
        # check if it has been calculated and is up to date
        num_families = self.num_families_total()
        if 'init_ancestor' in self.family_props:
            if len(self.family_props['init_ancestor']) == num_families:
                return self.family_props['init_ancestor']
        # if not, initialise the property and fill it
        self.family_props['init_ancestor'] = list(np.zeros(num_families + 1))
        # check each family's ancestor, going up the tree from the root
        for fam_ID, ancestor in enumerate(self.family_props['ancestor']):
            if fam_ID == 0:
                continue
            # all initial families have ancestor == 0 - their ID is the initial ancestor of the children
            if ancestor == 0:
                anc = fam_ID
            # otherwise, propagate own initial ancestor to children
            else:
                anc = self.family_props['init_ancestor'][fam_ID]
            # propagation step
            for child_fam_ID in self.family_props['descendants'][fam_ID]:
                self.family_props['init_ancestor'][child_fam_ID] = anc

        return self.family_props['init_ancestor']

    def calc_init_families_pop(self):
        """
        Calculate the current population per family that was initialised in the beginning of the simulation
        ("initial ancestor") and their descendants.
        :returns: (np.ndarray init_families_pop, np.ndarray init_ancestor_IDs)
                  init_families_pop: population per initial family
                  init_ancestor_IDs: family ID corresponding to each position in init_families_pop
        """
        # calculate population per family and initial family they descend from
        fam_pop = self.calc_family_pop_alive()
        init_ancestors = np.array(self.propagate_ancestor_to_descendants())
        # clip for indexing in case a late family already died out
        init_ancestors_alive = np.resize(init_ancestors, len(fam_pop))
        # initialise
        init_ancestor_IDs = np.arange(len(init_ancestors_alive))[init_ancestors_alive == 0][1:]  # ignore helper root family
        init_families_pop = np.zeros(len(init_ancestor_IDs), dtype=int)
        # filter living families by initial ancestor to add their population to the right array element
        for pos, init_fam in enumerate(init_ancestor_IDs):
            init_families_pop[pos] = fam_pop[init_ancestors_alive == init_fam].sum()
            # add population of ancestor itself
            init_families_pop[pos] += fam_pop[init_fam]
        return init_families_pop, init_ancestor_IDs

    def calc_init_families_pop_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate the current population per family that was initialised in the beginning of the simulation
        ("initial ancestor") and their descendants at each simulation step. Requires a previous timeevo with recordfampop=True.
        :param cutoff_abs exclude families that never exceeded this population size in absolute numbers
        :param cutoff_rel exclude families that never exceeded this population size,
                          expressed as a fraction of the total population
        cutoff_rel takes precedence over cutoff_abs
        :returns: (np.ndarray init_families_pop, np.ndarray init_ancestor_IDs)
                  init_families_pop: of shape (timesteps+1,number of initial families +1) population per initial family at each timestep
                  init_ancestor_IDs: family ID corresponding to each position on the family dimension in init_families_pop
        """
        # check existence of self.fam_pop_t
        if not hasattr(self, 'fam_pop_t'):
            raise AttributeError("LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before "
                                 "to calculate this.")

        # calculate population per family and initial family they descend from
        init_ancestors = np.array(self.propagate_ancestor_to_descendants())

        # initialise
        init_ancestor_IDs = np.arange(len(init_ancestors))[init_ancestors == 0][1:]  # ignore helper root family
        init_families_pop_t = np.zeros((self.fam_pop_t.shape[0], len(init_ancestor_IDs)), dtype=int)
        # filter families with population cutoff
        fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)
        # filter living families by initial ancestor to add their population to the right array element
        for pos, init_fam in enumerate(init_ancestor_IDs):
            init_families_pop_t[:, pos] = fam_pop_t[:, init_ancestors == init_fam].sum(-1)
            # add population of ancestor itself
            init_families_pop_t[:, pos] += fam_pop_t[:, init_fam]
        return init_families_pop_t, init_ancestor_IDs

    def propagate_family_prop_to_cells(self, prop: str):
        """
        Propagate a family property down to all cells that belong to the family. Uses contents of
        lgca.family_props[prop] to create a new cell property lgca.props[prop]
        :param prop: str, key of the family property
        """
        self.props[prop] = [self.family_props[prop][fam] for fam in self.props['family']]

    def add_family(self, ancestor_fam: int):
        """
        Create a new family and register it in the family tree.
        PLEASE DO NOT CALL THIS UNLESS YOU ARE WRITING AN INTERACTION FUNCTION THAT
        DEALS WITH MUTATIONS.
        :param ancestor_fam family that the new one descends from
        """
        if not hasattr(self, 'maxfamily') or not hasattr(self, 'family_props'):
            raise RuntimeError("No family tracking initialized. If you are writing an interaction function, please:\n"
                               "1) check if it deals with mutations. If not, you don't need this method.\n"
                               "2) if so, make sure to call self.init_families(...) in self.set_interaction(...).\n"
                               "If you are not writing an interaction function, please do not call this. It will "
                               "compromise the state of your LGCA.")
        # add new family
        self.maxfamily += 1
        # register ancestor of the new family
        self.family_props['ancestor'].append(ancestor_fam)
        # record new family as child of the old one
        self.family_props['descendants'][ancestor_fam].append(self.maxfamily)
        # create empty children list for the new family
        self.family_props['descendants'].append([])


class NoVE_LGCA_base(LGCA_base, ABC):
    """
    Base class for LGCA without volume exclusion.
    """
    def __init__(self, nodes=None, dims=None, restchannels=None, density=0.1, hom=None, bc='periodic', capacity=None, **kwargs):
        """
        Initialize class instance.
        :param nodes: np.ndarray initial configuration set manually
        :param dims: tuple determining lattice dimensions
        :param restchannels: number of resting channels
        :param density: float, if nodes is None, initialize lattice randomly with this particle density
        :param bc: boundary conditions
        :param r_int: interaction range
        """

        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
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
            # if self.restchannels > 0:  # what is happening here
            #     print('Interaction only works without restchannels and will crash.')
            if interaction == 'random_walk':
                self.interaction = random_walk
            # density-dependent interaction rule
            elif interaction == 'dd_alignment':
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
                self.interaction = di_alignment
                if self.restchannels > 0:
                    print('Interaction only works without restchannels and will crash.')

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
                    raise RuntimeError("No rest channels ({:d}) defined, interaction cannot be performed! Set number of rest channels with restchannels keyword.".format(self.restchannels))
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
                        "No rest channels ({:d}) defined, interaction cannot be performed! Set number of rest channels with restchannels keyword.".format(
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
                print('interaction', kwargs['interaction'], 'is not defined! Density-dependent alignment interaction used instead.')
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
        Distribute particles in the lattice according to a given density; can yield different cell numbers per lattice site
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
        It is calculated as the magnitude of the sum of the velocities of all particles normalized by the number of particles.
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
        It is calculated as the agreement in direction between the ﬂux of a lattice site and the ﬂux of the director ﬁeld
        summed up and normalized over all lattice sites.
        :return: Local alignment parameter: ranging from -1 (antiparallel alignment) through 0 (no alignment) to 1 (parallel alignment)
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


class NoVE_IBLGCA_base(NoVE_LGCA_base, IBLGCA_base, ABC):
    """
    Base class for identity-based LGCA without volume exclusion.
    """
    interactions = ['go_or_grow', 'birthdeath', 'randomwalk', 'steric_evolution']

    def __init__(self, nodes=None, dims=None, density=.1, restchannels=1, bc='periodic', **kwargs):
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
        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        self.props = {}
        self.length_checker = np.vectorize(len)
        self.set_bc(bc)
        self.interaction_params = {}
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        self.init_coords()
        self.init_nodes(density, nodes=nodes)
        self.calc_max_label()
        self.update_dynamic_fields()
        self.mean_prop_t = {}
        self.set_interaction(**kwargs)


    def update_dynamic_fields(self):
        self.channel_pop = self.length_checker(self.nodes)  # population of a channel
        self.cell_density = self.channel_pop.sum(-1)  # population of a node

    def convert_int_to_ib(self, occ):
        """
        Convert an array of integers representing the occupation numbers of an lgca to an array consisting of lists of
        individual cell labels, starting at 'starting_id'. The length of each list corresponds to the entry in 'occ'.
        :param occ: array of occupation numbers. must match the lgca dimensions
        : param starting_id: int > 0
        :return: array, where each entry is a list of individual cell labels.
        """
        ntot = occ.sum()
        labels = list(range(ntot))
        tempnodes = np.empty(occ.shape, dtype=object)
        counter = 0
        for ind, dens in np.ndenumerate(occ):
            tempnodes[ind] = labels[counter:counter+dens]
            counter += dens

        return tempnodes

    def random_reset(self, density):
        """
        Distribute particles in the lattice according to a given density; can yield different cell numbers per lattice site
        :param density: particle density in the lattice: average number of particles per channel
        """
        density = npr.poisson(lam=density, size=self.dims + (self.K,))
        tempnodes = self.convert_int_to_ib(density)
        self.nodes[self.nonborder] = tempnodes
        self.maxlabel = density.sum()
        self.update_dynamic_fields()

    def set_interaction(self, **kwargs):
        from lgca.nove_ib_interactions import randomwalk, birth, birthdeath, birthdeath_cancerdfe, go_or_grow, \
            evo_steric, go_or_grow_kappa
        from lgca.interactions import only_propagation
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction in ('random walk', 'random_walk', 'diffusion'):
                self.interaction = randomwalk
            elif interaction == 'only_propagation':
                self.interaction = only_propagation
            elif interaction == 'birth':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * self.maxlabel)

            elif interaction == 'birthdeath':
                self.interaction = birthdeath
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('capacity of channel set to ', self.interaction_params['capacity'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * (self.maxlabel + 1))

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'std' in kwargs:
                    self.interaction_params['std'] = kwargs['std']
                else:
                    self.interaction_params['std'] = 0.01
                    print('standard deviation set to = ', self.interaction_params['std'])
                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 0.
                    print('Rest channel weight set to gamma =', self.interaction_params['gamma'])

                Z = self.velocitychannels + np.exp(self.interaction_params['gamma']) * self.restchannels
                self.channel_weights = [1./Z] * self.velocitychannels + [np.exp(self.interaction_params['gamma'])/Z] * self.restchannels

            elif interaction == 'birthdeath_cancerdfe':
                self.interaction = birthdeath_cancerdfe
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('capacity of channel set to ', self.interaction_params['capacity'])

                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.2
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                self.props.update(r_b=[self.interaction_params['r_b']] * (self.maxlabel + 1))

                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = 0.02
                    print('death rate set to r_d = ', self.interaction_params['r_d'])

                if 'p_d' in kwargs:
                    self.interaction_params['p_d'] = kwargs['p_d']
                else:
                    self.interaction_params['p_d'] = 1.4e-5  # from macfarlane 2014
                    print('probability of drivers set to = ', self.interaction_params['p_d'])

                if 'p_p' in kwargs:
                    self.interaction_params['p_p'] = kwargs['p_p']
                else:
                    self.interaction_params['p_p'] = 0.1  # from macfarlane 2014
                    print('probability of passengers set to = ', self.interaction_params['p_p'])

                if 's_d' in kwargs:
                    self.interaction_params['s_d'] = kwargs['s_d']
                else:
                    self.interaction_params['s_d'] = .1 * self.interaction_params['r_b']  # from macfarlane 2014
                    print('driver strength set to = ', self.interaction_params['s_d'])

                if 's_p' in kwargs:
                    self.interaction_params['s_p'] = kwargs['s_p']
                else:
                    self.interaction_params['s_p'] = .001 * self.interaction_params['r_b']  # from macfarlane 2014
                    print('passenger strength set to = ', self.interaction_params['s_p'])

                if 'a_max' in kwargs:
                    self.interaction_params['a_max'] = kwargs['a_max']
                else:
                    self.interaction_params['a_max'] = 1.
                    print('Max. birth rate set to a_max =', self.interaction_params['a_max'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 0.
                    print('Rest channel weight set to gamma =', self.interaction_params['gamma'])

                Z = self.velocitychannels + np.exp(self.interaction_params['gamma']) * self.restchannels
                self.channel_weights = [1./Z] * self.velocitychannels + [np.exp(self.interaction_params['gamma'])/Z] * self.restchannels

            elif interaction == 'go_or_grow':
                self.interaction = go_or_grow
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    print('There must be exactly one rest channel for this interaction to work!')
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('node capacity set to ', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    print('std of kappa set to', self.interaction_params['kappa_std'])

                if 'theta_std' in kwargs:
                    self.interaction_params['theta_std'] = kwargs['theta_std']
                else:
                    self.interaction_params['theta_std'] = 0.05
                    print('std of theta set to', self.interaction_params['theta_std'])

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
                    kappa = kwargs['kappa']
                    if hasattr(kappa, '__iter__'):
                        self.interaction_params['kappa'] = list(kappa)
                    else:
                        self.interaction_params['kappa'] = [kappa] * (self.maxlabel + 1)
                else:
                    self.interaction_params['kappa'] = [5.] * (self.maxlabel + 1)
                    print('switch rate set to kappa = ', self.interaction_params['kappa'][0])

                self.props.update(kappa=np.array(self.interaction_params['kappa']))
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    if hasattr(theta, '__iter__'):
                        self.interaction_params['theta'] = list(theta)
                    else:
                        self.interaction_params['theta'] = [theta] * (self.maxlabel + 1)
                else:
                    self.interaction_params['theta'] = [0.5] * (self.maxlabel + 1)
                    print('switch threshold set to theta = ', self.interaction_params['theta'][0])
                self.props.update(theta=np.array(self.interaction_params['theta']))

            elif interaction == 'go_or_grow_kappa':
                self.interaction = go_or_grow_kappa
                try:
                    assert self.restchannels > 0
                except AssertionError:
                    print('There must be exactly one rest channel for this interaction to work!')
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 8
                    print('node capacity set to ', self.interaction_params['capacity'])

                if 'kappa_std' in kwargs:
                    self.interaction_params['kappa_std'] = kwargs['kappa_std']
                else:
                    self.interaction_params['kappa_std'] = 0.2
                    print('std of kappa set to', self.interaction_params['kappa_std'])

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
                    kappa = kwargs['kappa']
                    if hasattr(kappa, '__iter__'):
                        self.interaction_params['kappa'] = list(kappa)
                    else:
                        self.interaction_params['kappa'] = [kappa] * (self.maxlabel + 1)
                else:
                    self.interaction_params['kappa'] = [5.] * (self.maxlabel + 1)
                    print('switch rate set to kappa = ', self.interaction_params['kappa'][0])

                self.props.update(kappa=np.array(self.interaction_params['kappa']))
                if 'theta' in kwargs:
                    theta = kwargs['theta']
                    self.interaction_params['theta'] = theta
                else:
                    self.interaction_params['theta'] = 0.5
                    print('switch threshold set to theta = ', self.interaction_params['theta'])

            elif interaction == 'steric_evolution':
                self.interaction = evo_steric
                if 'r_b' in kwargs:
                    self.interaction_params['r_b'] = kwargs['r_b']
                else:
                    self.interaction_params['r_b'] = 0.1
                    print('birth rate set to r_b = ', self.interaction_params['r_b'])
                if 'r_m' in kwargs:
                    self.interaction_params['r_m'] = kwargs['r_m']
                else:
                    self.interaction_params['r_m'] = 1e-3
                    print('mutation rate set to r_m = ', self.interaction_params['r_m'])
                if 'r_d' in kwargs:
                    self.interaction_params['r_d'] = kwargs['r_d']
                else:
                    self.interaction_params['r_d'] = .98 * self.interaction_params['r_b']
                    print('death rate set to r_d = ', self.interaction_params['r_d'])
                if 'alpha' in kwargs:
                    self.interaction_params['alpha'] = kwargs['alpha']
                else:
                    self.interaction_params['alpha'] = 2.0
                    print('steric interaction strength set to alpha = ', self.interaction_params['alpha'])
                if 'gamma' in kwargs:
                    self.interaction_params['gamma'] = kwargs['gamma']
                else:
                    self.interaction_params['gamma'] = 3.0
                    print('rest channel weight set to gamma = ', self.interaction_params['gamma'])
                if 'capacity' in kwargs:
                    self.interaction_params['capacity'] = kwargs['capacity']
                else:
                    self.interaction_params['capacity'] = 512
                    print('deme capacity set to capacity = ', self.interaction_params['capacity'])
                self.init_families(type='homogeneous', mutation=True)
                self.props['family'][0] = 1  # there is no 'void' cell, so the cell w/ id = 0 also belongs to fam. 1
                self.family_props.update(r_b=[0] + [self.interaction_params['r_b']] * self.maxfamily)
                if 'fitness_increase' in kwargs:
                    self.interaction_params['fitness_increase'] = kwargs['fitness_increase']
                else:
                    self.interaction_params['fitness_increase'] = 1.1
                    print('fitness increase for driver mutations set to ',
                          self.interaction_params['fitness_increase'])
            else:
                print('interaction', kwargs['interaction'], 'is not defined! Random walk used instead.')
                print('Implemented interactions:', self.interactions)
                self.interaction = randomwalk

        else:
            print('Random walk interaction is used.')
            self.interaction = randomwalk

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, recordchanneldens=False,
                showprogress=True, recordfampop=False):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = get_arr_of_empty_lists((timesteps +1,) + self.dims + (self.K,))
            self.nodes_t[0, ...] = copy(self.nodes[self.nonborder])
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.uint)
            self.n_t[0] = self.cell_density[self.nonborder].sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1,) + self.dims, dtype=np.uint)
            self.dens_t[0, ...] = self.cell_density[self.nonborder]
        if recordchanneldens:
            self.channel_pop_t = np.zeros((timesteps + 1,) + self.dims + (self.K,), dtype=np.uint)
            self.channel_pop_t[0, ...] = self.channel_pop[self.nonborder]
        if recordfampop:
            from lgca.nove_ib_interactions import evo_steric
            # this needs to include all interactions that can increase the number of recorded families!
            if self.interaction in [evo_steric]:
                # if mutations are allowed, this is a list because it will be ragged due to increasing family numbers
                self.fam_pop_t = [self.calc_family_pop_alive()]
                is_mutating = True
            else:
                if 'family' not in self.props:
                    raise RuntimeError("Interaction does not deal with families, "
                                       "family population can therefore not be recorded.")
                # otherwise standard procedure
                self.fam_pop_t = np.zeros((timesteps + 1, self.maxfamily + 1))
                self.fam_pop_t[0, ...] = self.calc_family_pop_alive()
                is_mutating = False

        for t in tqdm(iterable=range(1, timesteps + 1), disable=1-showprogress):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = copy(self.nodes[self.nonborder])
            if recordN:
                self.n_t[t] = self.cell_density[self.nonborder].sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.nonborder]
            if recordfampop:
                if is_mutating:
                    # append to the ragged nested list
                    self.fam_pop_t.append(self.calc_family_pop_alive())
                else:
                    # standard procedure
                    try:
                        self.fam_pop_t[t, ...] = self.calc_family_pop_alive()
                    except ValueError as e:
                        raise ValueError("Number of families has increased, interaction must be included in the case " +
                                         "distinction for the recordfampop keyword in the IBLGCA base timeevo function!") from e
        if recordfampop and is_mutating:
            self.straighten_family_populations()

    def calc_max_label(self):
        cells = self.nodes.sum()
        if len(cells) == 0:
            self.maxlabel = None

        else: self.maxlabel = max(cells)

    def get_prop(self, nodes=None, props=None, propname=None):
        """
        Return array of property "propname" on every node, given the lattice configuration "nodes" and the property
        dictionary "props".
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(self.props))

        prop = np.array(props[propname])
        proparray = prop[nodes.sum()]
        return proparray

    def calc_prop_mean(self, nodes=None, props=None, propname=None):
        """
        Calculate the mean value of a property "propname" on every node, given the node configuration "nodes" and
        the properties "props"
        """
        cells = nodes.sum(-1)
        mean_prop = np.ma.masked_all(cells.shape)
        proparray = np.array(props[propname])
        for ind, loccells in np.ndenumerate(cells):
            if loccells:
                nodeprops = proparray[loccells]
                mean_prop[ind] = nodeprops.mean()
                mean_prop.mask[ind] = 0
        return mean_prop

    def calc_prop_mean_spatiotemp(self, nodes_t=None, props=None):
        """
        Given the lgca states nodes_t, calculate the mean of individual cell properties at each node for each time point.
        :param nodes_t: Recorded node states. If omitted, use recorded state of latest timeevo.
        :param props: dictionary of individual cell properties. If omitted, props dictionary of current instance is used
        :return:
        """
        if nodes_t is None:
            nodes_t = self.nodes_t
        if props is None:
            props = self.props
        tmax = nodes_t.shape[0]
        for key in self.props:
            self.mean_prop_t[key] = np.ma.masked_all((tmax, *self.dims))
            # self.mean_prop_vel_t[key] = np.zeros([tmax,l])
            # self.mean_prop_rest_t[key] = np.zeros([tmax,l])
            self.mean_prop_t[key] = self.calc_prop_mean(propname=key, props=props, nodes=nodes_t)
            # self.mean_prop_vel_t[key][t] = self.calc_prop_mean(propname=key, props=props, nodes=nodes_t[t][...,0:(self.K-1)])
            # self.mean_prop_rest_t[key][t] = self.calc_prop_mean(propname=key, props=props, nodes=nodes_t[t][...,(self.K-1):])

        return self.mean_prop_t


    def plot_prop_timecourse(self, nodes_t=None, props=None, propname=None, figindex=None, figsize=None, **kwargs):
        """
        Plot the time evolution of the cell property 'propname'
        :param nodes_t:
        :param props:
        :param propname:
        :param figindex:
        :param figsize:
        :param kwargs: keyword arguments for the matplotlib.plot command
        :return:
        """

        if nodes_t is None:
            nodes_t = self.nodes_t

        if props is None:
            props = self.props

        if propname is None:
            propname = next(iter(self.props))

        proparray = np.array(props[propname])
        prop_t = [proparray[nodes.sum()] for nodes in nodes_t]
        mean_prop_t = np.array([np.mean(prop) if len(prop) > 0 else np.nan for prop in prop_t])
        std_mean_prop_t = np.array \
            ([np.std(prop, ddof=1) / np.sqrt(len(prop)) if len(prop) > 0 else np.nan for prop in prop_t])
        if figindex is None:
            fig = plt.gcf()

        else:
            fig = plt.figure(num=figindex)

        if figsize is not None:
            fig.set_size_inches(figsize)

        tmax = nodes_t.shape[0]
        yerr = std_mean_prop_t
        x = np.arange(tmax)
        y = mean_prop_t

        plt.xlabel('$t$')
        plt.ylabel('${}$'.format(propname))
        plt.title('Time course of the cell property')
        line = plt.plot(x, y, **kwargs)
        errors = plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, antialiased=True, interpolate=True)
        return line, errors

    def plot_prop_hist(self, nodes=None, props=None, propname=None, figindex=None, figsize=None, **kwargs):
        """
        Plot histogram of cell property 'propname' of cells in 'nodes'. Per default, the current lgca state and the
        first property is shown.
        :param nodes:
        :param props:
        :param propname:
        :param figindex:
        :param figsize:
        :param kwargs:
        :return:
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propname is None:
            propname = next(iter(props))

        propvals = [props[propname][id] for id in nodes.sum()]
        plt.figure(num=figindex, figsize=figsize)
        plt.hist(propvals, **kwargs)
        plt.xlabel('{}'.format(propname))
        plt.ylabel('Count')

    def plot_prop_2dhist(self, nodes=None, props=None, propnames=None, figindex=None, figsize=None, **kwargs):
        """
        Plot a 2d-histogram of two cell properties given by 'propnames' of all cells in 'nodes'. By default, the current
        lgca state is used and the first two properties are shown.
        :param nodes:
        :param props:
        :param propnames:
        :param figindex:
        :param figsize:
        :param kwargs:
        :return:
        """
        import seaborn as sns
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propnames is None:
            names = iter(props)
            propname1 = next(names)
            propname2 = next(names)

        ids = [id for id in nodes.sum()]
        propvals1, propvals2 = [props[propname1][id] for id in ids], [props[propname2][id] for id in ids]
        # plt.figure(num=figindex, figsize=figsize)
        sns.jointplot(x=propvals1, y=propvals2, marginal_ticks=True, kind='hist', **kwargs)
        plt.xlabel('{}'.format(propname1))
        plt.ylabel('{}'.format(propname2))

    def calc_family_pop_alive(self):
        """
        Calculate how many cells of each family are alive.
        :returns: np.ndarray fam_pop_array - array of family population counts indexed by family ID
        """
        if 'family' not in self.props:
            raise RuntimeError("Family properties are not recorded by the LGCA, choose suitable interaction.")

        cells_alive = np.array(self.nodes[self.nonborder].sum()) # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
        fam_alive, fam_pop = np.unique(cell_fam_alive, return_counts=True)  # count number of cells for each family
        # transform into array with population entry for all families that ever existed
        fam_pop_array = np.zeros(self.maxfamily+1, dtype=int)
        fam_pop_array[fam_alive] = fam_pop
        return fam_pop_array
        # alternative: look up family for each live cell, then do unique on those altered nodes

    def list_families_alive(self):
        """
        Calculate which families are alive.
        :returns: np.ndarray - array of family IDs in ascending order
        """
        cells_alive = np.array(self.nodes[self.nonborder].sum(-1))  # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
        return np.unique(cell_fam_alive) # remove duplicate entries
