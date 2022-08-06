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
"""

import sys
from abc import ABC, abstractmethod
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from numpy import random as npr
from sympy.utilities.iterables import multiset_permutations
from copy import copy, deepcopy
from lgca.plots import muller_plot
import warnings

# configure matplotlib style
plt.style.use('default')


def update_progress(progress: float):
    """
    Update the progress bar on the standard output.

    Writes progress strings like ``"[####----] 50%"`` to the standard output, replacing 
    the currently active line. Used in :py:meth:`LGCA_base.timeevo`.

    Parameters
    -----------
    progress : float
        Fraction of the work done. A value <0 indicates a halt of the process. 
        Values >=1 are interpreted as a finished process.

    """
    # Modify this to change the length of the progress bar in signs
    barLength = 20
    # determine the status from the progress input
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "Progress variable must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    # assemble the string to print
    # \r carriage return causes it to overwrite the line each time it is updated
    block = int(round(barLength * progress))
    text = "\rProgress: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 1),
                                               status)
    # print to standard output
    sys.stdout.write(text)
    sys.stdout.flush()


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
    if ncolors > 101:
        stride = 10
    elif ncolors > 51:
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
        by the default value for the figure size in :py:meth:`setup_figure` of the 
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


def calc_nematic_tensor(v):
    # Todo Simon
    return np.einsum('...i,...j->...ij', v, v) - 0.5 * np.diag(np.ones(2))[None, ...]


class LGCA_base(ABC):
    """
    Abstract base class for classical LGCA with volume exclusion. 

    It holds all methods and attributes that are common for all geometries. 
    Cannot simulate on its own. 
    If you want to use it, instantiate one of the geometry-specific derived classes. 

    Parameters
    ----------
    bc : {'absorbing', 'reflecting', 'periodic', 'inflow'}, default='periodic'
        Boundary conditions. Not all bc are supported in all geometries (yet). 

        Aliases: absorbing: ``'absorb', 'abs', 'abc'``; reflecting: ``'reflect', 'refl', 'rbc'``; 
        periodic: ``'pbc'``.
    density : float, default=0.1
        If `nodes` is None, initialize lattice randomly with this particle density.
    dims : tuple or int
        Lattice dimensions. Must match with specified geometry. An integer for a 2D geometry is interpreted as 
        ``(dims, dims)``.
    nodes : :py:class:`numpy.ndarray`
        Custom initial lattice configuration.
    restchannels : int, default=0
        Number of resting channels.
    **kwargs
        Further arguments for the initial condition and/or the interaction.

    Attributes
    ----------
    apply_boundaries : callable
        Function implementing the boundary conditions.
    c
    cell_density : :py:class:`numpy.ndarray`
        Number of particles at each lattice node in the current LGCA state. Computed field. Dimensions: 
        :py:attr:`lgca.dims`.
    cij : :py:class:`numpy.ndarray`
        Nematic tensor. Element-wise multiplication of neighborhood vectors with themselves. Computed from the geometry. 
        Dimensions: 
        ``(lgca.c.shape[1], lgca.c.shape[0], lgca.c.shape[0})``. First dimension: neighborhood vector, second and 
        third dimension: combination of x and y components of the vector as 
        ``[[cix*cix, cix*ciy], [ciy*cix, ciy*ciy]]``. 
    concentration
        Internal variable for the chemotaxis interaction.
    dens_t : :py:class:`numpy.ndarray`
        Number of particles at each lattice node for all timesteps in the previous simulation. 
        Only available after a simulation performed with ``timeevo(recorddens=True)``. 
        Dimensions: ``(timesteps,) + lgca.dims``. 
    dims : tuple
        Lattice dimensions/size of the lattice as ``(xdim,)`` (1D LGCA) or ``(xdim, ydim)`` (2D LGCA), 
        excluding shadow nodes on the border. 
    guiding_tensor
        Internal variable for the contact_guidance interaction.
    interaction : callable
        Interaction rule assigned to the LGCA.
    interactions
    j : list of :py:class:`numpy.ndarray`
        Flux for each possible channel configuration. Dimensions: ``(lgca.K + 1, len(lgca.c))``.
    K : int
        Number of channels per node. Equal to ``lgca.velocitychannels + lgca.restchannels``.
    n_crit
        Internal variable for the wetting interaction.
    n_t : :py:class:`numpy.ndarray`
        Sum of particles in the lattice for all timesteps in the previous simulation. 
        Only available after a simulation performed with ``timeevo(recordN=True)``. Dimensions: ``(timesteps,)``. 
    nodes : :py:class:`numpy.ndarray`
        State of the lattice, configuration of all channels. Dimensions: ``(lgca.l + 2*lgca.r_int, lgca.K)`` 
        (in 1D LGCA) or ``(lgca.lx + 2*lgca.r_int, lgca.ly + 2*lgca.r_int, lgca.K)``. Includes shadow nodes 
        on all borders for implementing boundary conditions. 
    nodes_t : :py:class:`numpy.ndarray`
        Full lattice configuration of non-border nodes for all timesteps in the previous simulation. 
        Only available after a simulation performed with ``timeevo(record=True)``. 
        Dimensions: ``(timesteps,) + lgca.dims + (K,)``. 
    nonborder : tuple of :py:class:`numpy.ndarray`
        Indices of non-border nodes in the :py:attr:`lgca.nodes` array as ``(x-indices,)`` (in 1D LGCA) or 
        ``(x-indices, y-indices)`` (in 2D LGCA), i.e. all nodes excluding shadow nodes for boundary conditions. 
        Both arrays x-indices and y-indices have the dimensions :py:attr:`lgca.dims`. 
    permutations : list of :py:class:`numpy.ndarray`
        All possible configurations for a lattice site with :py:attr:`lgca.K` channels. Dimensions: 
        ``(lgca.K + 1, n, lgca.K)``. n is the number of possible permutations for the 
        node if x channels are occupied, where x is given by the first dimension. 
    r_int : int, default=1
        Interaction radius. Must be at least 1 to handle propagation.
    restchannels : int
        Number of resting channels.
    si : list of :py:class:`numpy.ndarray`
        Nematic tensor for all possible node configurations, obtained from :py:attr:`lgca.permutations` and 
        :py:attr:`py.cij`. Dimensions: ``(lgca.K + 1, n, len(lgca.c), len(lgca.c))``. n is the number of possible 
        permutations for the node if x channels are occupied, where x is given by the first dimension. 
    velcells_t, restcells_t : :py:class:`numpy.ndarray`
        Sum of particles in velocity/rest channels, respectively, for all timesteps in the previous simulation. 
        Only available after a simulation performed with ``timeevo(recordpertype=True)``. 
        Dimensions: ``(timesteps,) + lgca.dims``.
    velocitychannels

    See Also
    --------
    lgca_1d.LGCA_1D : Classical LGCA in a 1D geometry.
    lgca_square.LGCA_Square : Classical LGCA in a 2D square geometry.
    lgca_hex.LGCA_Hex : Classical LGCA in a 2D hexagonal geometry.

    """

    @property
    @abstractmethod
    def interactions(self) -> list:
        """(Class attribute.) List of interaction functions suitable for this type of LGCA."""
        # This is only a helper class, it cannot simulate! Use one the following classes:
        # LGCA_1D, LGCA_Square, LGCA_Hex

        # ... notation as of https://stackoverflow.com/a/58321197
        ...

    @property
    @abstractmethod
    def velocitychannels(self) -> int:
        """(Class attribute.) Number of velocity channels."""
        ...

    @property
    @abstractmethod
    def c(self) -> np.ndarray:
        """(Class attribute.) Array of the velocity channel vectors. Dimensions: ``(dims, lgca.velocitychannels)``, 
        where dims is 1 or 2 depending on the geometry."""
        ...

    @abstractmethod
    def set_dims(self, dims=None, nodes=None, restchannels=0):
        """
        Set LGCA dimensions. In the implementation, set :py:attr:`self.K`, :py:attr:`self.restchannels` 
        and :py:attr:`self.dims` to meaningful and consistent values.

        Must match what is done in :py:meth:`init_coords` and :py:meth:`init_nodes`. 
        For arguments and attribute types see :py:class:`lgca.base.LGCA_base`.
        """
        ...

    @abstractmethod
    def init_coords(self):
        """
        Initialize LGCA coordinates. These are used to index the lattice nodes. In the implementation, 
        set :py:attr:`self.nonborder`, :py:attr:`self.xcoords`, :py:attr:`self.ycoords`, 
        and :py:attr:`self.coord_pairs` to meaningful and consistent values.

        Must match what is done in :py:meth:`set_dims` and :py:meth:`init_nodes`. 
        For the attribute types see :py:class:`lgca.base.LGCA_base`.
        """
        ...

    @abstractmethod
    def init_nodes(self, density, nodes=None, **kwargs):
        """
        Initialize LGCA lattice configuration. Create the lattice and then assign particles to 
        channels in the nodes. In the implementation, set :py:attr:`self.nodes`.

        Must match what is done in :py:meth:`set_dims` and :py:meth:`init_coords`. 
        For arguments and attribute types see :py:class:`lgca.base.LGCA_base`.
        """
        ...

    @abstractmethod
    def gradient(self, qty):
        """
        Compute the gradient of qty along all axes.

        Parameters
        ----------
        qty : :py:class:`numpy.ndarray`
            Quantity to take the gradient of. Needs to have the same number of dimensions as :py:attr:`self.nodes`. 
            If ``qty.shape == self.nodes.shape[:-1]`` the result can be indexed with the LGCA coordinates (see example).

        Returns
        -------
        :py:class:`numpy.ndarray`
            Computed gradient. Dimensions: ``qty.shape + (len(self.c),)``. If ``self`` and ``qty`` are 2D arrays, 
            ``gradient(qty)[...,0]`` is the gradient in x direction and ``gradient(qty)[...,1]`` the gradient in 
            y direction.

        Notes
        -----
        The gradient is calculated using :py:func:`numpy.gradient()` with stepwidth h=0.5 
        (s.t. no normalization takes place). 
        It is computed as the central finite difference with equidistant support points and supports one-sided 
        differences at the boundaries.

        In most cases this yields the simple difference between the two closest array elements in the given direction. 
        For example, the gradient at position 1 of ``np.array([1, 2, 4])`` would be (4 - 1)/(2 * 0.5) = 3.

        Examples
        --------
        If the input quantity has the same x (and y) dimensions as the LGCA's nodes, the gradient at each node 
        position can be accessed the same way as the node itself.

        >>> from lgca import get_lgca
        >>> import numpy as np
        >>> # define a square LGCA to illustrate dimensions
        >>> lgca = get_lgca(geometry='square', dims=(2,3))
        >>> lgca.nodes.shape  # (xdim, ydim, number of channels)
        (4, 5, 4)
        >>> my_qty = np.array([[0,0,0,0,0],
        >>>                    [1,1,1,1,1],
        >>>                    [2,2,2,3,2],
        >>>                    [3,3,3,3,3]])
        >>> my_qty.shape  # (xdim, ydim)
        (4, 5)
        >>> grad = lgca.gradient(my_qty)
        >>> grad.shape  # (xdim, ydim, number of dimensions)
        (4, 5, 2)
        >>> # address like internal LGCA fields: first dimension is x (printed vertically),
        >>> # second dimension is y (printed horizontally), this can be a bit confusing
        >>> for coord in lgca.coord_pairs:
        >>>     if np.any(grad[coord]>2):
        >>>         print("Gradient at index", coord, "is ", grad[coord])
        >>>         print("Configuration at index ", coord, " is ", lgca.nodes[coord],
        >>>               ", with cell density ", lgca.cell_density[coord])
        Gradient at index (1, 3) is  [3. 0.] 
        Configuration at index  (1, 3)  is  [False False False  True] , with cell density  1 

        The first element of the gradient holds the gradient in x direction, the second element the gradient in 
        y direction. Note that ``(1, 3)`` is the index corresponding to a logical non-border coordinate ``(0, 2)`` 
        if the interaction radius is 1. This is relevant for defining a custom field qty: Only the field values at 
        non-border indices will be "felt" by the particles in the LGCA if the interaction is defined accordingly, 
        but border nodes can be used to specify the field's boundary conditions. 

        The gradient in x direction is 3 = (3 - 0)/1. In y direction it is 0 = (1 - 1)/1.

        """
        ...

    @abstractmethod
    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.

        Propagate the particles by updating :py:attr:`self.nodes`, respecting the geometry. 
        Boundary conditions are enforced later by :py:meth:`apply_boundaries`.
        """
        ...

    def apply_pbc(self):
        """
        Apply periodic boundary conditions.

        Update :py:attr:`self.nodes`, using the shadow border nodes and respecting the geometry.
        """
        raise NotImplementedError("Periodic boundary conditions not yet implemented for class " +
                                  str(self.__class__)+".")

    def apply_rbc(self):
        """
        Apply reflecting boundary conditions.

        Update :py:attr:`self.nodes`, using the shadow border nodes and respecting the geometry.
        """
        raise NotImplementedError("Reflecting boundary conditions not yet implemented for class " +
                                  str(self.__class__)+".")

    def apply_abc(self):
        """
        Apply absorbing boundary conditions.

        Update :py:attr:`self.nodes`, using the shadow border nodes and respecting the geometry.
        """
        raise NotImplementedError("Absorbing boundary conditions not yet implemented for class " +
                                  str(self.__class__) + ".")

    def apply_inflowbc(self):
        """
        Apply inflow boundary conditions.

        Update :py:attr:`self.nodes`, using the shadow border nodes and respecting the geometry.

        """
        raise NotImplementedError("Inflow boundary conditions not yet implemented for class "+str(self.__class__)+".")

    def __init__(self, nodes=None, dims=None, restchannels=0, density=0.1, bc='periodic', **kwargs):
        """ Initialize class instance. See class docstring."""
        self.r_int: int = 1  # Interaction radius. Must be at least 1 to handle propagation.
        # set boundary conditions, set self.apply_boundaries
        self.set_bc(bc)

        # initialize lattice
        # define self.K, self.restchannels, self.dims, self.l or self.lx and self.ly
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        # define self.nonborder, self.xcoords, (self.ycoords), self.coord_pairs
        self.init_coords()
        # define self.init_nodes
        self.init_nodes(density=density, nodes=nodes, **kwargs)

        # compute initial value of fields computed from the lattice state
        # define self.cell_density
        self.update_dynamic_fields()

        # set interaction rule
        self.interaction_params = {}
        # define self.interaction and potentially self.permutations, self.j, self.cij, self.si
        self.set_interaction(**kwargs)

    def set_r_int(self, r):
        """
        Change the interaction radius. Update shadow border nodes accordingly.

        Extended Summary
        ----------------
        This has effects on :py:attr:`self.nodes`, the coordinates and the computed fields.

        Parameters
        ----------
        r : int
            New interaction radius.

        """
        self.r_int = r
        self.init_nodes(nodes=self.nodes[self.nonborder])
        self.init_coords()
        self.update_dynamic_fields()

    def set_interaction(self, **kwargs):
        """
        Set the interaction rule and respective needed parameters.

        Set :py:attr:`self.interaction` and possibly add entries in :py:attr:`self.interaction_params`. 
        Do not use this to specify a custom interaction. In order to do this (as of now), :py:attr:`self.interaction` 
        and :py:attr:`self.interaction_params` must be manipulated directly from an external script.

        Parameters
        ----------
        kwargs['interaction'] : str, default='random_walk'
            Name of the predefined interaction in :py:mod:`lgca.interactions`.
        **kwargs
            Interaction parameters.

        """
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
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                         'reflect'))
                    else:
                        source = npr.normal(self.l / 2, 1)
                        r = abs(self.xcoords - source)
                        self.concentration = np.exp(-2 * r / self.l)
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                         'reflect'))
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
                    self.interaction_params['gradient_field'] = np.zeros((self.lx + 2 * self.r_int,
                                                                          self.ly + 2 * self.r_int, 2))
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
        """
        Set the boundary conditions.

        Selects a method which is called every timestep to enforce boundary conditions. 
        The methods to select from are implemented in the derived classes. The chosen one is assigned to 
        :py:meth:`self.apply_boundaries`.

        Parameters
        ----------
        bc : {'absorbing', 'reflecting', 'periodic', 'inflow'}
            Boundary conditions. Not all bc are supported in all geometries (yet). 

            Aliases: absorbing: ``'absorb', 'abs', 'abc'``; reflecting: ``'reflect', 'refl', 'rbc'``; 
            periodic: ``'pbc'``.

        """
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
        """
        Calculate the flux vector for all lattice sites in `nodes`.

        The elements of the flux vectors are computed as the dot product between the LGCA's neighborhood vectors and 
        the velocity channel configuration in `nodes`.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`
            Lattice configuration to compute the flux for. Must have more than or the same number of 
            dimensions as :py:attr:`self.nodes` and ``nodes.shape[-1] >= self.velocitychannels``. 
            Is typically :py:attr:`self.nodes`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Array of flux vectors at each lattice site. Dimensions: ``nodes.shape[:-1] + (len(self.c),)``.

        """
        # Todo: add example using the flux
        # 1st+ dim: lattice sites, last dim: channels
        # dot product between c vectors and actual configuration of site
        return np.einsum('ij,...j', self.c, nodes[..., :self.velocitychannels])

    def print_interactions(self):
        """Print the list of pre-implemented interactions for this LGCA type."""
        print(self.interactions)

    def print_nodes(self):
        """Print the full lattice configuration as integers."""
        print(self.nodes.astype(int))

    def random_reset(self, density):
        """
        Initialize lattice nodes with average density `density`. Channels are occupied at random and nodes can 
        have different particle numbers.

        For each channel a random number is drawn. If it is lower than `density`, the channel is filled, 
        otherwise it stays empty.

        Parameters
        ----------
        density : float
            Desired average particle density of the lattice. 
            ``density = total_number_of_particles / (number_of_nodes * number_of_channels_per_node)``.

        See Also
        --------
        homogeneous_random_reset : Initialize the lattice randomly with a fixed number of particles per node.

        """
        self.nodes = npr.random(self.nodes.shape) < density
        self.apply_boundaries()
        self.update_dynamic_fields()

    def homogeneous_random_reset(self, density):
        """
        Initialize lattice nodes with average density `density`. Channels are occupied at random and all nodes 
        have the same particle number.

        The particle number per node that matches `density` most closely is determined. The configuration for one 
        node with this number of particles is then permutated to fill the lattice.


        Parameters
        ----------
        density : float
            Desired average density of the lattice. 
            ``density = total_number_of_particles / (number_of_nodes * number_of_channels_per_node)``. 
            Here also: ``density = number_of_particles_per_node / number_of_channels_per_node``.

        See Also
        --------
        random_reset : Initialize the lattice randomly with a varying number of particles per node.

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
        channels = np.array([npr.permutation(channels) for _ in range(n_nodes)])
        self.nodes = channels.reshape(self.nodes.shape)

        self.apply_boundaries()
        self.update_dynamic_fields()
        # achieved density
        # eff_dens = self.nodes[self.nonborder].sum() / (self.K * self.cell_density[self.nonborder].size)

    def update_dynamic_fields(self):
        """
        Update "fields" from the current LGCA state that store important variables to compute other dynamic steps.

        Computes :py:attr:`self.cell_density`, number of particles at each lattice node.
        """
        self.cell_density = self.nodes.sum(-1)

    def timestep(self):
        """
        Update the state of the LGCA from time k to k+1. Includes the interaction and propagation steps.
        """
        self.interaction(self)
        self.apply_boundaries()
        self.propagation()
        self.apply_boundaries()
        self.update_dynamic_fields()

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True,
                recordpertype=False):
        """
        Perform a simulation of the LGCA for `timesteps` timesteps.

        Different quantities can be recorded during the simulation, e.g. the total number of particles at each 
        timestep. They are stored in LGCA attributes.

        Parameters
        ----------
        timesteps : int, default=100
            How long the simulation should be performed.
        record : bool, default=False
            Record the full lattice configuration for each timestep in :py:attr:`self.nodes_t`.
        recorddens : bool, default=True
            Record the number of particles at each lattice site for each timestep in :py:attr:`self.dens_t`.
        recordN : bool, default=False
            Record the total number of particles in the lattice for each timestep in :py:attr:`self.n_t`.
        recordpertype : bool, default=False
            Record the number of particles in velocity channels/resting channels at each lattice site for 
            each timestep in :py:attr:`self.velcells_t` and :py:attr:`self.restcells_t`, respectively.
        showprogress : bool, default=True
            Show a simple progress bar with a percentage of performed timesteps in the standard output.

        """
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
        for t in range(1, timesteps + 1):
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
            if showprogress:
                update_progress(1.0 * t / timesteps)

    def calc_permutations(self):
        """
        Precompute quantities that only depend on the geometry and lattice definition, but not on the current 
        configuration, for reuse in interaction functions. This speeds up concerned interactions.

        Currently computed quantities are a list of all possible node configurations (:py:attr:`self.permutations`), 
        the flux for each possible node configuration (:py:attr:`self.j`), all nematic tensor possibilities 
        (:py:attr:`self.cij`) and the nematic tensor for all possible node configurations (:py:attr:`self.si`).

        """
        # list of all possible configurations for a lattice site
        self.permutations = [np.array(list(multiset_permutations([1] * n + [0] * (self.K - n))), dtype=np.int8)
                             for n in range(self.K + 1)]
                                                                # builds list with one configuration for particle number
                                                                # n in an array of size self.K (velocity + resting)
                                      # builds list with all possible permutations for this array
                             # for all possible particle numbers in an array of size self.K (velocity + resting)
        # first dim: number of particles
        # second dim: all permutations for this n
        # third dim: channels

        # array of flux for each permutation for each number of particles
        self.j = [np.dot(self.c, self.permutations[n][:, :self.velocitychannels].T) for n in range(self.K + 1)]
        # dot product between the neighborhood vectors and the particles in the velocity channels
        # for each possible number of particles
        # first dim: number of particles
        # second dim: flux vector for each permutation (directions as specified in c)

        # element-wise multiplication of neighborhood vectors with themselves
        self.cij = np.einsum('ij,kj->jik', self.c, self.c) - 0.5 * np.diag(np.ones(2))[None, ...]
        # 1st dim: neighborhood vector
        # 2nd and 3rd dim: combination of x and y components of the vector as [[xx, xy], [yx, yy]]


        self.si = [np.einsum('ij,jkl', self.permutations[n][:, :self.velocitychannels], self.cij) for n in
                   range(self.K + 1)]
        # filter out self.cij with occupation of velocity channels corresponding to the neighborhood vectors
        # list for all possible particle numbers
        # 1st dim: number of particles
        # 2nd dim: permutation
        # 3rd dim and 4th dim: nematic tensor for each configuration
        # -> combination of x and y components of the result as [[xx, xy], [yx, yy]]

    def total_population(self):
        """
        Calculate the amount of particles in the lattice.

        Returns
        -------
        int
            Total population size.
        """
        return int(self.cell_density[self.nonborder].sum())


class IBLGCA_base(LGCA_base, ABC):
    """
    Abstract base class for identity-based LGCA with volume exclusion.

    It holds all methods and attributes that are common for all geometries.
    Cannot simulate on its own.
    If you want to use it, instantiate one of the geometry-specific derived classes.

    Parameters
    ----------
    bc : {'absorbing', 'reflecting', 'periodic', 'inflow'}, default='periodic'
        Boundary conditions. Not all bc are supported in all geometries (yet).

        Aliases: absorbing: ``'absorb', 'abs', 'abc'``; reflecting: ``'reflect', 'refl', 'rbc'``;
        periodic: ``'pbc'``.
    density : float, default=0.1
        If `nodes` is None, initialize lattice randomly with this particle density.
    dims : tuple or int
        Lattice dimensions. Must match with specified geometry. An integer for a 2D geometry is interpreted as
        ``(dims, dims)``.
    nodes : :py:class:`numpy.ndarray`
        Custom initial lattice configuration.
    restchannels : int, default=0
        Number of resting channels.
    **kwargs
        Further arguments for the initial condition and/or the interaction.

    Attributes
    ----------
    # TODO!! ##############
    apply_boundaries : callable
        Function implementing the boundary conditions.
    c
    cell_density : :py:class:`numpy.ndarray`
        Number of particles at each lattice node in the current LGCA state. Computed field. Dimensions:
        :py:attr:`lgca.dims`.
    cij : :py:class:`numpy.ndarray`
        Nematic tensor. Element-wise multiplication of neighborhood vectors with themselves. Computed from the geometry.
        Dimensions:
        ``(lgca.c.shape[1], lgca.c.shape[0], lgca.c.shape[0})``. First dimension: neighborhood vector, second and
        third dimension: combination of x and y components of the vector as
        ``[[cix*cix, cix*ciy], [ciy*cix, ciy*ciy]]``.
    concentration
        Internal variable for the chemotaxis interaction.
    dens_t : :py:class:`numpy.ndarray`
        Number of particles at each lattice node for all timesteps in the previous simulation.
        Only available after a simulation performed with ``timeevo(recorddens=True)``.
        Dimensions: ``(timesteps,) + lgca.dims``.
    dims : tuple
        Lattice dimensions/size of the lattice as ``(xdim,)`` (1D LGCA) or ``(xdim, ydim)`` (2D LGCA),
        excluding shadow nodes on the border.
    guiding_tensor
        Internal variable for the contact_guidance interaction.
    interaction : callable
        Interaction rule assigned to the LGCA.
    interactions
    j : list of :py:class:`numpy.ndarray`
        Flux for each possible channel configuration. Dimensions: ``(lgca.K + 1, len(lgca.c))``.
    K : int
        Number of channels per node. Equal to ``lgca.velocitychannels + lgca.restchannels``.
    n_crit
        Internal variable for the wetting interaction.
    n_t : :py:class:`numpy.ndarray`
        Sum of particles in the lattice for all timesteps in the previous simulation.
        Only available after a simulation performed with ``timeevo(recordN=True)``. Dimensions: ``(timesteps,)``.
    nodes : :py:class:`numpy.ndarray`
        State of the lattice, configuration of all channels. Dimensions: ``(lgca.l + 2*lgca.r_int, lgca.K)``
        (in 1D LGCA) or ``(lgca.lx + 2*lgca.r_int, lgca.ly + 2*lgca.r_int, lgca.K)``. Includes shadow nodes
        on all borders for implementing boundary conditions.
    nodes_t : :py:class:`numpy.ndarray`
        Full lattice configuration of non-border nodes for all timesteps in the previous simulation.
        Only available after a simulation performed with ``timeevo(record=True)``.
        Dimensions: ``(timesteps,) + lgca.dims + (K,)``.
    nonborder : tuple of :py:class:`numpy.ndarray`
        Indices of non-border nodes in the :py:attr:`lgca.nodes` array as ``(x-indices,)`` (in 1D LGCA) or
        ``(x-indices, y-indices)`` (in 2D LGCA), i.e. all nodes excluding shadow nodes for boundary conditions.
        Both arrays x-indices and y-indices have the dimensions :py:attr:`lgca.dims`.
    permutations : list of :py:class:`numpy.ndarray`
        All possible configurations for a lattice site with :py:attr:`lgca.K` channels. Dimensions:
        ``(lgca.K + 1, n, lgca.K)``. n is the number of possible permutations for the
        node if x channels are occupied, where x is given by the first dimension.
    r_int : int, default=1
        Interaction radius. Must be at least 1 to handle propagation.
    restchannels : int
        Number of resting channels.
    si : list of :py:class:`numpy.ndarray`
        Nematic tensor for all possible node configurations, obtained from :py:attr:`lgca.permutations` and
        :py:attr:`py.cij`. Dimensions: ``(lgca.K + 1, n, len(lgca.c), len(lgca.c))``. n is the number of possible
        permutations for the node if x channels are occupied, where x is given by the first dimension.
    velcells_t, restcells_t : :py:class:`numpy.ndarray`
        Sum of particles in velocity/rest channels, respectively, for all timesteps in the previous simulation.
        Only available after a simulation performed with ``timeevo(recordpertype=True)``.
        Dimensions: ``(timesteps,) + lgca.dims``.
    velocitychannels

    See Also
    --------
    lgca_1d.IBLGCA_1D : Identity-based LGCA in a 1D geometry.
    lgca_square.IBLGCA_Square : Identity-based LGCA in a 2D square geometry.
    lgca_hex.IBLGCA_Hex : Identity-based LGCA in a 2D hexagonal geometry.

    """

    def __init__(self, nodes=None, dims=None, restchannels=0, density=0.1, bc='periodic', **kwargs):
        """ Initialize class instance. See class docstring."""
        self.r_int = 1  # Interaction radius. Must be at least 1 to handle propagation.
        # set boundary conditions, set self.apply_boundaries
        self.set_bc(bc)

        # initialize lattice
        # define self.K, self.restchannels, self.dims, self.l or self.lx and self.ly
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        # define self.nonborder, self.xcoords, (self.ycoords), self.coord_pairs
        self.init_coords()
        # define self.init_nodes
        self.init_nodes(density=density, nodes=nodes)

        # prepare structures for identity information
        self.props = {}  # dictionary for cells' properties
        self.maxlabel = self.nodes.max()  # maximum particle ID

        # compute initial value of fields computed from the lattice state
        # define self.cell_density and self.occupied
        self.update_dynamic_fields()

        # set interaction rule
        self.interaction_params = {}
        # define self.interaction and potentially self.permutations, self.j, self.cij, self.si
        self.set_interaction(**kwargs)

    def set_interaction(self, **kwargs):
        """
        Set the interaction rule and respective needed parameters.

        Set :py:attr:`self.interaction` and possibly add entries in :py:attr:`self.interaction_params` and
        :py:attr:`self.props`.
        If inheritance is involved, initialize :py:attr:`self.family_props` and :py:attr:`self.maxfamily`.
        Do not use this to specify a custom interaction. In order to do this (as of now), :py:attr:`self.interaction`
        and :py:attr:`self.interaction_params` must be manipulated directly from an external script.

        Parameters
        ----------
        kwargs['interaction'] : str, default='random_walk'
            Name of the predefined interaction in :py:mod:`lgca.interactions`.
        **kwargs
            Interaction parameters.

        """
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
        """
        Update "fields" from the current LGCA state that store important variables to compute other dynamic steps.

        Computes :py:attr:`self.cell_density`, number of particles at each lattice node, and :py:attr:`self.occupied`,
        a boolean array that shows which channels are occupied.
        """
        self.occupied = self.nodes > 0
        self.cell_density = self.occupied.sum(-1)

    @staticmethod
    def convert_bool_to_ib(conf):
        """
        Convert the booleans in the lattice configuration of a classical LGCA with volume
        exclusion into unique particle identifiers.

        Parameters
        ----------
        conf : :py:class:`numpy.ndarray`
            Lattice configuration for a classical LGCA with volume exclusion.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Node configuration of `conf` converted to an identity-based LGCA. Dimensions: ``conf.shape``.

        """
        conf = conf.astype(np.uint)
        conf[conf > 0] = np.arange(1, 1 + conf.sum(), dtype=np.uint)
        return conf

    def random_reset(self, density):
        """
        Initialize lattice nodes with average density `density`. Channels are occupied at random and nodes can
        have different particle numbers.

        For each channel a random number is drawn. If it is lower than `density`, the channel is filled,
        otherwise it stays empty.

        Parameters
        ----------
        density : float
            Desired average particle density of the lattice.
            ``density = total_number_of_particles / (number_of_nodes * number_of_channels_per_node)``.

        """
        occupied = npr.random(self.dims + (self.K,)) < density
        self.nodes[self.nonborder] = self.convert_bool_to_ib(occupied)
        self.apply_boundaries()

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True,
                recordfampop=False):
        """
        Perform a simulation of the LGCA for `timesteps` timesteps.

        Different quantities can be recorded during the simulation, e.g. the total number of particles at each
        timestep. They are stored in LGCA attributes.

        Parameters
        ----------
        timesteps : int, default=100
            How long the simulation should be performed.
        record : bool, default=False
            Record the full lattice configuration for each timestep in :py:attr:`self.nodes_t`.
        recorddens : bool, default=True
            Record the number of particles at each lattice site for each timestep in :py:attr:`self.dens_t`.
        recordfampop : bool, default=False
            Record the population of all families for each timestep in :py:attr:`self.fam_pop_t`.
        recordN : bool, default=False
            Record the total number of particles in the lattice for each timestep in :py:attr:`self.n_t`.
        showprogress : bool, default=True
            Show a simple progress bar with a percentage of performed timesteps in the standard output.

        """
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
                    raise RuntimeError("Interaction does not deal with families, " +
                                       "family population can therefore not be recorded.")
                # otherwise standard procedure
                self.fam_pop_t = np.zeros((timesteps + 1, self.maxfamily+1))
                self.fam_pop_t[0, ...] = self.calc_family_pop_alive()
                is_mutating = False
        for t in range(1, timesteps + 1):
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
                                         "distinction for the recordfampop keyword in the IBLGCA base " +
                                         "timeevo function!") from e
            if showprogress:
                update_progress(1.0 * t / timesteps)
        if recordfampop and is_mutating:
            self._straighten_family_populations()

    def calc_flux(self, nodes):
        """
        Calculate the flux vector for all lattice sites in `nodes`.

        The elements of the flux vectors are computed as the dot product between the LGCA's neighborhood vectors and
        the velocity channel configuration in `nodes`.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`
            Lattice configuration to compute the flux for. Must have more than or the same number of
            dimensions as :py:attr:`self.nodes` and ``nodes.shape[-1] >= self.velocitychannels``.
            Is typically :py:attr:`self.nodes`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Array of flux vectors at each lattice site. Dimensions: ``nodes.shape[:-1] + (len(self.c),)``.

        """
        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

        return super().calc_flux(nodes)

    def get_prop(self, nodes=None, props=None, propname=None):
        """
        Obtain the value of property `propname` for all particles in the lattice configuration `nodes`.

        The value is drawn from the property dictionary `props` and placed in the channel that the respective particle
        resides in.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`, default=``self.nodes[self.nonborder]``
            Lattice configuration with particle IDs.
        propname : hashable type, default=first key in `props`
            Key for the desired property in the `props` dictionary. If `props` is the default, `propname` needs to be a
            string.
        props : dict, default=``self.props``
            Property dictionary that maps a property identifier, e.g. a name like ``'r_b'``, to a 1d array or list of
            values for this property. Indices in the list/array must reflect the ID of the particle that the value
            is associated with. The value at the 0th element is used for empty lattice sites.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Lattice configuration with value for property `propname` associated to each particle instead of particle ID,
            dimensions: ``nodes.shape``.

        See Also
        --------
        calc_prop_mean : Calculate the mean of a particle property in each node.
        plot_prop_timecourse : Plot the timecourse of the mean and standard deviation of a particle property across the
            lattice.

        """
        # lattice configuration
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        # property dictionary to draw from
        if props is None:
            props = self.props

        # property
        if propname is None:
            propname = next(iter(props))

        # get property value for each particle ID
        prop = np.array(props[propname])
        # sort property value of particles into node configuration
        proparray = prop[nodes]
        return proparray

    def calc_prop_mean(self, nodes=None, props=None, propname=None):
        """
        Calculate the mean of particle property `propname` across the lattice configuration `nodes`.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`, default=``self.nodes[self.nonborder]``
            Lattice configuration with particle IDs.
        propname : hashable type, default=first key in `props`
            Key for the desired property in the `props` dictionary. If `props` is the default, `propname` needs to be a
            string.
        props : dict, default=``self.props``
            Property dictionary that maps a property identifier, e.g. a name like ``'r_b'``, to a 1d array or list of
            values for this property. Indices in the list/array must reflect the ID of the particle that the value
            is associated with. The value at the 0th element is used for empty lattice sites.

        Returns
        -------
        :py:class:`numpy.ma.core.MaskedArray`
            Mean value of property `propname` among particles in each node, dimensions: ``self.dims``.

        See Also
        --------
        lgca.lgca_1d.IBLGCA_1D.plot_prop_spatial : Plot mean value of a property in all nodes over time.
        lgca.lgca_square.IBLGCA_Square.plot_prop_spatial : Plot mean value of a property in all nodes. Also used for
            hexagonal IBLGCA.

        """
        # lattice configuration
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        # obtain value of property for all particles in the lattice configuration
        prop = self.get_prop(nodes=nodes, props=props, propname=propname)
        # mask out empty lattice sites
        occupied = nodes.astype(bool)
        mask = 1 - occupied
        prop = np.ma.array(prop, mask=mask)
        # calculate mean within each node
        mean_prop = prop.mean(-1)
        return mean_prop

    def plot_prop_timecourse(self, nodes_t=None, props=None, propname=None, figindex=None, figsize=None, **kwargs):
        """
        Plot the time course of the mean and standard deviation of particle property `propname` across the lattice.

        The mean is plotted as a line with the standard deviation shadowed in both directions.

        Parameters
        ----------
        figindex : int or str, optional
            An identifier for the figure (passed to :py:func:`matplotlib.pyplot.figure`). If it is a string, the
            figure label and the window title is set to this value.
        figsize : tuple of int or tuple of float with 2 elements, optional
            Desired figure size in inches ``(x, y)``.
        nodes_t : :py:class:`numpy.ndarray`, default=``self.nodes_t``
            Lattice configuration with particle IDs over time, first dimension: time.
        propname : hashable type, default=first key in `props`
            Key for the desired property in the `props` dictionary. If `props` is the default, `propname` needs to be a
            string.
        props : dict, default=``self.props``
            Property dictionary that maps a property identifier, e.g. a name like ``'r_b'``, to a 1d array or list of
            values for this property. Indices in the list/array must reflect the ID of the particle that the value
            is associated with. The value at the 0th element is used for empty lattice sites.
        **kwargs
            Keyword arguments for the :py:func:`matplotlib.pyplot.plot` command.

        Returns
        -------
        line : :py:class:`matplotlib.lines.Line2D`
            Plot of the mean property over time.
        errors : :py:class:`matplotlib.collections.PolyCollection`
            Polygons representing the standard deviation of the property over time, in both directions.

        """
        # set lattice configuration over time
        if nodes_t is None:
            nodes_t = self.nodes_t

        # get particle property values and calculate mean and standard deviation
        prop = self.get_prop(nodes=nodes_t, props=props, propname=propname)
        occupied = nodes_t.astype(bool)
        mask = 1 - occupied
        prop = np.ma.array(prop, mask=mask)
        tocollapse = tuple(range(1, prop.ndim))
        mean_prop_t = np.mean(prop, axis=tocollapse)
        std_mean_prop_t = np.std(prop, axis=tocollapse, ddof=1) / np.sqrt(np.sum(occupied, axis=tocollapse))

        # set up figure
        plt.figure(num=figindex, figsize=figsize)
        tmax = nodes_t.shape[0]
        x = np.arange(tmax)
        y = mean_prop_t
        yerr = std_mean_prop_t
        plt.xlabel('$t$')
        plt.ylabel('${}$'.format(propname))
        plt.title('Time course of the cell property')

        # plot
        line = plt.plot(x, y, **kwargs)[0]
        errors = plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, antialiased=True, interpolate=True)
        return line, errors

    def init_families(self, type='homogeneous', mutation=True):
        """
        Initialize structures to track inheritance between cells and/or cell families.

        Initializes the ``'family'`` property in :py:attr:`self.props` and, if needed, the counter
        :py:attr:`self.maxfamily` and the dictionary :py:attr:`self.family_props`. The ``'ancestor'`` and
        ``'descendants'`` properties in the latter can be used to reconstruct a family tree.

        Parameters
        ----------
        mutation : bool, default=True
            If true, mutation can occur so that offspring can have different properties and found its own family.
            The additional dictionary ``self.family_props`` and the counter ``self.maxfamily`` keep track of this.
            If false, the number of families is not supposed to increase during the simulation so that
            ``self.family_props`` and ``self.maxfamily`` are not needed.
        type : {'homogeneous', 'heterogeneous'}, default='homogeneous'
            Family composition for the initial population. If homogeneous, all initial cells belong to the same family
            and have the same properties. If heterogeneous, each initial cell founds its own family and can have
            individual properties that it passes on to its offspring.

        See Also
        --------
        add_family : Add a family to the family tree after a mutation has occurred.
        muller_plot : Draw a Muller plot for a simulation timecourse.

        """
        # all cells at the beginning of the simulation belong to the same family, have the same properties
        if type == 'homogeneous':
            # all cells belong to the same family
            families = [0] + [1] * self.maxlabel
            self.props.update(family=families)
            if mutation:
                self.family_props = {}  # properties of families
                # family 1 is the semantic root of the tree, family 0 is the root for implementation purposes
                self.family_props.update(ancestor=[0, 0])
                self.family_props.update(descendants=[[1], []])
                self.maxfamily = 1
        # each cell in the beginning of the simulation represents an own family with own properties
        elif type == 'heterogeneous':
            # each initial cell founds its own family
            families = list(np.arange(self.maxlabel+1))
            self.props.update(family=families)
            if mutation:
                self.family_props = {}  # properties of families
                # family 0 is the root for implementation purposes, all initial families descend from it
                self.family_props.update(ancestor=[0] * (self.maxlabel + 1))
                self.family_props.update(descendants=[families[1:]])
                # create lists for each family's future descendants
                for i in range(self.maxlabel):
                    self.family_props['descendants'].append([])
                self.maxfamily = self.maxlabel

    def calc_family_pop_alive(self):
        """
        Calculate how many cells of each family are alive.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Array of family population counts indexed by family ID.

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

    def _straighten_family_populations(self):
        """
        Utility for the :py:meth:`self.timeevo` method. Straighten out the ragged family population record over time
        (:py:attr:`self.fam_pop_t`) in the presence of mutations.

        It is ragged because new families can appear in the course of the simulation.

        """
        for pop_arr in self.fam_pop_t:
            if len(pop_arr) < self.maxfamily + 1:
                pop_arr.resize(self.maxfamily + 1, refcheck=False)
        self.fam_pop_t = np.array(self.fam_pop_t)

    @staticmethod
    def propagate_pop_to_parents(pop_t, ancestor):
        """
        Propagate family population in `pop_t` to all ancestors recorded in `ancestor`.

        Parameters
        ----------
        pop_t : :py:class:`numpy.ndarray`
            Population of each family for all timesteps, dimensions: ``(timesteps, num_families)``.
        ancestor : list
            Record of each family's parent (initially existing families have 0), length: ``num_families + 1``.

        Returns
        --------
        cum_pop_t : :py:class:`numpy.ndarray`
            Family populations over time summed up to all ancestors recursively, cum_pop_t[:, 0] holds total
            population over time.

        See Also
        --------
        muller_plot : Draw a Muller plot for a simulation timecourse.

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

    def muller_plot(self, t_start_index=None, t_slice=slice(None), prop=None, cutoff_abs=None, cutoff_rel=None,
                    **kwargs):
        """
        Draw a Muller plot from the LGCA simulation data (runtime, family tree, recorded family populations).

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.
        prop : hashable type, key of the dictionary self.family_props
            Colour the area for each family according to this family property. If None, colour area according to family
            identity.
        t_slice : slice
            Which parts of the simulation time to consider for the plot.
        t_start_index : int
            First number to appear on the x axis (time).
        **kwargs
            Keyword arguments to further style the plot drawn by :py:func:`lgca.plots.muller_plot`.

        Returns
        -------
        fig : :py:class:`matplotlib.figure.Figure`
            Matplotlib figure handle.
        ax : :py:class:`matplotlib.axes.Axes`
            Muller plot axis handle.
        ret
            Handle of legend, handle of colourbar or None. The separate colourbar axis handle can be retrieved as
            ``ret.ax``.

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
                "LGCA simulation must have been run before with lgca.timeevo(recordfampop=True) to draw a Muller plot.")
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
        Calculate the total population of cells at each simulation step. Requires a previous
        :py:meth:`self.timeevo` with ``recordfampop=True`` due to the filtering option.

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Number of cells at each timestep, dimensions: ``timesteps+1``.

        """
        if not hasattr(self, 'fam_pop_t'):
            raise AttributeError("LGCA simulation must have been run with lgca.timeevo(recordfampop=True) before "
                                 "in order to calculate this.")
        # filter families below cutoff
        fam_pop_t = self.filter_family_population_t(cutoff_abs=cutoff_abs, cutoff_rel=cutoff_rel)
        # fam_pop_t is of shape (timesteps + 1, number of families)
        # summing of population values along the family dimension yields total population
        return fam_pop_t.sum(-1)

    def num_families_total(self):
        """
        Calculate how many families there are in total, including extinct ones.

        Returns
        -------
        int
            Number of families.

        """
        return len(self.family_props['ancestor'])-1  # subtract placeholder 0 family

    def num_families_total_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate how many families there are in total at each timestep, including extinct ones.

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Number of families at each timestep, dimensions: ``timesteps+1``.

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
        Mask out families from `fam_pop_t` that never exceeded the given population threshold.

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.
        fam_pop_t : :py:class:`numpy.ndarray`, default=``self.fam_pop_t``
            Each family's population size over time, dimensions: ``(timesteps + 1, num_families + 1)``.

        Returns
        -------
        :py:class:`numpy.ndarray`
            `fam_pop_t` with all population values for too small families set to 0, dimensions: ``(timesteps + 1,
            num_families + 1)``. Original `fam_pop_t` if both cutoff parameters are None.

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

        Returns
        -------
        int
            Number of families alive.

        """
        cell_fam_alive = self.list_families_alive()  # list all live family IDs
        return len(cell_fam_alive)

    def num_families_alive_t(self, cutoff_abs=None, cutoff_rel=None):
        """
        Calculate how many families have been alive at each simulation step. Requires a previous
        :py:meth:`self.timeevo` with ``recordfampop=True``.

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Number of families at each timestep, dimensions: ``timesteps+1``.

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
        Calculate which families are alive and list their IDs.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Array of family IDs in ascending order.

        """
        cells_alive = self.nodes[self.nonborder][
            np.where(self.nodes[self.nonborder] > 0)]  # indices of live cells # nonborder needed for uniqueness
        cell_fam = np.array(self.props['family'])  # convert for indexing
        cell_fam_alive = cell_fam[cells_alive.astype(np.int)]  # filter family array for families of live cells
        return np.unique(cell_fam_alive)  # remove duplicate entries

    def calc_family_generations(self):
        """
        Calculate which generation each family belongs to, i.e. how many mutations have occurred. The list is
        returned and stored in :py:attr:`self.family_props['generation']`.

        Returns
        -------
        list
            Family generations indexed by family ID, helper root family = generation 0, families at the beginning of the
            simulation = generation 1.

        """
        # check if it has been calculated and is up to date
        num_families = self.num_families_total()
        if 'generation' in self.family_props:
            if len(self.family_props['generation']) == num_families:
                return self.family_props['generation']
        # if not, initialise the property and fill it
        self.family_props['generation'] = list(np.zeros(num_families+1))
        self._recurse_generation(0, 0)
        return self.family_props['generation']

    def _recurse_generation(self, fam_ID, gen):
        """
        Utility for calculating the generation of each family. Recursively updates the entries in
        :py:attr:`self.family_props['generation']`.

        Parameters
        ----------
        fam_ID : int
            Family ID to access properties in :py:attr:`self.family_props`.
        gen : int
            This family's generation.

        """
        # update generation for this family
        self.family_props['generation'][fam_ID] = gen
        # update children generations with increased counter
        gen += 1
        for desc in self.family_props['descendants'][fam_ID]:
            self._recurse_generation(desc, gen)

    def calc_generation_pop(self):
        """
        Calculate the current population per family generation.

        Returns
        -------
        `numpy.ndarray`
            Cell numbers per generation, indexed by generation (helper root family is generation 0, families present at
            initialization are generation 1).

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
        Calculate the population per family generation at each simulation step. Requires a previous
        :py:meth:`self.timeevo` with ``recordfampop=True``.

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Cell numbers per generation at each timestep (helper root family is generation 0, families present at
            initialization are generation 1). Dimensions: ``(timesteps+1, generations+1)``.

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
        ("initial ancestors") and which families descend from them, including the 2nd, 3rd, ... generations.

        Returns
        -------
        list
            Initial ancestors indexed by family ID. The helper root family and initial ancestors have 0, all others
            their ancestor's family ID.

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
        ("initial ancestor") and its descendants.

        Returns
        -------
        init_families_pop : :py:class:`numpy.ndarray`
            Population per initial family.
        init_ancestor_IDs : :py:class:`numpy.ndarray`
            Initial family ID corresponding to each position in init_families_pop.

        """
        # calculate population per family and initial family they descend from
        fam_pop = self.calc_family_pop_alive()
        init_ancestors = np.array(self.propagate_ancestor_to_descendants())
        # clip for indexing in case a late family already died out
        init_ancestors_alive = np.resize(init_ancestors, len(fam_pop))
        # initialise
        init_ancestor_IDs = np.arange(len(init_ancestors_alive))[init_ancestors_alive == 0][1:]
        # ignore helper root family
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
        ("initial ancestor") and its descendants at each simulation step. Requires a previous
        :py:meth:`self.timeevo` with ``recordfampop=True``.

        Parameters
        ----------
        cutoff_abs : int
            Exclude families that never exceeded this population size in absolute numbers. `cutoff_rel` takes precedence
            over `cutoff_abs`.
        cutoff_rel : float
            Exclude families that never exceeded this population size, expressed as a fraction of the total population.
            `cutoff_rel` takes precedence over `cutoff_abs`.

        Returns
        -------
        init_families_pop : :py:class:`numpy.ndarray`
            Population per initial family at each timestep, dimensions: ``(timesteps+1, num_init_families+1)``.
        init_ancestor_IDs : :py:class:`numpy.ndarray`
            Family ID corresponding to each position on the family dimension in `init_families_pop`.

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

    def propagate_family_prop_to_cells(self, prop):
        """
        Propagate a family property down to all cells that belong to the family. Uses contents of
        :py:attr:`self.family_props[prop]` to create a new cell property :py:attr:`self.props[prop]`

        Parameters
        ----------
        prop : str
            Key of the family property in :py:attr:`self.family_props[prop]`.

        """
        self.props[prop] = [self.family_props[prop][fam] for fam in self.props['family']]

    def add_family(self, ancestor_fam: int):
        """
        Create a new family and register it in the family tree.

        Parameters
        ----------
        ancestor_fam : int
            Family that the new one descends from.

        Warnings
        --------
        **Please do not call this unless you are writing an interaction function that deals with mutations.**

        It accesses :py:attr:`self.maxfamily` and :py:attr:`self.family_props` which do not exist if 
        :py:meth:`self.init_families` has not been called in the LGCA initialization.

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
    def __init__(self, nodes=None, dims=None, restchannels=None, density=0.1, hom=None, bc='periodic', capacity=None,
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
        self.set_bc(bc)
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes, capacity=capacity)
        self.init_coords()
        self.init_nodes(density=density, nodes=nodes, hom=hom)
        self.update_dynamic_fields()
        self.interaction_params = {}
        self.set_interaction(**kwargs)

    def set_interaction(self, **kwargs):
        from lgca.nove_interactions import dd_alignment, di_alignment, go_or_grow, go_or_rest
        from lgca.interactions import only_propagation
        # configure interaction
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            # density-dependent interaction rule
            if interaction == 'dd_alignment':
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
        for t in range(1, timesteps + 1):
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
            if showprogress:
                update_progress(1.0 * t / timesteps)

    def random_reset(self, density):
        """
        Distribute particles in the lattice according to a given density; can yield different cell numbers per
        lattice site
        :param density: particle density in the lattice: number of particles/(dimensions*capacity)
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
