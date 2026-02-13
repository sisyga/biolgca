# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2024 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
Classical LGCA base class module.

Contains helper functions and the LGCA_base class which defines properties and structure
for classical LGCA with volume exclusion. This is the base for geometry-independent LGCA behavior.
Cannot be used to simulate - use geometry-specific derived classes instead.
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


def get_cmap(density, ax=None, vmax=None, cmap='viridis', cbar=True, cbarlabel=''):
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
        cbar = plt.colorbar(cmap, ax=ax, extend='min', use_gridspec=True, boundaries=[0, 0.5, 1], values=[0, 1])
    else:
        cbar = plt.colorbar(cmap, ax=ax, extend='min', use_gridspec=True)
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
    """
    Given a vector field 'v', calculate the nematic tensor at each location. This tensor can be used to calculate
    transition probabilites or characterize order. See https://en.wikipedia.org/wiki/Liquid_crystal#Order_parameter
    Parameters
    ----------
    v: :py:class:`numpy.ndarray`
    Last two dimensions should be length 2.

    Returns
    -------
    Array of nematic tensors.
    If 'v' has shape (nx, ny, 2) then the array of tensors has the shape (nx, ny, 2, 2).
    """
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
    lgca.lgca_1d.LGCA_1D : Classical LGCA in a 1D geometry.
    lgca.lgca_square.LGCA_Square : Classical LGCA in a 2D square geometry.
    lgca.lgca_hex.LGCA_Hex : Classical LGCA in a 2D hexagonal geometry.

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

    def apply_pbc(self):
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

    def __init__(self, nodes=None, dims=None, restchannels=0, density=0.1, bc='periodic', seed=None, **kwargs):
        """ Initialize class instance. See class docstring."""
        self.r_int: int = 1  # Interaction radius. Must be at least 1 to handle propagation.
        self.rng = npr.default_rng(seed=seed)
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
        if bc in ['absorbing', 'absorb', 'abs', 'abc', 'fixed']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl', 'rbc', 'no_flux', 'noflux']:
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


