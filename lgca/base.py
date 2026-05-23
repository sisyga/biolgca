# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
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

import warnings
from abc import ABC, abstractmethod
from copy import copy, deepcopy
import difflib


class _MissingPlotLib:
    """Placeholder object for an optional plotting library."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, _):
        raise ImportError(
            f"Plotting requires {self._name}. Install extras with 'pip install -r plotting-requirements.txt'."
        )


import numpy as np
try:  # optional plotting dependencies
    import matplotlib.colors as colors
    from matplotlib import cm, pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:  # pragma: no cover - handled at runtime
    colors = cm = plt = ScalarMappable = MaxNLocator = FuncFormatter = make_axes_locatable = _MissingPlotLib(
        "matplotlib"
    )
from numpy import random as npr
from itertools import combinations
from tqdm.auto import tqdm

from lgca.plots import muller_plot, colorbar_index, cmap_discretize, estimate_figsize, get_cmap


# configure matplotlib style
# plt.style.use('default')


_VALID_CONSTRUCTOR_KWARGS = {
    "N",
    "a_max",
    "alpha",
    "bc",
    "beta",
    "density",
    "director",
    "dims",
    "drb",
    "effect",
    "fitness_increase",
    "gamma",
    "gradient",
    "include_center",
    "interaction",
    "kappa",
    "kappa_std",
    "nodes",
    "p_d",
    "p_p",
    "pmut",
    "propagation",
    "r_b",
    "r_d",
    "r_int",
    "r_m",
    "restchannels",
    "rho_0",
    "s_d",
    "s_p",
    "seed",
    "std",
    "theta",
    "theta_std",
    "track_inheritance",
}


def _as_numeric_array(value, name):
    """Return a numeric array for validation and raise a clear error otherwise."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return arr


def _validate_positive_int(value, name):
    """Validate a positive integer parameter and return it as ``int``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if int_value != value or int_value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int_value


def _validate_nonnegative_int(value, name):
    """Validate a non-negative integer parameter and return it as ``int``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if int_value != value or int_value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int_value


def _validate_probability(value, name):
    """Validate scalar or vector probabilities."""
    arr = _as_numeric_array(value, name)
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _validate_nonnegative(value, name):
    """Validate scalar or vector non-negative finite values."""
    arr = _as_numeric_array(value, name)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    return value


def _validate_positive(value, name):
    """Validate scalar or vector positive finite values."""
    arr = _as_numeric_array(value, name)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_density(density, max_density=None):
    """Validate random-initialization density before it is used as a probability or rate."""
    arr = _as_numeric_array(density, "density")
    if arr.shape != ():
        raise ValueError("density must be a scalar.")
    density_value = float(arr)
    if density_value < 0:
        raise ValueError("density must be non-negative.")
    if max_density is not None and density_value > max_density:
        raise ValueError(f"density must not exceed the available channel count ({max_density}).")
    return density


def _validate_vector_field_shape(field, expected_shape, name):
    """Validate user-supplied vector fields used by chemotaxis/contact guidance."""
    arr = np.asarray(field)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {arr.shape}.")
    return arr



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


def _generate_permutations(K: int, n: int) -> np.ndarray:
    """Generate all boolean permutations for ``K`` channels with ``n`` occupied.

    Parameters
    ----------
    K : int
        Total number of channels.
    n : int
        Number of occupied channels.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(C(K, n), K)`` containing all permutations.
    """
    combs = list(combinations(range(K), n))
    perms = np.zeros((len(combs), K), dtype=bool)
    for i, idx in enumerate(combs):
        perms[i, list(idx)] = True
    return perms


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
    propagation : bool, default=True
        Toggle whether the propagation step is executed during a timestep.
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

    _LOCAL_ENSEMBLE_INTERACTIONS = {
        "birth",
        "birthdeath",
        "go_and_grow",
        "go_or_grow",
        "go_or_rest",
        "only_propagation",
        "random_walk",
    }

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

    def _warn_nodes_shape(self, nodes):
        """Issue warnings if ``nodes`` do not match current geometry."""
        if nodes is None:
            return
        expected = self.dims + (self.K,)
        if nodes.shape != expected:
            warnings.warn(
                f"Provided nodes have shape {nodes.shape}, expected {expected}.",
                UserWarning,
            )

    def _ensure_bool_nodes(self, nodes):
        """Return ``nodes`` as boolean array, warn if non-boolean values found."""
        if not np.isin(nodes, [0, 1]).all():
            warnings.warn(
                "Provided nodes contain values other than 0 or 1. "
                "Interpreting values as booleans.",
                UserWarning,
            )
        return nodes.astype(bool)

    @classmethod
    def _get_valid_kwargs(cls):
        """Return keyword arguments that may be forwarded through constructors."""
        return set(_VALID_CONSTRUCTOR_KWARGS)

    @classmethod
    def _validate_kwargs(cls, kwargs):
        """Raise on unexpected forwarded kwargs before they can be silently ignored."""
        unknown = sorted(set(kwargs) - cls._get_valid_kwargs())
        if not unknown:
            return

        details = []
        valid = cls._get_valid_kwargs()
        for key in unknown:
            matches = difflib.get_close_matches(key, valid, n=1)
            if matches:
                details.append(f"{key!r} (did you mean {matches[0]!r}?)")
            else:
                details.append(repr(key))
        raise TypeError(
            f"{cls.__name__}.__init__() got unexpected keyword argument(s): "
            + ", ".join(details)
        )

    def _warn_if_nonlocal_ensemble_interaction(self, interaction_name):
        if getattr(self, "enable_propagation", True):
            return
        if interaction_name in self._LOCAL_ENSEMBLE_INTERACTIONS:
            return
        warnings.warn(
            "Disabling propagation is intended for local interactions only. "
            f"The interaction {interaction_name!r} may depend on neighbourhood state.",
            UserWarning,
            stacklevel=3,
        )

    def __init__(self, nodes=None, dims=None, restchannels=0, density=0.1,
                 bc='periodic', seed=None, propagation=True, **kwargs):
        """Initialize class instance. See class docstring."""
        self._validate_kwargs(kwargs)
        self.enable_propagation = propagation
        self.r_int: int = _validate_positive_int(kwargs.pop("r_int", 1), "r_int")
        self.rng = npr.default_rng(seed=seed)
        # set boundary conditions, set self.apply_boundaries
        self.set_bc(bc)

        # initialize lattice
        # define self.K, self.restchannels, self.dims, self.l or self.lx and self.ly
        restchannels = _validate_nonnegative_int(restchannels, "restchannels")
        self.set_dims(dims=dims, restchannels=restchannels, nodes=nodes)
        self._validate_model_setup(nodes=nodes)
        if nodes is None:
            _validate_density(density, max_density=self.K)
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

    def _validate_model_setup(self, nodes=None):
        """Validate dimensions and channel metadata after geometry-specific setup."""
        dims = getattr(self, "dims", None)
        if dims is None:
            raise ValueError("dims must be specified by the geometry setup.")
        for dim in dims:
            _validate_positive_int(dim, "dims")
        self.restchannels = _validate_nonnegative_int(self.restchannels, "restchannels")
        self.K = _validate_positive_int(self.K, "channels")
        if self.K < self.velocitychannels:
            raise ValueError(
                f"channels must be at least the velocity channel count ({self.velocitychannels})."
            )
        if hasattr(self, "capacity"):
            _validate_positive(self.capacity, "capacity")
        if nodes is not None:
            self._warn_nodes_shape(nodes)

    def _validate_interaction_params(self):
        """Validate common interaction parameters that represent probabilities or positive scales."""
        probability_params = {"r_b", "r_d", "r_m", "p_d", "p_p", "pmut"}
        nonnegative_params = {"std", "kappa_std", "theta_std", "drb", "s_d", "s_p"}
        positive_params = {"a_max", "capacity"}

        for name in probability_params & self.interaction_params.keys():
            _validate_probability(self.interaction_params[name], name)
        for name in nonnegative_params & self.interaction_params.keys():
            _validate_nonnegative(self.interaction_params[name], name)
        for name in positive_params & self.interaction_params.keys():
            _validate_positive(self.interaction_params[name], name)

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
        r = _validate_positive_int(r, "r_int")
        old_nodes = deepcopy(self.nodes[self.nonborder])
        self.r_int = r
        self.init_coords()
        self.init_nodes(density=0, nodes=old_nodes)
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
        from lgca.ms_interactions import excitable_medium_ms
        if 'interaction' in kwargs:
            interaction = kwargs['interaction'].replace(" ", "_")
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
                    self.interaction_params['gradient_field'] = _validate_vector_field_shape(
                        kwargs['gradient'],
                        self.nodes.shape[:-1] + (self.c.shape[0],),
                        "gradient",
                    )
                else:
                    if len(self.dims) == 2:
                        x_source = self.xcoords.mean()
                        y_source = self.ycoords.mean()
                        rx = self.xcoords - x_source
                        ry = self.ycoords - y_source
                        r = np.sqrt(rx ** 2 + ry ** 2)
                        self.concentration = np.exp(-2 * r / self.ly)
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                         'reflect'))
                    elif len(self.dims) == 1:
                        source = self.l / 2
                        r = abs(self.xcoords - source)
                        self.concentration = np.exp(-2 * r / self.l)
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                         'reflect'))
                        self.interaction_params['gradient_field'] /= self.interaction_params['gradient_field'].max()

                    elif len(self.dims) == 3:
                        x_source = self.xcoords.mean()
                        y_source = self.ycoords.mean()
                        z_source = self.zcoords.mean()
                        rx = self.xcoords - x_source
                        ry = self.ycoords - y_source
                        rz = self.zcoords - z_source
                        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
                        self.concentration = np.exp(-2 * r / self.ly)
                        self.interaction_params['gradient_field'] = self.gradient(np.pad(self.concentration, 1,
                                                                                         'reflect'))


            elif interaction == 'contact_guidance':
                if len(self.dims) != 2:
                    raise ValueError("contact_guidance is not supported for this geometry.")
                self.interaction = contact_guidance
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.interaction_params['beta'] = kwargs['beta']
                else:
                    self.interaction_params['beta'] = 2.
                    print('sensitivity set to beta = ', self.interaction_params['beta'])

                if 'director' in kwargs:
                    self.interaction_params['gradient_field'] = _validate_vector_field_shape(
                        kwargs['director'],
                        self.nodes.shape[:-1] + (2,),
                        "director",
                    )
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

            elif interaction == 'excitable_medium_ms':
                if getattr(self, "n_species", 1) != 2:
                    raise ValueError("excitable_medium_ms requires a multi-species LGCA with exactly two species.")
                if self.restchannels < 1:
                    raise ValueError("excitable_medium_ms requires at least one rest channel.")
                self.interaction = excitable_medium_ms
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
                raise ValueError(
                    "Unknown interaction {!r}. Implemented interactions: {}".format(
                        kwargs["interaction"], self.interactions
                    )
                )

        else:
            print('Random walk interaction is used.')
            interaction = 'random_walk'
            self.interaction = random_walk
        self._validate_interaction_params()
        self._warn_if_nonlocal_ensemble_interaction(interaction)

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
            self.bc = 'absorbing'
        elif bc in ['reflecting', 'reflect', 'refl', 'rbc', 'no_flux', 'noflux']:
            self.apply_boundaries = self.apply_rbc
            self.bc = 'reflecting'
        elif bc in ['periodic', 'pbc']:
            self.apply_boundaries = self.apply_pbc
            self.bc = 'periodic'
        elif bc in ['inflow']:
            self.apply_boundaries = self.apply_inflowbc
            self.bc = 'inflow'
        else:
            raise ValueError(
                "Unknown boundary condition {!r}. Use one of: absorbing, reflecting, periodic, inflow.".format(bc)
            )

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
        """Randomly fill channels so that each node has on average ``density`` particles.

        Each channel is independently occupied with probability ``density / self.K``.

        Parameters
        ----------
        density : float
            Desired average number of particles per node.
            ``density = total_number_of_particles / number_of_nodes``.
        """
        _validate_density(density, max_density=self.K)
        self.nodes = self.rng.random(self.nodes.shape) < (density / self.K)
        self.apply_boundaries()
        self.update_dynamic_fields()
        # achieved density example
        # eff_dens = self.nodes[self.nonborder].sum() / self.cell_density[self.nonborder].size

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
        if getattr(self, "enable_propagation", True):
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
        Initialize lazy computation structures for permutations.
        Only compute permutations when actually needed.
        """
        # For geometries with many channels, use lazy computation
        if self.K > 15:  # threshold for when precomputation becomes expensive
            self._permutation_cache = {}
            self._flux_cache = {}
            self._si_cache = {}
            self.permutations = None  # Signal that we're using lazy computation
            # Still compute cij as it's geometry-dependent and small
            self.cij = np.einsum('ij,kj->jik', self.c, self.c) - 0.5 * np.diag(np.ones(self.c.shape[0]))[None, ...]
        else:
            # Precompute all permutations for smaller geometries
            self.permutations = [_generate_permutations(self.K, n) for n in range(self.K + 1)]
            self.j = [np.dot(self.c, self.permutations[n][:, :self.velocitychannels].T) for n in range(self.K + 1)]
            self.cij = np.einsum('ij,kj->jik', self.c, self.c) - 0.5 * np.diag(np.ones(self.c.shape[0]))[None, ...]
            self.si = [np.einsum('ij,jkl', self.permutations[n][:, :self.velocitychannels], self.cij) for n in
                       range(self.K + 1)]

    def get_permutations(self, n_particles):
        """Get permutations for ``n_particles``.

        Parameters
        ----------
        n_particles : int
            Number of occupied channels.

        Returns
        -------
        numpy.ndarray
            Array of permutations for ``n_particles``.
        """
        try:
            return self.permutations[n_particles]
        except (AttributeError, TypeError):
            pass

        if n_particles not in self._permutation_cache:
            # Limit cache size to prevent memory issues
            if len(self._permutation_cache) > 50:
                # Remove least recently used (simple FIFO here)
                oldest_key = next(iter(self._permutation_cache))
                del self._permutation_cache[oldest_key]

            self._permutation_cache[n_particles] = _generate_permutations(self.K, n_particles)
        return self._permutation_cache[n_particles]

    def get_flux_permutations(self, n_particles):
        """Get flux permutations for ``n_particles``."""
        try:
            return self.j[n_particles]
        except (AttributeError, TypeError):
            pass

        if n_particles not in self._flux_cache:
            perms = self.get_permutations(n_particles)
            self._flux_cache[n_particles] = np.dot(self.c, perms[:, :self.velocitychannels].T)
        return self._flux_cache[n_particles]

    def total_population(self):
        """
        Calculate the amount of particles in the lattice.

        Returns
        -------
        int
            Total population size.
        """
        return int(self.cell_density[self.nonborder].sum())

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        geom = getattr(self, "geometry", "unknown")
        bc = getattr(self, "bc", "periodic")
        interaction = getattr(self.interaction, "__name__", str(self.interaction))
        prop_flag = getattr(self, "enable_propagation", True)
        return (
            f"{self.__class__.__name__}(geometry={geom}, dims={self.dims}, "
            f"rest={self.restchannels}, K={self.K}, r_int={self.r_int}, bc={bc}, "
            f"propagate={prop_flag}, interaction={interaction})"
        )

    def __str__(self) -> str:
        """Human readable summary of the LGCA."""
        geom = getattr(self, "geometry", "unknown")
        bc = getattr(self, "bc", "periodic")
        interaction = getattr(self.interaction, "__name__", str(self.interaction))
        capacity = getattr(self, "capacity", self.K)
        prop_mode = "ensemble" if getattr(self, "ensemble", False) else "normal"
        prop_flag = getattr(self, "enable_propagation", True)
        lines = [
            f"Model: {self.__class__.__name__}",
            f"Geometry: {geom}",
            f"Dimensions: {self.dims}",
            f"Rest channels: {self.restchannels}",
            f"Carrying capacity: {capacity}",
            f"Interaction radius: {self.r_int}",
            f"Boundary conditions: {bc}",
            f"Propagation mode: {prop_mode}",
            f"Propagation enabled: {prop_flag}",
            f"Interaction: {interaction}",
            f"Interaction parameters: {self.interaction_params}",
        ]
        return "\n".join(lines)


from .list_utils import get_arr_of_empty_lists


def __getattr__(name):
    """Lazily expose compatibility base-class reexports."""
    if name == "IBLGCA_base":
        from .ib_base import IBLGCA_base
        return IBLGCA_base
    if name == "NoVE_LGCA_base":
        from .nove_base import NoVE_LGCA_base
        return NoVE_LGCA_base
    if name == "NoVE_IBLGCA_base":
        from .nove_ib_base import NoVE_IBLGCA_base
        return NoVE_IBLGCA_base
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LGCA_base",
    "IBLGCA_base",
    "NoVE_LGCA_base",
    "NoVE_IBLGCA_base",
    "calc_nematic_tensor",
    "colorbar_index",
    "cmap_discretize",
    "estimate_figsize",
    "get_arr_of_empty_lists",
    "get_cmap",
    "muller_plot",
    "np",
]

