# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# It is made available under the BSD 3-clause license (see LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause).
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# Contact: simon.syga@tu-dresden.de or bianca.guettner@nct-dresden.de.
"""
biolgca is a Python package for simulating different types of lattice-gas
cellular automata (LGCA) in the biological context. It is under active development.

It is tightly coupled to the theoretical method presented by Deutsch et al. [1]_.
The get_lgca function returns an LGCA object of the requested type with the
given initial conditions. It can simulate for a given number of timesteps and has
different plotting functions for analysis, depending on the geometry. The
interaction function can be chosen from built-in ones or defined by the user.
Currently, classical LGCA and identity-based LGCA with and without volume
exclusion, respectively, are supported on 1D, 2D square, 2D hexagonal,
3D cubic and 3D Moore lattices.

References
----------
.. [1] Deutsch A, Nava-Sedeño JM, Syga S, Hatzikirou H (2021) BIO-LGCA: A cellular
    automaton modelling class for analysing collective cell migration.
PLoS Comput Biol 17(6): e1009066. https://doi.org/10.1371/journal.pcbi.1009066

"""

import difflib
import warnings
from typing import Tuple, Any


_VALID_KWARGS = {
    "N",
    "a_max",
    "alpha",
    "bc",
    "beta",
    "capacity",
    "density",
    "dims",
    "director",
    "drb",
    "effect",
    "fitness_increase",
    "gamma",
    "gradient",
    "include_center",
    "interaction",
    "kappa",
    "kappa_std",
    "mutation_matrix",
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


def _validate_kwargs(kwargs):
    """Raise on unknown factory kwargs and suggest likely typos."""
    unknown = sorted(set(kwargs) - _VALID_KWARGS)
    if not unknown:
        return

    details = []
    for key in unknown:
        matches = difflib.get_close_matches(key, _VALID_KWARGS, n=1)
        if matches:
            details.append(f"{key!r} (did you mean {matches[0]!r}?)")
        else:
            details.append(repr(key))
    raise TypeError(
        "get_lgca() got unexpected keyword argument(s): " + ", ".join(details)
    )


def _translate_dims(dims: Any, geom_key: str) -> Tuple[int, ...]:
    """Translate the user supplied ``dims`` argument to a tuple.

    Parameters
    ----------
    dims
        Dimension argument passed to :func:`get_lgca`.
    geom_key
        Canonicalized geometry identifier.

    Returns
    -------
    tuple of int
        Dimensions interpreted in the same way as in ``set_dims``.
    """
    if dims is None:
        return None
    if isinstance(dims, tuple):
        if geom_key == 'lin':
            return (dims[0],)
        if geom_key in {'square', 'hex'}:
            return (dims[0], dims[1]) if len(dims) > 1 else (dims[0], dims[0])
        if geom_key in {'cubic', 'moore'}:
            if len(dims) >= 3:
                return dims[0], dims[1], dims[2]
            if len(dims) == 2:
                return dims[0], dims[0], dims[1]
            return (dims[0],) * 3
    else:
        if geom_key == 'lin':
            return (int(dims),)
        if geom_key in {'square', 'hex'}:
            d = int(dims)
            return (d, d)
        if geom_key in {'cubic', 'moore'}:
            d = int(dims)
            return (d, d, d)
    return tuple(dims)


def _warn_on_node_mismatch(nodes, dims_arg, rest_arg, density_arg, geom_key, n_species=1):
    """Warn if provided ``nodes`` are inconsistent with ``dims`` or ``restchannels``."""
    if nodes is None:
        return

    velocity_lookup = {
        'lin': 2,
        'square': 4,
        'hex': 6,
        'cubic': 6,
        'moore': 26,
    }

    dims_from_nodes = nodes.shape[:-2] if n_species > 1 else nodes.shape[:-1]
    vel = velocity_lookup.get(geom_key)
    if vel is not None:
        rest_from_nodes = nodes.shape[-1] - vel
    else:
        rest_from_nodes = None

    dims_translated = _translate_dims(dims_arg, geom_key)
    if dims_translated is not None and dims_from_nodes != tuple(dims_translated):
        warnings.warn(
            f"Provided nodes with dimensions {dims_from_nodes} override ``dims``={dims_arg}.",
            UserWarning,
        )

    if rest_arg is not None and rest_from_nodes is not None and rest_from_nodes != rest_arg:
        warnings.warn(
            f"Provided nodes imply {rest_from_nodes} rest channels but ``restchannels``={rest_arg} was passed.",
            UserWarning,
        )

    if density_arg is not None:
        warnings.warn(
            f"Provided nodes override ``density``={density_arg}.",
            UserWarning,
        )




def get_lgca(geometry: str = 'hex', ib: bool = False, ve: bool = True, n_species: int = 1, **kwargs):
    """
    Build an LGCA with the specified geometry and initial conditions. Choose the correct LGCA subclass
    from the package and pass remaining keyword parameters on to it for initialization.

    Parameters
    ----------
    geometry : {'hex', 'square', 'lin', 'cubic', 'moore'}, default='hex'
        Lattice geometry. Supported are 1D, 2D square, 2D hexagonal,
        3D cubic and 3D Moore lattices.

        Aliases: 1D: ``'1D', '1d', 'linear'``; 2D square: ``'sq', 'rect', 'rectangular'``;
        2D hexagonal: ``'hexagonal', 'hx'``; 3D cubic: ``'cubic', 'cb'``;
        3D Moore: ``'moore3d'``.
    ib : bool, default=False
        If the LGCA should be identity-based (every particle can have individual properties).
    ve : bool, default=True
        If the LGCA should comply with the volume exclusion principle (only one particle per channel).
    n_species : int, default=1
        Number of species. Values greater than one select multi-species LGCA classes.
    **kwargs : dict
        Keyword arguments for dimensions, initial conditions and interaction. Used by the constructor of the LGCA subclass.

    Returns
    -------
    lgca : subclass of :py:class:`base.LGCA_base` instance
        LGCA simulator object.

    See Also
    --------
    base.LGCA_base.set_bc : Set boundary conditions. Processes `**kwargs`.
    base.LGCA_base.set_dims : Set the lattice geometry. Processes `**kwargs`.
    base.LGCA_base.init_nodes : Initialize the lattice.  Processes `**kwargs`.
    base.LGCA_base.set_interaction : Set the interaction and corresponding parameters. Processes `**kwargs`.

    Notes
    -----
    The function mediates between user and package. It picks the correct LGCA subclass
    and initializes an instance of it. Allowed types and values of keyword arguments
    for initialization may vary. Details can be found in the documentation of the subclasses.

    How to navigate: Subclasses are structured as follows (omitting geometry inheritance):

    - :py:class:`lgca.base.LGCA_base`: classical LGCA
        - :py:class:`lgca.lgca_1d.LGCA_1D`
        - :py:class:`lgca.lgca_square.LGCA_Square`
        - :py:class:`lgca.lgca_hex.LGCA_Hex`
    - :py:class:`lgca.base.IBLGCA_base`: identity-based LGCA
        - :py:class:`lgca.lgca_1d.IBLGCA_1D`
        - :py:class:`lgca.lgca_square.IBLGCA_Square`
        - :py:class:`lgca.lgca_hex.IBLGCA_Hex`
    - :py:class:`lgca.base.NoVE_LGCA_base`: classical LGCA without volume exclusion
        - :py:class:`lgca.lgca_1d.NoVE_LGCA_1D`
        - :py:class:`lgca.lgca_square.NoVE_LGCA_Square`
        - :py:class:`lgca.lgca_hex.NoVE_LGCA_Hex`
    - :py:class:`lgca.base.NoVE_IBLGCA_base`: identity-based LGCA without volume exclusion
        - :py:class:`lgca.lgca_1d.NoVE_IBLGCA_1D`
        - :py:class:`lgca.lgca_square.NoVE_IBLGCA_Square`
        - :py:class:`lgca.lgca_hex.NoVE_IBLGCA_Hex`

    Examples
    --------
    Request a classical LGCA with a hexagonal lattice and a random walk interaction.

    >>> from lgca import get_lgca
    >>> lgca = get_lgca(interaction='random_walk')
    Random walk interaction is used.

    Used default values for interactions are printed to the terminal.

    Request an identity-based LGCA in a linear geometry with a birth interaction.

    >>> lgca = get_lgca(ib=True, geometry='1d', interaction='birth')
    Birth rate set to r_b = 0.2
    Standard deviation set to std = 0.01
    Max. birth rate set to a_max = 1.0

    The returned LGCA object can then be used to simulate.

    >>> # simulate for 50 timesteps
    >>> lgca.timeevo(timesteps=50)
    Progress: [####################] 100% Done...

    """
    _validate_kwargs(kwargs)

    nodes = kwargs.get('nodes')
    rest_arg = kwargs.get('restchannels')
    dims_arg = kwargs.get('dims')
    density_arg = kwargs.get('density') if 'density' in kwargs else None

    geom_map = {
        '1d': 'lin',
        'lin': 'lin',
        'linear': 'lin',
        'square': 'square',
        'sq': 'square',
        'rect': 'square',
        'rectangular': 'square',
        'hex': 'hex',
        'hx': 'hex',
        'hexagonal': 'hex',
        'cubic': 'cubic',
        'cb': 'cubic',
        'moore': 'moore',
        'moore3d': 'moore',
    }

    geom_spec = geometry.lower() if isinstance(geometry, str) else geometry
    geom_key = geom_map.get(geom_spec, geom_spec)

    if isinstance(n_species, bool) or int(n_species) != n_species or n_species < 1:
        raise ValueError("n_species must be a positive integer.")
    n_species = int(n_species)
    interaction = kwargs.get("interaction")
    interaction_name = interaction.replace(" ", "_") if isinstance(interaction, str) else interaction

    if interaction_name == "excitable_medium_ms":
        if not ve:
            raise ValueError("excitable_medium_ms requires volume exclusion.")
        if n_species != 2:
            raise ValueError("excitable_medium_ms requires a multi-species LGCA with exactly two species.")
        if kwargs.get("restchannels", 0) < 1:
            raise ValueError("excitable_medium_ms requires at least one rest channel.")

    if n_species > 1:
        kwargs["n_species"] = n_species
        if ib:
            raise NotImplementedError("Multi-species identity-based LGCA is not implemented yet.")
        if ve:
            if geom_key == 'lin':
                from lgca.ms_1d import MSLGCA_1D as _Cls
            elif geom_key == 'square':
                from lgca.ms_square import MSLGCA_Square as _Cls
            elif geom_key == 'hex':
                from lgca.ms_hex import MSLGCA_Hex as _Cls
            elif geom_key == 'cubic':
                from lgca.ms_cubic import MSLGCA_Cubic as _Cls
            elif geom_key == 'moore':
                from lgca.ms_moore import MSLGCA_Moore as _Cls
            else:
                raise ValueError(
                    "Geometry specification is unknown. Try: '1d', 'lin', 'linear', 'square', 'sq', "
                    "'rect', 'rectangular', 'hex', 'hx',  'hexagonal', 'cubic', 'cb', or 'moore3d'."
                )
        else:
            if geom_key == 'lin':
                from lgca.ms_1d import MSLGCA_NoVE_1D as _Cls
            elif geom_key == 'square':
                from lgca.ms_square import MSLGCA_NoVE_Square as _Cls
            elif geom_key == 'hex':
                from lgca.ms_hex import MSLGCA_NoVE_Hex as _Cls
            elif geom_key == 'cubic':
                from lgca.ms_cubic import MSLGCA_NoVE_Cubic as _Cls
            elif geom_key == 'moore':
                from lgca.ms_moore import MSLGCA_NoVE_Moore as _Cls
            else:
                raise ValueError(
                    "Geometry specification is unknown. Try: '1d', 'lin', 'linear', 'square', 'sq', "
                    "'rect', 'rectangular', 'hex', 'hx',  'hexagonal', 'cubic', 'cb', or 'moore3d'."
                )
    elif not ve and not ib:
        if geom_key == 'lin':
            from lgca.lgca_1d import NoVE_LGCA_1D as _Cls
        elif geom_key == 'square':
            from lgca.square_ext import NoVE_LGCA_Square as _Cls
        elif geom_key == 'hex':
            from lgca.lgca_hex import NoVE_LGCA_Hex as _Cls
        elif geom_key == 'cubic':
            from lgca.cubic_ext import NoVE_LGCA_Cubic as _Cls
        elif geom_key == 'moore':
            from lgca.lgca_3dmoore import NoVE_LGCA_Moore as _Cls
        else:
            raise ValueError(
                "Geometry specification is unknown. Try: '1d', 'lin', 'linear', 'square', 'sq', "
                "'rect', 'rectangular', 'hex', 'hx',  'hexagonal', 'cubic', 'cb', or 'moore3d'."
            )
    elif ib and ve:
        if geom_key == 'lin':
            from lgca.lgca_1d import IBLGCA_1D as _Cls
        elif geom_key == 'square':
            from lgca.square_ext import IBLGCA_Square as _Cls
        elif geom_key == 'hex':
            from lgca.lgca_hex import IBLGCA_Hex as _Cls
        elif geom_key == 'cubic':
            from lgca.cubic_ext import IBLGCA_Cubic as _Cls
        elif geom_key == 'moore':
            from lgca.lgca_3dmoore import IBLGCA_Moore as _Cls
        else:
            raise ValueError(
                "Geometry specification is unknown. Try: '1d', 'lin', 'linear', 'square', 'sq', "
                "'rect', 'rectangular', 'hex', 'hx',  'hexagonal', 'cubic', 'cb', or 'moore3d'."
            )
    elif not ve and ib:
        if geom_key == 'lin':
            from lgca.lgca_1d import NoVE_IBLGCA_1D as _Cls
        elif geom_key == 'square':
            from lgca.square_ext import NoVE_IBLGCA_Square as _Cls
        elif geom_key == 'hex':
            from lgca.lgca_hex import NoVE_IBLGCA_Hex as _Cls
        elif geom_key == 'cubic':
            from lgca.cubic_ext import NoVE_IBLGCA_Cubic as _Cls
        elif geom_key == 'moore':
            from lgca.lgca_3dmoore import NoVE_IBLGCA_Moore as _Cls
        else:
            raise ValueError(
                "Geometry specification is unknown. Try: '1d', 'lin', 'linear', 'square', 'sq', "
                "'rect', 'rectangular', 'hex', 'hx',  'hexagonal', 'cubic', 'cb', or 'moore3d'."
            )
    else:
        if geom_key == 'lin':
            from lgca.lgca_1d import LGCA_1D as _Cls
        elif geom_key == 'square':
            from lgca.lgca_square import LGCA_Square as _Cls
        elif geom_key == 'hex':
            from lgca.lgca_hex import LGCA_Hex as _Cls
        elif geom_key == 'cubic':
            from lgca.lgca_cubic import LGCA_Cubic as _Cls
        elif geom_key == 'moore':
            from lgca.lgca_3dmoore import LGCA_3dMoore as _Cls
        else:
            raise ValueError(
                "Geometry specification is unknown. Try: '1d', 'lin', 'linear', 'square', 'sq', "
                "'rect', 'rectangular', 'hex', 'hx',  'hexagonal', 'cubic', 'cb', or 'moore3d'."
            )

    lgca = _Cls(**kwargs)
    _warn_on_node_mismatch(nodes, dims_arg, rest_arg, density_arg, geom_key, n_species)
    return lgca
