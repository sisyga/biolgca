# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.

"""
Interaction functions and helper functions for LGCA without volume exclusion.
"""

import numpy as np
from lgca.interactions import tanh_switch

def random_walk(lgca):
    """Perform a random walk rearrangement on the lattice.

    All particles of every occupied lattice site are redistributed
    uniformly over the velocity channels.

    Parameters
    ----------
    lgca : LGCA_1D, LGCA_Square, LGCA_Hex, or LGCA_Cubic
        LGCA instance that the interaction is applied to. The lattice may be
        one-, two-, or three-dimensional.

    Notes
    -----
    ``lgca.nodes`` is overwritten with the new lattice configuration.
    No other attributes are updated.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]

    weights = np.ones(lgca.K)
    weights /= lgca.K

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # reassign particle directions
        sample = lgca.rng.multinomial(n, weights,)

        newnodes[coord] = sample

    lgca.nodes = newnodes

def dd_alignment(lgca):
    """Density dependent alignment of particle directions.

    The local director field is obtained from the flux of all neighboring nodes
    (including the central node if ``nb_include_center`` is set).  Particles in
    each occupied node are then reoriented according to
    ``exp(beta * dot(g, c_i))`` where ``g`` is the director field and ``c_i`` are
    the velocity vectors.

    Parameters
    ----------
    lgca : LGCA_1D, LGCA_Square, LGCA_Hex, or LGCA_Cubic
        LGCA instance that the interaction is applied to. The lattice may be
        one-, two-, or three-dimensional. Requires the
        ``beta`` and ``nb_include_center`` entries in
        ``lgca.interaction_params``.

    Notes
    -----
    ``lgca.nodes`` is replaced by the sampled configuration.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # flux for each lattice site
    if lgca.interaction_params['nb_include_center']:
        g += lgca.nb_sum(g)
    else:
        g = lgca.nb_sum(g)
    # sum of flux of neighbors for each lattice site


    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij', g[coord], lgca.c))

        z = weights.sum()
        # to prevent divisions by zero if the weight is zero
        aux = np.nan_to_num(z)
        weights = np.nan_to_num(weights)
        weights = (weights / aux)
        # In case there are some rounding problems
        if weights.sum() > 1:
            weights = (weights / weights.sum())

        # reassign particle directions
        sample = lgca.rng.multinomial(n, weights,)

        newnodes[coord] = sample

    lgca.nodes = newnodes



def di_alignment(lgca):
    """Density independent alignment of particle directions.

    Similar to :func:`dd_alignment`, but the director field is
    normalized by the number of neighbors so that the reorientation
    does not depend on local density.

    Parameters
    ----------
    lgca : LGCA_1D, LGCA_Square, LGCA_Hex, or LGCA_Cubic
        LGCA instance that the interaction is applied to. The lattice may be
        one-, two-, or three-dimensional. Requires the
        ``beta`` and ``nb_include_center`` entries in
        ``lgca.interaction_params``.

    Notes
    -----
    ``lgca.nodes`` is replaced by the sampled configuration.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # flux for each lattice site
    if lgca.interaction_params['nb_include_center']:
        g += lgca.nb_sum(g)
        nsum = lgca.nb_sum(lgca.cell_density)[..., None] + lgca.cell_density[..., None]     # normalize director field by number of neighbors

    else:
        g = lgca.nb_sum(g)
        nsum = lgca.nb_sum(lgca.cell_density)[..., None]

    np.maximum(nsum, 1, out=nsum)   # avoid dividing by zero later
    g = g / nsum

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij', g[coord], lgca.c))
        z = weights.sum()
        # avoid division by zero if weights is zero
        aux = np.nan_to_num(z)
        weights = np.nan_to_num(weights)
        weights = (weights / aux)
        # In case there are rounding problems
        if weights.sum() > 1:
            weights = (weights / weights.sum())

        # reassign particle directions
        sample = lgca.rng.multinomial(n, weights, )

        newnodes[coord] = sample

    lgca.nodes = newnodes

def go_or_grow(lgca):
    """Interaction step of the go-or-grow model without volume exclusion.

    Cells switch between moving and resting states depending on the local
    density.  Resting cells may proliferate and both phenotypes can die.  After
    the switch and birth/death events, moving cells are reoriented uniformly
    among the velocity channels.

    Parameters
    ----------
    lgca : LGCA_1D, LGCA_Square, LGCA_Hex, or LGCA_Cubic
        LGCA instance that the interaction is applied to. The lattice may be
        one-, two-, or three-dimensional. The following keys in
        ``lgca.interaction_params`` are used: ``kappa``, ``theta``, ``r_b`` and
        ``r_d``.

    Notes
    -----
    ``lgca.nodes`` is overwritten with the updated node configuration.
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)

    for coord in zip(*coords):
        # determine cell number and moving and resting cell population at coordinate
        n = lgca.cell_density[coord]
        n_mxy = n_m[coord]
        n_rxy = n_r[coord]
        rho = n / lgca.capacity

        # phenotypic switch
        j_1 = lgca.rng.binomial(n_mxy, tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                              theta=lgca.interaction_params['theta']))
        j_2 = lgca.rng.binomial(n_rxy, 1 - tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                                  theta=lgca.interaction_params['theta']))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2

        # death
        n_mxy -= lgca.rng.binomial(n_mxy * np.heaviside(n_mxy, 0), lgca.interaction_params['r_d'])
        n_rxy -= lgca.rng.binomial(n_rxy * np.heaviside(n_rxy, 0), lgca.interaction_params['r_d'])

        # birth
        n_rxy += lgca.rng.binomial(n_rxy * np.heaviside(n_rxy, 0), np.maximum(lgca.interaction_params['r_b']*(1-rho), 0))

        # reorientation
        v_channels = lgca.rng.multinomial(n_mxy, [1/lgca.velocitychannels]*lgca.velocitychannels)

        # add resting cells and assign new content of node at the end of interaction step
        r_channels = np.array([n_rxy])
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


def go_or_rest(lgca):
    """Simplified go-or-grow interaction without birth and death.

    Only the switching between moving and resting states and the
    reorientation of moving cells are performed.

    Parameters
    ----------
    lgca : LGCA_1D, LGCA_Square, LGCA_Hex, or LGCA_Cubic
        LGCA instance that the interaction is applied to. The lattice may be
        one-, two-, or three-dimensional. Uses the ``kappa`` and
        ``theta`` values from ``lgca.interaction_params``.

    Notes
    -----
    ``lgca.nodes`` is overwritten with the updated node configuration.
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)

    for coord in zip(*coords):
        # determine cell number and moving and resting cell population at coordinate
        n = lgca.cell_density[coord]
        n_mxy = n_m[coord]
        n_rxy = n_r[coord]
        rho = n / lgca.capacity

        # phenotypic switch
        j_1 = lgca.rng.binomial(n_mxy, tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                              theta=lgca.interaction_params['theta']))
        j_2 = lgca.rng.binomial(n_rxy, 1 - tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                                  theta=lgca.interaction_params['theta']))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2

        # reorientation
        v_channels = lgca.rng.multinomial(n_mxy, [1/lgca.velocitychannels]*lgca.velocitychannels)

        # add resting cells and assign new content of node at the end of interaction step
        r_channels = np.array([n_rxy])
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node
