# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""
Interaction functions and helper functions for LGCA without volume exclusion.
"""

import numpy as np
from scipy.special import softmax
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
    weights = np.full(lgca.K, 1 / lgca.K)

    nb_nodes = lgca.cell_density[lgca.nonborder]
    newnodes[lgca.nonborder] = lgca.rng.multinomial(nb_nodes, weights)

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
    beta = lgca.interaction_params['beta']
    g = lgca.calc_flux(lgca.nodes)
    if lgca.interaction_params['nb_include_center']:
        g += lgca.nb_sum(g)
    else:
        g = lgca.nb_sum(g)

    weights = softmax(beta * np.einsum('...i,ij->...j', g, lgca.c), axis=-1)

    newnodes = lgca.nodes.copy()
    nb_density = lgca.cell_density[lgca.nonborder]
    newnodes[lgca.nonborder] = lgca.rng.multinomial(nb_density, weights[lgca.nonborder])

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
    beta = lgca.interaction_params['beta']
    g = lgca.calc_flux(lgca.nodes)
    if lgca.interaction_params['nb_include_center']:
        g += lgca.nb_sum(g)
        nsum = lgca.nb_sum(lgca.cell_density)[..., None] + lgca.cell_density[..., None]
    else:
        g = lgca.nb_sum(g)
        nsum = lgca.nb_sum(lgca.cell_density)[..., None]

    np.maximum(nsum, 1, out=nsum)
    g = g / nsum

    weights = softmax(beta * np.einsum('...i,ij->...j', g, lgca.c), axis=-1)

    newnodes = lgca.nodes.copy()
    nb_density = lgca.cell_density[lgca.nonborder]
    newnodes[lgca.nonborder] = lgca.rng.multinomial(nb_density, weights[lgca.nonborder])

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
    nb_nodes = lgca.nodes[lgca.nonborder]
    n_m = nb_nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = nb_nodes[..., lgca.velocitychannels:].sum(-1)
    rho = (n_m + n_r) / lgca.capacity

    prob = tanh_switch(rho, kappa=lgca.interaction_params['kappa'], theta=lgca.interaction_params['theta'])
    j_1 = lgca.rng.binomial(n_m, prob)
    j_2 = lgca.rng.binomial(n_r, 1 - prob)
    n_m = n_m + j_2 - j_1
    n_r = n_r + j_1 - j_2

    n_m -= lgca.rng.binomial(n_m, lgca.interaction_params['r_d'])
    n_r -= lgca.rng.binomial(n_r, lgca.interaction_params['r_d'])
    birth_prob = np.clip(lgca.interaction_params['r_b'] * (1 - rho), 0, 1)
    n_r += lgca.rng.binomial(n_r, birth_prob)

    weights = np.full(lgca.velocitychannels, 1 / lgca.velocitychannels)
    v_channels = lgca.rng.multinomial(n_m, weights)
    r_channels = n_r[..., None]
    nb_new = np.concatenate((v_channels, r_channels), axis=-1)
    lgca.nodes[lgca.nonborder] = nb_new.astype(lgca.nodes.dtype)


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
    nb_nodes = lgca.nodes[lgca.nonborder]
    n_m = nb_nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = nb_nodes[..., lgca.velocitychannels:].sum(-1)
    rho = (n_m + n_r) / lgca.capacity

    prob = tanh_switch(rho, kappa=lgca.interaction_params['kappa'], theta=lgca.interaction_params['theta'])
    j_1 = lgca.rng.binomial(n_m, prob)
    j_2 = lgca.rng.binomial(n_r, 1 - prob)
    n_m = n_m + j_2 - j_1
    n_r = n_r + j_1 - j_2

    weights = np.full(lgca.velocitychannels, 1 / lgca.velocitychannels)
    v_channels = lgca.rng.multinomial(n_m, weights)
    r_channels = n_r[..., None]
    nb_new = np.concatenate((v_channels, r_channels), axis=-1)
    lgca.nodes[lgca.nonborder] = nb_new.astype(lgca.nodes.dtype)
