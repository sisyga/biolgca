# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""
Interaction functions and helper functions for classical LGCA with volume exclusion.
"""

from bisect import bisect_left
from scipy.special import binom as binom_coeff, softmax
import numpy as np


def disarrange(a: np.ndarray, axis=-1):
    """
    Shuffle a in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of a. Each one-dimensional
    slice is shuffled independently.

    THIS IS A LEGACY FUNCTION, USE THE RANDOM WALK FUNCTION INSTEAD FOR A MORE EFFICIENT IMPLEMENTATION!

    Parameters
    ----------
    a : numpy.ndarray
        The array to shuffle
    axis : int, optional, default: -1
           Along which axis to shuffle `a`. The default is -1, which implies the
           last axis.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis. `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    rng = npr.default_rng()
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        rng.shuffle(b[ndx])
    return


def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


def ent_prod(x):
    return x * np.log(x, where=x > 0, out=np.zeros_like(x, dtype=float))



def random_walk(lgca):
    """Apply a random walk to all nodes.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Notes
    -----
    The ``lgca`` object is modified in place. This interaction does not use
    ``lgca.interaction_params``.

    Returns
    -------
    None
    """
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


def birth(lgca):
    """Perform a simple birth step followed by a random walk.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float
        Birth probability per empty channel; effective
        probability is ``r_b * n / lgca.K`` for local density ``n``.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    birth = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * lgca.cell_density[..., None] / lgca.K
    np.add(lgca.nodes, (1 - lgca.nodes) * birth, out=lgca.nodes, casting='unsafe')
    random_walk(lgca)


def birthdeath(lgca):
    """Perform a birth--death step followed by a random walk.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float
        Birth probability per empty channel.
    r_d : float
        Death probability per occupied channel.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    birth = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * lgca.cell_density[..., None] / lgca.K
    death = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_d']
    ds = (1 - lgca.nodes) * birth - lgca.nodes * death
    np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
    lgca.update_dynamic_fields()
    random_walk(lgca)


def alignment(lgca):
    """Align moving cells with their neighbours.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Alignment strength with neighbouring velocities.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    beta = lgca.interaction_params['beta']
    g = lgca.nb_sum(lgca.calc_flux(lgca.nodes))

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]
    flux = g[lgca.nonborder]
    density = lgca.cell_density[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n

        j = lgca.get_flux_permutations(n)
        weights = softmax(beta * (flux[mask] @ j), axis=1)
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = lgca.get_permutations(n)[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes


def persistent_walk(lgca):
    """Rearrange nodes to implement persistent motion.
    Vectorized implementation.
    See also: `alignment`. This is a special case of alignment where the interaction radius is 0.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of alignment with the previous velocity direction.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    beta = lgca.interaction_params['beta']
    g = lgca.calc_flux(lgca.nodes)

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]
    flux = g[lgca.nonborder]
    density = lgca.cell_density[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n

        j = lgca.get_flux_permutations(n)
        weights = softmax(beta * (flux[mask] @ j), axis=1)
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = lgca.get_permutations(n)[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes



def chemotaxis(lgca):
    """Rearrange nodes in response to an external gradient.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of the bias toward the gradient.
    gradient_field : numpy.ndarray
        External gradient field influencing motion.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    beta = lgca.interaction_params['beta']
    grad = lgca.interaction_params['gradient_field']

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]
    gradients = grad[lgca.nonborder]
    density = lgca.cell_density[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n
        j = lgca.get_flux_permutations(n)
        weights = softmax(beta * (gradients[mask] @ j), axis=1)
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = lgca.get_permutations(n)[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes


def contact_guidance(lgca):
    """Align cells with a predefined guiding axis.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Alignment strength toward the guiding axis.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    beta = lgca.interaction_params['beta']

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]
    tensors = lgca.guiding_tensor[lgca.nonborder]
    density = lgca.cell_density[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n
        si = lgca.si[n]
        weights = softmax(beta * np.einsum('nij,pij->np', tensors[mask], si), axis=1)
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = lgca.get_permutations(n)[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes


def nematic(lgca):
    """Implement nematic alignment of neighbouring cells.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of nematic alignment.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    beta = lgca.interaction_params['beta']

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]
    s = np.einsum('...k,kxy->...xy', lgca.nodes[..., :lgca.velocitychannels], lgca.cij)
    sn = lgca.nb_sum(s)
    tensors = sn[lgca.nonborder]
    density = lgca.cell_density[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n
        si = lgca.si[n]
        weights = softmax(beta * np.einsum('nij,pij->np', tensors[mask], si), axis=1)
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = lgca.get_permutations(n)[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes


def aggregation(lgca):
    """Bias movement towards higher cell density.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of the bias toward higher density regions.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    beta = lgca.interaction_params['beta']
    g = lgca.gradient(lgca.cell_density)

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]
    grad = g[lgca.nonborder]
    density = lgca.cell_density[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n
        j = lgca.get_flux_permutations(n)
        weights = softmax(beta * (grad[mask] @ j), axis=1)
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = lgca.get_permutations(n)[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes


def wetting(lgca):
    """Model wetting dynamics on an adhesive surface.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float
        Birth probability used inside the spheroid region.
    rho_0 : float
        Homeostatic density determining the pressure gradient.
    alpha : float
        ECM degradation rate.
    beta : float
        Adhesion strength weighting flux alignment.
    gamma : float
        Strength of the pressure gradient term.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    if hasattr(lgca, 'spheroid'):
        birth = lgca.rng.random(lgca.nodes[lgca.spheroid].shape) < lgca.interaction_params['r_b']
        ds = (1 - lgca.nodes[lgca.spheroid]) * birth
        lgca.nodes[lgca.spheroid, :] = np.add(lgca.nodes[lgca.spheroid, :], ds, casting='unsafe')
        lgca.update_dynamic_fields()

    newnodes = lgca.nodes.copy()
    nb_nodes = newnodes[lgca.nonborder]

    nbs = lgca.nb_sum(lgca.cell_density)
    nbs *= np.clip(1 - nbs / lgca.n_crit, a_min=0, a_max=None) / lgca.n_crit * 2
    g_adh = lgca.gradient(nbs)
    pressure = (np.clip(lgca.cell_density - lgca.interaction_params['rho_0'], a_min=0., a_max=None) /
                (lgca.K - lgca.interaction_params['rho_0']))
    g_pressure = -lgca.gradient(pressure)

    resting = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    resting = lgca.nb_sum(resting) / lgca.velocitychannels / lgca.interaction_params['rho_0']
    g = lgca.nb_sum(lgca.calc_flux(lgca.nodes))

    density = lgca.cell_density[lgca.nonborder]
    flux = g[lgca.nonborder]
    rest_nb = resting[lgca.nonborder]
    g_adh_nb = g_adh[lgca.nonborder]
    g_press_nb = g_pressure[lgca.nonborder]
    ecm_nb = lgca.ecm[lgca.nonborder]

    unique = np.unique(density)
    unique = unique[(unique > 0) & (unique < lgca.K)]
    for n in unique:
        mask = density == n
        perms = lgca.get_permutations(n)
        restc = perms[:, lgca.velocitychannels:].sum(-1)
        j = lgca.j[n]
        weights = softmax(
            lgca.interaction_params['beta'] * (flux[mask] @ j) / lgca.velocitychannels / 2
            + lgca.interaction_params['beta'] * rest_nb[mask, None] * restc
            + lgca.interaction_params['beta'] * np.einsum('nd,dp->np', g_adh_nb[mask], j)
            + restc * ecm_nb[mask, None]
            + lgca.interaction_params['gamma'] * np.einsum('nd,dp->np', g_press_nb[mask], j),
            axis=1,
        )
        cumw = weights.cumsum(axis=1)
        rnd = lgca.rng.random(mask.sum())
        ind = (rnd[:, None] < cumw).argmax(axis=1)
        nb_nodes[mask] = perms[ind]

    newnodes[lgca.nonborder] = nb_nodes
    lgca.nodes = newnodes
    lgca.ecm -= lgca.interaction_params['alpha'] * lgca.ecm * lgca.cell_density / lgca.K


def excitable_medium(lgca):
    """Simulate an excitable medium following Barkley's model.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    alpha : float
        Controls the excitability of the medium.
    beta : float
        Interaction coefficient between species.
    N : int
        Number of sub-steps performed in each interaction.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    n_x = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_y = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    rho_x = n_x / lgca.velocitychannels
    rho_y = n_y / lgca.restchannels
    p_xp = rho_x ** 2 * (1 + (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha'])
    p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha']
    p_yp = rho_x
    p_ym = rho_y
    dn_y = (lgca.rng.random(n_y.shape) < p_yp).astype(np.int8)
    dn_y -= lgca.rng.random(n_y.shape) < p_ym
    for _ in range(lgca.interaction_params['N']):
        dn_x = (lgca.rng.random(n_x.shape) < p_xp).astype(np.int8)
        dn_x -= lgca.rng.random(n_x.shape) < p_xm
        n_x += dn_x
        rho_x = n_x / lgca.velocitychannels
        p_xp = rho_x ** 2 * (1 + (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha'])
        p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha']

    n_y += dn_y

    newnodes = np.zeros_like(lgca.nodes)
    v_idx = np.arange(lgca.velocitychannels)
    r_idx = np.arange(lgca.restchannels)
    newnodes[..., :lgca.velocitychannels] = (v_idx < n_x[..., None]).astype(lgca.nodes.dtype)
    newnodes[..., lgca.velocitychannels:] = (r_idx < n_y[..., None]).astype(lgca.nodes.dtype)
    newnodes[..., :lgca.velocitychannels] = lgca.rng.permuted(newnodes[..., :lgca.velocitychannels], axis=-1)
    lgca.nodes = newnodes



def go_or_grow(lgca):
    """Perform the go-or-grow switching interaction.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float # This parameter is not used in the provided snippet for go_or_grow, but kept for consistency if it
    was intended.
        Birth probability for resting cells. 
    r_d : float # This parameter is not used in the provided snippet for go_or_grow
        Death probability for both states.
    beta : float # Renamed from kappa in some contexts, this is the sensitivity for switching
        Steepness of the switching function / sensitivity to entropy change.
    theta : float # This parameter is not used here, tanh_switch is not directly called for entropy part
        Threshold density for switching.

    Notes
    -----
    The ``lgca`` object is modified in place.
    This version attempts to use the entropy-based switching logic.
    Assumes lgca.velocitychannels and lgca.restchannels are defined.
    Uses a placeholder `_s_binom_entropy_like` for the undefined `s_binom`.
    """
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    M1 = np.minimum(n_m, lgca.restchannels - n_r)
    M2 = np.minimum(n_r, lgca.velocitychannels - n_m)

    rho = lgca.cell_density / lgca.K
    prob = tanh_switch(rho, kappa=lgca.interaction_params['kappa'], theta=lgca.interaction_params['theta'])
    j_1 = lgca.rng.binomial(M1, prob)
    j_2 = lgca.rng.binomial(M2, 1 - prob)
    n_m = n_m + j_2 - j_1
    n_r = n_r + j_1 - j_2
    n_m -= lgca.rng.binomial(n_m, lgca.interaction_params['r_d'])
    n_r -= lgca.rng.binomial(n_r, lgca.interaction_params['r_d'])
    M = np.minimum(n_r, lgca.restchannels - n_r)
    n_r += lgca.rng.binomial(M, lgca.interaction_params['r_b'])

    newnodes = np.zeros_like(lgca.nodes)
    v_idx = np.arange(lgca.velocitychannels)
    r_idx = np.arange(lgca.restchannels)
    newnodes[..., :lgca.velocitychannels] = (v_idx < n_m[..., None]).astype(lgca.nodes.dtype)
    newnodes[..., lgca.velocitychannels:] = (r_idx < n_r[..., None]).astype(lgca.nodes.dtype)
    newnodes[..., :lgca.velocitychannels] = lgca.rng.permuted(newnodes[..., :lgca.velocitychannels], axis=-1)
    lgca.nodes = newnodes

def p_binom(k, n, p):
    pb = binom_coeff(n, k) * p ** k * (1 - p) ** (n - k)
    pb[n < k] = 0.
    return pb


def s_binom(n, p0, kmax):
    n = n[..., None]
    p0 = p0[..., None]
    k = np.arange(kmax + 1)
    p = p_binom(k, n, p0)
    return -ent_prod(p).sum(-1)


def go_or_rest(lgca):
    """Switch cells between moving and resting states based on local density.
    Uses tanh_switch function. Assumes lgca.nodes is (K, dims...).
    No birth or death in this version.
    """
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    M1 = np.minimum(n_m, lgca.restchannels - n_r)
    M2 = np.minimum(n_r, lgca.velocitychannels - n_m)

    rho = lgca.cell_density / lgca.K
    prob = tanh_switch(rho, kappa=lgca.interaction_params['kappa'], theta=lgca.interaction_params['theta'])
    j_1 = lgca.rng.binomial(M1, prob)
    j_2 = lgca.rng.binomial(M2, 1 - prob)
    n_m = n_m + j_2 - j_1
    n_r = n_r + j_1 - j_2

    newnodes = np.zeros_like(lgca.nodes)
    v_idx = np.arange(lgca.velocitychannels)
    r_idx = np.arange(lgca.restchannels)
    newnodes[..., :lgca.velocitychannels] = (v_idx < n_m[..., None]).astype(lgca.nodes.dtype)
    newnodes[..., lgca.velocitychannels:] = (r_idx < n_r[..., None]).astype(lgca.nodes.dtype)
    newnodes[..., :lgca.velocitychannels] = lgca.rng.permuted(newnodes[..., :lgca.velocitychannels], axis=-1)
    lgca.nodes = newnodes


def only_propagation(lgca):
    """Placeholder interaction that performs no rearrangement.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Notes
    -----
    The ``lgca`` object is modified in place but remains unchanged. This
    interaction does not use ``lgca.interaction_params``.

    Returns
    -------
    None
    """
    pass