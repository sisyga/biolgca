# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""
Interaction functions and helper functions for classical LGCA with volume exclusion.
"""

from scipy.special import binom as binom_coeff, softmax
import numpy as np


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

    Other Parameters
    ----------------
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

    Other Parameters
    ----------------
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

    Other Parameters
    ----------------
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

    Other Parameters
    ----------------
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

    Other Parameters
    ----------------
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

    Other Parameters
    ----------------
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
        si = lgca.get_si_permutations(n)
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

    Other Parameters
    ----------------
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
        si = lgca.get_si_permutations(n)
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

    Other Parameters
    ----------------
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


def excitable_medium(lgca):
    """Simulate an excitable medium following Barkley's model.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Other Parameters
    ----------------
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
    """Switch cells between moving and resting, then apply death and resting-cell birth.

    Moving cells switch to free rest channels with probability
    ``tanh_switch(rho, kappa, theta)``, where ``rho`` is the node density
    divided by ``K``; resting cells switch to free velocity channels with the
    complementary probability. Every cell then dies with probability ``r_d``,
    and each resting cell divides into a free rest channel with probability
    ``r_b``. Moving cells are finally shuffled over the velocity channels.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Other Parameters
    ----------------
    r_b : float
        Division probability of a resting cell per time step.
    r_d : float
        Death probability of every cell per time step.
    kappa : float
        Steepness of the density-dependent switch.
    theta : float
        Relative density at which the switch probability is one half.

    Notes
    -----
    The ``lgca`` object is modified in place.
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
