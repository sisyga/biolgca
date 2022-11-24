# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
Interaction functions and helper functions for classical LGCA with volume exclusion.
"""

from bisect import bisect_left
from random import random

import numpy as np
import numpy.random as npr
from scipy.special import binom as binom_coeff


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
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


def ent_prod(x):
    return x * np.log(x, where=x > 0, out=np.zeros_like(x, dtype=float))


def random_walk(lgca):
    """
    Shuffle config in the last axis, modeling a random walk.
    :return:
    """
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)

    # disarrange(lgca.nodes, axis=-1)
    # relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
    #            (lgca.cell_density[lgca.nonborder] < lgca.K)
    # coords = [a[relevant] for a in lgca.nonborder]
    # for coord in zip(*coords):
    #     npr.shuffle(lgca.nodes[coord])


def birth(lgca):
    """
    Simple birth process coupled to a random walk
    :return:
    """
    birth = npr.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * lgca.cell_density[..., None] / lgca.K
    np.add(lgca.nodes, (1 - lgca.nodes) * birth, out=lgca.nodes, casting='unsafe')
    random_walk(lgca)


def birthdeath(lgca):
    """
    Simple birth-death process coupled to a random walk
    :return:
    """
    birth = npr.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * lgca.cell_density[..., None] / lgca.K
    death = npr.random(lgca.nodes.shape) < lgca.interaction_params['r_d']
    ds = (1 - lgca.nodes) * birth - lgca.nodes * death
    np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
    lgca.update_dynamic_fields()
    random_walk(lgca)


def persistent_walk(lgca):
    """
    Rearrangement step for persistent motion (alignment with yourlgca)
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    newnodes = lgca.nodes.copy()
    g = lgca.calc_flux(lgca.nodes)
    for coord in zip(*coords):
        n = lgca.cell_density[coord]

        permutations = lgca.permutations[n]
        j = lgca.j[n]
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij', g[coord], j)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes


def chemotaxis(lgca):
    """
    Rearrangement step for chemotaxis to external gradient field
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        n = lgca.cell_density[coord]

        permutations = lgca.permutations[n]
        j = lgca.j[n]
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij',
                                                                     lgca.interaction_params['gradient_field'][coord],
                                                                     j)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes


def contact_guidance(lgca):
    """
    Rearrangement step for contact guidance interaction. Cells are guided by an external axis
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        sni = lgca.guiding_tensor[coord]
        permutations = lgca.permutations[n]
        si = lgca.si[n]
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('ijk,jk', si, sni)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes


def alignment(lgca):
    """
    Rearrangement step for alignment interaction
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    # gives ndarray of boolean values
    coords = [a[relevant] for a in lgca.nonborder]
    # a is an array of numbers, array can be indexed with another array of same size with boolean specification if
    # element should be included. Returns only the relevant elements and coords is a list here
    g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    g = lgca.nb_sum(g)  # calculates sum of flux of neighbors for each lattice site
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        j = lgca.j[n]  # flux per permutation
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij', g[coord], j)).cumsum()
        # multiply neighborhood flux with the flux for each possible permutation
        # np.exp for probability
        # cumsum() for cumulative distribution function
        ind = bisect_left(weights, random() * weights[-1])
        # inverse transform sampling method
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes


def nematic(lgca):
    """
    Rearrangement step for nematic interaction
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]

    s = np.einsum('ijk,klm', lgca.nodes[..., :lgca.velocitychannels], lgca.cij)
    sn = lgca.nb_sum(s)

    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        sni = sn[coord]
        permutations = lgca.permutations[n]
        si = lgca.si[n]
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('ijk,jk', si, sni)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes


def aggregation(lgca):
    """
    Aggregation interaction.

    Parameters
    ----------
    lgca: LGCA_1D or LGCA_Square or LGCA_Hex
          LGCA instance that the interaction is applied to
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]

    g = np.asarray(lgca.gradient(lgca.cell_density))  # np.asarray not needed
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        j = lgca.j[n]
        weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij', g[coord], j)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes

def wetting(lgca):
    """
    Wetting of a surface for different levels of E-cadherin
    :param lgca:
    :return:
    """
    if hasattr(lgca, 'spheroid'):
        birth = npr.random(lgca.nodes[lgca.spheroid].shape) < lgca.interaction_params['r_b']
        ds = (1 - lgca.nodes[lgca.spheroid]) * birth
        lgca.nodes[lgca.spheroid, :] = np.add(lgca.nodes[lgca.spheroid, :], ds, casting='unsafe')
        lgca.update_dynamic_fields()
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    nbs = lgca.nb_sum(lgca.cell_density)  # + lgca.cell_density
    nbs *= np.clip(1 - nbs / lgca.n_crit, a_min=0, a_max=None) / lgca.n_crit * 2
    g_adh = lgca.gradient(nbs)
    pressure = np.clip(lgca.cell_density - lgca.interaction_params['rho_0'], a_min=0., a_max=None) / \
               (lgca.K - lgca.interaction_params['rho_0'])
    g_pressure = -lgca.gradient(pressure)

    resting = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    resting = lgca.nb_sum(resting) / lgca.velocitychannels / lgca.interaction_params['rho_0']
    g = lgca.calc_flux(lgca.nodes)
    g = lgca.nb_sum(g)

    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        restc = permutations[:, lgca.velocitychannels:].sum(-1)
        j = lgca.j[n]
        j_nb = g[coord]
        weights = np.exp(
            lgca.interaction_params['beta'] * (j_nb[0] * j[0] + j_nb[1] * j[1]) / lgca.velocitychannels / 2
            + lgca.interaction_params['beta'] * resting[coord] * restc
            # * np.clip(1 - restc / lgca.interaction_params['rho_0'] / 2, a_min=0, a_max=None) * 2
            + lgca.interaction_params['beta'] * np.einsum('i,ij', g_adh[coord], j)
            # + lgca.interaction_params['alpha'] * np.einsum('i,ij', g_subs[coord], j)
            + restc * lgca.ecm[coord]
            + lgca.interaction_params['gamma'] * np.einsum('i,ij', g_pressure[coord], j)
        ).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes
    lgca.ecm -= lgca.interaction_params['alpha'] * lgca.ecm * lgca.cell_density / lgca.K


def excitable_medium(lgca):
    """
    Model for an excitable medium based on Barkley's PDE model.
    :return:
    """
    n_x = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_y = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    rho_x = n_x / lgca.velocitychannels
    rho_y = n_y / lgca.restchannels
    p_xp = rho_x ** 2 * (1 + (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha'])
    p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha']
    p_yp = rho_x
    p_ym = rho_y
    dn_y = (npr.random(n_y.shape) < p_yp).astype(np.int8)
    dn_y -= npr.random(n_y.shape) < p_ym
    for _ in range(lgca.interaction_params['N']):
        dn_x = (npr.random(n_x.shape) < p_xp).astype(np.int8)
        dn_x -= npr.random(n_x.shape) < p_xm
        n_x += dn_x
        rho_x = n_x / lgca.velocitychannels
        p_xp = rho_x ** 2 * (1 + (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha'])
        p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha']

    n_y += dn_y

    newnodes = np.zeros(lgca.nodes.shape, dtype=lgca.nodes.dtype)
    for coord in lgca.coord_pairs:
        newnodes[coord + (slice(0, n_x[coord]),)] = 1
        newnodes[coord + (slice(lgca.velocitychannels, lgca.velocitychannels + n_y[coord]),)] = 1

    newv = newnodes[..., :lgca.velocitychannels]
    disarrange(newv, axis=-1)
    newnodes[..., :lgca.velocitychannels] = newv
    lgca.nodes = newnodes


def go_or_grow(lgca):
    """
    interactions of the go-or-grow model.
    :return:
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    M1 = np.minimum(n_m, lgca.restchannels - n_r)
    M2 = np.minimum(n_r, lgca.velocitychannels - n_m)
    for coord in zip(*coords):
        # node = lgca.nodes[coord]
        n = lgca.cell_density[coord]

        n_mxy = n_m[coord]
        n_rxy = n_r[coord]

        rho = n / lgca.K
        j_1 = npr.binomial(M1[coord], tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                                  theta=lgca.interaction_params['theta']))
        j_2 = npr.binomial(M2[coord], 1 - tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                                      theta=lgca.interaction_params['theta']))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2
        n_mxy -= npr.binomial(n_mxy, lgca.interaction_params['r_d'])
        n_rxy -= npr.binomial(n_rxy, lgca.interaction_params['r_d'])
        M = min([n_rxy, lgca.restchannels - n_rxy])
        n_rxy += npr.binomial(M, lgca.interaction_params['r_b'])

        v_channels = [1] * n_mxy + [0] * (lgca.velocitychannels - n_mxy)
        v_channels = npr.permutation(v_channels)
        r_channels = np.zeros(lgca.restchannels)
        r_channels[:n_rxy] = 1
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


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


def leup_test(lgca):
    """
    Go-or-grow with least-environmental uncertainty principle. cells try to minimize their entropy with the environment,
    by changing their state between moving and resting. resting cells can proliferate. all cells die at a constant rate.
    :return:
    """
    if lgca.interaction_params['r_b'] > 0 or lgca.interaction_params['r_d'] > 0:
        n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
        n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
        birth = np.zeros_like(lgca.nodes)
        birth[..., lgca.velocitychannels:] = npr.random(n_r.shape + (lgca.restchannels,)) \
                                             < lgca.interaction_params['r_b'] * n_r[..., None] / lgca.restchannels
        death = npr.random(birth.shape) < lgca.interaction_params['r_d'] * (
                n_m[..., None] / lgca.velocitychannels + n_r[..., None] / lgca.restchannels) / 2
        ds = (1 - lgca.nodes) * birth - lgca.nodes * death
        np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
        lgca.update_dynamic_fields()

    # if lgca.interaction_params['r_b'] > 0: # or lgca.interaction_params['r_d'] > 0:
    #     n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    #     n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    #     birth = npr.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * n_r[..., None] / lgca.restchannels
    #     death = npr.random(birth.shape) < lgca.interaction_params['r_d'] * (n_r[..., None] / lgca.restchannels + n_m[..., None] / lgca.velocitychannels)
    #     ds = (1 - lgca.nodes) * birth - lgca.nodes * death
    #     np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
    #     lgca.update_dynamic_fields()

    relevant = (lgca.cell_density[lgca.nonborder] > 0) & (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    n = lgca.cell_density
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    M1 = np.minimum(n_m, lgca.restchannels - n_r)
    M2 = np.minimum(n_r, lgca.velocitychannels - n_m)

    p0 = np.divide(n_r, n, where=n > 0, out=np.zeros_like(n, dtype=float))
    s = s_binom(n, p0, lgca.velocitychannels)
    p10 = np.divide(n_r + 1, n, where=n > 0, out=np.zeros_like(n, dtype=float))
    s10 = s_binom(n, p10, lgca.velocitychannels)
    ds1 = s10 - s

    p01 = np.divide(n_r - 1, n, where=n > 0, out=np.zeros_like(n, dtype=float))
    s01 = s_binom(n, p01, lgca.velocitychannels)
    ds2 = s01 - s
    #
    #
    # s = ent_prod(n_r) + ent_prod(n_m)
    # ds1 = np.divide(s - ent_prod(n_r + 1) - ent_prod(n_m - 1), n, where=n > 0)
    p1 = 1 / (1 + np.exp(lgca.interaction_params['beta'] * ds1))
    p1[M1 == 0] = 0.
    # ds2 = np.divide(s - ent_prod(n_r - 1) - ent_prod(n_m + 1), n, where=n > 0)
    p2 = 1 / (1 + np.exp(lgca.interaction_params['beta'] * ds2))
    p2[M2 == 0] = 0.
    try:
        j_1 = npr.binomial(M1, p1)

    except:
        print('Error!')
        ind = np.isnan(p1) | (p1 > 0) | (p1 > 0)
        print(M1[ind], p1[ind])

    try:
        j_2 = npr.binomial(M2, p2)

    except:
        print('Error!')
        ind = np.isnan(p2) | (p2 > 0) | (p2 > 0)
        print(M2[ind], p2[ind])

    n_m += j_2 - j_1
    n_r += j_1 - j_2

    for coord in zip(*coords):
        # # node = lgca.nodes[coord]
        # n = lgca.cell_density[coord]
        #
        n_mxy = n_m[coord]
        n_rxy = n_r[coord]
        #
        # s = ent_prod(n_rxy) + ent_prod(n_mxy)
        # ds1 = (s - ent_prod(n_rxy+1) - ent_prod(n_mxy-1)) / n
        # p1 = 1 / (1 + exp(lgca.interaction_params['beta'] * ds1))
        # # switch to velocity channel
        # ds2 = (s - ent_prod(n_rxy-1) - ent_prod(n_mxy+1)) / n
        # p2 = 1 / (1 + exp(lgca.interaction_params['beta'] * ds2))
        #
        # j_1 = npr.binomial(M1[coord], p1)
        # j_2 = npr.binomial(M2[coord], p2)
        # n_mxy += j_2 - j_1
        # n_rxy += j_1 - j_2
        v_channels = npr.choice(lgca.velocitychannels, n_mxy, replace=False)

        # v_channels = [1] * n_mxy + [0] * (lgca.velocitychannels - n_mxy)
        # v_channels = npr.permutation(v_channels)
        # r_channels = np.zeros(lgca.restchannels)
        # r_channels[:n_rxy] = 1
        node = np.zeros(lgca.K, dtype='bool')
        node[v_channels] = 1
        node[lgca.velocitychannels:lgca.velocitychannels + n_rxy] = 1
        # node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


def go_or_rest(lgca):
    """
    Interactions of the go-or-grow model without birth and death, i.e. only the switch and random walk.
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    M1 = np.minimum(n_m, lgca.restchannels - n_r)
    M2 = np.minimum(n_r, lgca.velocitychannels - n_m)

    for coord in zip(*coords):
        n = lgca.cell_density[coord]

        n_mxy = n_m[coord]
        n_rxy = n_r[coord]

        rho = n / lgca.K
        j_1 = npr.binomial(M1[coord], tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                                  theta=lgca.interaction_params['theta']))
        j_2 = npr.binomial(M2[coord], 1 - tanh_switch(rho, kappa=lgca.interaction_params['kappa'],
                                                      theta=lgca.interaction_params['theta']))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2

        v_channels = [1] * n_mxy + [0] * (lgca.velocitychannels - n_mxy)
        v_channels = npr.permutation(v_channels)
        r_channels = np.zeros(lgca.restchannels)
        r_channels[:n_rxy] = 1
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


def only_propagation(lgca):
    pass
