from bisect import bisect_left
from random import random

import numpy as np
import numpy.random as npr


def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
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

def random_walk(lgca):
    """
    Shuffle config in the last axis, modeling a random walk.
    :return:
    """
    # disarrange(lgca.nodes, axis=-1)
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        npr.shuffle(lgca.nodes[coord])


def birth(lgca):
    """
    Simple birth process coupled to a random walk
    :return:
    """
    birth = npr.random(lgca.nodes.shape) < lgca.r_b * lgca.cell_density[..., None] / lgca.K
    np.add(lgca.nodes, (1 - lgca.nodes) * birth, out=lgca.nodes, casting='unsafe')
    random_walk(lgca)


def birthdeath(lgca):
    """
    Simple birth-death process coupled to a random walk
    :return:
    """
    birth = npr.random(lgca.nodes.shape) < lgca.r_b * lgca.cell_density[..., None] / lgca.K
    death = npr.random(lgca.nodes.shape) < lgca.r_d
    ds = (1 - lgca.nodes) * birth - lgca.nodes * death
    np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
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
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], j)).cumsum()
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
        weights = np.exp(lgca.beta * np.einsum('i,ij', lgca.g[coord], j)).cumsum()
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
        weights = np.exp(lgca.beta * np.einsum('ijk,jk', si, sni)).cumsum()
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
    coords = [a[relevant] for a in lgca.nonborder]
    g = lgca.calc_flux(lgca.nodes)
    g = lgca.nb_sum(g)
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        j = lgca.j[n]
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], j)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
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
        weights = np.exp(lgca.beta * np.einsum('ijk,jk', si, sni)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes


def aggregation(lgca):
    """
    Rearrangement step for aggregation interaction
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]

    g = np.asarray(lgca.gradient(lgca.cell_density))
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        j = lgca.j[n]
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], j)).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes

def wetting(lgca):
    """
    Wetting of a surface for different levels of E-cadherin
    :param n_crit:
    :param lgca:
    :return:
    """
    if hasattr(lgca, 'spheroid'):
        rho = lgca.cell_density[lgca.spheroid, None] / lgca.K
        birth = npr.random(lgca.nodes[lgca.spheroid].shape) < lgca.r_b  # / (1 - rho)
        # nbs = lgca.nb_sum(lgca.cell_density) + lgca.cell_density - 1
        # nbs /= lgca.n_crit
        # death = npr.random(lgca.nodes[lgca.spheroid].shape) < lgca.r_b / (lgca.rho_0 + 1)
        death = 0
        ds = (1 - lgca.nodes[lgca.spheroid]) * birth - lgca.nodes[lgca.spheroid] * death
        lgca.nodes[lgca.spheroid, :] = np.add(lgca.nodes[lgca.spheroid, :], ds, casting='unsafe')
        lgca.update_dynamic_fields()
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)  # & \
    # (np.invert(lgca.ecm[lgca.nonborder]))
    coords = [a[relevant] for a in lgca.nonborder]
    # nbs = np.clip(lgca.nb_sum(lgca.cell_density) + lgca.cell_density, a_min=None, a_max=lgca.n_crit)
    nbs = lgca.nb_sum(lgca.cell_density)
    nbs *= np.clip(1 - nbs / lgca.n_crit, a_min=0, a_max=None) / lgca.n_crit * 2
    # adh_weight = lgca.channel_weight(nbs) - nbs[..., None]
    g_adh = lgca.gradient(nbs)
    pressure = np.clip(lgca.cell_density - lgca.rho_0, a_min=0., a_max=None) / (lgca.K - lgca.rho_0)
    g_pressure = -lgca.gradient(pressure)
    # pressure_weight = np.clip(lgca.cell_density[..., None] - 1 - lgca.rho_0, a_min=0, a_max=None) - lgca.channel_weight(pressure)

    if hasattr(lgca, 'ecm'):
        ecm_weight = -lgca.channel_weight(lgca.ecm)

    else:
        ecm_weight = lgca.channel_weight(lgca.cell_density) * 0


    resting = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    resting = lgca.nb_sum(resting) / lgca.velocitychannels / lgca.rho_0
    # resting *= np.clip(1 - resting / lgca.rho_0 / lgca.velocitychannels / 2, a_min=0, a_max=None) / lgca.velocitychannels / lgca.rho_0
    #resting = np.clip(resting, a_min=0, a_max=lgca.velocitychannels * lgca.rho_0) / lgca.velocitychannels / lgca.rho_0
    # resting = np.clip(resting, a_min=None, a_max=lgca.n_crit)
    g = lgca.calc_flux(lgca.nodes)
    g = lgca.nb_sum(g)
    #g = np.clip(g, a_max=lgca.n_crit, a_min=-lgca.n_crit)
    # g = np.divide(g, lgca.cell_density[..., None], where=lgca.cell_density[..., None]>0, out=np.zeros_like(g))
    # g = np.divide(g, nbs[..., None], where=nbs[..., None]>0, out=np.zeros_like(g))

    # g = lgca.gradient(np.clip(lgca.cell_density.astype(float), 0, lgca.n_crit))
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        # velocityc = permutations[:, :lgca.velocitychannels]
        restc = permutations[:, lgca.velocitychannels:].sum(-1)
        j = lgca.j[n]
        j_nb = g[coord]
        weights = np.exp(
            lgca.beta * (j_nb[0] * j[0] + j_nb[1] * j[1]) / lgca.velocitychannels / 2
            + lgca.beta * resting[coord] * restc  #* np.clip(1 - restc / lgca.rho_0 / 2, a_min=0, a_max=None) * 2
            # + lgca.beta * np.dot(permutations[:, :lgca.velocitychannels], adh_weight[coord])
            + lgca.beta * np.einsum('i,ij', g_adh[coord], j)
            # + lgca.alpha * np.einsum('i,ij', g_subs[coord], j)
            + lgca.alpha * restc
            + lgca.gamma * np.einsum('i,ij', g_pressure[coord], j)
            #+ lgca.gamma * np.dot(permutations[:, :lgca.velocitychannels], ecm_weight[coord])
            #+ lgca.gamma * np.dot(permutations[:, :lgca.velocitychannels], subs_weight[coord])
        ).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    # reflection at ecm sites
    # relevant = (lgca.cell_density[lgca.nonborder] > 0) & (lgca.ecm[lgca.nonborder])
    # coords = [a[relevant] for a in lgca.nonborder]
    # for coord in zip(*coords):
    #     oldcellnode = lgca.nodes[coord]
    #     newcellnode = newnodes[coord]
    #     newcellnode[:lgca.velocitychannels // 2] = oldcellnode[lgca.velocitychannels // 2:lgca.velocitychannels]
    #     newcellnode[lgca.velocitychannels // 2:lgca.velocitychannels] = oldcellnode[:lgca.velocitychannels // 2]
    #     newnodes[coord] = newcellnode
    lgca.nodes = newnodes


def excitable_medium(lgca):
    """
    Model for an excitable medium based on Barkley's PDE model.
    :return:
    """
    n_x = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_y = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    rho_x = n_x / lgca.velocitychannels
    rho_y = n_y / lgca.restchannels
    p_xp = rho_x ** 2 * (1 + (rho_y + lgca.beta) / lgca.alpha)
    p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.beta) / lgca.alpha
    p_yp = rho_x
    p_ym = rho_y
    dn_y = (npr.random(n_y.shape) < p_yp).astype(np.int8)
    dn_y -= npr.random(n_y.shape) < p_ym
    for _ in range(lgca.N):
        dn_x = (npr.random(n_x.shape) < p_xp).astype(np.int8)
        dn_x -= npr.random(n_x.shape) < p_xm
        n_x += dn_x
        rho_x = n_x / lgca.velocitychannels
        p_xp = rho_x ** 2 * (1 + (rho_y + lgca.beta) / lgca.alpha)
        p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.beta) / lgca.alpha

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
        j_1 = npr.binomial(M1[coord], tanh_switch(rho, kappa=lgca.kappa, theta=lgca.theta))
        j_2 = npr.binomial(M2[coord], 1 - tanh_switch(rho, kappa=lgca.kappa, theta=lgca.theta))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2
        n_mxy -= npr.binomial(n_mxy * np.heaviside(n_mxy, 0), lgca.r_d)
        n_rxy -= npr.binomial(n_rxy * np.heaviside(n_rxy, 0), lgca.r_d)
        M = min([n_rxy, lgca.restchannels - n_rxy])
        n_rxy += npr.binomial(M * np.heaviside(M, 0), lgca.r_b)

        v_channels = [1] * n_mxy + [0] * (lgca.velocitychannels - n_mxy)
        v_channels = npr.permutation(v_channels)
        r_channels = np.zeros(lgca.restchannels)
        r_channels[:n_rxy] = 1
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node
