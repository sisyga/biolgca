from bisect import bisect_left
from math import log, exp
from random import random
import numpy as np
from copy import copy
import numpy.random as npr
from scipy.special import binom as binom_coeff
from mypackage.ECM import nb_ECM, nb_coord
from data import DataClass


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

def ent_prod(x):
    return x * np.log(x, where=x > 0, out=np.zeros_like(x, dtype=float))


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

# def contact_guidance(lgca):
#     """
#     Rearrangement step for contact guidance interaction. Cells are guided by an external axis
#     :return:
#     """
#     beta_a = 0.1
#     scalar = lgca.scalar_field
#
#     newnodes = lgca.nodes.copy()
#     relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
#                (lgca.cell_density[lgca.nonborder] < lgca.K)
#     coords = [a[relevant] for a in lgca.nonborder]
#
#     g = np.asarray(lgca.gradient(lgca.cell_density))
#
#     for coord in zip(*coords):
#
#         n = lgca.cell_density[coord]
#         for i in npr.random(n):
#             if i <0.1:
#                 n = n+1
#         sni = lgca.guiding_tensor[coord]
#         permutations = lgca.permutations[n]
#         j = lgca.j[n]
#         si = lgca.si[n]
#
#         # weights from aggregation
#         weights_aggr = np.exp(beta_a * np.einsum('i,ij', g[coord], j)).cumsum()
#
#         # weights from contact guidance
#         weights = np.exp(lgca.beta * np.einsum('ijk,jk', si, sni)).cumsum()
#
#         # adjustment for underlying scalarfield
#         # get scalar from respective coords to change weights accordingly
#
#         scalar_adj = np.zeros(len(permutations))
#
#         # ugly --> to be rewritten; but works
#         for counter, perm in enumerate(permutations):
#
#             y1 = scalar[((coord[1] + perm[1], coord[0]))]
#             y2 = scalar[((coord[1] - perm[3], coord[0]))]
#             x1 = scalar[((coord[1], coord[0] + perm[0]))]
#             x2 = scalar[((coord[1], coord[0] - perm[2]))]
#
#             if scalar[((coord[1], coord[0]))] == 0:
#                 if y1[0][0] + y2[0][0] + x1[0][0] + x2[0][0] == lgca.cell_density[coord]:
#                     scalar_adj[counter] = 1
#                 else:
#                     scalar_adj[counter] = 0
#
#             if scalar[((coord[1], coord[0]))] != 0:
#                 if y1[0][0] + y2[0][0] + x1[0][0] + x2[0][0] == 4:
#                     scalar_adj[counter] = 1
#                 else:
#                     scalar_adj[counter] = 0
#
#         # adjusted weights
#         diff = np.insert(np.diff(weights), 0 , weights[0])
#
#         diff_aggr = np.insert(np.diff(weights_aggr), 0 , weights_aggr[0])
#
#         weights_adj_1 = np.multiply(diff, diff_aggr)
#         weights_adj = (np.multiply(weights_adj_1, scalar_adj)).cumsum()
#         rand_ = random() * weights_adj[-1]
#         ind = bisect_left(weights_adj, rand_)
#
#         newnodes[coord] = permutations[ind]
#
#     lgca.nodes = newnodes

def ind_coord(ind_array, coord_array, ind):
    list_all = []
    for i in ind:
        for counter, j in enumerate(ind_array):
            if j == i:
                index = counter
                break
        list_all.append(coord_array[index])

    return list_all

def contact_guidance(lgca, ecm, data):
    """
    Rearrangement step for contact guidance interaction. Cells are guided by an external axis
    :return:
    """

    # connection, path_index = ecm.simple_percolation()
    # paths, length, all_paths = ecm.strands_fixed_border()
    # scalar_field_values = ecm.scalar_field[lgca.nonborder].ravel()
    # perco = (scalar_field_values >= 0.9).astype(int)
    # ecm.perco_ratio.append(sum(perco)/900)
    # #+
    # # print(connection)
    # if connection:
    #     ecm.p_inf.append(len(ecm.paths[path_index[0]]))
    #     # ecm.perco_ratio.append((sum(perco)/900))
    # else:
    #     ecm.p_inf.append(0)
    # # #
    # # # ecm.list_paths_number.append(paths)
    # # # ecm.list_paths_length.append(length)
    # ecm.paths = all_paths

    d = ecm.d
    d_neigh = ecm.d_neigh

    newscalar = copy(ecm.scalar_field)
    newscalar2 = copy(ecm.scalar_field)
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    g = np.asarray(lgca.gradient(lgca.cell_density))
    lgca.guiding_tensor = ecm.tensor_field

    Abstand_d = []
    # coords_all = list(zip(*coords))
    #
    # list_coords = ind_coord(ecm.coord_pairs, ecm.coord_pairs_hex, coords_all)
    # array_squared = []
    # print(list_coords)
    #
    # for i in list_coords:
    #     array = np.array([i[0] - 15, i[1] - 13])
    #     array_squared.append(np.dot(array, array))
    #
    # print(np.mean(array_squared))
    #
    # data.Abstand[data.x_counter][data.y_counter][ecm.t] = (np.mean(array_squared))
    for coord in zip(*coords):

        neigh = nb_coord(coord, lgca)
        nb_ecm = nb_ECM(ecm.scalar_field, coord, ecm.restchannels)


        # beta_prolif = np.exp(np.sum(nb_ecm/1.5))*lgca.r_b
        H = 5
        Km = 3**H
        v0 = 1
        v1 = 1
        n = lgca.cell_density[coord]

        beta_agg_MM = lgca.beta_agg*v0 * np.sum(nb_ecm)**H/(Km + np.sum(nb_ecm)**H)
        prolif_MM =  lgca.r_b*v1 * np.sum(nb_ecm)**H/(Km + np.sum(nb_ecm)**H)
        g_rate = log_growth(lgca.K, lgca.r_b,  n)

        p_apop = 0.0

        # if ecm.t >= 25:
        #     p_apop = 1 - 2 * np.sum(nb_ecm) ** H / (Km + np.sum(nb_ecm) ** H)


        # for i in npr.random(n):
        #    if n < lgca.K:
        if npr.random() < (g_rate):
            if npr.random() < p_apop:
                n -= 1
            if npr.random() >= p_apop:
                n += 1

        sni = lgca.guiding_tensor[coord]
        permutations = lgca.permutations[n]
        si = lgca.si[n]
        restc = permutations[:, lgca.velocitychannels:].sum(-1)
        # print(coord, permutations)
        j = lgca.j[n]
        nbs = np.asarray(list(reversed(nb_ecm)))
        nbs[-ecm.restchannels:] = ecm.restchannels*[0]
        hind = np.multiply((-nbs/ecm.fc), permutations)

        beta_guid = lgca.beta * ecm.vector_field[coord][2]
        weights_adj_tot = np.exp(beta_guid * np.einsum('ijk,jk', si, sni)
                                 + np.einsum('ij -> i', hind)
                                 + lgca.beta_rest * restc
                                 + beta_agg_MM * np.einsum('i,ij', g[coord], j)).cumsum()

        ind = bisect_left(weights_adj_tot, random() * weights_adj_tot[-1])
        if ecm.d != 0 or ecm.d_neigh != 0:
            for i in neigh:
                if i == coord:
                    delta = - (d * n * newscalar[i])/ lgca.K
                    newscalar2[i] += delta
                    if newscalar2[i] <= 0:
                        newscalar2[i] = 0
                else:
                    if newscalar[i] != 1:
                        delta = (d_neigh * n* newscalar[i])/ lgca.K
                        newscalar2[i] += delta
                        if newscalar2[i] >= 1:
                            newscalar2[i] = 1

        newnodes[coord] = permutations[ind]

        real_coord = ind_coord(lgca.coord_pairs, lgca.coord_pairs_hex, [coord])
        data.MSD.append(np.array([real_coord[0][0], real_coord[0][1]]))

    # data.tumor_vol[ecm.t] += np.sum(lgca.cell_density[lgca.nonborder])
    ecm.t = ecm.t + 1
    ecm.scalar_field = newscalar2
    ecm.scalar_field_t[ecm.t] = newscalar2
    ecm.periodic_rb()
    ecm.tensor_update(ecm.t)

    lgca.nodes = newnodes
    # ecm.t = ecm.t + 1

def contact_guidance_old(lgca):
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
        rand_ = random() * weights[-1]
        ind = bisect_left(weights, rand_)
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
    g1 = lgca.calc_flux(lgca.nodes)
    g = lgca.nb_sum(g1)
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
        birth = npr.random(lgca.nodes[lgca.spheroid].shape) < lgca.r_b
        ds = (1 - lgca.nodes[lgca.spheroid]) * birth
        lgca.nodes[lgca.spheroid, :] = np.add(lgca.nodes[lgca.spheroid, :], ds, casting='unsafe')
        lgca.update_dynamic_fields()
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    nbs = lgca.nb_sum(lgca.cell_density)  # + lgca.cell_density
    nbs *= np.clip(1 - nbs / lgca.n_crit, a_min=0, a_max=None) / lgca.n_crit * 2
    g_adh = lgca.gradient(nbs)
    pressure = np.clip(lgca.cell_density - lgca.rho_0, a_min=0., a_max=None) / (lgca.K - lgca.rho_0)
    g_pressure = -lgca.gradient(pressure)

    resting = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    resting = lgca.nb_sum(resting) / lgca.velocitychannels / lgca.rho_0
    g = lgca.calc_flux(lgca.nodes)
    g = lgca.nb_sum(g)

    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        permutations = lgca.permutations[n]
        restc = permutations[:, lgca.velocitychannels:].sum(-1)
        j = lgca.j[n]
        j_nb = g[coord]
        weights = np.exp(
             lgca.beta * (j_nb[0] * j[0] + j_nb[1] * j[1]) / lgca.velocitychannels / 2
            + lgca.beta * resting[coord] * restc  #* np.clip(1 - restc / lgca.rho_0 / 2, a_min=0, a_max=None) * 2
            + lgca.beta * np.einsum('i,ij', g_adh[coord], j)
            # + lgca.alpha * np.einsum('i,ij', g_subs[coord], j)
            + restc * lgca.ecm[coord]
            + lgca.gamma * np.einsum('i,ij', g_pressure[coord], j)
        ).cumsum()
        ind = bisect_left(weights, random() * weights[-1])
        newnodes[coord] = permutations[ind]

    lgca.nodes = newnodes
    lgca.ecm -= lgca.alpha * lgca.ecm * lgca.cell_density / lgca.K


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
        n_mxy -= npr.binomial(n_mxy, lgca.r_d)
        n_rxy -= npr.binomial(n_rxy, lgca.r_d)
        M = min([n_rxy, lgca.restchannels - n_rxy])
        n_rxy += npr.binomial(M, lgca.r_b)

        v_channels = [1] * n_mxy + [0] * (lgca.velocitychannels - n_mxy)
        v_channels = npr.permutation(v_channels)
        r_channels = np.zeros(lgca.restchannels)
        r_channels[:n_rxy] = 1
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


def p_binom(k, n, p):
    pb = binom_coeff(n, k) * p**k * (1 - p)**(n-k)
    pb[n<k] = 0.
    return pb

def s_binom(n, p0, kmax):
    n = n[..., None]
    p0 = p0[..., None]
    k = np.arange(kmax + 1)
    p = p_binom(k, n, p0)
    return -ent_prod(p).sum(-1)




def simplelinear(scalar):
    return 1 - scalar

def exphinderanc(scalar, crit):
    return (np.exp(-scalar/crit))

def log_func(x, c=1, ex1=1 , ex2=1, K=1):
    return c * x**(ex1)/K * (1 - x**(ex2)/K)

def log_growth(K, g, n_cells):
    if n_cells <= K:
        return float(g) * n_cells / K * (1 - n_cells / K)
    else:
        return 0

def leup_test(lgca):
    """
    Go-or-grow with least-environmental uncertainty principle. cells try to minimize their entropy with the environment,
    by changing their state between moving and resting. resting cells can proliferate. all cells die at a constant rate.
    :return:
    """
    if lgca.r_b > 0 or lgca.r_d > 0:
        n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
        n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
        birth = np.zeros_like(lgca.nodes)
        birth[..., lgca.velocitychannels:] = npr.random(n_r.shape + (lgca.restchannels, )) < lgca.r_b * n_r[..., None] / lgca.restchannels
        death = npr.random(birth.shape) < lgca.r_d * (n_m[..., None] / lgca.velocitychannels + n_r[..., None] / lgca.restchannels) / 2
        ds = (1 - lgca.nodes) * birth - lgca.nodes * death
        np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
        lgca.update_dynamic_fields()

    # if lgca.r_b > 0: # or lgca.r_d > 0:
    #     n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    #     n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)
    #     birth = npr.random(lgca.nodes.shape) < lgca.r_b * n_r[..., None] / lgca.restchannels
    #     death = npr.random(birth.shape) < lgca.r_d * (n_r[..., None] / lgca.restchannels + n_m[..., None] / lgca.velocitychannels)
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

    p0 = np.divide(n_r, n, where=n>0, out=np.zeros_like(n, dtype=float))
    s = s_binom(n, p0, lgca.velocitychannels)
    p10 = np.divide(n_r+1, n, where=n>0, out=np.zeros_like(n, dtype=float))
    s10 = s_binom(n, p10, lgca.velocitychannels)
    ds1 = s10 - s

    p01 = np.divide(n_r-1, n, where=n>0, out=np.zeros_like(n, dtype=float))
    s01 = s_binom(n, p01, lgca.velocitychannels)
    ds2 = s01 - s
    #
    #
    # s = ent_prod(n_r) + ent_prod(n_m)
    # ds1 = np.divide(s - ent_prod(n_r + 1) - ent_prod(n_m - 1), n, where=n > 0)
    p1 = 1 / (1 + np.exp(lgca.beta * ds1))
    p1[M1 == 0] = 0.
    # ds2 = np.divide(s - ent_prod(n_r - 1) - ent_prod(n_m + 1), n, where=n > 0)
    p2 = 1 / (1 + np.exp(lgca.beta * ds2))
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
        # p1 = 1 / (1 + exp(lgca.beta * ds1))
        # # switch to velocity channel
        # ds2 = (s - ent_prod(n_rxy-1) - ent_prod(n_mxy+1)) / n
        # p2 = 1 / (1 + exp(lgca.beta * ds2))
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



