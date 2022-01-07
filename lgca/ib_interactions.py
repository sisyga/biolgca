from random import choices
import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm

from lgca.interactions import tanh_switch


def random_walk(lgca):
    """
    Particles are randomly redistributed among channels with the same probability for each channel.
    In combination with deterministic propagation it produces an unbiased random walk.
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        npr.shuffle(lgca.nodes[coord])



def trunc_gauss(lower, upper, mu, sigma=.1, size=1):
    """
    Sample from a truncated Gaussian distribution.
    :param lower: lower limit of truncation
    :param upper: upper limit of truncation
    :param mu: mean of the distribution
    :param sigma: standard deviation of the distribution
    :param size: desired sample size
    :returns: (array) of size size, samples from the described truncated Gaussian
    """
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)

def birth(lgca):
    """
    Simple birth process
    """

    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]

        # choose cells that proliferate
        r_bs = np.array([lgca.props['r_b'][i] for i in node])
        proliferating = npr.random(lgca.K) < r_bs

        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for label in node[proliferating]:
            ind = npr.choice(lgca.K)
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props['r_b'][label]
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        lgca.nodes[coord] = node
    random_walk(lgca)


def birthdeath(lgca):
    """
    Simple birth-death process with evolutionary dynamics towards a higher proliferation rate.
    Family membership of cells can be tracked by setting birthdeath.track_inheritance = True.
    """
    # death process
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()

    # birth
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        occ = lgca.occupied[coord]

        # choose cells that proliferate
        r_bs = np.array([lgca.props['r_b'][i] for i in node])
        proliferating = (npr.random(lgca.K) * occ) < r_bs
        n_p = proliferating.sum()
        if n_p == 0:
            continue
        targetchannels = npr.choice(lgca.K, size=n_p, replace=False)  # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for i, label in enumerate(node[proliferating]):
            ind = targetchannels[i]
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props['r_b'][label]
                if lgca.std > 0:
                    lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))
                else:
                    lgca.props['r_b'].append(r_b)
                if birthdeath.track_inheritance:
                    fam = lgca.props['family'][label]
                    lgca.props['family'].append(fam)
        lgca.nodes[coord] = node
    random_walk(lgca)

def birthdeath_discrete(lgca):
    """
    Simple birth-death process with evolutionary dynamics towards a higher proliferation rate
    """
    # determine which cells will die
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        occ = lgca.occupied[coord]

        # choose cells that proliferate

        r_bs = np.array([lgca.props['r_b'][i] for i in node])
        proliferating = (npr.random(lgca.K) * occ) < r_bs
        n_p = proliferating.sum()
        if n_p == 0:
            continue
        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        targetchannels = npr.choice(lgca.K, size=n_p, replace=False)

        for i, label in enumerate(node[proliferating]):
            ind = targetchannels[i]
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props['r_b'][label]
                if r_b < lgca.a_max:
                    lgca.props['r_b'].append(choices((r_b-lgca.drb, r_b+lgca.drb, r_b), weights=(lgca.pmut/2, lgca.pmut/2,
                                                                                                 1-lgca.pmut))[0])
                else:
                    lgca.props['r_b'].append(choices((r_b-lgca.drb, r_b), weights=(lgca.pmut/2, 1-lgca.pmut/2))[0])

        lgca.nodes[coord] = node

    random_walk(lgca)

def go_or_grow(lgca):
    """
    interactions of the go-or-grow model. formulation too complex for 1d, but to be generalized.
    """

    # death
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    lgca.nodes[dying] = 0

    # birth
    lgca.update_dynamic_fields()  # routinely update
    n_m = lgca.occupied[..., :lgca.velocitychannels].sum(-1)  # number of cells in rest channels for each node
    n_r = lgca.occupied[..., lgca.velocitychannels:].sum(-1)  # -"- velocity -"-
    relevant = (lgca.cell_density[lgca.nonborder] > 0)  # only nodes that are not empty
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):  # loop through all relevant nodes
        node = lgca.nodes[coord]
        vel = node[:lgca.velocitychannels]
        rest = node[lgca.velocitychannels:]
        n = lgca.cell_density[coord]

        rho = n / lgca.K

        # determine cells to switch to rest channels and cells that switch to moving state
        # kappas = np.array([lgca.props['kappa'][i] for i in node])
        # r_s = tanh_switch(rho, kappa=kappas, theta=lgca.theta)

        free_rest = lgca.restchannels - n_r[coord]
        free_vel = lgca.velocitychannels - n_m[coord]
        # choose a number of cells that try to switch. the cell number must fit to the number of free channels
        can_switch_to_rest = npr.permutation(vel[vel > 0])[:free_rest]
        can_switch_to_vel = npr.permutation(rest[rest > 0])[:free_vel]

        for cell in can_switch_to_rest:
            if npr.random() < tanh_switch(rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell]):
                # print 'switch to rest', cell
                rest[np.where(rest == 0)[0][0]] = cell
                vel[np.where(vel == cell)[0][0]] = 0

        for cell in can_switch_to_vel:
            if npr.random() < 1 - tanh_switch(rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell]):
                # print 'switch to vel', cell
                vel[np.where(vel == 0)[0][0]] = cell
                rest[np.where(rest == cell)[0][0]] = 0

        # cells in rest channels can proliferate
        can_proliferate = npr.permutation(rest[rest > 0])[:(rest == 0).sum()]
        for cell in can_proliferate:
            if npr.random() < lgca.r_b:
                lgca.maxlabel += 1
                rest[np.where(rest == 0)[0][0]] = lgca.maxlabel
                kappa = lgca.props['kappa'][cell]
                if lgca.kappa_std == 0:
                    lgca.props['kappa'].append(kappa)
                else:
                    lgca.props['kappa'].append(npr.normal(loc=kappa, scale=lgca.kappa_std))
                theta = lgca.props['theta'][cell]
                if lgca.theta_std == 0:
                    lgca.props['theta'].append(theta)
                else:
                    lgca.props['theta'].append(npr.normal(loc=theta, scale=lgca.theta_std))

        v_channels = npr.permutation(vel)
        r_channels = npr.permutation(rest)
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node

def go_and_grow_mutations(lgca):
    """
    Simple birth-death process with tracked family membership of cells. New families develop by mutations.
    If lgca.effect == 'passenger_mutation': no change in proliferation rate, but mutations found new families
    If lgca.effect == 'driver_mutation': evolutionary dynamics towards a higher proliferation rate
    """
    # dying process
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()

    # birth process
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        # choose cells that proliferate
        if lgca.effect == 'driver_mutation':
            # use rb of family
            r_bs = np.array([lgca.family_props['r_b'][lgca.props['family'][i]] for i in node])
        else:
            # rb is constant
            r_bs = lgca.r_b * node.astype(bool)
        proliferating = npr.random(lgca.K) < r_bs
        n_p = proliferating.sum()
        if n_p == 0:
            continue
        # pick a random channel for each proliferating cell
        targetchannels = npr.choice(lgca.K, size=n_p, replace=False)
        for i, label in enumerate(node[proliferating]):
            # If the picked channel is empty, place the daughter cell there
            ind = targetchannels[i]
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel  # new cell
                # mother cell: label
                # family: lgca.props['family'][label]
                fam = lgca.props['family'][label]

                mutation = npr.random() < lgca.r_m
                if mutation:
                    # add new family
                    lgca.maxfamily += 1
                    lgca.props['family'].append(int(lgca.maxfamily))

                    if lgca.effect == 'driver_mutation':
                        # give the new family an rb: rb of mother cell's family * fitness increase
                        lgca.family_props['r_b'].append(lgca.family_props['r_b'][fam] * lgca.fitness_increase)
                    # register ancestor of the new family
                    lgca.family_props['ancestor'].append(fam)
                    # record new family as child of the old one
                    lgca.family_props['descendants'][fam].append(lgca.maxfamily)
                    # create empty children list for the new family
                    lgca.family_props['descendants'].append([])
                else:
                    # record family of new cell = family of mother cell
                    lgca.props['family'].append(fam)
        lgca.nodes[coord] = node

    random_walk(lgca)
