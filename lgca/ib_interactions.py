from random import choices
import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm

try:
    from .interactions import tanh_switch
except ImportError:
    from interactions import tanh_switch


def randomwalk(lgca):
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        npr.shuffle(lgca.nodes[coord])


def trunc_gauss(lower, upper, mu, sigma=.1, size=1):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)

def birth(lgca):
    """
    Simple birth process
    :return:
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
                # lgca.props['r_b'].append(np.clip(npr.normal(loc=r_b, scale=lgca.std), 0, 1))
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        lgca.nodes[coord] = node
    randomwalk(lgca)


def birthdeath(lgca):
    """
    Simple birth-death process with evolutionary dynamics towards a higher proliferation rate
    :return:
    """
    # death process
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    # lgca.update_dynamic_fields()
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
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        lgca.nodes[coord] = node

    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()
    randomwalk(lgca)

def birthdeath_discrete(lgca):
    """
    Simple birth-death process with evolutionary dynamics towards a higher proliferation rate
    :return:
    """
    # determine which cells will die
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    # lgca.update_dynamic_fields()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        occ = lgca.occupied[coord]

        # choose cells that proliferate

        r_bs = np.array([lgca.props['r_b'][i] for i in node])
        proliferating = npr.random(lgca.K) < r_bs
        n_p = proliferating.sum()
        if n_p == 0:
            continue
        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        targetchannels = npr.choice(lgca.K, n_p, replace=False)

        for i, label in enumerate(node[proliferating]):
            ind = targetchannels[i]
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props['r_b'][label]
                if r_b < lgca.a_max:
                    lgca.props['r_b'].append(choices((r_b-lgca.drb, r_b+lgca.drb, r_b), weights=(lgca.pmut, lgca.pmut,
                                                                                                 1-2*lgca.pmut))[0])
                else:
                    lgca.props['r_b'].append(choices((r_b-lgca.drb, r_b), weights=(lgca.pmut, 1-lgca.pmut))[0])

        lgca.nodes[coord] = node

    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()
    randomwalk(lgca)

def go_or_grow(lgca):
    """
    interactions of the go-or-grow model. formulation too complex for 1d, but to be generalized.
    :return:
    """

    # death
    dying = (npr.random(size=lgca.nodes.shape) < lgca.r_d) & lgca.occupied
    lgca.nodes[dying] = 0

    # birth
    lgca.update_dynamic_fields()  # routinely update
    n_m = lgca.occupied[..., :lgca.velocitychannels].sum(-1)  # number of cells in rest channels for each node
    #   .sum(-1) ? specifies dimension, -1 -> 1 dimension ? SIMON: .sum(-1) sums over the last axis (= sum over channels!)
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
                lgca.props['kappa'].append(npr.normal(loc=kappa, scale=0.2))
                theta = lgca.props['theta'][cell]
                lgca.props['theta'].append(npr.normal(loc=theta, scale=0.05))

        v_channels = npr.permutation(vel)
        r_channels = npr.permutation(rest)
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


if __name__ =='__main__':
    from scipy.special import erf
    from math import sqrt
    def gaussian(x):
        y = np.exp(-0.5 * x ** 2) / sqrt(2 * np.pi)
        return y
    def cdf_gaussian(x):
        y = 0.5 * (1 + erf(x / sqrt(2)))
        return y
    def trunc_gaussian(x, mu, sigma, a=0, b=1):
        xi = (x - mu) / sigma
        beta = (b - mu) / sigma
        alpha = (a - mu) / sigma
        y = gaussian(xi) / sigma
        y /= cdf_gaussian(beta) - cdf_gaussian(alpha)
        return y

    from matplotlib import pyplot as plt
    r_b = 0.1
    lower = 0
    mu = 0.1
    upper = 1
    sigma = 0.1
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    pdf = truncnorm(a, b, loc=mu, scale=sigma).pdf
    randn = trunc_gauss(0, 1., r_b, sigma=0.1, size=1000)
    # trunc_gauss(0, 1., r_b, sigma=0.1)
    plt.hist(randn, range=(0, 1), bins=20, density=True)
    x = np.linspace(0, 1)
    tgauss = trunc_gaussian(x, mu, sigma)
    plt.plot(x, tgauss)
    plt.plot(x, pdf(x))
    plt.show()