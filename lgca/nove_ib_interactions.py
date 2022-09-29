from random import choices, random, shuffle, randrange
import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm
from copy import deepcopy
from numba import jit
import numba
from lgca.interactions import tanh_switch

def trunc_gauss(lower, upper, mu, sigma=.1, size=1):
    """
    Draw random variables from a truncated Gaussian distribution. The distribution is normalized between the 'lower'
    and 'upper' bound, hast he mean value 'mu' and the standard deviation 'sigma'.
    :param lower: lower bound
    :param upper: upper bound
    :param mu: mean value
    :param sigma: standard deviation
    :param size: number of samples
    :return:
    """
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    vals = truncnorm(a, b, loc=mu, scale=sigma).rvs(size)
    if size != 1:
        return vals
    else:
        return float(vals)


def randomwalk(lgca):
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        cells = node.sum()

        channeldist = npr.multinomial(len(cells), [1. / lgca.K] * lgca.K).cumsum()
        shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)


def birth(lgca):
    """
    Apply a birth step. Each cell proliferates following a logistic growth law using its individual birth rate r_b and
    a capacity 'capacity', that is constant for all cells. Daughter cells receive an individual proliferation rate that
    is drawn from a truncated Gaussian distribution between 0 and a_max, whose mean is equal to the mother cell's r_b,
    with standard deviation 'std'.
    :param lgca:
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = node.sum()

        for cell in cells:
            r_b = lgca.props['r_b'][cell]
            if random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                cells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        channeldist = npr.multinomial(len(cells), [1. / lgca.K] * lgca.K).cumsum()
        shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)


def birthdeath(lgca):
    """
    Apply a birth-death step. Each cell proliferates following a logistic growth law using its individual birth rate r_b and
    a capacity 'capacity', that is constant for all cells. All cells die with a constant probability 'r_d'.
    Daughter cells receive an individual proliferation rate that is drawn from a truncated Gaussian distribution between
    0 and a_max, whose mean is equal to the mother cell's r_b, with standard deviation 'std'.
    :param lgca:
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = node.sum()
        newcells = cells.copy()
        for cell in cells:
            if random() < lgca.interaction_params['r_d']:
                newcells.remove(cell)

            r_b = lgca.props['r_b'][cell]
            if random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.interaction_params['a_max'], r_b,
                                                           sigma=lgca.interaction_params['std'])))

        channeldist = npr.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        shuffle(newcells)
        newnode = [newcells[:channeldist[0]]] + [newcells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)

def birthdeath_numba(lgca):
    """
    Apply a birth-death step. Each cell proliferates following a logistic growth law using its individual birth rate r_b and
    a capacity 'capacity', that is constant for all cells. All cells die with a constant probability 'r_d'.
    Daughter cells receive an individual proliferation rate that is drawn from a truncated Gaussian distribution between
    0 and a_max, whose mean is equal to the mother cell's r_b, with standard deviation 'std'.
    :param lgca:
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        cells = node.sum()

        lgca.nodes[coord], lgca.maxlabel, newprops = onenodebirthdeath(cells, density, lgca.props['r_b'], lgca.maxlabel,
                                                             lgca.channel_weights, lgca.interaction_params['capacity'],
                                                             lgca.interaction_params['r_d'])
        newalphas = [float(trunc_gauss(0, lgca.interaction_params['a_max'], lgca.props['r_b'][cell],
                                                           sigma=lgca.interaction_params['std'])) for cell in newprops]

        lgca.props['r_b'].extend(newalphas)

@jit(nopython=True)
def onenodebirthdeath(cells, density, props, maxlabel, channel_weights, capacity, r_d):
    rho = density / capacity
    newcells = cells.copy()
    newprops = []
    for cell in cells:
        if random() < r_d:
            newcells.remove(cell)

        r_b = props[cell]
        if random() < r_b * (1 - rho):
            maxlabel += 1
            newcells.append(maxlabel)
            newalpha = cell # trunc_gauss(0, a_max, r_b, sigma=std)
            newprops.append(newalpha)

    channeldist = npr.multinomial(len(newcells), channel_weights).cumsum()
    npr.shuffle(np.array(newcells))
    newnode = [newcells[:channeldist[0]]] + [newcells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]
    return newnode, maxlabel, newprops

def go_or_grow(lgca):
    """
    Apply the evolutionary "go-or-grow" interaction. Cells switch from a migratory to a resting phenotype and vice versa
    depending on their individual properties and the local cell density. Resting cells proliferate with a constant
    proliferation rate. Each cell dies with a constant rate. Daughter cells inherit their switch properties from the
    mother cells with some small variations given by a (truncated) Gaussian distribution.
    :param lgca:
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = np.array(node.sum())
        # R1: cell death
        tobekilled = npr.random(size=density) < 1. - lgca.interaction_params['r_d']
        cells = cells[tobekilled]
        if cells.size == 0:
            lgca.nodes[coord] = [[] for _ in range(lgca.K)]
            continue
        # R2: switch (using old density, as switching happens faster than death)
        velcells = []
        restcells = []
        for cell in cells:
            if random() < tanh_switch(rho=rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell]):
                restcells.append(cell)
            else:
                velcells.append(cell)

        # R3: birth
        rho = len(cells) / lgca.interaction_params['capacity']  # update density after deaths for birth
        for cell in restcells:
            if random() < lgca.interaction_params['r_b'] * (1 - rho):
                lgca.maxlabel += 1
                restcells.append(lgca.maxlabel)
                lgca.props['kappa'].append(npr.normal(loc=lgca.props['kappa'][cell], scale=lgca.interaction_params['kappa_std']))
                lgca.props['theta'].append(float(trunc_gauss(0, 1, mu=lgca.props['theta'][cell], sigma=lgca.interaction_params['theta_std'])))

        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        for cell in velcells:
            node[randrange(lgca.velocitychannels)].append(cell)
        lgca.nodes[coord] = deepcopy(node)
