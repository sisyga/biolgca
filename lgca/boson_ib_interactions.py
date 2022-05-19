from random import choices, random, shuffle, randrange
import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm
from copy import deepcopy

try:
    from .interactions import tanh_switch
except ImportError:
    from interactions import tanh_switch


def trunc_gauss(lower, upper, mu, sigma=.1, size=1):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)


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
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.capacity
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
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.capacity
        cells = node.sum()

        for cell in cells:
            if random() < lgca.r_d:
                cells.remove(cell)

            r_b = lgca.props['r_b'][cell]
            if random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                cells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        channeldist = npr.multinomial(len(cells), [1. / lgca.K] * lgca.K).cumsum()
        shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)


def go_or_grow(lgca):
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.capacity
        cells = np.array(node.sum())
        # R1: cell death
        tobekilled = npr.random(size=density) < 1. - lgca.r_d
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
        rho = len(cells) / lgca.capacity  # update density after deaths for birth
        for cell in restcells:
            if random() < lgca.r_b * (1 - rho):
                lgca.maxlabel += 1
                restcells.append(lgca.maxlabel)
                lgca.props['kappa'].append(npr.normal(loc=lgca.props['kappa'][cell], scale=lgca.kappa_std))
                lgca.props['theta'].append(float(trunc_gauss(0, 1, mu=lgca.props['theta'][cell], sigma=lgca.theta_std)))

        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        for cell in velcells:
            node[randrange(lgca.velocitychannels)].append(cell)
        lgca.nodes[coord] = deepcopy(node)
