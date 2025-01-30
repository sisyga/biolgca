# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.

"""
Interaction functions and helper functions for identity-based LGCA without volume exclusion.
"""

# from random import random, shuffle, randrange
import numpy as np
from scipy.stats import truncnorm, truncexpon, expon
from copy import deepcopy
from numba import jit
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

        channeldist = lgca.rng.multinomial(len(cells), [1. / lgca.K] * lgca.K).cumsum()
        lgca.rng.shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)


def evo_steric(lgca):
    """
    Apply a birth-death step, then cells move under steric interactions.
    Each cell proliferates with its individual birth rate r_b following logistic growth until
    a capacity 'capacity' is reached, that is constant for all cells.
    All cells die with a constant probability 'r_d'.
    During proliferation there can be a mutation on either mother or daughter cell.
    Mutations can be beneficial (driver mutations) or deleterious to neutral (passenger mutations).
    These mutations manifest in a changed proliferation rate.
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
        velchannelweights = -lgca.interaction_params['alpha'] * lgca.channel_weight(lgca.cell_density)
        channelweights = np.append(velchannelweights, np.full(lgca.cell_density.shape,
                                                              lgca.interaction_params['gamma'])[..., None], axis=-1)
        channelprobs = np.exp(channelweights)
        channelprobs /= np.sum(channelprobs, axis=-1)[..., None]
        for cell in cells:
            if lgca.rng.random() < lgca.interaction_params['r_d']:
                newcells.remove(cell)

            # r_b = lgca.props['r_b'][cell]
            fam = lgca.props['family'][cell]
            r_b = lgca.family_props['r_b'][fam]
            # mother cell: cell
            # family: lgca.props['family'][cell]

            if lgca.rng.random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                if lgca.rng.random() < lgca.interaction_params['r_m']:
                    # driver mutation mother cell
                    lgca.add_family(fam)
                    # record family of new cell = new family
                    lgca.props['family'].append(int(lgca.maxfamily))
                    lgca.family_props['r_b'].append(lgca.family_props['r_b'][fam] * \
                                                    lgca.interaction_params['fitness_increase'])
                else:
                    # record family of new cell = family of mother cell
                    lgca.props['family'].append(fam)

        channelprob = channelprobs[coord]
        channeldist = lgca.rng.multinomial(len(newcells), channelprob).cumsum()
        lgca.rng.shuffle(newcells)
        newnode = [newcells[:channeldist[0]]] + [newcells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)


def birth(lgca):
    """
    Apply a birth step. Each cell proliferates following a logistic growth law using its individual birth rate r_b and
    a capacity 'capacity', that is constant for all cells.
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
            r_b = lgca.props['r_b'][cell]
            if lgca.rng.random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.interaction_params['a_max'], r_b,
                                                           sigma=lgca.interaction_params['std'])))

        # channeldist = lgca.rng.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        channeldist = lgca.rng.multinomial(len(newcells), lgca.channel_weights).cumsum()
        lgca.rng.shuffle(newcells)
        newnode = [newcells[:channeldist[0]]] + [newcells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

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
            if lgca.rng.random() < lgca.interaction_params['r_d']:
                newcells.remove(cell)

            r_b = lgca.props['r_b'][cell]
            if lgca.rng.random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.interaction_params['a_max'], r_b,
                                                           sigma=lgca.interaction_params['std'])))

        # channeldist = lgca.rng.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        channeldist = lgca.rng.multinomial(len(newcells), lgca.channel_weights).cumsum()
        lgca.rng.shuffle(newcells)
        newnode = [newcells[:channeldist[0]]] + [newcells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)

def birthdeath_cancerdfe(lgca):
    """
    Apply a birth-death step. Each cell proliferates following a logistic growth law using its individual birth rate r_b and
    a capacity 'capacity', that is constant for all cells. All cells die with a constant probability 'r_d'.
    Daughter cells receive an individual proliferation rate that is the mother cell's r_b, with a deviation caused by a
    mutation. The mutation can either be a driver mutation, which increases the proliferation rate, or a passenger mutation,
    which slightly decreases the proliferation rate.
    Both mutations are exponentially distributed with mean s_d and s_p, respectively.
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
            if lgca.rng.random() < lgca.interaction_params['r_d']:
                newcells.remove(cell)

            r_b = lgca.props['r_b'][cell]
            if lgca.rng.random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                passenger = 0.
                driver = 0.
                if lgca.rng.random() < lgca.interaction_params['p_p']:
                    passenger = float(expon.rvs(scale=lgca.interaction_params['s_p']))
                    # lgca.props['r_b'].append(max(0., r_b-float(expon.rvs(scale=lgca.interaction_params['s_p']))))

                if lgca.rng.random() < lgca.interaction_params['p_d']:
                    driver = float(expon.rvs(scale=lgca.interaction_params['s_d']))
                    # lgca.props['r_b'].append(r_b+float(truncexpon.rvs(lgca.interaction_params['a_max']-r_b,
                    #                                                   scale=lgca.interaction_params['s_d'])))

                lgca.props['r_b'].append(min(r_b - passenger + driver, lgca.interaction_params['a_max']))


        # channeldist = lgca.rng.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        channeldist = lgca.rng.multinomial(len(newcells), lgca.channel_weights).cumsum()
        lgca.rng.shuffle(newcells)
        newnode = [newcells[:channeldist[0]]] + [newcells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)


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
        node = lgca.nodes[coord]
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = np.array(node.sum())
        # R1: cell death
        notkilled = lgca.rng.random(size=density) < 1. - lgca.interaction_params['r_d']
        cells = cells[notkilled]
        if len(cells) == 0:
            lgca.nodes[coord] = [[] for _ in range(lgca.K)]
            continue
        # R2: switch (using old density, as switching happens faster than death)
        kappas = lgca.props['kappa'][cells]
        thetas = lgca.props['theta'][cells]
        switch = lgca.rng.random(len(cells)) < tanh_switch(rho=rho, kappa=kappas, theta=thetas)
        restcells, velcells = list(cells[switch]), list(cells[~switch])
        # R3: birth
        rho = len(cells) / lgca.interaction_params['capacity']  # update density after deaths for birth
        n_prolif = lgca.rng.binomial(len(restcells), max(lgca.interaction_params['r_b'] * (1 - rho), 0))
        if n_prolif > 0:
            proliferating = lgca.rng.choice(restcells, size=n_prolif, replace=False, shuffle=False)
            lgca.maxlabel += n_prolif
            new_cells = np.arange(lgca.maxlabel - n_prolif + 1, lgca.maxlabel + 1)
            lgca.props['kappa'] = np.concatenate((lgca.props['kappa'],
                                                  lgca.rng.normal(loc=lgca.props['kappa'][proliferating],
                                                             scale=lgca.interaction_params['kappa_std'])))
            # lgca.props['theta'] = np.concatenate((lgca.props['theta'],
            #                                       trunc_gauss(0, 1, mu=lgca.props['theta'][proliferating],
            #                                                   sigma=lgca.interaction_params['theta_std'])))
            lgca.props['theta'] = np.concatenate((lgca.props['theta'],
                                                  lgca.rng.normal(loc=lgca.props['theta'][proliferating],
                                                            scale=lgca.interaction_params['theta_std'])))
            restcells.extend(list(new_cells))

        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        for cell in velcells:
            node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

        lgca.nodes[coord] = node

def go_or_grow_kappa(lgca):
    """
    Apply the evolutionary "go-or-grow" interaction. Cells switch from a migratory to a resting phenotype and vice versa
    depending on their individual properties and the local cell density. Resting cells proliferate with a constant
    proliferation rate. Each cell dies with a constant rate. Daughter cells inherit their switch properties from the
    mother cells with some small variations given by a (truncated) Gaussian distribution.

    :param lgca: The lattice-gas cellular automata object.
    :return: None. The function modifies the lgca object in-place.
    """
    # Identify the relevant cells (those with non-zero density)
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # Calculate the average density in the neighborhood
    nbdensity = lgca.nb_sum(lgca.cell_density, addCenter=True) / ((lgca.velocitychannels+1) * lgca.interaction_params['capacity']) # average density in neighborhood
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        density = lgca.cell_density[coord]
        nbdens = nbdensity[coord]
        # rho = density / lgca.interaction_params['capacity']
        # Get the list of cells at the current node
        cells = np.array(node.sum())
        # R1: cell death
        # Determine which cells survive
        notkilled = lgca.rng.random(size=density) < 1. - lgca.interaction_params['r_d']
        cells = cells[notkilled]
        # If all cells at the current node died, continue to the next node
        if len(cells) == 0:
            lgca.nodes[coord] = [[] for _ in range(lgca.K)]
            continue

        # Determine which cells switch phenotype based on their individual properties and the local cell density
        kappas = lgca.props['kappa'][cells]
        switch = lgca.rng.random(len(cells)) < tanh_switch(rho=nbdens, kappa=kappas, theta=lgca.interaction_params['theta'])
        restcells, velcells = list(cells[switch]), list(cells[~switch])
        # Update the density after deaths for birth
        rho = len(cells) / lgca.interaction_params['capacity']  # update density after deaths for birth
        # Determine the number of proliferating cells
        n_prolif = lgca.rng.binomial(len(restcells), max(lgca.interaction_params['r_b'] * (1 - rho), 0))
        # If there are proliferating cells, generate new cells
        if n_prolif > 0:
            proliferating = lgca.rng.choice(restcells, n_prolif, replace=False)
            lgca.maxlabel += n_prolif
            new_cells = np.arange(lgca.maxlabel - n_prolif + 1, lgca.maxlabel + 1)
            # Update the kappa properties of the new cells
            lgca.props['kappa'] = np.concatenate((lgca.props['kappa'],
                                                  lgca.rng.normal(loc=lgca.props['kappa'][proliferating],
                                                             scale=lgca.interaction_params['kappa_std'])))
            # Add the new cells to the list of resting cells
            restcells.extend(list(new_cells))

        # Initialize the node with empty channels and add the resting cells
        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        # Assign the migrating cells to random velocity channels
        for cell in velcells:
            node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

        # Update the node in the lgca object
        lgca.nodes[coord] = deepcopy(node)


@jit(nopython=True)
def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


@jit(nopython=True)
def nb_sum(qty, addCenter):
    sum = np.zeros(qty.shape)
    sum[:-1, ...] += qty[1:, ...]
    sum[1:, ...] += qty[:-1, ...]

    if addCenter:
        sum += qty
    return sum


def go_or_grow_kappa_chemo(lgca):
    """
    Apply the evolutionary "go-or-grow" interaction. Cells switch from a migratory to a resting phenotype and vice versa
    depending on their individual properties and the local cell density. Resting cells proliferate with a constant
    proliferation rate. Migrating cells move along the cell density gradient.
    Each cell dies with a constant rate. Daughter cells inherit their switch properties from the
    mother cells with some small variations given by a (truncated) Gaussian distribution.
    :param lgca:
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    g = lgca.gradient(lgca.cell_density / lgca.interaction_params['capacity'])  # density gradient for each lattice site
    nbdensity = lgca.nb_sum(lgca.cell_density, addCenter=True) / (lgca.velocitychannels * lgca.interaction_params['capacity']) # density of neighbors
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        density = lgca.cell_density[coord]
        nbdens = nbdensity[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = np.array(node.sum())
        # R1: cell death
        notkilled = lgca.rng.random(size=density) < 1. - lgca.interaction_params['r_d']
        cells = cells[notkilled]
        if len(cells) == 0:
            lgca.nodes[coord] = [[] for _ in range(lgca.K)]
            continue

        kappas = lgca.props['kappa'][cells]
        switch = lgca.rng.random(len(cells)) < tanh_switch(rho=nbdens, kappa=kappas, theta=lgca.interaction_params['theta'])
        restcells, velcells = list(cells[switch]), list(cells[~switch])

        rho = len(cells) / lgca.interaction_params['capacity']  # update density after deaths for birth
        n_prolif = lgca.rng.binomial(len(restcells), max(lgca.interaction_params['r_b'] * (1 - rho), 0))
        if n_prolif > 0:
            proliferating = lgca.rng.choice(restcells, n_prolif, replace=False)
            lgca.maxlabel += n_prolif
            new_cells = np.arange(lgca.maxlabel - n_prolif + 1, lgca.maxlabel + 1)
            lgca.props['kappa'] = np.concatenate((lgca.props['kappa'],
                                                  lgca.rng.normal(loc=lgca.props['kappa'][proliferating],
                                                             scale=lgca.interaction_params['kappa_std'])))
            restcells.extend(list(new_cells))

        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        if len(velcells) > 0:
            gloc = g[coord]
            weights = np.exp(lgca.interaction_params['beta'] * np.einsum('i,ij', gloc, lgca.c))

            z = weights.sum()
            weights /= z  # normalize
            # to prevent divisions by zero if the weight is zero
            # aux = np.nan_to_num(z)
            # weights = np.nan_to_num(weights)
            # weights = (weights / aux)
            # # In case there are some rounding problems
            # if weights.sum() > 1:
            #     weights = (weights / weights.sum())

            # reassign particle directions
            sample = lgca.rng.multinomial(len(velcells), weights)
            lgca.rng.shuffle(velcells)
            for i in range(lgca.velocitychannels):
                node[i].extend(velcells[:sample[i]])
                velcells = velcells[sample[i]:]

        lgca.nodes[coord] = node