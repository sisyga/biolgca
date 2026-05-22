# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""
Interaction functions and helper functions for identity-based LGCA without volume exclusion.
"""

# from random import random, shuffle, randrange
from itertools import chain

import numpy as np
from scipy.stats import truncnorm
from lgca.interactions import tanh_switch


def _cells_from_node(node):
    """Return a flat list of cell ids from all channels in a node."""
    return list(chain.from_iterable(node))


def _split_cells_into_channels(cells, channeldist):
    """Split a shuffled cell list at cumulative channel counts."""
    return [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]


def trunc_gauss(lower, upper, mu, sigma=.1, size=1, rng=None):
    """Draw samples from a truncated normal distribution.

    Parameters
    ----------
    lower : float
        Lower bound of the distribution.
    upper : float
        Upper bound of the distribution.
    mu : float
        Mean of the underlying normal distribution.
    sigma : float, optional
        Standard deviation of the underlying normal distribution. ``0.1`` by
        default.
    size : int, optional
        Number of samples to draw. ``1`` by default.

    Returns
    -------
    float or numpy.ndarray
        If ``size`` equals ``1`` a single float is returned, otherwise an array
        of shape ``(size,)`` with the drawn samples.
    """
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    vals = truncnorm(a, b, loc=mu, scale=sigma).rvs(size, random_state=rng)
    if size != 1:
        return vals
    else:
        return vals[0]


def random_walk(lgca):
    """Move cells by uniformly redistributing them among velocity channels.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that is modified in-place.

    Returns
    -------
    None
    """

    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        cells = _cells_from_node(node)

        channeldist = lgca.rng.multinomial(len(cells), [1. / lgca.K] * lgca.K).cumsum()
        lgca.rng.shuffle(cells)

        lgca.nodes[coord] = _split_cells_into_channels(cells, channeldist)


def evo_steric(lgca):
    """Birth--death dynamics with mutations and steric movement.

    Cells proliferate with their individual birth rates following a logistic
    growth law limited by ``capacity`` and die with probability ``r_d``. During
    proliferation mutations may occur which modify the proliferation rate. After
    the birth--death step cells redistribute among velocity channels according to
    steric interactions controlled by ``alpha`` and ``gamma``.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    velchannelweights = -lgca.interaction_params['alpha'] * lgca.channel_weight(lgca.cell_density)
    channelweights = np.append(velchannelweights, np.full(lgca.cell_density.shape,
                                                          lgca.interaction_params['gamma'])[..., None], axis=-1)
    channelprobs = np.exp(channelweights)
    channelprobs /= np.sum(channelprobs, axis=-1)[..., None]
    for coord in zip(*coords):
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = _cells_from_node(lgca.nodes[coord])
        newcells = cells.copy()
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

        lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)


def birth(lgca):
    """Logistic birth process with inheritable proliferation rates.

    Each cell divides with probability ``r_b`` scaled by the available capacity.
    The proliferation rate of each daughter cell is drawn from a truncated
    Gaussian distribution centred at the mother's rate.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = _cells_from_node(lgca.nodes[coord])
        newcells = cells.copy()
        for cell in cells:
            r_b = lgca.props['r_b'][cell]
            if lgca.rng.random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.interaction_params['a_max'], r_b,
                                                           sigma=lgca.interaction_params['std'],
                                                           rng=lgca.rng)))

        # channeldist = lgca.rng.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        channeldist = lgca.rng.multinomial(len(newcells), lgca.channel_weights).cumsum()
        lgca.rng.shuffle(newcells)

        lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)


def birthdeath(lgca):
    """Birth and death step with heterogenous proliferation rates.

    Cells divide according to their individual birth rate and the available
    capacity, while every cell dies with probability ``r_d``. The proliferation
    rate of each daughter cell is drawn from a truncated Gaussian distribution
    around the mother's rate.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = _cells_from_node(lgca.nodes[coord])
        newcells = cells.copy()
        for cell in cells:
            if lgca.rng.random() < lgca.interaction_params['r_d']:
                newcells.remove(cell)

            r_b = lgca.props['r_b'][cell]
            if lgca.rng.random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                newcells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.interaction_params['a_max'], r_b,
                                                           sigma=lgca.interaction_params['std'],
                                                           rng=lgca.rng)))

        # channeldist = lgca.rng.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        channeldist = lgca.rng.multinomial(len(newcells), lgca.channel_weights).cumsum()
        lgca.rng.shuffle(newcells)

        lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)

def birthdeath_cancerdfe(lgca):
    """Birth--death step with driver and passenger mutations.

    Proliferation follows a logistic law and death occurs with probability
    ``r_d``. Daughter cells inherit the mother's birth rate plus a deviation
    drawn from exponential distributions representing driver (beneficial) or
    passenger (deleterious) mutations.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = _cells_from_node(lgca.nodes[coord])
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
                    passenger = float(lgca.rng.exponential(scale=lgca.interaction_params['s_p']))

                if lgca.rng.random() < lgca.interaction_params['p_d']:
                    driver = float(lgca.rng.exponential(scale=lgca.interaction_params['s_d']))

                lgca.props['r_b'].append(min(r_b - passenger + driver, lgca.interaction_params['a_max']))


        # channeldist = lgca.rng.multinomial(len(newcells), [1. / lgca.K] * lgca.K).cumsum()
        channeldist = lgca.rng.multinomial(len(newcells), lgca.channel_weights).cumsum()
        lgca.rng.shuffle(newcells)

        lgca.nodes[coord] = _split_cells_into_channels(newcells, channeldist)


def go_or_grow(lgca):
    """Evolutionary ``go-or-grow`` interaction.

    Cells stochastically switch between a migratory and a resting phenotype
    according to a sigmoidal function of the local density (``tanh_switch``) with
    individual parameters ``kappa`` and ``theta``. Resting cells proliferate with
    a constant rate ``r_b`` and all cells die with rate ``r_d``. Offspring inherit
    the mother's switching parameters with Gaussian noise.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    new_kappa_chunks = []
    new_theta_chunks = []
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        density = lgca.cell_density[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = np.asarray(_cells_from_node(node), dtype=int)
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
            new_kappa_chunks.append(
                lgca.rng.normal(
                    loc=lgca.props['kappa'][proliferating],
                    scale=lgca.interaction_params['kappa_std'],
                )
            )
            # lgca.props['theta'] = np.concatenate((lgca.props['theta'],
            #                                       trunc_gauss(0, 1, mu=lgca.props['theta'][proliferating],
            #                                                   sigma=lgca.interaction_params['theta_std'])))
            new_theta_chunks.append(
                lgca.rng.normal(
                    loc=lgca.props['theta'][proliferating],
                    scale=lgca.interaction_params['theta_std'],
                )
            )
            restcells.extend(list(new_cells))

        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        for cell in velcells:
            node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

        lgca.nodes[coord] = node
    if new_kappa_chunks:
        lgca.props['kappa'] = np.concatenate((lgca.props['kappa'], *new_kappa_chunks))
        lgca.props['theta'] = np.concatenate((lgca.props['theta'], *new_theta_chunks))

def go_or_grow_kappa(lgca):
    """``Go-or-grow`` interaction using neighbourhood density.

    Phenotype switching is determined by the average density in the Moore
    neighbourhood rather than only the local density. Only the slope parameter
    ``kappa`` evolves, whereas the threshold ``theta`` is fixed globally.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    # Identify the relevant cells (those with non-zero density)
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # Calculate the average density in the neighborhood
    nbdensity = lgca.nb_sum(lgca.cell_density, addCenter=True) / ((lgca.velocitychannels+1) * lgca.interaction_params['capacity']) # average density in neighborhood
    new_kappa_chunks = []
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        density = lgca.cell_density[coord]
        nbdens = nbdensity[coord]
        # rho = density / lgca.interaction_params['capacity']
        # Get the list of cells at the current node
        cells = np.asarray(_cells_from_node(node), dtype=int)
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
            new_kappa_chunks.append(
                lgca.rng.normal(
                    loc=lgca.props['kappa'][proliferating],
                    scale=lgca.interaction_params['kappa_std'],
                )
            )
            # Add the new cells to the list of resting cells
            restcells.extend(list(new_cells))

        # Initialize the node with empty channels and add the resting cells
        node = [[] for _ in range(lgca.velocitychannels)]
        node.append(restcells)
        # Assign the migrating cells to random velocity channels
        for cell in velcells:
            node[lgca.rng.integers(lgca.velocitychannels)].append(cell)

        # Update the node in the lgca object
        lgca.nodes[coord] = node
    if new_kappa_chunks:
        lgca.props['kappa'] = np.concatenate((lgca.props['kappa'], *new_kappa_chunks))


def tanh_switch(rho, kappa=5.0, theta=0.8):
    """Sigmoidal switching function.

    Parameters
    ----------
    rho : float or numpy.ndarray
        Local (or neighbourhood) density.
    kappa : float, optional
        Steepness of the transition. Default is ``5.0``.
    theta : float, optional
        Density threshold at which the switch probability is ``0.5``.
        Default is ``0.8``.

    Returns
    -------
    float or numpy.ndarray
        Switching probability with the same shape as ``rho``.
    """
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))

def go_or_grow_kappa_chemo(lgca):
    """``Go-or-grow`` interaction with chemotactic movement.

    Switching dynamics are identical to :func:`go_or_grow_kappa`, but migrating
    cells move preferentially along the density gradient according to a Boltzmann
    weight with parameter ``beta``.

    Parameters
    ----------
    lgca : object
        Lattice-gas cellular automaton that will be modified in-place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    g = lgca.gradient(lgca.cell_density / lgca.interaction_params['capacity'])  # density gradient for each lattice site
    nbdensity = lgca.nb_sum(lgca.cell_density, addCenter=True) / (lgca.velocitychannels * lgca.interaction_params['capacity']) # density of neighbors
    new_kappa_chunks = []
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        density = lgca.cell_density[coord]
        nbdens = nbdensity[coord]
        rho = density / lgca.interaction_params['capacity']
        cells = np.asarray(_cells_from_node(node), dtype=int)
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
            new_kappa_chunks.append(
                lgca.rng.normal(
                    loc=lgca.props['kappa'][proliferating],
                    scale=lgca.interaction_params['kappa_std'],
                )
            )
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
    if new_kappa_chunks:
        lgca.props['kappa'] = np.concatenate((lgca.props['kappa'], *new_kappa_chunks))
