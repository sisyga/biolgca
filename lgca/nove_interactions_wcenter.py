import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# the same as nove_interactions.py except that in all nb_sum() methods the center has to be included



def dd_alignment(lgca):
    """
    Rearrangement step for density-dependent alignment interaction including the central lattice site.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # flux for each lattice site
    g = lgca.nb_sum(g)  # sum of flux of neighbors for each lattice site


    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c))

        z = weights.sum()
        # to prevent divisions by zero if the weight is zero
        aux = np.nan_to_num(z)
        weights = np.nan_to_num(weights)
        weights = (weights / aux)
        # In case there are some rounding problems
        if weights.sum() > 1:
            weights = (weights / weights.sum())

        # reassign particle directions
        sample = npr.multinomial(n, weights,)

        newnodes[coord] = sample

    lgca.nodes = newnodes



def di_alignment(lgca):
    """
    Rearrangement step for density-independent alignment interaction including the central lattice site.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # flux for each lattice site
    g = lgca.nb_sum(g)  # sum of flux of neighbors for each lattice site

    # normalize director field by number of neighbors
    nsum = lgca.nb_sum(lgca.cell_density)[None, ...]
    np.maximum(nsum, 1, out=nsum)   # avoid dividing by zero later
    g = g / nsum.T

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c))
        z = weights.sum()
        # avoid division by zero if weights is zero
        aux = np.nan_to_num(z)
        weights = np.nan_to_num(weights)
        weights = (weights / aux)
        # In case there are rounding problems
        if weights.sum() > 1:
            weights = (weights / weights.sum())

        # reassign particle directions
        sample = npr.multinomial(n, weights, )

        newnodes[coord] = sample

    lgca.nodes = newnodes

