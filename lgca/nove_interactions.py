import numpy as np
import numpy.random as npr


def dd_alignment(lgca):
    """
    Rearrangement step for density-dependent alignment interaction excluding the central lattice site.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # flux for each lattice site
    g = lgca.nb_sum(g, False)  # sum of flux of neighbors for each lattice site

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c))
        z = weights.sum()
        weights = (weights / z)
        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)

        # reassign particle directions
        #sample = npr.choice(lgca.c[0], size=(n,), replace=True, p=weights)

        sample = npr.multinomial(lgca.c[0,0], weights, size = (n,) )

        print("Sample:")
        print(sample)
        newnodes[coord] = np.array([np.count_nonzero(sample == 1), np.count_nonzero(sample == -1)])
    lgca.nodes = newnodes


def di_alignment(lgca):
    """
    Rearrangement step for density-independent alignment interaction excluding the central lattice site.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # flux for each lattice site
    g = lgca.nb_sum(g, False)  # sum of flux of neighbors for each lattice site
    # normalize director field by number of neighbors
    nsum = lgca.nb_sum(lgca.cell_density, False)[None, ...]
    np.maximum(nsum, 1, out=nsum) # avoid dividing by zero later
    g = g/ nsum.T

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c))
        z = weights.sum()
        weights = (weights / z)
        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)

        # reassign particle directions
        #sample = npr.choice(lgca.c[0], size=(n,), replace=True, p=weights)

        sample = npr.multinomial(lgca.c[0,0], weights, size = (n,) )

        #print("Sample:")
        #print(sample)
        newnodes[coord] = np.array([np.count_nonzero(sample == 1), np.count_nonzero(sample == -1)])
    lgca.nodes = newnodes

