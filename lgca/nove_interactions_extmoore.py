from bisect import bisect_left
from random import random

import numpy as np
import numpy.random as npr


def dd_alignment(lgca):
    """
    Rearrangement step for density-dependent alignment interaction
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    # # gives ndarray of boolean values
    coords = [a[relevant] for a in lgca.nonborder]
    # # a is an array of numbers, array can be indexed with another array of same size with boolean specification if element
    # # should be included. Returns only the relevant elements and coords is a list here
    g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    g = lgca.nb_sum(g, True)  # calculates sum of flux of neighbors for each lattice site
    # 1st dim: nodes
    # 2nd dim: flux vectors
    #print("Before:")
    #lgca.print_nodes()
    for coord in zip(*coords): #Todo: can do all at once?
        n = lgca.cell_density[coord]
    #     permutations = lgca.permutations[n]
    #     j = lgca.j[n]  # flux per permutation
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c)) #TODO: make it appropriate for >1D
        z = weights.sum()
        weights = (weights / z)
        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)
        sample = npr.choice(lgca.c[0], size=(n,), replace=True, p=weights) #TODO: only works for 1D; multinomial dist.?
        #print("Sample:")
        #print(sample)
        #direction, counts = np.unique(sample, return_counts=True)
        newnodes[coord] = np.array([np.count_nonzero(sample == 1), np.count_nonzero(sample == -1)]) #TODO: only works for 1D
        # multiply neighborhood flux with the flux for each possible direction
        # np.exp for probability
    lgca.nodes = newnodes
    #print("After:")
    #lgca.print_nodes()


def di_alignment(lgca):
    """
    Rearrangement step for density-independent alignment interaction
    :return:
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    # # gives ndarray of boolean values
    coords = [a[relevant] for a in lgca.nonborder]
    # # a is an array of numbers, array can be indexed with another array of same size with boolean specification if element
    # # should be included. Returns only the relevant elements and coords is a list here
    g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    g = lgca.nb_sum(g, True)  # calculates sum of flux of neighbors for each lattice site
    #print(g)
    # 1st dim: nodes
    # 2nd dim: flux vectors
    nsum = lgca.nb_sum(lgca.cell_density, True)[None, ...] #Todo: hier anders?
    np.maximum(nsum, 1, out=nsum) #avoid dividing by zero later
    #print(nsum.T)
    g = g/ nsum.T #todo: T kann vielleicht weg und numpy.divide benutzen
    #print(g)
    #print("Before:")
    #lgca.print_nodes()
    for coord in zip(*coords): #Todo: can do all at once?
        n = lgca.cell_density[coord]
    #     permutations = lgca.permutations[n]
    #     j = lgca.j[n]  # flux per permutation
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c)) #TODO: make it appropriate for >1D
        z = weights.sum()
        weights = (weights / z)
        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)
        sample = npr.choice(lgca.c[0], size=(n,), replace=True, p=weights) #TODO: only works for 1D
        #print("Sample:")
        #print(sample)
        #direction, counts = np.unique(sample, return_counts=True)
        newnodes[coord] = np.array([np.count_nonzero(sample == 1), np.count_nonzero(sample == -1)]) #TODO: only works for 1D
        # multiply neighborhood flux with the flux for each possible direction
        # np.exp for probability
    lgca.nodes = newnodes
    #print("After:")
    #lgca.print_nodes()