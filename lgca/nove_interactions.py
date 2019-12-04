from bisect import bisect_left
from random import random

import numpy as np
import numpy.random as npr

def dd_alignment(lgca):
    """
    Rearrangement step for density-dependent alignment interaction
    :return:
    """
    print("The chosen one")

    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    # # gives ndarray of boolean values
    coords = [a[relevant] for a in lgca.nonborder]
    # # a is an array of numbers, array can be indexed with another array of same size with boolean specification if element
    # # should be included. Returns only the relevant elements and coords is a list here
    g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    g = lgca.nb_sum(g)  # calculates sum of flux of neighbors for each lattice site
    for coord in zip(*coords): #Todo: can do all at once?
         n = lgca.cell_density[coord]
    #     permutations = lgca.permutations[n]
    #     j = lgca.j[n]  # flux per permutation
         weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c)) #TODO: make it appropriate for >1D
         Z = weights.sum()
         weights = (weights / Z).cumsum()
         # multiply neighborhood flux with the flux for each possible direction
         # np.exp for probability
         # cumsum() for cumulative distribution function
         # Todo: sample number of particles for each direction np.random.choice()
    #     newnodes[coord] = permutations[ind]
    #
    lgca.nodes = newnodes


def di_alignment(lgca):
    """
    Rearrangement step for density-independent alignment interaction
    :return:
    """
    print("The chosen one")

    newnodes = lgca.nodes.copy()
    # relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
    #            (lgca.cell_density[lgca.nonborder] < lgca.K)
    # # gives ndarray of boolean values
    # coords = [a[relevant] for a in lgca.nonborder]
    # # a is an array of numbers, array can be indexed with another array of same size with boolean specification if element
    # # should be included. Returns only the relevant elements and coords is a list here
    # g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    # g = lgca.nb_sum(g)  # calculates sum of flux of neighbors for each lattice site
    # for coord in zip(*coords):
    #     n = lgca.cell_density[coord]
    #     permutations = lgca.permutations[n]
    #     j = lgca.j[n]  # flux per permutation
    #     weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], j)).cumsum()
    #     # multiply neighborhood flux with the flux for each possible permutation
    #     # np.exp for probability
    #     # cumsum() for cumulative distribution function
    #     ind = bisect_left(weights, random() * weights[-1])
    #     # inverse transform sampling method
    #     newnodes[coord] = permutations[ind]
    #
    lgca.nodes = newnodes