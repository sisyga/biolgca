import numpy as np
import numpy.random as npr

# the same as nove_interactions.py except that in all nb_sum() methods the second argument has to be True! (to include center)

def dd_alignment(lgca):
    """
    Rearrangement step for density-dependent alignment interaction including the central lattice site.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0) # gives ndarray of boolean values
    coords = [a[relevant] for a in lgca.nonborder]
    # a is an array of numbers, array can be indexed with another array of same size with boolean specification if element
    # should be included. Returns only the indices of the relevant lattice sites and coords is a list here
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    g = lgca.nb_sum(g, True)  # calculates sum of flux of neighbors for each lattice site
    # 1st dim: nodes
    # 2nd dim: flux vectors
    #print("Before:")
    #lgca.print_nodes()
    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords): #Todo: can do all at once?
        # number of particles
        n = lgca.cell_density[coord]
    #     permutations = lgca.permutations[n]
    #     j = lgca.j[n]  # flux per permutation
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c)) #TODO: make it appropriate for >1D
        z = weights.sum()
        weights = (weights / z)
        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)
        # reassign particle directions
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
    Rearrangement step for density-independent alignment interaction including the central lattice site.
    """
    newnodes = lgca.nodes.copy()
    # filter for nodes that are not virtual border lattice sites
    relevant = (lgca.cell_density[lgca.nonborder] > 0) # gives ndarray of boolean values
    coords = [a[relevant] for a in lgca.nonborder]
    # a is an array of numbers, array can be indexed with another array of same size with boolean specification if element
    # should be included. Returns only the indices of the relevant lattice sites and coords is a list here
    # calculate director field
    g = lgca.calc_flux(lgca.nodes)  # calculates flux for each lattice site
    g = lgca.nb_sum(g, True)  # calculates sum of flux of neighbors for each lattice site
    #print(g)
    # 1st dim: nodes
    # 2nd dim: flux vectors
    # normalize director field by number of neighbors
    nsum = lgca.nb_sum(lgca.cell_density, True)[None, ...] #Todo: hier anders?
    np.maximum(nsum, 1, out=nsum) #avoid dividing by zero later
    #print(nsum.T)
    g = g/ nsum.T #todo: T kann vielleicht weg und numpy.divide benutzen
    #print(g)
    #print("Before:")
    #lgca.print_nodes()
    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords): #Todo: can do all at once?
        # number of particles
        n = lgca.cell_density[coord]
    #     permutations = lgca.permutations[n]
    #     j = lgca.j[n]  # flux per permutation
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c)) #TODO: make it appropriate for >1D
        z = weights.sum()
        weights = (weights / z)
        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)
        # reassign particle directions
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