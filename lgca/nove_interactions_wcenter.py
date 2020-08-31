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

    polar_alignment = lgca.calc_polar_alignment_parameter()
    #mean_alignment = lgca.calc_mean_alignment()
    #nonodes = lgca.nbofnodes()
    #vsum = lgca.vectorsum()

    print(" ")
    #print("number of nodes")
    #print(nonodes)

    #print("vector sum")
    #print(vsum)


    entropy = lgca.calc_normalized_entropy()  #For some reason this function returns a vector
    print("entropy")
    print(entropy)



    print("polar alignment")
    print(polar_alignment)

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c))

       # print("Weights:")
        #print(weights)

        z = weights.sum()
        #print(z)

        aux = np.nan_to_num(z)


        weights = np.nan_to_num(weights)

        weights = (weights / aux)

       # print("Weights2:")
        #print(weights)


        if weights.sum() > 1:
          #  print("Sum")
           # print(weights.sum())
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
    #Se tiene que volver a poner el addself en nbsum para que funcione bien - sí, esto es la única diferencia entre este código y nove_interactions.py
    #ayyyy hablas español! Jajaja dejame un mensaje cuando leas esto
    # normalize director field by number of neighbors
    nsum = lgca.nb_sum(lgca.cell_density)[None, ...]
    np.maximum(nsum, 1, out=nsum) #avoid dividing by zero later
    g = g/ nsum.T
    Palignment = []

    # loop through lattice sites and reassign particle directions
    for coord in zip(*coords):
        # number of particles
        n = lgca.cell_density[coord]
        # calculate transition probabilities for directions
        weights = np.exp(lgca.beta * np.einsum('i,ij', g[coord], lgca.c))
        z = weights.sum()

        aux = np.nan_to_num(z)

        weights = np.nan_to_num(weights)

        weights = (weights / aux)

        if weights.sum() > 1:
            weights = (weights / weights.sum())


        # reassign particle directions

        sample = npr.multinomial(n, weights, )


        polar_alignment = lgca.calc_polar_alignment_parameter()  # Its getting the same for every iteration


        newnodes[coord] = sample

       # newnodes[coord] = np.array([np.count_nonzero(sample == 1), np.count_nonzero(sample == -1)])
    lgca.nodes = newnodes

