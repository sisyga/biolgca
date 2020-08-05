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

    Palignment = []
    polar_alignment = lgca.calc_polar_alignment_parameter()  # Its getting the same for every iteration
    nonodes = lgca.nbofnodes()
    vsum = lgca.vectorsum()

    print(" ")
    print("number of nodes")
    print(nonodes)

    print("vector sum")
    print(vsum)

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


 #       if z > 9e+300:        # To prevent z = inf -> weights / inf
  #          z = 9e+300
   #     if z < 9e-300:
    #        z = 9e-300






       # for x in range(len(weights)):
        #    if weights[x] > 9e+300:  # To prevent z = inf -> weights / inf
         #       weights[x] = 9e+300
          #  if weights[x] < 9e-300:
           #     weights[x] = 9e-300

        weights = np.nan_to_num(weights)

        weights = (weights / aux)

       # print("Weights2:")
        #print(weights)


        if weights.sum() > 1:
          #  print("Sum")
           # print(weights.sum())
            weights = (weights / weights.sum())



    #    for x in range(len(weights)):
     #       if weights[x] == float('nan') or weights[x] >= 1:        # To solve Z = inf -> weights = nan
      #          for y in range(len(weights)):
       #             weights[y] = 0
        #        weights[x] = 1

       # print("Weights3:")
        #print(weights)



        #print("Coord:")
        #print(coord)

        # print("Weights:")
        # print(weights)
        # reassign particle directions
        #sample = npr.choice(lgca.c[0], size=(n,), replace=True, p=weights) #TODO: only works for 1D; multinomial dist.?

        sample = npr.multinomial(n, weights,)
        #print("Sample:")
        #print(sample)
        #print(np.count_nonzero(sample == -1))

        #newnodes[coord] = np.array([np.count_nonzero(sample ==1), np.count_nonzero(sample == -1)]) only works in 1D - have to change nsum



        newnodes[coord] = sample




        #print(newnodes[coord])


    polar_alignment = lgca.calc_polar_alignment_parameter()  # Its getting the same for every iteration
    Palignment.append(polar_alignment)



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

        #       if z > 9e+300:        # To prevent z = inf -> weights / inf
        #          z = 9e+300
        #     if z < 9e-300:
        #        z = 9e-300

        # for x in range(len(weights)):
        #    if weights[x] > 9e+300:  # To prevent z = inf -> weights / inf
        #       weights[x] = 9e+300
        #  if weights[x] < 9e-300:
        #     weights[x] = 9e-300

        weights = np.nan_to_num(weights)

        weights = (weights / aux)

        #print("Weights2:")
        #print(weights)

        if weights.sum() > 1:
            #  print("Sum")
            # print(weights.sum())

            weights = (weights / weights.sum())

        #   if z > 9e+300:        # To prevent z = inf -> weights / inf
      #      z = 9e+300
       # if z < 9e-300:
        #    z = 9e-300







        #print("Coord:")
        #print(coord)
        #print("Weights:")
        #print(weights)

        # reassign particle directions
        #sample = npr.choice(lgca.c[0], size=(n,), replace=True, p=weights) #TODO: only works for 1D

        sample = npr.multinomial(n, weights, )

        #print("Sample:")
        #print(sample)
        polar_alignment = lgca.calc_polar_alignment_parameter()  # Its getting the same for every iteration
        print("polar alignment")
        print(polar_alignment)

        newnodes[coord] = sample

       # newnodes[coord] = np.array([np.count_nonzero(sample == 1), np.count_nonzero(sample == -1)])
    lgca.nodes = newnodes

