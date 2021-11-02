import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
try:
    from .interactions import tanh_switch
except ImportError:
    from interactions import tanh_switch

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
    g = lgca.nb_sum(g, addCenter=lgca.nb_include_center)  # sum of flux of neighbors for each lattice site


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
    g = lgca.nb_sum(g, addCenter=lgca.nb_include_center)  # sum of flux of neighbors for each lattice site

    # normalize director field by number of neighbors
    nsum = lgca.nb_sum(lgca.cell_density, addCenter=lgca.nb_include_center)[None, ...]
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

def go_or_grow(lgca):
    """
    interactions (switch, reorientation, birth, death) of the go-or-grow model for volume exclusion free LGCA
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)

    for coord in zip(*coords):
        # determine cell number and moving and resting cell population at coordinate
        n = lgca.cell_density[coord]
        n_mxy = n_m[coord]
        n_rxy = n_r[coord]
        rho = n / lgca.capacity

        # phenotypic switch
        j_1 = npr.binomial(n_mxy, tanh_switch(rho, kappa=lgca.kappa, theta=lgca.theta))
        j_2 = npr.binomial(n_rxy, 1 - tanh_switch(rho, kappa=lgca.kappa, theta=lgca.theta))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2

        # death
        n_mxy -= npr.binomial(n_mxy * np.heaviside(n_mxy, 0), lgca.r_d)
        n_rxy -= npr.binomial(n_rxy * np.heaviside(n_rxy, 0), lgca.r_d)

        # birth
        n_rxy += npr.binomial(n_rxy * np.heaviside(n_rxy, 0), np.maximum(lgca.r_b*(1-rho), 0))

        # reorientation
        v_channels = npr.multinomial(n_mxy, [1/lgca.velocitychannels]*lgca.velocitychannels)

        # add resting cells and assign new content of node at the end of interaction step
        r_channels = np.array([n_rxy])
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node


def go_or_rest(lgca):
    """
    interactions (switch, reorientation) of the go-or-rest model for volume exclusion free LGCA
    go-or-rest is a go-or-grow model without birth and death
    """
    relevant = lgca.cell_density[lgca.nonborder] > 0
    coords = [a[relevant] for a in lgca.nonborder]
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(-1)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(-1)

    for coord in zip(*coords):
        # determine cell number and moving and resting cell population at coordinate
        n = lgca.cell_density[coord]
        n_mxy = n_m[coord]
        n_rxy = n_r[coord]
        rho = n / lgca.capacity

        # phenotypic switch
        j_1 = npr.binomial(n_mxy, tanh_switch(rho, kappa=lgca.kappa, theta=lgca.theta))
        j_2 = npr.binomial(n_rxy, 1 - tanh_switch(rho, kappa=lgca.kappa, theta=lgca.theta))
        n_mxy += j_2 - j_1
        n_rxy += j_1 - j_2

        # reorientation
        v_channels = npr.multinomial(n_mxy, [1/lgca.velocitychannels]*lgca.velocitychannels)

        # add resting cells and assign new content of node at the end of interaction step
        r_channels = np.array([n_rxy])
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node
