import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm
from copy import deepcopy as copy


try:
    from .interactions import tanh_switch
except ImportError:
    from interactions import tanh_switch


def trunc_gauss(lower, upper, mu, sigma=.1, size=1):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)

def birth(lgca):
    """
    Simple birth process
    :return:
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    inds = np.arange(lgca.K)
    for coord in zip(*coords):
        n = lgca.cell_density[coord]
        node = lgca.nodes[coord]

        # choose cells that proliferate
        r_bs = [lgca.props['r_b'][i] for i in node]
        proliferating = npr.random(lgca.K) < r_bs
        dn = proliferating.sum()
        n += dn
        # assure that there are not too many cells. if there are, randomly kick enough of them
        while n > lgca.K:
            p = proliferating.astype(float)
            Z = p.sum()
            p /= Z
            ind = npr.choice(inds, p=p)
            proliferating[ind] = 0
            n -= 1

        # distribute daughter cells randomly in channels
        for label in node[proliferating]:
            p = 1. - lgca.occupied[coord]
            Z = p.sum()
            p /= Z
            ind = npr.choice(inds, p=p)
            lgca.maxlabel += 1
            node[ind] = lgca.maxlabel
            r_b = lgca.props['r_b'][label]
            lgca.props['r_b'].append(npr.normal(loc=r_b, scale=0.2 * r_b))

        lgca.nodes[coord] = node
        npr.shuffle(lgca.nodes[coord])


def birthdeath(lgca):
    """
    Simple birth-death process with evolutionary dynamics towards a higher proliferation rate
    :return:
    """
    # death process
    dying = npr.random(lgca.nodes.shape) < lgca.r_d
    lgca.nodes[dying] = 0
    lgca.update_dynamic_fields()
    # birth
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]

        # choose cells that proliferate
        r_bs = np.array([lgca.props['r_b'][i] for i in node])
        proliferating = npr.random(lgca.K) < r_bs

        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for label in node[proliferating]:
            ind = npr.choice(lgca.K)
            if node[ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props['r_b'][label]
                # lgca.props['r_b'].append(np.clip(npr.normal(loc=r_b, scale=lgca.std), 0, 1))
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        lgca.nodes[coord] = node
        npr.shuffle(lgca.nodes[coord])


def go_or_grow_interaction(lgca):
    """
    interactions of the go-or-grow model. formulation too complex for 1d, but to be generalized.
    :return:
    """

    # death
    dying = npr.random(lgca.nodes.shape) < lgca.r_d
    lgca.nodes[dying] = 0

    # birth
    lgca.update_dynamic_fields()  # routinely update
    n_m = lgca.occupied[:, :2].sum(-1)  # number of cells in rest channels for each node
    #   .sum(-1) ? specifies dimension, -1 -> 1 dimension ?
    n_r = lgca.occupied[:, 2:].sum(-1)  # -"- velocity -"-
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)  # only nodes that are not empty or full
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):  # loop through all relevant nodes
        node = lgca.nodes[coord]
        vel = node[:lgca.velocitychannels]
        rest = node[lgca.velocitychannels:]
        n = lgca.cell_density[coord]

        rho = n / lgca.K

        # determine cells to switch to rest channels and cells that switch to moving state
        # kappas = np.array([lgca.props['kappa'][i] for i in node])
        # r_s = tanh_switch(rho, kappa=kappas, theta=lgca.theta)

        free_rest = lgca.restchannels - n_r[coord]
        free_vel = lgca.velocitychannels - n_m[coord]
        # choose a number of cells that try to switch. the cell number must fit to the number of free channels
        can_switch_to_rest = npr.permutation(vel[vel > 0])[:free_rest]
        can_switch_to_vel = npr.permutation(rest[rest > 0])[:free_vel]

        for cell in can_switch_to_rest:
            if npr.random() < tanh_switch(rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell]):
                # print 'switch to rest', cell
                rest[np.where(rest == 0)[0][0]] = cell
                vel[np.where(vel == cell)[0][0]] = 0

        for cell in can_switch_to_vel:
            if npr.random() < 1 - tanh_switch(rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell]):
                # print 'switch to vel', cell
                vel[np.where(vel == 0)[0][0]] = cell
                rest[np.where(rest == cell)[0][0]] = 0

        # cells in rest channels can proliferate
        can_proliferate = npr.permutation(rest[rest > 0])[:(rest == 0).sum()]
        for cell in can_proliferate:
            if npr.random() < lgca.r_b:
                lgca.maxlabel += 1
                rest[np.where(rest == 0)[0][0]] = lgca.maxlabel
                kappa = lgca.props['kappa'][cell]
                lgca.props['kappa'].append(npr.normal(loc=kappa, scale=0.2))
                theta = lgca.props['theta'][cell]
                lgca.props['theta'].append(npr.normal(loc=theta, scale=0.05))

        v_channels = npr.permutation(vel)
        r_channels = npr.permutation(rest)
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node

def inheritance(lgca):
    """
    r_d = const
    """
    chronicle = True   #Ausgabe der einzelnen Schritte für chronicle = True

    rel_nodes = lgca.nodes[lgca.r_int:-lgca.r_int]
    if chronicle:
        print('rel_nodes ', rel_nodes)
    dying = npr.random(rel_nodes.shape) < lgca.r_d
    for label in rel_nodes[dying]:
        if label > 0:
            if chronicle:
                print('cell with label %d dies' % label)
            labmoth = lgca.props['lab_m'][label]
            lgca.props['num_off'][labmoth] -= 1
            # lgca.diedcells += 1
            if chronicle:
                print('lab_m dazu ist', labmoth)
    rel_nodes[dying] = 0

    # birth
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        if chronicle:
            print('look at node', node)
        # choose cells that proliferate
        r_bs = [lgca.props['r_b'][i] for i in node]
        proliferating = npr.random(lgca.K) < r_bs
        if chronicle:
            print('prolif ', proliferating)
        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for label in node[proliferating]:
            ind = npr.choice(lgca.K)
            if node[ind] == 0:
                if chronicle:
                    print('es proliferiert Zelle', label)
                lgca.maxlabel += 1
                # lgca.borncells += 1

                node[ind] = lgca.maxlabel
                lgca.apply_boundaries()

                if chronicle:
                    print('%d is born' %(lgca.maxlabel))
                    print('with ancestor ', lgca.props['lab_m'][label])

                labm = lgca.props['lab_m'][label]
                lgca.props['lab_m'].append(labm)
                lgca.props['num_off'][labm] += 1

                if lgca.variation:
                    r_b = lgca.props['r_b'][label]
                    lgca.props['r_b'].append(np.clip(npr.normal(loc=r_b, scale=lgca.std), 0, 1))
                else:
                    lgca.props['r_b'].append(lgca.r_b)
            if chronicle:
                print('nodes after birth: ', lgca.nodes)
        lgca.nodes[coord] = node

    #reorientation:
    if chronicle:
        print('vor shuffle', lgca.nodes[1:-1])
    for a in lgca.nonborder:
        for c in a:
            npr.shuffle(lgca.nodes[c])
    if chronicle:
        print('nach shuffle', lgca.nodes[1:-1])

def passenger_mutations_deprecated(lgca):
    if lgca.density != 1:
        print('maxlabel und maxfam nicht mehr aussagekräftig!')
    """
    r_d = const, r_b = const, r_m = const, new families will develop by mutations
    """
    chronicle = True   #Ausgabe der einzelnen Schritte für chronicle = True

    rel_nodes = lgca.nodes[lgca.r_int:-lgca.r_int]
    # if chronicle:
    #     print('rel_nodes ', rel_nodes)

    #dying process
    dying = npr.random(rel_nodes.shape) < lgca.r_d
    # if chronicle:
    #     print('dying ', dying)
    for label in rel_nodes[dying]:
        if label > 0:
            if chronicle:
                print('cell with label %d dies' % label)

            fam = lgca.props['lab_m'][label]
            lgca.props['num_off'][fam] -= 1
            if chronicle:
                print('family dazu ist', fam)
    rel_nodes[dying] = 0

    # birth process
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        # if chronicle:
        #     print('look at node', node)

        # choose cells that proliferate
        r_bs = [0] * len(node)      #TODO: besserer Weg?
        for i in range(0, len(node)):
            if node[i] > 0:
                r_bs[i] = lgca.r_b

        # new = [0] * len(node)
        # TODO: besser?!
        # indexes = [index for index in range(len(node)) if node[index] > 0]
        # for entry in indexes:
        #     new[entry] = r_b

        proliferating = npr.random(lgca.K) < r_bs
        # if chronicle:
        #     print('prolif ', proliferating)

        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for label in node[proliferating]:
            ind = npr.choice(lgca.K)
            if node[ind] == 0:

                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                lgca.apply_boundaries()

                if chronicle:
                    print('es proliferiert Zelle', label)
                    print('der Familie', lgca.props['lab_m'][label])
                    print('%d is born' % lgca.maxlabel)

                mutation = npr.random() < lgca.r_m
                if mutation:
                    lgca.maxfamily += 1
                    if chronicle:
                        print('mit Mutation und neuer family ', lgca.maxfamily)
                    lgca.props['num_off'].append(1)
                    lgca.props['lab_m'].append(int(lgca.maxfamily))
                    lgca.tree_manager.register(lgca.props['lab_m'][label])
                else:
                    fam = lgca.props['lab_m'][label]
                    lgca.props['lab_m'].append(fam)
                    lgca.props['num_off'][fam] += 1
                if chronicle:
                    print('labsm', lgca.props['lab_m'])

            # if chronicle:
            #     # print('nodes after birth: ', lgca.nodes)
            #     print('props after birth: ', lgca.props)
        lgca.nodes[coord] = node

    #reorientation:
    # if chronicle:
    #     print('vor shuffle', lgca.nodes[1:-1])
    for a in lgca.nonborder:
        for c in a:
            npr.shuffle(lgca.nodes[c])
    if chronicle:
        print('props after t ', lgca.props['num_off'])
        print(lgca.tree_manager.tree)

    # if chronicle:
    #     print('nach shuffle', lgca.nodes[1:-1])

def passenger_mutations(lgca):
    if lgca.density != 1:
        print('maxlabel und maxfam nicht mehr aussagekräftig!')
    """
    r_d = const, r_b = const, r_m = const, new families will develop by mutations
    """
    chronicle = False  # Ausgabe der einzelnen Schritte für chronicle = True
    rel_nodes = lgca.nodes[lgca.r_int:-lgca.r_int]

    # dying process
    dying = npr.random(rel_nodes.shape) < lgca.r_d

    for label in rel_nodes[dying]:
        if label > 0:
            if chronicle:
                print('cell with label %d dies' % label)
            fam = lgca.props['lab_m'][label]
            lgca.props['num_off'][fam] -= 1
            if chronicle:
                print('family dazu ist', fam)
    rel_nodes[dying] = 0

    # birth process
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]

        # choose cells that proliferate
        proliferating = [x for x in node if x > 0 and np.random.random(1) < lgca.r_b]

        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for label in proliferating:
            ind = npr.choice(lgca.K)
            if node[ind] == 0:

                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                lgca.apply_boundaries()

                if chronicle:
                    print('es proliferiert Zelle', label)
                    print('der Familie', lgca.props['lab_m'][label])
                    print('%d is born' % lgca.maxlabel)

                mutation = npr.random() < lgca.r_m
                if mutation:
                    lgca.maxfamily += 1
                    if chronicle:
                        print('mit Mutation und neuer family ', lgca.maxfamily)
                    lgca.props['num_off'].append(1)
                    lgca.props['lab_m'].append(int(lgca.maxfamily))
                    lgca.tree_manager.register(lgca.props['lab_m'][label])
                else:
                    fam = lgca.props['lab_m'][label]
                    lgca.props['lab_m'].append(fam)
                    lgca.props['num_off'][fam] += 1
                if chronicle:
                    print('labsm', lgca.props['lab_m'])
        lgca.nodes[coord] = node

    #reorientation:
    for a in lgca.nonborder:
        for c in a:
            npr.shuffle(lgca.nodes[c])
    if chronicle:
        print('props after t ', lgca.props['num_off'])
        print(lgca.tree_manager.tree)


