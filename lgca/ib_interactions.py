import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm

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
            if lgca.occupied[coord, ind] == 0:
                lgca.maxlabel += 1
                node[ind] = lgca.maxlabel
                r_b = lgca.props['r_b'][label]
                # lgca.props['r_b'].append(np.clip(npr.normal(loc=r_b, scale=lgca.std), 0, 1))
                lgca.props['r_b'].append(float(trunc_gauss(0, 1, r_b, sigma=lgca.std)))

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
    lgca.update_dynamic_fields()
    n_m = lgca.occupied[:, :2].sum(-1)
    n_r = lgca.occupied[:, 2:].sum(-1)
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
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
            if npr.random() < tanh_switch(rho, kappa=lgca.props['kappa'][cell], theta=lgca.theta):
                # print 'switch to rest', cell
                rest[np.where(rest == 0)[0][0]] = cell
                vel[np.where(vel == cell)[0][0]] = 0

        for cell in can_switch_to_vel:
            if npr.random() < 1 - tanh_switch(rho, kappa=lgca.props['kappa'][cell], theta=lgca.theta):
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
                lgca.props['kappa'].append(float(npr.normal(loc=kappa)))

        v_channels = npr.permutation(vel)
        r_channels = npr.permutation(rest)
        node = np.hstack((v_channels, r_channels))
        lgca.nodes[coord] = node

def inheritance(lgca):
    """
    r_d = const
    """
    chronicle = False   #Ausgabe der einzelnen Schritte für chronicle = True

    # death process, cell dies -> correct value of prop[num_off]
    rel_nodes = lgca.nodes[lgca.r_int:-lgca.r_int]
    # if chronicle:
    #     print('rel_nodes ', rel_nodes)
    dying = npr.random(rel_nodes.shape) < lgca.r_d
    for label in rel_nodes[dying]:
        lgca.props = {
            'r_b': lgca.props['r_b'].copy(),
            'lab_m': lgca.props['lab_m'].copy(),
            'num_off': lgca.props['num_off'].copy()
        }
        if label > 0:
            if chronicle:
                print('cell with label %d dies' % label)
            labmoth = lgca.props['lab_m'][label]
            lgca.props['num_off'][labmoth] -= 1
            lgca.diedcells += 1
            if chronicle:
                print('lab_m dazu ist', labmoth)
    rel_nodes[dying] = 0
    lgca.apply_boundaries()

    # birth
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]

        # choose cells that proliferate
        r_bs = [lgca.props['r_b'][i] for i in node]
        proliferating = npr.random(lgca.K) < r_bs
        # pick a random channel for each proliferating cell. If it is empty, place the daughter cell there
        for label in node[proliferating]:
            ind = npr.choice(lgca.K)
            occ = lgca.nodes > 0
            if occ[coord, ind] == 0:
                # lgca.occupied[coord, ind] wurde nie aktualisiert?!)
                if chronicle:
                    print('es proliferiert Zelle', label)
                lgca.maxlabel += 1
                lgca.borncells += 1
                node[ind] = lgca.maxlabel
                lgca.apply_boundaries()

                if chronicle:
                    print('%d is born' %(lgca.maxlabel))
                    print('with ancestor ', lgca.props['lab_m'][label])

                lgca.props = {
                    'r_b': lgca.props['r_b'].copy(),
                    'lab_m': lgca.props['lab_m'].copy(),
                    'num_off': lgca.props['num_off'].copy()
                }

                labm = lgca.props['lab_m'][label]
                lgca.props['lab_m'].append(labm)
                lgca.props['num_off'][labm] += 1

                if lgca.variation:
                    r_b = lgca.props['r_b'][label]
                    lgca.props['r_b'].append(np.clip(npr.normal(loc=r_b, scale=lgca.std), 0, 1))
                else:
                    lgca.props['r_b'].append(lgca.r_b)
            # if chronicle:
            #     print('nodes after birth: ', lgca.nodes)
        lgca.nodes[coord] = node
        npr.shuffle(lgca.nodes[coord])
    # print('props', lgca.props['num_off'])
    #lgca.plot_prop_numoff()

