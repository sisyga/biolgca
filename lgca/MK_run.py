import pickle

from lgca.lgca_1d import *

par = {'dims': 100, 'rest': [2], 'b': [0.2], 'bd': [0.05], 'runs': 2000, 't_max': 100, 'vel': 2, 'ini': 'full'}
# dims:  size of the lattice in nodes
# rest:  number of rest channels
# b:     birth rate: chance of giving birth in a time step (only applies if in rest)
# bd:    birth rate divided by death rate: relation of death to birth (although death may apply in any state)
# runs:  how many times the simulation is repeated (with different random initialization if there are such)
# t_max: how many time steps the simulations will last
# vel:   number of velocity channels, !!! MUST fit geometry !!!
# ini:   cells to start with
# 'full': all channels are filled
# integer value: filles all channels of that many nodes in the center

ALL = par
ALL['data'] = []
ALL['kappas'] = []
ALL['thetas'] = []

# initiale distribution of kappa and theta
cen_k = 0  # center of the initial distribution of kappa
ran_k = 12  # range of kappa around the center
cen_t = 0.5  # center of theta
max_t = 2  # range of theta

for i in range(par['runs']):  # repeat for each run
    lgcalist = []
    thetas = []
    kappas = []
    print(i)
    for r in par['rest']:
        n_channels = r + par['vel']
        if isinstance(par['ini'], int):
            nodes = np.zeros((par['dims'], n_channels), dtype=bool)  # create lattice, all channels false
            left_node = int(len(nodes) / 2 - par['ini'] / 2)  # most left node that will be filled
            nodes[left_node:left_node + par['ini'], [0, -1]] = 1
            #   only par['ini'] center nodes filled
            #   filling may be one of each channel, if second dimension is [0,-1]
        for b in par['b']:
            for d in par['bd']:
                KAPPAS = npr.random(par['dims'] * n_channels) * 24 - 12
                THETAS = npr.random(par['dims'] * n_channels) * 3 - 1
                if par['ini'] == 'full':
                    lgca2 = IBLGCA_1D(density=1, bc='reflect', interaction='go_or_grow', kappa=list(KAPPAS), r_b=b,
                                      r_d=b * d, theta=list(THETAS), restchannels=r, dims=par['dims'])
                else:
                    use_par = par['ini'] * 2  # indicates how many channels are currently filled
                    #   if each 'filled' node has one of each channels, multiple by 2
                    #   otherwise (all channels are filled) multiply by n_channels
                    lgca2 = IBLGCA_1D(nodes=nodes, bc='reflect', interaction='go_or_grow', kappa=list(KAPPAS[:use_par]),
                                      r_b=b, r_d=b * d, theta=list(THETAS[:use_par]), restchannels=r, dims=par['dims'])

                lgca2.timeevo(timesteps=par['t_max'], recordLast=True)
                # lgcalist.append(lgca2)
                kappas.append(lgca2.props_t[-1]['kappa'])
                thetas.append(lgca2.props_t[-1]['theta'])

    # ALL['data'].append(lgcalist)
    ALL['kappas'].append(kappas)
    ALL['thetas'].append(thetas)

    with open('S01_' + str(par['rest']) + str(par['b']) + str(par['bd']) + str(par['runs']) + '_' + str(
            par['t_max']) + '_' + str(par['ini']) + '_WIDE.pkl', 'wb') as handle:
        pickle.dump(ALL, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('S01_CURRENT.pkl', 'wb') as handle:
    pickle.dump(ALL, handle, protocol=pickle.HIGHEST_PROTOCOL)
# from P01 import plot01
# plot01(ALL=ALL)
