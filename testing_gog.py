from lgca.base import *
import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from lgca.nove_ib_interactions import go_or_grow_kappa_chemo
import pickle as pkl

def timeevo(self, N, timesteps=100, record=False, recordN=False, recorddens=True, recordchanneldens=False):
    self.update_dynamic_fields()
    if record:
        self.nodes_t = []
        self.nodes_t.append(copy(self.nodes[self.nonborder]))
    if recordN:
        self.n_t = []
        self.n_t.append(self.cell_density[self.nonborder].sum())
    if recorddens:
        self.dens_t = []
        self.dens_t.append(self.cell_density[self.nonborder])
    if recordchanneldens:
        self.channel_pop_t = []
        self.channel_pop_t.append(self.channel_pop[self.nonborder])

    tmax = 1
    while self.cell_density[self.nonborder].sum() <= N:
        tmax += 1
        self.timestep()
        if record:
            self.nodes_t.append(copy(self.nodes[self.nonborder]))
        if recordN:
            self.n_t.append(self.cell_density[self.nonborder].sum())
        if recorddens:
            self.dens_t.append(self.cell_density[self.nonborder])

    if record:
        temp = get_arr_of_empty_lists((tmax, ) + self.dims + (self.K,))
        temp[...] = self.nodes_t[:]
        self.nodes_t = temp
    if recordN:
        self.n_t = np.array(self.n_t)
    if recorddens:
        self.dens_t = np.array(self.dens_t)



restchannels = 1
l = 100
dims = l,
capacity = 2000
# interaction parameters
r_b = .5 # initial birth rate
r_d = 0.25 # initial death rate
# r_b = 0 # initial birth rate
# r_d = 0.# initial death rate
nodes = np.zeros(dims+(6+restchannels,), dtype=int)
# nodes[l//2, l//2, -1] = capacity

nodes = np.zeros((l,)+(2+restchannels,), dtype=int)
nodes[0, -1] = capacity / 2
# kappa = np.random.random(nodes.sum()) * 8. - 4

# index = 4, 2, 0, 0
# PATH = '.\\data\\gog\\nonlocaldensity_10reps\\'
# p = PATH + 'data{}.pkl'.format(index)
# with open(p, 'rb') as f:
#     d = pkl.load(f)
#
# parameters = np.load(PATH + 'params.npz', allow_pickle=True)
# constparams = parameters['constparams'].item()
# r_ds = parameters['r_ds']
# thetas = parameters['thetas']
# theta = thetas[index[1]]
# r_d = r_ds[index[0]]
# print('r_d = {}, theta = {}'.format(r_d, theta))
# del constparams['nodes']
# rhoeq = 1 - r_d / r_b
# lgca = get_lgca(ib=True, bc='reflect', interaction='go_or_grow', dims=dims, nodes=nodes, ve=False, geometry='hx',
#                 r_b=r_b, capacity=capacity, r_d=r_d, kappa=kappa, theta_std=1e-6)
# lgca = get_lgca(ib=True, bc='reflect', interaction='go_or_grow_kappa', dims=l, nodes=nodes, ve=False, geometry='lin',
#                 r_b=r_b, capacity=capacity, r_d=r_d, kappa=kappa, theta=.3, kappa_std=1)
lgca = get_lgca(ib=True, bc='reflect', interaction='birthdeath_cancerdfe', dims=l, nodes=nodes, ve=False, geometry='lin',
                capacity=capacity, r_b=r_b, r_d=r_d, gamma=14)
lgca.timeevo(50000, record=True, recordN=True, recorddens=True)
# lgca = get_lgca(**constparams, theta=theta, nodes=d['nodes_t'][1:-1])
# lgca.props['kappa'] = d['kappa']
# # lgca.nodes = d['nodes_t']
# N = lgca.cell_density[lgca.nonborder].sum()
# lgca.interaction_params['r_d'] = .99
# lgca.timestep()
# lgca.interaction_params['r_d'] = r_d
# timeevo(lgca, N, record=True, recordN=False, recorddens=False)
# print('Recurrence time: ', lgca.nodes_t.shape[0])
# # concatenate lgca.nodes_t with d['nodes_t'] along first dimension
# lgca.nodes_t = np.concatenate((d['nodes_t'][None, 1:-1, ...], lgca.nodes_t, ))
#
# channelpop = lgca.length_checker(lgca.nodes_t)
# lgca.dens_t = channelpop.sum(axis=-1)
# plt.imshow(lgca.dens_t)
# kappas = lgca.get_prop(propname='kappa')
# anim = lgca.animate_density()
# plt.plot(lgca.n_t)
# plt.hist(kappas, bins='auto')
# plt.show()
# plt.figure()
# lgca.plot_prop_spatial()
# # lgca.plot_density()
#
plt.figure()
# # lgca.plot_prop_spatial(propname='kappa')
# lgca.plot_density(vmax=lgca.interaction_params['capacity'])
plt.plot(lgca.dens_t)
# plt.show()
#
# plt.figure()
# lgca.plot_prop_timecourse()
# plt.show()

plt.figure()
plt.plot(lgca.n_t)
plt.show()

