from lgca import get_lgca
from lgca.ib_interactions import driver_mut, passenger_mut
from lgca.helpers import *
from lgca.helpers2d import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from os import environ as env
from uuid import uuid4 as uuid

def create_inh(name, dim, rc, save=False):
    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',
                    variation=False, density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1)
    lgca.timeevo_until_hom(spatial=True)

    if save:
        np.save('saved_data/' + name + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca.offsprings)
        np.save('saved_data/' + name + '_nodes', lgca.nodes_t)

def read_inh(name):
    fams = np.load('saved_data/' + name + '_families.npy')
    offs = np.load('saved_data/' + name + '_offsprings.npy')
    nodes = np.load('saved_data/' + name + '_nodes.npy')
    return fams, offs, nodes

def create_pm(name, dim, rc, save=False):
    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting', density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1, r_m=0.3,
                    pop={1: 1})
    lgca.timeevo(timesteps=20, record=True)

    if save:
        np.save('saved_data/' + name + '_tree', lgca.tree_manager.tree)
        np.save('saved_data/' + name + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca.offsprings)
        np.save('saved_data/' + name + '_nodes', lgca.nodes_t)

def read_pm(name):
    tree = np.load('saved_data/' + name + '_tree.npy')
    fams = np.load('saved_data/' + name + '_families.npy')
    offs = np.load('saved_data/' + name + '_offsprings.npy')
    nodes = np.load('saved_data/' + name + '_nodes.npy')
    return tree, fams, offs, nodes

# create_pm(name='Probe', dim=3, rc=1, save=True)
# print(read_pm('Probe'))

# datanames = {'inh': 'todo1', 'pm': 'todo2'}

dim = 4
rc = 2

nodes = np.zeros((dim, dim, 6+rc))
for i in range(0, 6+rc):
    nodes[dim//2, dim//2, i] = i+1

# print(nodes)
# nodes[1,3,1] = 4

lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                r_b=0, r_d=0, r_m=0.5, effect=passenger_mut)
# # lgca_hex.timeevo(timesteps=2, record=True)
#
# lgca_hex.plot_density()
# n = np.array([x[1:-1] for x in lgca_hex.nodes[lgca_hex.r_int:-lgca_hex.r_int]])
# # print(n)
# # for t in range(0, 3):
lgca_hex.timeevo(timesteps=2, record=True)
# # print('nt', lgca_hex.nodes_t)
for t in range(0, 3):
    # lgca_hex.plot_test(nodes_t=lgca_hex.nodes_t[t])
    lgca_hex.plot_density(density=lgca_hex.dens_t[t])
# print(lgca_hex.props)
# print(lgca_hex.nodes_t)
# lgca_hex.plot_density()
# lgca_hex.plot_test()


# print(lgca_hex.props)
# lgca_hex.timeevo(timesteps=2, record=True)
# print(lgca_hex.nodes_t)
# print(lgca_hex.nodes_t[0])
# print(lgca_hex.nodes_t[-1])
# print(lgca_hex.props)
# lgca_hex.timeevo(timesteps=1, record=True)
# print(lgca_hex.nodes_t)
# print(lgca_hex.props)



# lgca_hex.plot_test()
# lgca_hex.plot_density()
# lgca_hex.timeevo(timesteps=1, record=True)
# lgca_hex.plot_test()
# print(lgca_hex.nodes_t)
# tend, lx, ly, K = lgca_hex.nodes_t.shape
# print('tend, lx, ly, K', tend, lx, ly, K)

# lgca_hex.plot_test()


# plot_popsize(data=lgca_hex.offsprings, plotmax=0)   #dim*dim*lgca_hex.K)
# print(lgca_hex.dens_t)
# data = np.array([[[-99]*lx]*ly]*tend)
# print(data)
# lgca_hex.plot_test()
# lgca_hex.live_animate_density()
# t = 10
# lgca_hex.timeevo(timesteps=t, record=True)
# steps = np.arange(0, t, 4)
# # for step in steps:
# #     lgca_hex.plot_density(lgca_hex.dens_t[step])
# # print('popmax', dim*dim*lgca_hex.K)
# # plot_popsize(data=lgca_hex.offsprings, plotmax=dim*dim*lgca_hex.K)
# plot_families(lgca_hex.nodes_t)
