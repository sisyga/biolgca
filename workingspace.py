from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from os import environ as env
from uuid import uuid4 as uuid

def create_inh(name, dim, rc):
    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',
                    variation=False, density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1)
    lgca.timeevo_until_hom(spatial=True)

    np.save('saved_data/' + name + '_families', lgca.props['lab_m'])
    np.save('saved_data/' + name + '_offsprings', lgca.offsprings)
    np.save('saved_data/' + name + '_nodes', lgca.nodes_t)

def read_inh(name):
    fams = np.load('saved_data/' + name + '_families.npy')
    offs = np.load('saved_data/' + name + '_offsprings.npy')
    nodes = np.load('saved_data/' + name + '_nodes.npy')
    return fams, offs, nodes

def create_pm(name, dim, rc):
    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',
                    variation=False, density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1, r_m=0.3,
                    pop={1:1})
    lgca.timeevo(timesteps=20, record=True)

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

datanames = {'inh': 'todo1', 'pm': 'todo2'}
dim = 3
rc = 1

nodes = np.zeros((dim, dim, 4+rc))

nodes[dim//2, dim//2, :] = 1

print(nodes)
lgca = get_lgca(geometry='square', density=0.25, nodes=nodes, bc='absorbing')
lgca.plot_density()
print(lgca.nodes)
print(lgca.dims, lgca.velocitychannels, lgca.restchannels)
lgca.timeevo(timesteps=1, record=True)
lgca.plot_density()
