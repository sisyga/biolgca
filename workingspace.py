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
    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting', density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1, r_m=0.3,
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
dim = 2
rc = 1

nodes = np.zeros((dim, dim, 6+rc))
for i in range(0, 6+rc):
    nodes[1, 1, i] = i+1

lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                r_b=0.8, r_d=0.3, r_m=0.8, mut=True)
print(lgca_hex.props)
print(lgca_hex.maxlabel, lgca_hex.maxfamily_init)
print(lgca_hex.nodes[lgca_hex.nonborder])
lgca_lin = get_lgca(ib=True, geometry='lin', interaction='mutations', bc='reflecting', density=1, dims=3,
                restchannels=1, r_b=0.8, r_d=0.1, r_m=0.3)
# print(lgca_lin.nodes)

lgca_hex.timeevo(timesteps=2, record=True)
print(lgca_hex.nodes[lgca_hex.nonborder])

