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

def create_hex(dim, rc, steps, driver=False, save=False):
    nodes = np.zeros((dim, dim, 6 + rc))
    for i in range(0, 6 + rc):
        nodes[dim//2, dim//2, i] = i+1
    name = str(dim) + 'x' + str(dim) + '_rc=' + str(rc) + '_steps=' + str(steps) + '_Test'

    if driver:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                        r_m=0.01, effect=driver_mut)
        name += '_driver'
    else:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                            effect=passenger_mut)
        name += '_passenger'

    lgca_hex.timeevo(timesteps=steps, record=True)
    print('maxfam', len(lgca_hex.tree_manager.tree))
    if save:
        np.save('saved_data/' + name + '_tree', lgca_hex.tree_manager.tree)
        np.save('saved_data/' + name + '_families', lgca_hex.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca_hex.offsprings)
        np.save('saved_data/' + name + '_nodes', lgca_hex.nodes_t)
    for t in range(0, steps + 1, steps):
        lgca_hex.plot_test(nodes_t=lgca_hex.nodes_t[t], save=save, id=name + '_step' + str(t))
        # lgca_hex.plot_density(density=lgca_hex.dens_t[t], save=save, id=name + '_step' + str(t))


create_hex(dim=50, rc=1, steps=50, driver=True, save=False)
# create_hex(dim=50, rc=1, steps=50, driver=False, save=True)

# name = '50x50_rc=1_steps=70_Test'
# tree, fams, offs, nodes = read_pm(name=name)
# offs = correct(offs)
# print(len(offs), len(nodes))
#
