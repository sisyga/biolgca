from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid

#todo 2. Start vollbesetzt, passenger -> funktioniert noch?
def create(name, dim, rc):
    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',
                    variation=False, density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1)
    t = lgca.timeevo_until_hom(spatial=True)
    print(t)
    np.save('saved_data/' + name + '_families', lgca.props['lab_m'])
    np.save('saved_data/' + name + '_offsprings', lgca.offsprings)
    np.save('saved_data/' + name + '_nodes', lgca.nodes_t)

def read(name):
    fams = np.load('saved_data/' + name + '_families.npy')
    offs = np.load('saved_data/' + name + '_offsprings.npy')
    nodes = np.load('saved_data/' + name + '_nodes.npy')
    return fams, offs, nodes

name = 'todo1'
dim = 2
rc = 2
# create(name, dim, rc)
fams, offs, nodes = read(name)
print(offs)
spacetime_plot(nodes, fams, figsize=(10,10))
mullerplot(offs)
