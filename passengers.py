from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
from copy import deepcopy as copy

import matplotlib.pyplot as plt
import math as m

# dens = 1
# birthrate = 0.5
# deathrate = 0.02
def control(lgca, t):
    steps, n, c = lgca.nodes_t.shape
    if len(lgca.offsprings) != len(lgca.nodes_t):
        exit(900) # offs länge != nodest länge
    if steps != t + 1:
        exit(901) # offs länge != timesteps
    for i in range(0, t + 1):
        anz = 0
        for j in range(0, n):
            anz += len(lgca.nodes_t[i, j][lgca.nodes_t[i, j] > 0])

        if sum(lgca.offsprings[i][1:]) != anz:
            print(lgca.offsprings)
            print(lgca.nodes_t)
            print(anz)
            exit(902)   #anz zellen != offs
    if lgca.maxfamily - lgca.maxfamily_init != len(lgca.offsprings[-1]) - len(lgca.offsprings[0]):
        exit(903)   #mutationen stimmen nicht


# dim = 1
# rc = 3
#
# lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
#            density=1, dims=dim, restchannels=rc, r_d=0.1, r_m=0.5, r_b=0.8)
# t = lgca.timeevo_until_pseudohom(spatial=True)
# print('tree', lgca.tree_manager.tree)
# print(t)
# print('Mutationen: ', lgca.maxfamily - lgca.maxfamily_init)
# # print(lgca.tree_manager.tree)
# id = 'passengermutations'
# np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
# np.save('saved_data/' + str(id) + '_nodest', lgca.nodes_t)
# np.save('saved_data/' + str(id) + '_labels', lgca.props['lab_m'])
# np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
# control(lgca, t)

offs = np.load('saved_data/passengermutations_offsprings.npy')
nodest = np.load('saved_data/passengermutations_nodest.npy')
tree = np.load('saved_data/passengermutations_tree.npy')
labels = np.load('saved_data/passengermutations_labels.npy')

spacetime_plot(nodest, labels, figsize=(10, 140))
oris = []
# print(tree)
# print(tree.item().get(42))
# print(tree.item().get(42)['origin'])
for entry in labels[1:]:
    ori = tree.item().get(entry)['origin']
    oris.append(ori)

print(labels[:20])
print(oris[:20]) #TODO Kontrolle!

# spacetime_plot(nodest, oris, figsize=(10, 140)) #TODO fix!


