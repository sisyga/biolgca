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


dim = 1
rc = 178

lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           density=1, dims=dim, restchannels=rc)
t = lgca.timeevo_until_pseudohom(spatial=True)
print('tree', lgca.tree_manager.tree)
print(t)
print('Mutationen: ', lgca.maxfamily - lgca.maxfamily_init)
# # # print(lgca.tree_manager.tree)
id = 'real180_bsp'
np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
np.save('saved_data/' + str(id) + '_nodest', lgca.nodes_t)
np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
control(lgca, t)

# # data = 'passengermutations'
# data = 'passengermutationsdim10rc3'
# offs = np.load('saved_data/' + data + '_offsprings.npy')
# nodest = np.load('saved_data/' + data + '_nodest.npy')
# tree = np.load('saved_data/' + data + '_tree.npy')
# labels = np.load('saved_data/' + data + '_labels.npy')
#
#
# oris = [-99]
#
# for entry in labels[1:]:
#     ori = tree.item().get(entry)['origin']
#     oris.append(ori)
#
# saving = False
# # spacetime_plot(nodest, labels, figsize=(5, 5), tend=50, save=saving, id=data)
# # spacetime_plot(nodest, oris, figsize=(5, 5), tend=50, save=saving, id=data+'_oris')
#
# spacetime_plot(nodest, labels, figsize=(4, 11), save=saving, id=data)
# spacetime_plot(nodest, oris, figsize=(4, 11), save=saving, id=data+'_oris')
#




