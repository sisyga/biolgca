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


dim = 3
rc = 2

lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.3, r_m=1)
t = lgca.timeevo_until_pseudohom(spatial=True)
print(t)
print('Mutationen: ', lgca.maxfamily - lgca.maxfamily_init)
print(lgca.tree_manager.tree)

control(lgca, t)
# print(lgca.offsprings)
# # print(np.shape(lgca.offsprings))
# np.save('saved_data/offs', lgca.offsprings) #TODO WHY?!
real = np.load('saved_data/offs.npy')
print(real)
# mullerplot(real)
# #
# # spacetime_plot(lgca.nodes_t, lgca.props['lab_m'], figsize=(10,10))
# offs = np.load('saved_data/Testdaten_1__offsprings.npy')
# print(offs)
# mullerplot(offs)

# p1 = {'labs': [1,2,3,4], 'offs':  [1,1,1,1]}
# p2 = {'labs': [1,2,3,4,1], 'offs':  [2,1,1,1]}
# p3 = {'labs': [1,2,3,4,1], 'offs':  [2,0,1,1]}
#
# off = [p1['offs']]
# print(off)
# off.append(p2['offs'])
# print(off)
# off.append(copy(p3['offs']))
# print(off)
#
# p3['offs'].append(99)
#
# off.append(copy(p3['offs']))
# print(off)
#
# np.save('saved_data/offsprings', off)
# o = np.load('saved_data/offsprings.npy')
# print(o)
#
# test = [[1,2], [3,4], [4,5,6]]
# print(test)
# np.save('saved_data/test', test)
# t = np.load('saved_data/test.npy')
# print(t)
# print(t[0])