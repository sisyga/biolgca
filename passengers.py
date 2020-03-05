from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
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
        if sum(lgca.offsprings[i][1:]) != len(lgca.nodes_t[i, 0][lgca.nodes_t[i, 0] > 0]):
            print(lgca.offsprings)
            print(lgca.nodes_t)
            exit(902)   #anz zellen != offs
    if lgca.maxfamily - lgca.maxfamily_init != len(lgca.offsprings[-1]) - len(lgca.offsprings[0]):
        exit(903)   #mutationen stimmen nicht


dim = 1
rc = 2

lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           density=1, dims=dim, restchannels=rc, r_d=0.08, r_m=0.8)
t = lgca.timeevo_until_pseudohom(spatial=True)
print(t)
print('Mutationen: ', lgca.maxfamily - lgca.maxfamily_init)
control(lgca, t)
print(lgca.offsprings)
print(np.shape(lgca.offsprings))
np.save('saved_data/offs', lgca.offsprings) #TODO WHY?!
real = np.load('saved_data/offs.npy')
print(real)
# mullerplot(real)
# #
# # spacetime_plot(lgca.nodes_t, lgca.props['lab_m'], figsize=(10,10))
# offs = np.load('saved_data/Testdaten_1__offsprings.npy')
# print(offs)
# mullerplot(offs)

