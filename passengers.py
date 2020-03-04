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
        exit(900)
    if steps != t + 1:
        exit(901)
    for i in range(0, t + 1):
        if sum(lgca.offsprings[i][1:]) != len(lgca.nodes_t[i, 0][lgca.nodes_t[i,0]>0]):
            print(sum(lgca.offsprings[i][1:]))
            print(lgca.nodes_t[i, 0][lgca.nodes_t[i,0]>0])
            exit(902)
    return exit(777)
dim = 1
rc = 2

lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           density=1, dims=dim, restchannels=rc, r_d=0.08, r_m=0.5)
t = lgca.timeevo_until_pseudohom(spatial=True)
print(t)
print(lgca.tree)
print(len(lgca.tree))
print('anzahl mutationen ', lgca.maxfamily - lgca.maxfamily_init)
off = (lgca.offsprings)
for line in range(0, len(off)):
    print(off[line])
n = lgca.nodes_t
print(n)

print('nodes, offs', n.shape, len(off))

control(lgca, t)

