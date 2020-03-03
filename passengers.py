from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math as m

# dens = 1
# birthrate = 0.5
# deathrate = 0.02

dim = 2
rc = 1

lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           density=1, dims=dim, restchannels=rc, r_d=0.1, r_m=0.5)
print(cond_oneancestor(lgca))

# print(lgca.nodes[lgca.r_int:-lgca.r_int])
print(lgca.timeevo_until_pseudohom())
# print('anzahl mutationen ', lgca.maxfamily - lgca.maxfamily_init)
print(lgca.tree)
# print(lgca.nodes[lgca.r_int:-lgca.r_int])

print(cond_oneancestor(lgca))
