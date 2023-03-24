import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from lgca.nove_ib_interactions import go_or_grow_kappa_chemo

restchannels = 1
l = 100
dims = l, l
capacity = 50
# interaction parameters
r_b = 1. # initial birth rate
r_d = 0.4 * r_b / 2 # initial death rate
# r_b = 0 # initial birth rate
# r_d = 0.# initial death rate
nodes = np.zeros(dims+(6+restchannels,), dtype=int)
nodes[l//2, l//2, -1] = capacity
#
# nodes = np.zeros((l,)+(2+restchannels,), dtype=int)
# nodes[l//2, -1] = capacity
kappa = np.random.random(capacity) * 8 - 4

rhoeq = 1 - r_d / r_b
lgca = get_lgca(ib=True, bc='reflect', interaction='go_or_grow_kappa', dims=dims, nodes=nodes, ve=False, geometry='hx',
                r_b=r_b, capacity=capacity, r_d=r_d, kappa=kappa, theta=rhoeq/2)
# lgca = get_lgca(ib=True, bc='reflect', interaction='go_or_grow_kappa', dims=l, nodes=nodes, ve=False, geometry='lin',
#                 r_b=r_b, capacity=capacity, r_d=r_d, kappa=kappa, kappa_std=.2)

lgca.timeevo(200, record=True, recordN=False)

kappas = lgca.get_prop(propname='kappa')
# anim = lgca.animate_density()
# plt.plot(lgca.n_t)
plt.hist(kappas, bins='auto')
plt.figure()
lgca.plot_prop_spatial(propname='kappa')
# lgca.plot_density()
plt.show()

