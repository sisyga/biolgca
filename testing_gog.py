import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca

geom = 'hx'
restchannels = 1
l = 1000
dims = l, l
capacity = 100
# interaction parameters
r_b = 1. # initial birth rate
r_d = 0.2 * r_b / 2 # initial death rate

# nodes = np.zeros(dims+(6+restchannels,), dtype=int)
# nodes[l//2, l//2, -1] = capacity

nodes = np.zeros((l,)+(2+restchannels,), dtype=int)
nodes[l//2, -1] = capacity
kappa = np.random.random(capacity) * 8 - 4

# lgca = get_lgca(ib=True, bc='reflect', interaction='go_or_grow', dims=dims, nodes=nodes, ve=False, geometry='hx',
#                 r_b=r_b, capacity=capacity, r_d=r_d, kappa=kappa)
lgca = get_lgca(ib=True, bc='reflect', interaction='go_or_grow', dims=l, nodes=nodes, ve=False, geometry='lin',
                r_b=r_b, capacity=capacity, r_d=r_d, kappa=kappa, theta_std=1e-6)


lgca.timeevo(700, record=True, recordN=True)

kappas = lgca.get_prop(propname='kappa')
# anim = lgca.animate_density()
# plt.plot(lgca.n_t)
plt.hist(kappas)
plt.figure()
lgca.plot_prop_spatial(propname='kappa')

plt.show()

