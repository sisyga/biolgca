import matplotlib.pyplot as plt

from lgca import get_lgca
import numpy as np

lx = 100
ly = round(lx * 1.1547)  # distance in y direction is smaller than in x direction
ly += 1 if ly % 2 == 1 else 0
dims = lx, ly
capacity = 25
tmax = 200
restchannels = 1
nodes = np.zeros(dims + (6 + restchannels,), dtype=int)
nodes[lx // 2, ly // 2, -1] = capacity
kappa_max = 4
kappa_std = 0.05 * kappa_max
kappa = np.random.random(capacity) * kappa_max * 2 - kappa_max
r_d = 0.2


# %%

lgca = get_lgca(geometry='hx', dims=dims, interaction='go_or_grow_kappa', ve=False, ib=True, bc='reflect', restchannels=1,
                r_b=1, capacity=100, nodes=nodes, kappa=kappa, kappa_std=kappa_std, r_d=r_d, theta=0.25)


# %%
lgca.timeevo(tmax, record=True)
# %%
lgca.plot_density(cbar=True)
plt.show()
# %%

lgca.plot_flux()
plt.show()
# %%
lgca.plot_prop_spatial()
plt.show()
# %%
plt.hist(lgca.get_prop())
plt.show()
# %%


