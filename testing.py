from lgca import get_lgca
import numpy as np
from matplotlib import pyplot as plt

# geometry
geom = 'hx'
restchannels = 1
l = 100
dims = l, l
# model parameters

r_d = 0.1
r_b = 0.5
kappa = 0.
theta = 0.5

# simulation parameters
dens = 0.2 #starting condition
time = 100
nodes = np.zeros(dims+(6+restchannels,), dtype=int)
nodes[l//2, l//2, :] = 8


lgca = get_lgca(interaction='go_or_grow', bc='periodic', density=dens, geometry=geom, dims=dims,
                restchannels=restchannels, ve=False, ib=True,
                r_d=r_d, r_b=r_b, kappa=kappa, theta=theta, nodes=nodes)
lgca.timeevo(200, record=True)
# lgca.plot_config()
lgca.plot_prop_spatial()
# ani = lgca.live_animate_config(interval=500, grid=True)
# ani = lgca.animate_flux()
plt.show()

