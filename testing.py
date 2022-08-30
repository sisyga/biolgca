from lgca import get_lgca
import numpy as np
from matplotlib import pyplot as plt

#test
# geometry
geom = 'hx'
restchannels = 6
l = 50
dims = l, l
# model parameters

r_d = 0.
r_b = 0.
kappa = 0.
theta = 0.5

# simulation parameters
dens = 0.1 #starting condition
beta = 1.2
# time = 100
# nodes = np.zeros(dims+(6+restchannels,), dtype=int)
# nodes[l//2, l//2, 0] = 1
# nodes[l//4, l//4, 0] = 1
# nodes[l//4 * 3, l//4 * 3, 0] = 1



lgca = get_lgca(interaction='alignment', bc='periodic', density=dens, geometry=geom, dims=dims,
                restchannels=restchannels, ve=1, ib=0, beta=beta,
                r_d=r_d, r_b=r_b, kappa=kappa, theta=theta)
# lgca.timeevo(50, record=1)
# lgca.plot_config()
# lgca.plot_prop_spatial()
# lgca.plot_config(grid=1)
# ani = lgca.animate_config(interval=500, grid=1)
ani = lgca.live_animate_flux()
# ani = lgca.animate_density()
# lgca.plot_flux(cbar=0)
# plt.gca().axis('off')
# plt.tight_layout()
# plt.savefig('alignment_art.svg')
plt.show()

