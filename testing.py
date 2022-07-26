from lgca import get_lgca
import numpy as np
from matplotlib import pyplot as plt

# geometry
geom = 'hx'
restchannels = 0
l = 100
dims = 200, 300
# model parameters

r_d = 0.1
r_b = 0.5
kappa = 0.
theta = 0.5

# simulation parameters
dens = 0.5 #starting condition
beta = 1.2
# time = 100
# nodes = np.zeros(dims+(6+restchannels,), dtype=int)
# nodes[l//2, l//2, 0] = 1
# nodes[l//4, l//4, 0] = 1
# nodes[l//4 * 3, l//4 * 3, 0] = 1



lgca = get_lgca(interaction='alignment', bc='periodic', density=dens, geometry=geom, dims=dims,
                restchannels=restchannels, ve=True, ib=False, beta=beta,
                r_d=r_d, r_b=r_b, kappa=kappa, theta=theta)
lgca.timeevo(50, record=0)
# lgca.plot_config()
# lgca.plot_prop_spatial()
# ani = lgca.live_animate_config(interval=500, grid=True)
# ani = lgca.animate_flux()
# ani = lgca.animate_density()
lgca.plot_flux(cbar=0)
plt.gca().axis('off')
plt.tight_layout()
# plt.savefig('alignment_art.svg')
plt.show()

