from lgca import get_lgca
import numpy as np
from matplotlib import pyplot as plt


# geometry
geom = 'hx'
restchannels = 1
l = 6
dims = l, l
# model parameters


kappa = 0.
theta = 0.5

# simulation parameters
dens = 0.1 #starting condition
beta = 1.2
# time = 100

# setup =
from lgca import get_lgca
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
a_max = 1.
a_min = 0.
na = 401
alpha, dalpha = np.linspace(a_min, a_max, num=na, retstep=True)
r_d = 0.05
r_b = 0.1
var = 0.01**2
tmax = 5000
Da = var / 2
dens0 = 1 - r_d / r_b
ts = np.linspace(0, tmax, num=101)
K = 100

l = 50
dims= l, l
nodes = np.zeros(dims+(6+restchannels,), dtype=int)
nodes[l//2, l//2, -1] = 100
lgca = get_lgca(ib=True, bc='reflect', interaction='steric_evolution', dims=dims, nodes=nodes, ve=False, geometry='hx',
                r_m=0.01, r_b=0.01, capacity=12, gamma=3)
# print((lgca.props['family'][99]), max(lgca.nodes.sum()))
# test_code = lgca.timeevo(timesteps=tmax, record=True)

# time = timeit.repeat(setup=setup, stmt=test_code, repeat=1, number=1)
# print('exec time = {}'.format(time))

# lgca = get_lgca(interaction='birthdeath', bc='periodic', density=dens, geometry=geom, dims=dims,
                # restchannels=restchannels, ve=0, ib=1 , beta=beta,
                # r_d=r_d, r_b=r_b, kappa=kappa, theta=theta)
lgca.timeevo(500, recordfampop=True, record=True)
# lgca.plot_config()
# lgca.plot_prop_spatial()
# lgca.plot_config(grid=1)
# ani = lgca.animate_config(interval=500, grid=1)
# ani = lgca.live_animate_flux()
# ani = lgca.animate_density()
# lgca.plot_flux(cbar=0)
# plt.gca().axis('off')
# plt.tight_layout()
# plt.savefig('alignment_art.svg')
#
lgca.muller_plot()
plt.show()
lgca.plot_prop_spatial()
plt.show()

