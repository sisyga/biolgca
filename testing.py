from lgca import get_lgca
import numpy as np
from matplotlib import pyplot as plt
import timeit
#test
# geometry
geom = 'hx'
restchannels = 1
l = 6
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
setup = """
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

l = 2
dims= l, l
lgca = get_lgca(ib=True, density=dens0*K/3, bc='reflect', interaction='birthdeath', std=sqrt(var), ve=False, capacity=K,
                r_d=r_d, r_b=r_b, a_max=a_max, geometry='hx', dims=dims, restchannels=1)
"""
test_code = """lgca.timeevo(timesteps=tmax, record=True)"""

time = timeit.repeat(setup=setup, stmt=test_code, repeat=1, number=1)
print('exec time = {}'.format(time))

# lgca = get_lgca(interaction='birthdeath', bc='periodic', density=dens, geometry=geom, dims=dims,
                # restchannels=restchannels, ve=0, ib=1 , beta=beta,
                # r_d=r_d, r_b=r_b, kappa=kappa, theta=theta)
# lgca.timeevo(50, record=1)
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
# plt.show()

