from lgca import get_lgca
from lgca.interactions import leup_test
import numpy as np
import matplotlib.pyplot as plt

nodes = np.zeros((50, 50, 12), dtype=bool)
nodes[25, 25, -6:] = 1
lgca = get_lgca(interaction='alignment', beta=4, restchannels=6, density=0.02)#, nodes=nodes)
lgca.interaction = leup_test
lgca.r_b = 0.1
lgca.r_d = 0.01
ani = lgca.live_animate_density()
plt.show()
lgca.timeevo(5, record=True)
lgca.plot_density(lgca.nodes_t[-1, ..., lgca.velocitychannels:].sum(-1), vmax=lgca.restchannels)
plt.title('Resting cells')
plt.figure()
lgca.plot_density(lgca.nodes_t[-1, ..., :lgca.velocitychannels].sum(-1), vmax=lgca.velocitychannels)
plt.title('Moving cells')

plt.show()