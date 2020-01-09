from lgca import get_lgca
from lgca.interactions import leup_test
import numpy as np
import matplotlib.pyplot as plt

nodes = np.zeros((50, 50, 12), dtype=bool)
nodes[25, 25, -6:] = 1
lgca = get_lgca(interaction='alignment', beta=160, restchannels=6, density=0.1)#, nodes=nodes)
lgca.interaction = leup_test
lgca.r_b = 0.05
lgca.r_d = 0.01
lgca.timeevo(2000, record=True)
ani = lgca.animate_density(channels=slice(6, None), repeat=False, vmax=6, cbarlabel='Resting cells')
# ani = lgca.live_animate_density(channels=slice(-1))  # show everything
# ani = lgca.live_animate_density(channels=slice(6, None
#                                                ), vmax=6)  # show only rest channels
ani.save('lgca_pattern_with_growth.mp4')
plt.show()
# lgca.timeevo(100, record=True)
# lgca.plot_density(lgca.nodes_t[-1, ..., lgca.velocitychannels:].sum(-1), vmax=lgca.restchannels)
# plt.title('Resting cells')
# plt.figure()
# lgca.plot_density(lgca.nodes_t[-1, ..., :lgca.velocitychannels].sum(-1), vmax=lgca.velocitychannels)
# plt.title('Moving cells')
# plt.show()