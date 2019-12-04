#import matplotlib
import matplotlib.pyplot as plt
from lgca import get_lgca
import numpy as np

#matplotlib.use('Qt5Agg')
plt.interactive(False)
lgca = get_lgca(geometry='square')
#lgca.plot_density()
#lgca.plot_flux()
#lgca.plot_flow()
#plt.show()
#plt.show() is needed to display the plots after the function call -> alternatively assign result to variable

nodes = np.zeros((2, 2, 4))
#first dimension: x
#2nd dim: y
#3rd dim: 0-4 directions, 5+ rest channels
# 0 right
# 1 up
# 2 left
# 3 down
nodes[0, 0, :] = 1
nodes[1, 1, 4:] = 1
nodes[0, 1, :4] = 1
nodes[1, 0, 2] = 1
print(nodes)
lgca2 = get_lgca(geometry='square', density=0.25, nodes=nodes)
#lgca2.plot_config()


lgca3 = get_lgca(geometry='hex', interaction='aggregation', dims=(10, 10), restchannels=0)
lgca3.timeevo(record=True, timesteps=100)
#lgca3.animate_density(interval=100)
#lgca3.animate_flux(interval=100)
#lgca3.animate_flow()
#lgca3.animate_config()
ani = lgca3.live_animate_density()
plt.show()
#how do I progress the animation? -> I assign the output of the function to a variable, then it works