import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca

"""
# fixed 2D square colour plotting for high particle numbers > number of colours in colourmap
# fixed 2D square colourbar for maximal density of 0 and 1 particle
dens = np.arange(0,50,2).reshape((5,5))
dens[3,3] = 400
#dens = np.ones((5,5))
print(dens)
lgca = get_lgca(geometry='lin', ve=False, dims=10, interaction='dd_alignment', kappa=-4, theta=0.5)
lgca.plot_density(density_t=dens)
plt.show()
"""

"""
# fixed entropy and generalised to any dimensions
dens = np.ones((5))
dens[1]=3
lgca = get_lgca(geometry='lin', ve=False, dims=3, interaction='go_or_rest', kappa=-4, theta=0.5, restchannels=1)
lgca.cell_density = dens
print(lgca.calc_entropy(base=np.exp(1)))
"""

"""
# polar alignment parameter and mean alignment
one = np.array([1,1,0,0])
two = np.array([0,0,1,0])
three = np.array([0,0,0,1])
four = np.array([1,0,0,1])
five = np.array([1,0,0,0])
six = np.array([0,0,0,1])
line1 = np.array([one,two])
line2 = np.array([three,four])
line3 = np.array([five,six])
nodes = np.array([line1,line2,line3])
#print(nodes)
#nodes [0,1,1] = 3
lgca = get_lgca(geometry='square', ve=False, nodes=nodes, interaction='dd_alignment', kappa=-4, theta=0.5, restchannels=0) #dims=3, , density=0.5
#lgca.print_nodes()
print(lgca.calc_polar_alignment_parameter())
print(lgca.calc_mean_alignment())
"""

lgca = get_lgca(geometry='lin', ve=False, dims=20, interaction='go_or_rest', kappa=-4, theta=0.5, restchannels=1, density=0.5)
lgca.timeevo(50)
lgca.plot_density()
plt.show()
lgca.plot_density(offset_t=10, offset_x=10)
plt.show()