import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca

#"""
# fixed 2D square colour plotting for high particle numbers > number of colours in colourmap
# fixed 2D square colourbar for maximal density of 0 and 1 particle
dens = np.arange(0,50,2).reshape((5,5))
dens[3,3] = 400
#dens = np.ones((5,5))
print(dens)
lgca = get_lgca(geometry='lin', ve=False, dims=10, interaction='dd_alignment', kappa=-4, theta=0.5)
lgca.plot_density(density_t=dens)
plt.show()
#"""

