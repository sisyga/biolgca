from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

"""
Script to verify density for a random initial state
Mean density plotted against number of lgcas and lattice size of 100 lgcas
"""

"""TIMESTEPPING"""
mean_densities = []
steps = [1, 5, 10, 50, 100, 200, 500]
for st in steps:
    densities = []
    for i in range (st):
        lgca1 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', dims=100, beta=2.0)
        densities.append(lgca1.eff_dens)
    mean_densities.append(np.mean(densities))
#ani = lgca2.plot_density()
#ani2 = lgca2.plot_flux()
#plt.show()
fig, axes = plt.subplots(nrows=2, sharey=True)
fig.suptitle("Effective density for random initialisation in dependence on number of lgcas and lattice size")
axes[0].set_xlabel("Number of lgcas")
axes[0].set_ylabel("Mean effective density")
axes[0].plot(steps, mean_densities)
axes[0].grid()
axes[0].legend()

mean_densities = []
steps = [5, 10, 50, 100, 200, 500, 750, 1000]
for st in steps:
    densities = []
    for i in range (100):
        lgca1 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', dims=st, beta=2.0)
        densities.append(lgca1.eff_dens)
    mean_densities.append(np.mean(densities))
#ani = lgca2.plot_density()
#ani2 = lgca2.plot_flux()
#plt.show()
axes[1].set_xlabel("Number of lattice sites")
axes[1].set_ylabel("Effective density, mean over 100 realizations")
axes[1].plot(steps, mean_densities)
plt.legend()
plt.grid()
plt.show()