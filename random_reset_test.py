from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

"""INITIAL STATE"""
#nodes = np.array([[0,0],[0,0],[0,0],[0,0],[0,1],[1,0],[3,0],[0,0]])

#lgca2 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='dd_alignment', nodes=nodes, beta=2.0)



"""TIMESTEPPING"""
mean_densities = []
steps = [1, 5, 10, 50, 100, 200, 500]
for st in steps:
    densities = []
    for i in range (st):
        lgca = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', dims=100, beta=2.0)
        densities.append(lgca.eff_dens)
    mean_densities.append(np.mean(densities))
#ani = lgca2.plot_density()
#ani2 = lgca2.plot_flux()
#plt.show()
fig, axes = plt.subplots(nrows=2)
axes[0].plot(steps, mean_densities, label = 'amount of lgcas')
axes[0].grid()
axes[0].legend()

mean_densities = []
steps = [5, 10, 50, 100, 200, 500, 750, 1000]
for st in steps:
    densities = []
    for i in range (100):
        lgca = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', dims=st, beta=2.0)
        densities.append(lgca.eff_dens)
    mean_densities.append(np.mean(densities))
#ani = lgca2.plot_density()
#ani2 = lgca2.plot_flux()
#plt.show()
axes[1].plot(steps, mean_densities, label='dimension')
plt.legend()
plt.grid()
plt.show()