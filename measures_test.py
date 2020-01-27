from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

"""INITIAL STATE"""
nodes = np.array([[2,0],[1,1],[5,0]])

lgca2 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='dd_alignment', nodes=nodes, beta=2.0)
#lgca2 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', nodes=nodes, beta=2.0)


"""TIMESTEPPING"""
print(lgca2.calc_mean_alignment())
lgca2.print_nodes()
lgca2.timeevo(timesteps=1, record=True)
#lgca2.print_nodes()
ani = lgca2.plot_density()
ani2 = lgca2.plot_flux()
plt.show()