from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

"""INITIAL STATE"""
nodes = np.array([[0,0],[0,0],[0,0],[0,0],[0,1],[1,0],[3,0],[0,0]])

lgca2 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='dd_alignment', nodes=nodes, beta=2.0)
#lgca2 = get_lgca(density=0.1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', nodes=nodes, beta=2.0)


"""TIMESTEPPING"""
lgca2.print_nodes()
lgca2.timeevo(timesteps=5, record=True)
lgca2.print_nodes()
ani = lgca2.plot_density()
ani2 = lgca2.plot_flux()
plt.show()