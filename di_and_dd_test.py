from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

"""INITIAL STATE"""

nodes = np.array([[0,0],[1,0],[0,2],[0,0],[0,1],[2,0],[3,0],[0,1],[0,2],[0,0],[0,1],[1,0],[1,0],[0,0]])

lgca2 = get_lgca(density=3, ve=False, geometry='hex', bc='refl', interaction='dd_alignment', nodes=None, beta=3)

#lgca2 = get_lgca(density=1, ve=False, geometry='lin', bc='refl', interaction='di_alignment', nodes=None, beta=0)





"""TIMESTEPPING"""
lgca2.print_nodes()
lgca2.timeevo(timesteps=35, record=True)
lgca2.print_nodes()

#ani = lgca2.plot_density()
#ani2 = lgca2.plot_flux()




ani = lgca2.animate_density()
ani2 = lgca2.animate_flow()
ani3 = lgca2.animate_flux()

#ali = lgca2.calc_mean_alignment()  is missing 2d
#ali = lgca2.calc_polar_alignment_parameter()  #works fine - how to get it at every timestep



plt.show()

