from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

#nodes = np.array([[0,0],[2,0],[0,0],[4,0],[1,0],[0,0],[0,0],[0,0],[0,0],[30,0],[14,0],[0,0]])
#nodes = np.array([[0,0],[0,0],[0,0],[0,0],[0,1],[1,0],[3,0],[0,0]])

"""Setup"""
beta = 5
density = 0.2
dims = 100
timesteps = 100

"""INITIAL STATE"""
lgca = get_lgca(interaction='dd_alignment', ve=False, bc='periodic', density=density, geometry='lin', dims=dims, beta=beta)

"""TIMESTEPPING"""
lgca.timeevo(timesteps=timesteps, record=True)
ani = lgca.plot_density()
#ani2 = lgca.plot_flux_fancy()
#ani3 = lgca.plot_flux()
plt.show()

#Todo:
# plot title reflect parameters - done
# flux plot title, too
# and for Simon's plots also
# list of parameters - done
# dd/di for ve? what did Simon implement?
# way to store initial conditions permanently for repeating experiments - done
# fancy flux plotting
# how do I characterize clusters?