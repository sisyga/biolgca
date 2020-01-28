from lgca import get_lgca
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""give it a name before you run it!"""

"""INITIAL STATE"""
density = 0.7
mode = "dd"
mode_num="110_2"
beta = 2
dims = 70
timesteps = 500

lgca_moore = get_lgca(density=density, ve=False, geometry='lin', bc='periodic', interaction=(mode+'_alignment'), beta=beta, dims=dims, no_ext_moore=True)
nodes = lgca_moore.nodes[lgca_moore.nonborder]
lgca_moore_ext = get_lgca(density=density, ve=False, geometry='lin', bc='periodic', interaction=(mode+'_alignment'), nodes=nodes, beta=beta)

savestr = mode_num + "_" + str(dims) + "_" + '{0:.6f}'.format(density) + "_" + '{0:.6f}'.format(\
            beta) + "_" + str(timesteps)

"""TIMESTEPPING"""
lgca_moore.timeevo(timesteps=timesteps, record=True, recordnove=True, showprogress=False)
ani = lgca_moore.plot_density()
plt.savefig('./images/' + savestr + "_moore_dens.png")
ani2 = lgca_moore.plot_flux()
plt.savefig('./images/' + savestr + "_moore_flux.png")
lgca_moore_ext.timeevo(timesteps=timesteps, record=True, recordnove=True, showprogress=False)
ani = lgca_moore_ext.plot_density()
plt.savefig('./images/' + savestr + "_mooreExt_dens.png")
ani2 = lgca_moore_ext.plot_flux()
plt.savefig('./images/' + savestr + "_mooreExt_flux.png")

pd.to_pickle(lgca_moore, "./saved_lgcas/" + savestr + "_" + "moore.pkl")
pd.to_pickle(lgca_moore_ext, "./saved_lgcas/" + savestr + "_" + "mooreExt.pkl")
#read with:
#densities_new = pd.read_pickle("./pickles/" + savestr + ".pkl")
#plt.show()

#plt.figure(fig_norment.number)
#fig_ent = plt.figure("Entropy")