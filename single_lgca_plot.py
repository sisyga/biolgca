from lgca import get_lgca
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np
#mpl.use('Agg')
"""give it a name before you run it!"""

"""INITIAL STATE"""
density = 0.1
mode = "dd"
mode_num="100_TALK_10"
beta = 0.1
dims = 70
timesteps = 100

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


""" #for examining and saving:
/usr/bin/python3.6 /snap/pycharm-community/175/plugins/python-ce/helpers/pydev/pydevconsole.py --mode=client --port=34267
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/bianca/repos/biolgca'])
Python 3.6.8 (default, Apr  9 2019, 04:59:38) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.5.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.5.0
Python 3.6.8 (default, Apr  9 2019, 04:59:38) 
[GCC 8.3.0] on linux
runfile('/home/bianca/repos/biolgca/single_lgca_plot.py', wdir='/home/bianca/repos/biolgca')
Backend TkAgg is interactive backend. Turning interactive mode on.
Density: 0.6
{'interaction': 'dd_alignment', 'beta': 0.5, 'no_ext_moore': True, 've': False}
Density: 0.6
{'interaction': 'dd_alignment', 'beta': 0.5, 've': False}
/home/bianca/repos/biolgca/lgca/lgca_1d.py:392: RuntimeWarning: divide by zero encountered in log
  a = np.where(rel_freq > 0, np.log(rel_freq), 0)
nodes = lgca_moore.nodes[lgca_moore.nonborder]
lgca_moore_long = get_lgca(density=density, ve=False, geometry='lin', bc='periodic', interaction=(mode+'_alignment'), nodes=nodes, beta=beta, no_ext_moore=True)
Density: 0.6
{'interaction': 'dd_alignment', 'beta': 0.5, 'no_ext_moore': True, 've': False}
lgca_moore_long.timeevo(timesteps=100000, record=True, recordnove=True, showprogress=True)
Progress: [####################] 100% Done...
lgca_moore_long.calc_mean_alignment()
Out[13]: -0.6533333333333333
lgca_moore_long.calc_polar_alignment_parameter()
Out[14]: 0.013333333333333334
lgca_moore_long.plot_density()
Out[15]: <matplotlib.image.AxesImage at 0x7efcd9ff03c8>
import pandas as pd
pd.to_pickle(lgca_moore_long, "./saved_lgcas/" + savestr + "_" + "moore_long.pkl")
plt.plot(lgca_moore_long.polAlParam_t)
Out[18]: [<matplotlib.lines.Line2D at 0x7efcd9b17eb8>]
plt.plot(lgca_moore_long.meanAlign_t)
Out[19]: [<matplotlib.lines.Line2D at 0x7efcda90a160>]
"""