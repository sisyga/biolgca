import numpy as np
from matplotlib import pyplot as plt
from lgca import get_lgca
plt.style.use('seaborn-paper')
#%%

def kc(x):
    return np.arccos(0.25 / x)

x = np.linspace(1, 0.25, num=100, endpoint=False)

k = kc(x)
lam = 2 * np.pi / k
np.nan_to_num(lam, copy=False)
textwidth = 4.792
figsize = textwidth, textwidth/2
fig, ax = plt.subplots(ncols=2, figsize=figsize)


restchannels = 2
lgca = get_lgca(geometry='1d', interaction='aggregation', beta=100, density=0, bc='pbc',
                restchannels=restchannels, dims=50)
lgca.nodes[lgca.nonborder, -1] = 1
# lgca.timeevo(100)
plt.sca(ax[1])
lgca.timeevo(lgca.dims[0])
lgca.plot_density(cmap='viridis', figsize=figsize, colorbarwidth=0.1)
plt.vlines(np.arange(1, 50, 2 * np.pi / kc(100)), -1, 51, colors='orange', lw=1., ls='--',
           label='Mean-field prediction')
plt.legend(frameon=False)


plt.sca(ax[0])
plt.plot(x, lam)
plt.hlines(4, 0, 2, ls='--', colors='gray')
plt.vlines(0.25, 0, lam.max(), ls='--', colors='gray')
plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('$\\lambda_c$')
plt.xlabel('$\\beta\\bar\\rho$')
# fig.set_size_inches((textwidth, textwidth/3.2))
plt.tight_layout(pad=0)
plt.show()