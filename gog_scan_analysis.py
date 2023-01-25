import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from multiprocess import PATH

parameters = np.load(PATH+'params.npz', allow_pickle=True)

constparams = parameters['constparams'].item()
r_ds = parameters['r_ds']
thetas = parameters['thetas']
data = np.load(PATH+'n_pr.npy', allow_pickle=True)
# print(data)
# create a grid of figures with size of 'data'
fig, axes = plt.subplots(data.shape[0], data.shape[1], figsize=(10, 10), sharex=True, sharey=True)
lgca = get_lgca(**constparams)
# iterate over the grid
for ax, d in zip(axes.flat, data[..., 0].flat):
    lgca.nodes_t = d['nodes_t']
    lgca.props['kappa'] = d['kappa']
    # plot the data
    # activate current axis
    plt.sca(ax)
    # plt.hist(d['kappa'][d['nodes_t'][-1].sum()])
    # plot title as parameters
    ax.set_title(fr'r_d={d["interaction_params"]["r_d"]:.2f}, $\theta$={d["interaction_params"]["theta"]:.2f}')
    lgca.plot_prop_spatial(propname='kappa', cbar=1, cbarlabel=r'$\kappa$')
    # remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.show()







