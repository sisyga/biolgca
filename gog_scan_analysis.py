import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from itertools import product
from multiprocess import PATH
import pickle as pkl
from copy import deepcopy as copy

parameters = np.load(PATH+'params.npz', allow_pickle=True)

constparams = parameters['constparams'].item()
r_ds = parameters['r_ds']
thetas = parameters['thetas']
# data = np.load(PATH+'n_pr.npy', allow_pickle=True)
stepsize = 2
# create a grid of figures with size of 'data'
fig, axes = plt.subplots(len(r_ds)//stepsize, len(thetas)//stepsize, figsize=(10, 10), sharex=True, sharey=False)
lgca = get_lgca(**constparams)
# iterate over the grid
for ax, index in zip(axes.flat, product(np.arange(1, len(r_ds), stepsize), np.arange(1, len(thetas), stepsize))):
    with open(PATH+'data{}.pkl'.format(index+(0,0,)), 'rb') as f:
        d = pkl.load(f)
    # lgca.nodes_t = d['nodes']
    # lgca.props['kappa'] = d['kappa']
    # plot the data
    # activate current axis
    plt.sca(ax)
    plt.hist(d['kappa'][d['nodes_t'].sum()], density=True)  # not really nodes_t, naming error! it's just the nodes of the last time step
    # plot title as parameters
    # ax.set_title(fr'$r_d$={d["lgca_params"]["r_d"]:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    ax.set_title(fr'$\bar\rho$={(1-d["lgca_params"]["r_d"])/2:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    # lgca.plot_prop_spatial(propname='kappa', cbar=1, cbarlabel=r'$\kappa$')
    # remove axis labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel('')
    # ax.set_ylabel('')

plt.show()

nodes_t = np.empty((constparams['tmax'], constparams['l'], 3), dtype=object)
lgca.props = {}
for ax, index in zip(axes.flat, product(np.arange(1, len(r_ds), stepsize), np.arange(1, len(thetas), stepsize))):
    with open(PATH+'data{}.pkl'.format(index+(0,0,)), 'rb') as f:
        d = pkl.load(f)

    nodes_t[:] = d['nodes_t'][None, 1:-1, :]
    lgca.nodes_t = nodes_t
    lgca.props['kappa'] = d['kappa']
    lgca.mean_prop_t = lgca.calc_prop_mean_spatiotemp()
    # plot the data
    # activate current axis
    plt.sca(ax)
    # plt.hist(d['kappa'][d['nodes_t'].sum()], density=True)  # not really nodes_t, naming error! it's just the nodes of the last time step
    # plot title as parameters
    # ax.set_title(fr'$r_d$={d["lgca_params"]["r_d"]:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    ax.set_title(fr'$\bar\rho$={(1-d["lgca_params"]["r_d"])/2:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    lgca.plot_prop_spatial(propname='kappa', cbar=False, cbarlabel=r'$\kappa$')
    # remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.show()







