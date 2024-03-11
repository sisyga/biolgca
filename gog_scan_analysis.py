import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from itertools import product
from multiprocess import PATH
import pickle as pkl
from copy import deepcopy as copy
PATH = '.\\data\\gog\\nonlocaldensity_10reps\\'
parameters = np.load(PATH+'params.npz', allow_pickle=True)

constparams = parameters['constparams'].item()
r_ds = parameters['r_ds']
thetas = parameters['thetas']
reps = 10
# data = np.load(PATH+'n_pr.npy', allow_pickle=True)
stepsize = 1
lgca = get_lgca(**constparams)
# create a grid of figures with size of 'data'
fig, axes = plt.subplots((len(r_ds))//stepsize, (len(thetas))//stepsize, figsize=(10, 10), sharex=True, sharey=True)

# iterate over the grid
# print(r_ds.shape, thetas.shape, len(r_ds)//stepsize, len(thetas)//stepsize, len(axes.flat))
for ax, index in zip(axes.flat, product(np.arange(0, len(r_ds), stepsize), np.arange(0, len(thetas), stepsize))):
    mean_props = []
    for i in range(reps):
        p = PATH+'data{}.pkl'.format(index+(i, 0,))
        with open(p, 'rb') as f:
            d = pkl.load(f)
            mean_props.append(lgca.calc_prop_mean(d['nodes_t'], propname='kappa', props={'kappa': d['kappa']}))

    len(mean_props)
    mean_props = np.ma.array(mean_props)
    mean_prop = np.mean(mean_props, axis=0)
    std_prop = np.std(mean_props, axis=0)  # calculate the standard deviation of the mean
    n_sample = np.sum(~mean_props.mask, axis=0)  # calculate the number of samples
    std_mean = std_prop / np.sqrt(n_sample)  # calculate the standard error of the mean
    # plot the data
    # activate current axis
    plt.sca(ax)

    # ax.plot(np.ma.array(mean_props).T, color='k', alpha=0.5, lw=.5)
    plt.plot([0, len(mean_prop)], [0, 0], 'k--')
    ax.plot(mean_prop, lw=1)
    # fill the area between the mean and the standard deviation
    ax.fill_between(np.arange(len(mean_prop)), mean_prop-std_prop, mean_prop+std_prop, alpha=.5)
    # ax.errorbar(np.arange(len(mean_prop)), mean_prop, yerr=std_mean, capsize=2)
    # ax.hist(d['kappa'][d['nodes_t'].sum()], density=True, log=True)  # not really nodes_t, naming error! it's just the nodes of the last time step
    # plot title as parameters
    # ax.set_title(fr'$r_d$={d["lgca_params"]["r_d"]:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    # ax.set_title(fr'$\bar\rho$={(1-d["lgca_params"]["r_d"])/2:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    # lgca.plot_prop_spatial(propname='kappa', cbar=1, cbarlabel=r'$\kappa$')
    # remove axis labels
    ax.set_ylim(-12, 12)
    ax.set_xticks([0, constparams['l']])
    ax.set_xticklabels([0, '$L$'])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel('')
    # ax.set_ylabel('')
# set sup labels
fig.supxlabel(r'Space $x$')
fig.supylabel(r'Average switch parameter $\langle \kappa \rangle$')
# set title for first row to corresponding theta value
for ax, t in zip(axes[0, :], thetas[::stepsize]):
    ax.set_title(fr'$\theta={t:.1f}$')

# set y label for first column to corresponding r_d value
for ax, r in zip(axes[:, 0], r_ds[::stepsize]):
    ax.set_ylabel(fr'$\delta={r:.2f}$')
fig.tight_layout()
# plt.show()

fig, axes = plt.subplots((len(r_ds))//stepsize, (len(thetas))//stepsize, figsize=(10, 10), sharex=True, sharey=True)

nodes_t = np.empty((constparams['tmax'], constparams['l'], 3), dtype=object)
lgca.props = {}
for ax, index in zip(axes.flat, product(np.arange(0, len(r_ds), stepsize), np.arange(0, len(thetas), stepsize))):
    migrating_cells = []
    resting_cells = []
    for i in range(reps):
        p = PATH+'data{}.pkl'.format(index+(i, 0,))
        with open(p, 'rb') as f:
            d = pkl.load(f)

        lgca.nodes = d['nodes_t']
        lgca.update_dynamic_fields()
        m = lgca.channel_pop[:, :lgca.velocitychannels].sum(-1)
        migrating_cells.append(m)
        resting_cells.append(lgca.cell_density - m)

    mean_migrating_cells = np.mean(migrating_cells, axis=0)
    std_migration = np.std(migrating_cells, axis=0) / np.sqrt(reps)
    mean_resting_cells = np.mean(resting_cells, axis=0)
    std_resting = np.std(resting_cells, axis=0) / np.sqrt(reps)
    # nodes_t[:] = d['nodes_t'][None, 1:-1, :]
    # lgca.nodes_t = nodes_t
    # lgca.props['kappa'] = d['kappa']
    # lgca.mean_prop_t = lgca.calc_prop_mean_spatiotemp()
    # plot the data
    # activate current axis
    plt.sca(ax)
    ax.plot(mean_resting_cells, label='resting')
    ax.fill_between(np.arange(len(mean_resting_cells)), mean_resting_cells-std_resting, mean_resting_cells+std_resting, alpha=.5)
    ax.plot(mean_migrating_cells, label='migrating')
    ax.fill_between(np.arange(len(mean_migrating_cells)), mean_migrating_cells-std_migration, mean_migrating_cells+std_migration, alpha=.5)
    # plt.hist(d['kappa'][d['nodes_t'].sum()], density=True)  # not really nodes_t, naming error! it's just the nodes of the last time step
    # plot title as parameters
    # ax.set_title(fr'$r_d$={d["lgca_params"]["r_d"]:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    # ax.set_title(fr'$\bar\rho$={(1-d["lgca_params"]["r_d"])/2:.2f}, $\theta$={d["lgca_params"]["theta"]:.2f}')
    # lgca.plot_prop_spatial(propname='kappa', cbar=False, cbarlabel=r'$\kappa$')
    # remove axis labels
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 100])
    ax.set_yticklabels([0, '$K$'])
    # set xticks and ticklabels to 0 and L
    ax.set_xticks([0, constparams['l']])
    ax.set_xticklabels([0, '$L$'])


# axes[0, 0].legend()
fig.supylabel(r'Cell number')
fig.supxlabel('Space $x$')
for ax, t in zip(axes[0, :], thetas[::stepsize]):
    ax.set_title(fr'$\theta={t:.1f}$')

# set y label for first column to corresponding r_d value
for ax, r in zip(axes[:, 0], r_ds[::stepsize]):
    ax.set_ylabel(fr'$\delta={r:.2f}$')
fig.tight_layout()
plt.show()
