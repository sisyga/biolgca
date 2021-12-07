from lgca import get_lgca
import numpy as np
import matplotlib.pyplot as plt

def entropy_product(x):
    return x * np.log(x, where=x > 0, out=np.zeros_like(x, dtype=float))

path = '/home/simon/Dokumente/projects/leup_go_or_grow/param_scan/'


params = np.load(path + 'params.npz')

densities = params['densities']
betas = params['betas']
runs = params['runs']
dims = params['dims']

lgca = get_lgca(dims=dims)

lb = len(betas)
ld = len(densities)
s = np.empty((lb, ld))

# nodes_nt = np.load(path + 'beta_{}_dens_{}.npy'.format(betas[-1], densities[-1]))
#
# nodes_t = nodes_nt[0]
# resting = nodes_t[..., 6:].sum(-1)
# ani = lgca.animate_density(density_t=resting, vmax=6)
# plt.show()

for i, beta in enumerate(betas):
    for j, density in enumerate(densities):
        nodes_nt = np.load(path + 'beta_{}_dens_{}.npy'.format(betas[i], densities[j]))
        n_nt = nodes_nt.sum(-1)
        N_nt = n_nt.sum(axis=(-1, -2))
        # n_nt = np.ma.masked_equal(n_nt, 0, copy=False)

        # ng_nt = nodes_nt[..., :6].sum(-1)
        # nr_nt = n_nt - ng_nt
        # pr_nt = nr_nt / n_nt
        # pg_nt = 1 - pr_nt
        # s_nt = -entropy_product(pr_nt) - entropy_product(pg_nt)
        # s_nt.mask = n_nt.mask
        # s_nt *= n_nt
        s_nt = np.sum(-entropy_product(n_nt / N_nt[..., None, None]), axis=(-1, -2))
        # s[i, j] = s_nt.sum() / n_nt.sum()
        # s[i, j] = s_nt.mean()
        smax = np.log(dims[0] * dims[1])
        s[i, j] = (1 - s_nt / smax).mean()

np.save(path + 'entropy_cluster.npy', s)


# lgca.plot_density(n_nt[0, -1], vmax=n_nt[0, -1].max())

# plt.figure()
# lgca.plot_scalarfield(s_nt[0, -1], vmin=0, vmax=np.log(2))
# plt.show()



