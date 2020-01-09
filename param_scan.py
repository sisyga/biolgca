from lgca import get_lgca
from lgca.interactions import leup_test
import numpy as np
import matplotlib.pyplot as plt

path = '/home/simon/Dokumente/projects/leup_go_or_grow/param_scan/'

lgca = get_lgca(interaction='alignment', beta=0, restchannels=6, density=0.2)#, nodes=nodes)
lgca.interaction = leup_test
lgca.r_b = 0.
lgca.r_d = 0.

runs = 10
ld = 20
lb = 20
tmax = 50
dims = lgca.dims



densities = np.linspace(0.5, 0, ld, endpoint=False)[::-1]
betas = np.linspace(0, 10, lb, endpoint=False)


np.savez(path + 'params.npz', densities=densities, betas=betas, runs=runs, dims=dims, tmax=tmax)

for beta in betas:
    lgca.beta = beta
    for density in densities:
        nodes_nt = np.empty((runs, tmax + 1) + dims + (lgca.K,), dtype=bool)
        for n in range(runs):
            lgca.random_reset(density=density)
            lgca.timeevo(50, recorddens=False, showprogress=False)
            lgca.timeevo(tmax, showprogress=False, recorddens=False, record=True)
            nodes_nt[n] = lgca.nodes_t

        np.save(path + 'beta_{}_dens_{}.npy'.format(beta, density), nodes_nt)




