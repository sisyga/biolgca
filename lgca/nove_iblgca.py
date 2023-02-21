import numpy as np
from lgca import get_lgca
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lx = 21
    ly = lx
    restchannels = 1
    capacity = 50
    nodes = np.zeros((lx, ly, 6 + restchannels))
    nodes[lx//2, ly//2] = capacity / 5
    lgca = get_lgca(ve=False, ib=True, geometry='hx', restchannels=restchannels, lx=lx, ly=ly, bc='rbc', nodes=nodes,
                         interaction='go_or_grow', theta=.5, density=.01, kappa=-4, capacity=capacity, r_d=0.08)
    lgca.timeevo(100, record=True)
    # print(lgca.cell_density.sum())
    # lgca.plot_prop_spatial(propname='r_b', cbarlabel='$r_b$')
    # print(lgca.cell_density[lgca.nonborder])
    # ani = lgca.animate_flow(interval=50)
    # ani = lgca.animate_flux(interval=100)
    # ani = lgca.animate_density()
    # ani = lgca.live_animate_flux()
    ani = lgca.animate_config()
    # lgca.plot_density()
    # ani = lgca.plot_prop_spatial()
    # print(lgca.mean_prop_t['kappa'][-1].min(), lgca.mean_prop_t['kappa'][-1].max())
    # lgca.plot_config(grid=True)
    plt.show()
