from lgca import get_lgca
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r_d=0.1
r_b=0.6
kappa = 4.
theta = 0.25
density=0.5
capacity=40
dims=50

# from simulation script with master branch:
#lgca = get_lgca(interaction='go_or_grow', bc='periodic', geometry=geom, restchannels=restchannels, r_d=r_d, r_b=r_b,
#                kappa=kappa, theta=theta, nodes=nodes)
#lgca.timeevo(timesteps=timesteps, recorddens=True, showprogress=False)

# test go or grow
#lgca = get_lgca(interaction='go_or_grow', ve=False, bc='periodic', geometry='lin', density=density, restchannels=1, r_d=r_d, r_b=r_b, kappa=kappa, theta=theta, nodes=None, dims=dims, capacity=capacity)

# test go or rest
lgca = get_lgca(interaction='go_or_rest', ve=False, bc='periodic', geometry='lin', density=density, hom=True, restchannels=1, kappa=kappa, theta=theta, nodes=None, dims=dims, capacity=capacity)
#print(lgca.nodes.sum(-1))
#lgca.print_nodes()

lgca.timeevo(timesteps=50, record=True, showprogress=True, recordpertype=True)

#lgca1 = get_lgca(interaction='dd', ve=False, bc='pbc', density=0.5, geometry='hex', nodes=None, beta=0.2, dims=(25, 25))
#lgca1.timeevo(timesteps=150, record=True)

#pd.to_pickle(lgca,'./pickles/1D_test_goorrest_long2_all_stab_switchset_1.pkl')
#lgca.print_nodes()
#test = np.random.randint(83,size=(20,20))
#print(test.max())
lgca.plot_density()
plt.title("Density")
maxval = lgca.dens_t.max()
lgca.plot_density(density_t=lgca.velcells_t, absolute_max=maxval)
plt.title("Migrating cells")
lgca.plot_density(density_t=lgca.restcells_t, absolute_max=maxval)
plt.title("Resting cells")
lgca.plot_density(density_t=lgca.dens_t[25:,30:], offset_t=25, offset_x=30)
plt.title("Slice")

print("Hey!")

#lgca.plot_flux()
#print(lgca.dens_t.sum(-1))



#ani = lgca1.animate_flow()
#ani2 = lgca1.animate_flux()
#ani3 = lgca1.animate_density()
"""
fentropy = lgca1.calc_entropy()
fnormentropy = lgca1.calc_normalized_entropy()
fpalignment = lgca1.calc_polar_alignment_parameter()
fmpalignment = lgca1.calc_mean_alignment()

print("final entropies")
print(fentropy)
print("final total alignments")
print(fpalignment)
print("final local alignments")
print(fmpalignment)
print("final normalized entropies")
print(fnormentropy)
"""
plt.show()