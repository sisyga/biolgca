import pickle

from lgca.lgca_1d import *

with open('lgca_1d_channels.pkl', 'rb') as config_dictionary_file:
    # Step 3
    lgcalist = pickle.load(config_dictionary_file)
"""
lgca2.plot_prop_timecourse(propname='kappa')
lgca2.plot_prop_timecourse(propname='theta')
plt.show()
"""
size = 4
for i in np.arange(0, size - 1):
    shape = lgcalist[i].nodes_t.shape
    # print(shape)
    KAPPAS = np.asarray(lgcalist[i].props_t[-1]['kappa'][:])
    THETAS = np.asarray(lgcalist[i].props_t[-1]['theta'][:])

    plt.subplot(size, 3, i * 3 + 1)
    plt.hist2d(THETAS, KAPPAS, range=((-2, 3), (-16, 16)), bins=25)
    plt.colorbar()

    a = lgcalist[i].nodes[lgcalist[i].nodes > 0]
    plt.subplot(size, 3, i * 3 + 2)
    plt.hist2d(THETAS[a], KAPPAS[a], bins=25)
    plt.colorbar()

    plt.subplot(size, 3, i * 3 + 2)
    plt.plot(np.arange(len(lgcalist[i].dens_t)), np.array(lgcalist[i].dens_t).mean(-1))
plt.show()

"""
#lgca2.plot_prop_spatial()
#plt.show()
#
plt.figure
plt.plot(lgca2.props_t[1]['theta'][:])
plt.show()
plt.figure
plt.plot(lgca2.props_t[100]['theta'][lgca2.nodes_t])

#plt.ylabel('$\\kappa$')
#plt.savefig('gng_mean_alpha.png', dpi=600)
plt.show()
"""
