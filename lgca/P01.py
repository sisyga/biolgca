import pickle
from lgca.lgca_1d import *
import pickle

from lgca.lgca_1d import *


def plot01(save_file, p_start=0.5, p_end=1):
    try:
        with open(save_file, 'rb') as config_dictionary_file:
            # Step 3
            ALL = pickle.load(config_dictionary_file)
    except:
        print('no file loaded')

    size_b = len(ALL['b'])
    size_d = len(ALL['bd'])
    size_c = len(ALL['rest'])
    size_run = ALL['runs']

    for c in np.arange(0, size_c):
        plt.figure(c)
        for rd in np.arange(0, size_b * size_d):

            KAPPAS = []
            THETAS = []
            for run in range(size_run):
                i = c * (size_b * size_d) + rd
                K = []
                T = []
                try:
                    K = ALL['kappas'][run][i]
                    T = ALL['thetas'][run][i]
                except:
                    K = ALL['data'][run][i].props_t[-1]['kappa']
                    T = ALL['data'][run][i].props_t[-1]['theta']

                KAPPAS.extend(K[int(len(K) * p_start):int(len(K) * p_end)])
                print(len(KAPPAS))
                THETAS.extend(T[int(len(T) * p_start):int(len(K) * p_end)])

            plt.subplot(size_b, size_d, rd + 1)
            # plt.hist2d(THETAS[int(-len(THETAS)/2):],KAPPAS[int(-len(KAPPAS)/2):], range=((-1,2),(-16,16)), bins=25)
            plt.hist2d(THETAS, KAPPAS, bins=100, range=((-1, 2), (-16, 16)), )
            # plt.hist2d(THETAS[-1000:], KAPPAS[-1000:], range=((-2, 3), (-16, 16)), bins=25)
            if run == size_run - 1:
                plt.plot([-1, 2], [0, 0])
                plt.plot([-1, 2], [-8, -8], ls='--', c=[0.1, 0, 0])
                plt.plot([-1, 2], [8, 8], ls='--', c=[0.1, 0, 0])
                plt.plot([0, 0], [-16, 16], ls='--', c=[0.1, 0, 0])
                plt.plot([1, 1], [-16, 16], ls='--', c=[0.1, 0, 0])
                cb = plt.colorbar()
                if run != 0:
                    plt.title(
                        str(ALL['dims']) + ' nodes, ' + str(ALL['t_max']) + ' ts, ' + str(ALL['runs']) + ' runs, ' +
                        str(ALL['rest']) + ' rest, b=' + str(ALL['b'][rd]) + ', d=' + str(
                            round(ALL['bd'][rd] * ALL['b'][rd], 3)))
                    plt.xlabel('Theta')
                    plt.xticks([0, 1])
                    plt.ylabel('Kappa')
                    plt.yticks([-8, 0, 8])
                else:
                    cb.ax.tick_params(labelsize=6)
                    plt.xticks([])
                    plt.yticks([])

                # plt.plot(np.arange(len(lgcalist[i].dens_t)), np.array(lgcalist[i].dens_t).mean(-1))
    plt.show()


"""
lgca2.plot_prop_timecourse(propname='kappa')
lgca2.plot_prop_timecourse(propname='theta')
plt.show()
"""

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

# plot for _channel files
"""
size=4
for i in np.arange(0,size):
    shape=lgcalist[i].nodes_t.shape
    #print(shape)

    KAPPAS = np.asarray(lgcalist[i].props_t[-1]['kappa'])
    THETAS = np.asarray(lgcalist[i].props_t[-1]['theta'])
    #print(KAPPAS)
    plt.subplot(size,3,i*3+1)
    plt.hist2d(THETAS,KAPPAS, range=((-2,3),(-16,16)), bins=25)
    plt.colorbar()

    #a = lgcalist[i].nodes[lgcalist[i].nodes > 0]
    plt.subplot(size,3,i*3+2)
    print(-len(THETAS)/2)
    plt.hist2d(THETAS[int(-len(THETAS)/2):],KAPPAS[int(-len(KAPPAS)/2):], range=((-2,3),(-16,16)), bins=25)
    plt.colorbar()

    plt.subplot(size,3,i*3+3)
    plt.plot(np.arange(len(lgcalist[i].dens_t)), np.array(lgcalist[i].dens_t).mean(-1))

plt.show()
"""
