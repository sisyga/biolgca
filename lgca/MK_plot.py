from lgca.lgca_1d import *
import pickle
import statistics as stat

from lgca.lgca_1d import *


def plot01(save_file=[], ALL=[], p_start=0.5, p_end=1, img_x=[-1, 2], img_y=[-16, 16]):
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

    m_k = np.zeros((size_c * size_b * size_d))
    m_t = np.zeros((size_c * size_b * size_d))
    for c in np.arange(0, size_c):
        plt.figure(c)
        for rd in np.arange(0, size_b * size_d):
            i = c * (size_b * size_d) + rd
            KAPPAS = []
            THETAS = []
            for run in range(size_run):

                K = []
                T = []
                try:
                    K = ALL['kappas'][run][i]
                    T = ALL['thetas'][run][i]
                except:
                    K = ALL['data'][run][i].props_t[-1]['kappa']
                    T = ALL['data'][run][i].props_t[-1]['theta']

                KAPPAS.extend(K[int(len(K) * p_start):int(len(K) * p_end)])
                THETAS.extend(T[int(len(T) * p_start):int(len(K) * p_end)])
            m_k[i] = stat.median(KAPPAS)
            m_t[i] = stat.median(THETAS)
            plt.subplot(size_b, size_d, rd + 1)
            plt.hist2d(THETAS, KAPPAS, bins=100, range=(img_x, img_y), )
            if run == size_run - 1:
                plt.plot([-1, 2], [0, 0])
                plt.plot([-1, 2], [-8, -8], ls='--', c=[0.1, 0, 0])
                plt.plot([-1, 2], [8, 8], ls='--', c=[0.1, 0, 0])
                plt.plot([0, 0], [-16, 16], ls='--', c=[0.1, 0, 0])
                plt.plot([1, 1], [-16, 16], ls='--', c=[0.1, 0, 0])
                plt.plot(m_t[i], m_k[i], marker='+', c=[1, 0, 0])
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

    plt.figure(size_c)
    plt.subplot(1, 2, 1)
    plt.plot(m_t)
    plt.subplot(1, 2, 2)
    plt.plot(m_k)
    plt.show()


plot01(save_file='S02_CURRENT.pkl', img_x=[-2, 3])

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
