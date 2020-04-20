from .analysis import *
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

farben = {
    'si':       'gold',
    'gi':       'seagreen',
    'sh':       'red',
    'eve':      'red',
    'hill_25':  'rosybrown',
    'hill_5':   'lawngreen',
    'hill_75':  'aqua',
    'hill_1':   'mediumblue',
    'hill_2':   'sienna',
    'hill_3':   'coral',
    'rich':     'darkmagenta',
    'onenode':  'darkred',
    'onerc':    'orange'
}

def aloha(who):
    print('aloha', who)

def calc_mullplotdata(data, int_length):
    """
    calculate the required data for mullerplot()
    :param data: lgca offsprings
    :param int_length: length of interval
    :return: array of size familynumber*timesteps with number of familymembers per timepoint
    """
    maxfamily = len(data[-1]) - 1
    tend = len(data)

    fig, ax = plt.subplots()
    val = np.zeros((maxfamily, tend))

    for t in range(0, tend):
        maxlab = len(data[t]) - 1   #data unterschiedlich lang!
        for lab in range(0, maxlab):
            val[lab, t] = data[t][lab + 1]
    return val



def mullerplot(data, id=0, save=False, int_length=1):
    """
    create mullerplot-like barstack-plot;
    only use in the case of: decreasing family number (without mutations)
    for simulation which include mutations: mullplot_data.py & muller_create.R
    for now: only 1d-data
    :param data: array of lgca offsprings
    :param int_length: desired length of interval
    :param id: filename for saving
    :param save: saves the plot if true
    """
    tend = len(data)

    maxlab = len(data[-1]) - 1
    if len(data[0]) != len(data[-1]):
        calc_mullplotdata(data, int_length)

    fig, ax = plt.subplots()
    val = np.zeros((maxlab, tend))

    for t in range(0, tend):
        for lab in range(0, maxlab):
            val[lab, t] = data[t][lab + 1]

    if int_length == 1:
        xrange = range(0, tend)
        pop = val
    else:
        int_num = ((tend - 1) // int_length)
        xrange = [0]
        for i in range(int_num):
            xrange.append(i * int_length + 0.5 * int_length)
        if int_num * int_length != tend:
            xrange.append((tend - 1 + int_num * int_length) / 2)
        xrange.append(tend-1)

        acc_val = np.zeros((maxlab, len(xrange) - 2)) + -999
        for i in range(0, int_num):
            for lab in range(0, maxlab):
                acc_val[lab, i] = np.sum(val[lab, i * int_length:1 + (i+1)*int_length])
        if int_num * int_length != tend:
            for lab in range(0, maxlab):
                acc_val[lab, -1] = np.sum(val[lab, int_length * int_num:tend])

        pop = np.zeros((maxlab, len(xrange))) + -777
        pop[:, 0] = val[:, 0]
        pop[:, 1:-1] = acc_val
        pop[:, -1] = val[:, -1]

    popdic = {str(i): pop[i] for i in range(0, maxlab)}
    data = pd.DataFrame(popdic, index=xrange)
    data_perc = data.divide(data.sum(axis=1), axis=0)
    plt.xlabel('timesteps')
    plt.ylabel(' frequency of families')

    # plot
    plt.xlim(0, xrange[-1])
    plt.ylim(0, 1)
    if xrange[-1] <= 15:
        plt.xticks(np.arange(0, xrange[-1], 1))
    elif xrange[-1] <= 100:
        plt.xticks(np.arange(0, xrange[-1], 5))
    elif xrange[-1] >= 10000:
        plt.xticks(np.arange(0, xrange[-1], 2000))
    elif xrange[-1] >= 6000:
        plt.xticks(np.arange(0, xrange[-1], 1000))
    elif xrange[-1] >= 1000:
        plt.xticks(np.arange(0, xrange[-1], 500))
    elif xrange[-1] >= 100:
        plt.xticks(np.arange(0, xrange[-1], 50))

    plt.stackplot(xrange, *[data_perc[str(f)] for f in range(0, maxlab)], \
                  labels=list(range(0, maxlab)))
    if save:
        save_plot(fig, str(id) + '_' + ' mullerplot with intervall=' + str(int_length) + '.jpg')
    plt.show()

def spacetime_plot(nodes_t, labels, tbeg=None, tend=None, save=False, id=0,\
                   figsize=None, figindex=None, cmap='nipy_spectral'):
    """
    create plot with distribution of the cells in the lattice per timepoint
    for now: only 1d-data

    :param nodes_t: lgca.node per timepoint
    :param labels: data from lgca.props['labm] -> family of each cell
    :param tbeg: desired starting time
    :param tend: desired end time
    :param save: saves the plot if true
    :param id: filename for saving
    :param figsize: size of figure
    :param figindex: index of figure
    :param cmap: colormap
    """
    tmax, dim, c = nodes_t.shape
    vc = 2
    rc = c - vc
    print('steps, nodes, rc', tmax, dim, rc)
    if tbeg is None:
        tbeg = 0
    if tend is None:
        tend = tmax
    if figsize is None:
        if tend-tbeg<=100:
            fx = 4.5  #for c == 180
            fy = (tend - tbeg) / 40
            figsize = (fx, fy)
        elif tend-tbeg <= 500:
            fx = 4.5  #for c == 180
            fy = (tend - tbeg) / 55
            figsize = (fx, fy)

    val = np.zeros((tmax, dim*c))

    for t in range(0, tmax):
        for x in range(dim):
            node = nodes_t[t, x]
            # print(node)
            for pos in range(len(node)):
                lab = node[pos]
                # print('lab', lab)
                if pos == 0 or pos == 1:
                    if pos == 0:
                        if lab == 0:
                            val[t, (c-1) + x * c] = None
                        else:
                            val[t,  (c-1) + x * c] = labels[lab]

                    elif pos == 1:
                        if lab == 0:
                            val[t, x * c] = None
                        else:
                            val[t, x * c] = labels[lab]
                else:
                    if lab == 0:
                        val[t, x*c + pos - 1] = None
                    else:
                        val[t, x*c + pos - 1] = labels[lab]
                    # print('stückchen val', val[t, x*c + pos - 1])
        # print('val[t]', val[t])
    # print('val', val)

    fig = plt.figure(num=figindex, figsize=figsize)
    ax = fig.add_subplot(111)
    plot = ax.matshow(val, cmap=cmap)
    ###
    # vmin = 1
    # vmax = max(labels)
    # print('Legende von ', vmin, ' bis ', vmax)
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # plot.set_norm(norm)
    #
    # cbar = fig.colorbar(plot, ax=ax)
    # cbar.ax.get_yaxis().set_ticks([])
    # # for j, lab in enumerate(range(1, vmax+1)):
    # #     cbar.ax.text(3, j+1.075, lab, ha='center', va='center')
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('family', rotation=270)
    ###
    plt.ylabel('timesteps', fontsize=15, labelpad=10) #15
    ax.xaxis.set_label_position('top')
    plt.xlabel('lattice site', fontsize=15, labelpad=10) #, fontsize=12

    # nur "Knotenanfang"
    plt.xlim(-0.5, dim * c - 0.5)

    if dim >= 20:
        x1 = np.arange(0, dim*c, 20*c)
        x2 = np.zeros(len(x1)).astype(int)
        for i in range(0, len(x1)):
            x2[i] = (x1[i]/c) + 1
        ax.set_xticks(x1)
        ax.set_xticklabels(x2, minor=False, fontsize=12)
    elif dim > 1:
        x1 = (np.arange(0, dim * c, c))
        x2 = np.zeros(len(x1)).astype(int)
        for i in range(0, len(x1)):
            x2[i] = (x1[i] / c) + 1
        ax.set_xticks(x1)
        ax.set_xticklabels(x2, minor=False, fontsize=12)
    else:
        plt.xticks((np.arange(0, dim*c, c)+1), fontsize=12)

    plt.ylim(tend-0.5, tbeg-0.5)
    if tend - tbeg > 700:
        plt.yticks(np.arange(tbeg, tend, 100))
    elif tend - tbeg > 100:
        plt.yticks(np.arange(tbeg, tend, 50), fontsize=12)
    elif tend - tbeg <= 100:
        plt.yticks(np.arange(tbeg, tend, 10), fontsize=11)
    if save:
        save_plot(fig, str(id) + '_spacetimeplot_' + str(tbeg) + '-' + str(tend) + '.jpg')
    plt.show()

def plot_sth(data, save=False, id=0, ylabel='index', savename=None):
    """
    plot of variable indices
    :param data: structure {'name1': index_data, 'name2': index_data}
    """
    tend = len(list(data.values())[0])
    x = np.arange(0, tend)
    maxy = 0
    filename = list(data.keys())
    fig, ax = plt.subplots(figsize=(12,4))
    for name in data:
        m = max(data[name])
        if m > maxy:
            maxy = m + 0.1
        if name=='eve':
            plt.plot(x, data[name], farben[name], label=name, linewidth=0.75, linestyle=(0, (1, 10)))
        else:
            plt.plot(x, data[name], farben[name], label=name, linewidth=0.75)
    ax.set(xlabel='timesteps', ylabel=ylabel)
    ax.legend(loc='upper left')
    plt.xlim(0, tend-1)
    if tend >= 10000:
        plt.xticks(np.arange(0, tend, 5000))
    elif tend >= 100:
        plt.xticks(np.arange(0, tend, 50))

    plt.ylim(0, maxy)
    if save:
        if savename is None:
            save_plot(plot=fig, filename=str(id) + '_comparing_' + str(filename) + '.jpg')
        else:
            save_plot(plot=fig, filename=savename + '.jpg')

    plt.show()

def plot_index(index_data, which, save=False, id=0):
    """
    plot desired diversity index
    :param index_data: array of indices per timestep
    :param which: desired y-label
    :param save: saving plot if true
    :param id: filename for saving
    """
    time = len(index_data)
    x = np.arange(0, time, 1)
    y = index_data[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='timestep', ylabel=str(which))
    plt.xlim(0, time-1)

    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    else:
        plt.xticks(np.arange(0, time, 2))
    plt.ylim(0, max(y))

    if save:
        save_plot(fig, str(id) + str(which) + '.jpg')
    plt.show()

# def plot_hillnumbers_together(hill_1, hill_2, hill_3, hill_5, hill_25, hill_75, save=False, id=0):
def plot_hillnumbers_together(hill_2, hill_25, hill_75, save=False, id=0):
    """
    plot hillnumbers of order 1,2,3 together
    :param hill_1: array of hillnumbers 1st order
    :param hill_2: array of hillnumbers 2nd order
    :param hill_3: array of hillnumbers 3rd order
    :param save: saves plot if true
    :param id: filename for saving
    """
    time = len(hill_2)
    x = np.arange(0, time, 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    # plt.plot(x, hill_1, farben['hill_1'], label='order 1', linewidth=1)
    plt.plot(x, hill_2, farben['hill_2'], label='order 2', linewidth=1)
    # plt.plot(x, hill_3, farben['hill_3'], label='order 3', linewidth=1)
    # plt.plot(x, hill_5, farben['hill_5'], label='order 0.5', linewidth=1)
    plt.plot(x, hill_25, farben['hill_25'], label='order 0.25', linewidth=1)
    plt.plot(x, hill_75, farben['hill_75'], label='order 0.75', linewidth=1)

    ax.set(xlabel='timesteps', ylabel='Hillnumbers')
    ax.legend(handlelength=2.5)
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 5000))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    plt.axvline(x=23062, ymax=0.9, linestyle='--')
    plt.axvline(x=36157, ymax=0.9, linestyle='--')
    plt.axvline(x=37562, ymax=0.9, linestyle='--')
    plt.text(23062-250, 7.3, '$k_1$')
    plt.text(36157-250, 7.3, '$k_2$')
    plt.text(37562-250, 7.3, '$k_3$')
    plt.ylim(1, max(hill_25) + 0.5, 10)
    if save:
        save_plot(plot=fig, filename= str(id) + '_comparing hillnumbers_hilfslinien' + '.jpg')
    plt.show()

def plot_entropies_together(gini, shannon, save=False, id=0, simpson=None):
    """
    plot simpson index, gini-simpson index and shannon index together
    :param simpson: array of simpson index
    :param gini: array of ginisimpson index
    :param shannon: array of shannon index
    :param save: saves plot if true
    :param id: filename for saving
    """
    if save is None:
        save = False
    if id is None:
        id = 0
    time = len(gini)
    x = np.arange(0, time, 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(x, shannon, farben['sh'], label='Shannonindex', linewidth=0.75)
    if simpson is not None:
        plt.plot(x, simpson, farben['si'], label='Simpsonindex', linewidth=0.75)
    plt.plot(x, gini, farben['gi'], label='GiniSimpsonindex', linewidth=0.75)

    ax.set(xlabel='timesteps', ylabel='Index')
    ax.legend()
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 5000))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))

    plt.ylim(0, max(shannon) * 1.1)
    if save:
        save_plot(fig, str(id) + '_comparing entropies' + '.jpg')
    plt.show()

def plot_selected_entropies(shannon, hill2, gini, save=False, id=0):
    """
    plt shannonindex, hillnumber 2nd order and ginisimpson index together with different y-axes
    :param shannon: shannonindices
    :param hill2: hillnumbers 2nd order
    :param gini: ginisimpson indices
    :param save: save plot if true
    :param id: filename for saving
    """
    time = len(shannon)
    x = np.arange(0, time, 1)

    fig, host = plt.subplots(figsize=(12, 4))
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.1))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    host.set_xlim(0, time - 1)
    host.set_ylim(bottom=1, top=max(hill2)*1.1)
    par1.set_ylim(bottom=0, top=max(gini)*1.1)
    par2.set_ylim(bottom=0, top=max(shannon)*1.1)

    host.set_xlabel("timesteps")
    host.set_ylabel("Hillnumber")
    par1.set_ylabel("GiniSimpsonindex")
    par2.set_ylabel("Shannonindex")

    p1, = par2.plot(x, shannon, farben['sh'], linewidth=0.75, label="Shannonindex")
    p2, = par1.plot(x, gini, farben['gi'], linewidth=0.75, label="GiniSimpsonindex")
    p3, = host.plot(x, hill2, farben['hill_2'], linewidth=0.75, label="Hillnumber of order 2")
    # p4, = host.plot(x, hill5, farben['hill_5'], linewidth=0.75, label="Hillnumber of order 0.5")



    host.yaxis.label.set_color(p3.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p1.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p1.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines])
    if save:
        filename = str(id) + '_comparing sh, gi, hh' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename, bbox_inches='tight')
    plt.show()


def make_patch_spines_invisible(ax):
    """
    required for plot_selected_entropies
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_popsize(data, save=False, id=0, plotmax=0):
    """
    plot of population size during time
    :param data: lgca.offsprings
    :param save: saves plot if true
    :param id: filename for saving
    """
    time = len(data)
    x = np.arange(0, time, 1)
    size = np.zeros(time)
    for t in range(time):
        size[t] = sum(data[t][1:])
    y = size[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlim(0, time - 1)
    if plotmax != 0:
        plt.ylim(0, plotmax + 10)
    else:
        plt.ylim(0, max(size) * 1.1)
    ax.set(xlabel='timestep', ylabel='total number of living cells')
    if plotmax != 0:
        plt.plot(x, [plotmax]*len(x), 'seagreen')
    if save:
        save_plot(fig, str(id) + '_population size ' + '.jpg')

    plt.show()

def plot_histogram_thom(thom, int_length, save=False, id=0):
    """
    plots histogram of times of homogeneity
    :param thom: array with times of homogeneity
    :param int_length: desired length of interval
    """
    max = thom.max().astype(int)
    l = len(thom)

    #number of intervalls
    ni = (max / int_length + 1).astype(int)
    count = np.zeros(ni+1)

    for entry in thom:
        c = (entry / int_length).astype(int)
        count[c] += 1
    if count.sum() != l:
        print('FEHLER!')

    fig, ax = plt.subplots()
    x = np.arange(0, max + int_length, int_length)
    y = count[(x/int_length).astype(int)]
    int_max = x[(y==y.max())]
    for entry in int_max:
        print('max in intervall [%d, %d]' %(entry, entry + int_length))
    print('with total= ', y.max())

    plt.bar(x+int_length/2, y, width=int_length, color='black', alpha=0.5)
    plt.xlim(0, max + int_length)
    plt.ylim(0, y.max()+1)
    ax.set(xlabel='timesteps', ylabel='absolut')
    if save:
        filename = str(id) + '_distribution with int_length=' + str(int_length) + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename)
    plt.show()

def thom_all(time_array, int_length, save=False, id=0):
    """
    coordinates plot of thom-plots for different variations in lattice structue
    :param time_array: structure like data = {'rc=02': thom02, 'rc=01': thom01}
    :param int_length: desired length of intervals
    """
    maxx = max([x.max() for x in time_array.values()])
    x = np.arange(0, maxx + int_length, int_length) + int_length / 2
    smoothie = {}
    for name, entry in time_array.items():
        c = create_count(int_length, entry)
        smoothie[name] = np.append(c, np.zeros(len(x) - len(c)))
    thom_all_plot(time_arrays=smoothie, xrange=x, save=save, id=id)

def thom_all_plot(time_arrays, xrange, save, id):
    """
    plot of times of homogeneity for different lattice variation;
    uses middle of bars from histograms as x-values;
    called by thom_all
    :param time_arrays: structure like data = {'rc=02': thom02, 'rc=01': thom01}
    :param xrange: used x-values
    """
    colors = ['darkred', 'orange', 'olivedrab', 'darkturquoise']
    # colors = ['darkred', 'olivedrab', 'darkturquoise']
    fig, ax = plt.subplots()
    data = pd.DataFrame({**{'range': xrange}, **time_arrays})
    for index, (name, thom) in enumerate(time_arrays.items()):
        plt.plot('range', name, data=data, marker='', color=colors[index], linewidth=1, label=name)
    plt.legend()
    plt.xlim(0, xrange.max() + xrange[0])
    plt.ylim(bottom=0)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel('absolute frequency', fontsize=15) #15
    plt.xlabel('thom', fontsize=15)
    # ax.set(xlabel='thom', ylabel='absolute frequency')

    if save:
        filename = str(id) + '_compared distribution' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename)

    plt.show()


def create_count(int_length, thom):
    """
    utility function of thom_all, calculate bars
    called by thom_all
    """
    max = thom.max().astype(int)
    l = len(thom)
    # anz intervalle
    ni = (max / int_length + 1).astype(int)  #
    count = np.zeros(ni + 1)

    for entry in thom:
        c = (entry / int_length).astype(int)
        count[c] += 1
    if count.sum() != l:
        print('FEHLER!')

    return count

def plot_lognorm_distribution(thom, int_length, save=False, id=0, c='seagreen'):
    """
    plot histogram of thom & lognormal distribution
    :param thom: timepoints of homogeneity
    :param int_length: desired length of intervals
    :param c: desired color of distribution function
    """
    max = thom.max().astype(int)
    fig, ax = plt.subplots()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('thom', fontsize=15)
    plt.ylabel('absolute frequency', fontsize=15)

    fitted_data, maxy, y = calc_lognormaldistri(thom=thom, int_length=int_length)
    maxfit = fitted_data.max()
    x = np.arange(0, max, int_length) + int_length/2

    plt.xlim(0, max + int_length/2)
    plt.bar(x, y, width=int_length, color='grey', alpha=0.5)
    print('x', x)
    print('y', y)
    barerr = calc_barerrs(y)
    #todo plt.errorbar(x, y, yerr=barerr, color='magenta')
    plt.plot(x, fitted_data * maxy / maxfit, color=c, label=id)
    # sqderr = calc_quaderr(fitted_data * maxy / maxfit, y)
    # print('q', sqderr)
    # scale = int_length * 5
    # print((barerr * sqderr).max())
    # weightederr = barerr * sqderr #/ scale #skaliert und gewichtet
    # print(weightederr.max())
    # print('w', weightederr)
    # print(fitted_data * maxy / maxfit)
    # print(x)
    # plt.errorbar(x, y, yerr=weightederr, lw=1, capsize=2, capthick=1, color=c)
    # plt.errorbar(x, fitted_data * maxy / maxfit, yerr=weightederr, lw=1, capsize=2, capthick=1, color=c)

    plt.legend()

    if save:
        filename = str(id) + '_interval=' + str(int_length) + '_lognormal_distribution' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename)
    plt.show()

def plot_all_lognorm(thomarray, colorarray, int_length, save=False):
    """
    plot_lognormal_distribution for numerous cases
    :param thomarray: structure like data = {'rc=02': thom02, 'rc=01': thom01}
    :param colorarray: structure like data = {'rc=02': 'red', 'rc=01': 'blue'}
    :param int_length: desired length of interval
    """
    fig, ax = plt.subplots()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('thom', fontsize=15)
    plt.ylabel('absolute frequency', fontsize=15)
    filename = ''

    for index, name in enumerate(thomarray):
        thom = thomarray[name]
        filename += (str(name)) + ',' + (str(len(thom))) + '_'

        fitted_data, maxy, _ = calc_lognormaldistri(thom=thom, int_length=int_length)
        maxfit = fitted_data.max()
        x = np.arange(0, thom.max(), int_length) + int_length/2
        plt.plot(x, fitted_data * maxy / maxfit, color=colorarray[index], label=name)
        print('a', maxy / maxfit)
        plt.xlim(0, thom.max() + int_length)

    plt.ylim(0)
    plt.legend()

    if save:
        filename = str(filename) + 'lognormal_all_intervall=' + str(int_length) + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename)
    plt.show()

def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

def save_plot(plot, filename=None):
    """
    manages the saving of plots
    :param plot: desired plot
    :param filename: individual filename
    """
    if filename is None:
        filename = 'no_name'

    plt.savefig(pathlib.Path('pictures').resolve() / filename)


