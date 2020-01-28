import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from datetime import datetime
import pathlib

def mullerplot(props, id=0, save=False):
    tend = len(props)
    time = range(0, tend)
    maxlab = len(props[0]['num_off']) - 1

    val = np.zeros((maxlab, tend))
    for t in range(0, tend):
        for lab in range(0, maxlab):
            val[lab, t] = props[t]['num_off'][lab + 1]
    valdic = {str(i): val[i] for i in range(0, maxlab)}
    data = pd.DataFrame(valdic, index=time)
    data_perc = data.divide(data.sum(axis=1), axis=0)
    fig = plt.subplot()
    plt.ylabel(' frequency of families')
    plt.xlabel('timesteps')
    plt.xlim(0, tend - 1)
    plt.ylim(0, 1)
    if tend <= 15:
        plt.xticks(np.arange(0, tend, 1))
    elif tend <= 100:
        plt.xticks(np.arange(0, tend, 5))
    elif tend >= 5000:
        plt.xticks(np.arange(0, tend, 1000))
    elif tend >= 1000:
        plt.xticks(np.arange(0, tend, 500))
    elif tend >= 100:
        plt.xticks(np.arange(0, tend, 50))


    plt.stackplot(time, *[data_perc[str(f)] for f in range(0, maxlab)],\
                  labels=list(range(0, maxlab)))
    plt.show()


    if save:
        save_plot(fig, str(id) + '_' + ' mullerplot' + '.jpg')

def mullerplot_extended(props, id=0, save=False, int_range=1, off=False):
    tend = len(props)
    if off:
        maxlab = len(props[0]) - 1
    else:
        maxlab = len(props[0]['num_off']) - 1
    fig, ax = plt.subplots()
    # xrange = range(0,tend, int_range)
    val = np.zeros((maxlab, tend))
    if off:
        for t in range(0, tend):
            for lab in range(0, maxlab):
                val[lab, t] = props[t, lab + 1]
    else:
        for t in range(0, tend):
            for lab in range(0, maxlab):
                val[lab, t] = props[t]['num_off'][lab + 1]

    plt.xlabel('timesteps')

    if int_range == 1:
        xrange = range(0, tend)
        pop = val
    else:
        int_num = ((tend - 1) // int_range + 1)
        xrange = np.arange(0, tend, int_range) + int_range / 2
        xrange = np.append(np.append(np.zeros(1), xrange), tend)
        mean_val = np.zeros((maxlab, int_num)) + -999
        for i in range(0, int_num):
            for lab in range(0, maxlab):
                mean_val[lab, i] = np.sum(val[lab, i * int_range:(i + 1) * int_range]) / int_range
        for lab in range(0, maxlab):
            mean_val[lab, int_num - 1] = \
                np.sum(val[lab, (int_num - 1) * int_range:]) / (tend - (int_num - 1) * int_range)

        pop = np.zeros((maxlab, int_num + 2)) + -777
        pop[:, 0] = val[:, 0]
        pop[:, 1:-1] = mean_val
        pop[:, -1] = val[:, -1]

    popdic = {str(i): pop[i] for i in range(0, maxlab)}
    data = pd.DataFrame(popdic, index=xrange)
    data_perc = data.divide(data.sum(axis=1), axis=0)

    plt.ylabel(' frequency of families')

    # plot einstellungen
    plt.xlim(0, xrange[-1])
    plt.ylim(0, 1)
    if xrange[-1] <= 15:
        plt.xticks(np.arange(0, xrange[-1], 1))
    elif xrange[-1] <= 100:
        plt.xticks(np.arange(0, xrange[-1], 5))
    elif xrange[-1] >= 1000:
        plt.xticks(np.arange(0, xrange[-1], 500))
    elif xrange[-1] >= 100:
        plt.xticks(np.arange(0, xrange[-1], 50))

    plt.stackplot(xrange, *[data_perc[str(f)] for f in range(0, maxlab)], \
                  labels=list(range(0, maxlab)))
    plt.show()

    if save:
        save_plot(fig, str(id) + '_' + ' mullerplot with il=' + str(int_range) + '.jpg')

def plot_index(index_data, which, save=False, id=0):
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



def plot_hillnumbers_together(hill_1, hill_2, hill_3, save=False, id=0):
    time = len(hill_1)
    x = np.arange(0, time, 1)

    fig, ax = plt.subplots()
    plt.plot(x, hill_1, 'b-', label='order 1')
    plt.plot(x, hill_2, 'c--', label='order 2')
    plt.plot(x, hill_3, 'm:', label='order 3')

    ax.set(xlabel='timesteps', ylabel='Hillnumbers')
    ax.legend()
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))

    plt.ylim(1, max(hill_1)*1.1, 10)
    if save:
        save_plot(plot=fig, filename= str(id) + '_comparing hillnumbers' + '.jpg')
    plt.show()



def plot_entropies_together(simpson, gini, shannon, save=False, id=0):
    time = len(gini)
    x = np.arange(0, time, 1)

    fig, ax = plt.subplots()
    plt.plot(x, shannon, 'b-', label='Shannonindex')
    plt.plot(x, simpson, 'c--', label='Simpsonindex')
    plt.plot(x, gini, 'm:', label='GiniSimpsonindex')

    ax.set(xlabel='timesteps', ylabel='Index')
    ax.legend()
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))

    plt.ylim(0, max(shannon) * 1.1)
    if save:
        save_plot(fig, str(id) + '_comparing entropies' + '.jpg')
    plt.show()



def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_selected_entropies(shannon, hill2, gini, save=False, id=0):
    time = len(shannon)
    x = np.arange(0, time, 1)

    fig, host = plt.subplots()
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    p1, = host.plot(x, shannon, "m", linewidth=0.7, label="Shannonindex")
    p2, = par1.plot(x, gini, "b", linewidth=0.7, label="GiniSimpsonindex")
    p3, = par2.plot(x, hill2, "c", linewidth=0.7, label="Hillnumber of order 2")

    host.set_xlim(0, time - 1)
    host.set_ylim(bottom=0)
    par1.set_ylim(bottom=0)
    par2.set_ylim(bottom=0)

    host.set_xlabel("timesteps")
    host.set_ylabel("Shannonindex")
    par1.set_ylabel("GiniSimpsonindex")
    par2.set_ylabel("Hillnumber of order 2")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines])
    if save:
        filename = str(id) + '_comparing sh, gi, hh' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename, bbox_inches='tight')
    plt.show()

    #     plt.plot(x, shan, 'b-', label='Shannonindex')
    #     plt.plot(x, gini, 'm:', label='GiniSimpsonindex')
    #     plt.plot(x, hh, 'c--', label='Hillnumber of order 2')

    #     ax.set(xlabel='timesteps', ylabel='Index')
    #     ax.legend()
    #     plt.xlim(0, time - 1)
    #     if time <= 15:
    #         plt.xticks(np.arange(0, time, 1))
    #     elif time <= 100:
    #         plt.xticks(np.arange(0, time, 5))
    #     elif time >= 1000:
    #         plt.xticks(np.arange(0, time, 500))
    #     elif time >= 100:
    #         plt.xticks(np.arange(0, time, 50))
    #     plt.ylim(0, np.exp(shanmax) * 1.1, 0.5)




def plot_popsize(data, save=False, id=0):
    time = len(data)
    x = np.arange(0, time, 1)
    size = np.zeros(time)
    for t in range(time):
        size[t] = sum(data[t][1:])
    y = size[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlim(0, time - 1)
    plt.ylim(0, max(size) * 1.1)
    ax.set(xlabel='timestep', ylabel='total number of living cells')
    if save:
        save_plot(fig, str(id) + '_population size ' + '.jpg')

    plt.show()



def spacetime_plot(nodes_t, labels, figsize=None, cmap='nipy_spectral', tbeg=None, tend=None, save=False, id=0):
    tmax, dim, c = nodes_t.shape
    vc = 2
    rc = c - vc
    print('tmax, Knoten, rc', tmax, dim, rc)
    if tbeg is None:
        tbeg = 0
    if tend is None:
        tend = tmax

    val = np.zeros((tmax, dim*c))

    for t in range(0, tmax):
        for x in range(dim):
            node = nodes_t[t, x]
            occ = node.astype(np.bool)
            # print('occ', occ)
            if occ.sum() == 0:
                # TODO: neu
                # val[t, x * c: x * c + c] = None  ???
                i = 0
                while i < c:
                    val[t, x * c + i] = None
                    i = i + 1
                continue
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
    # fig.colorbar(plot, shrink = 0.5)

    plt.ylabel('timesteps')
    plt.xlabel('lattice site')
    # nur "Knotenanfang"
    plt.xlim(-0.5, dim*c-0.5)
    plt.xticks((np.arange(0, dim*c, c)))

    plt.ylim(tend-0.5, tbeg-0.5)
    if tend - tbeg > 700:
        plt.yticks(np.arange(tbeg, tend, 100))
    elif tend - tbeg > 100:
        plt.yticks(np.arange(tbeg, tend, 50))
    elif tend - tbeg <= 100:
        plt.yticks(np.arange(tbeg, tend, 10))
    plt.show()
    if save:
        save_plot(fig, str(id) + '_spacetimeplot' + '.jpg')

def save_plot(plot, filename=None):
    if filename is None:
        filename = 'no_name'

    plt.savefig(pathlib.Path('pictures').resolve() / filename)


def aloha(who):
    print('aloha', who)


