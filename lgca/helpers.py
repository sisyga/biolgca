import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from datetime import datetime
import pathlib

def mullerplot(data, id=0, save=False, int_length=1):
    tend = len(data)
    maxlab = len(data[0]) - 1

    fig, ax = plt.subplots()
    val = np.zeros((maxlab, tend))

    for t in range(0, tend):
        for lab in range(0, maxlab):
            val[lab, t] = data[t, lab + 1]

    if int_length == 1:
        xrange = range(0, tend)
        print('x', xrange)
        pop = val
        print('pop', pop)
    else:
        int_num = ((tend - 1) // int_length)
        print('anz intervalle', int_num)
        xrange = [0]
        for i in range(int_num):
            xrange.append(i * int_length + 0.5 * int_length)
        print('xrange1', xrange)
        # xrange = np.append(np.append(np.zeros(1), xrange), tend)
        if int_num * int_length != tend:
            xrange.append((tend - 1 + int_num * int_length) / 2)
        xrange.append(tend-1)
        print('xrange2', xrange[0:3])

        acc_val = np.zeros((maxlab, len(xrange) - 2)) + -999 #todo: 0
        for i in range(0, int_num):
            for lab in range(0, maxlab):
                acc_val[lab, i] = np.sum(val[lab, i * int_length:1 + (i+1)*int_length])
        if int_num * int_length != tend:
            for lab in range(0, maxlab):
                acc_val[lab, -1] = np.sum(val[lab, int_length * int_num:tend])

        print('mean_val', acc_val)
        pop = np.zeros((maxlab, len(xrange))) + -777
        pop[:, 0] = val[:, 0]
        pop[:, 1:-1] = acc_val
        pop[:, -1] = val[:, -1]
        print('pop', pop)

    popdic = {str(i): pop[i] for i in range(0, maxlab)}
    data = pd.DataFrame(popdic, index=xrange)
    data_perc = data.divide(data.sum(axis=1), axis=0)
    # print(data_perc)
    plt.xlabel('timesteps')
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
    if save:
        save_plot(fig, str(id) + '_' + ' mullerplot with intervall=' + str(int_length) + '.jpg')
    plt.show()



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



def plot_entropies_together(simpson, gini, shannon, save, id):
    if save is None:
        save = False
    if id is None:
        id = 0
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

def spacetime_plot(nodes_t, labels, figsize=None, figindex=None, cmap='nipy_spectral', tbeg=None, tend=None, save=False, id=0):
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
    if save:
        save_plot(fig, str(id) + '_spacetimeplot_' + str(tbeg) + '-' + str(tend) + '.jpg')
    plt.show()


def thom_all_plot(time_arrays, xrange, save, id):
    colors = ['seagreen', 'magenta', 'cyan', 'blue']
    fig, ax = plt.subplots()
    data = pd.DataFrame({**{'range': xrange}, **time_arrays})
    for index, (name, thom) in enumerate(time_arrays.items()):
        plt.plot('range', name, data=data, marker='', color=colors[index], linewidth=0.7, label=name)
    plt.legend()
    plt.xlim(0, xrange.max() + xrange[0])
    plt.ylim(bottom=0)
    ax.set(xlabel='timesteps', ylabel='absolut frequencies')

    if save:
        filename = str(id) + '_compared distribution' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename)

    plt.show()

def create_count(int_length, thom):
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

def thom_all(time_array, int_length, save=False, id=0):
    maxx = max([x.max() for x in time_array.values()])
    x = np.arange(0, maxx + int_length, int_length) + int_length / 2
    smoothie = {}
    for name, entry in time_array.items():
        c = create_count(int_length, entry)
        smoothie[name] = np.append(c, np.zeros(len(x) - len(c)))
    thom_all_plot(smoothie, x, save, id)

def save_plot(plot, filename=None):
    if filename is None:
        filename = 'no_name'

    plt.savefig(pathlib.Path('pictures').resolve() / filename)


def aloha(who):
    print('aloha', who)


