import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from datetime import datetime
import pathlib


def errors(lgca):
    print('---errors?---')
    inh_l = False
    for i in range(lgca.maxlabel.astype(int) + 1):
        if lgca.props['lab_m'][i] <= lgca.maxlabel_init:
            inh_l = True
        else:
            inh_l = False
    if inh_l:
        print('---')
    else:
        print('Fehler: inheritance label passen nicht!')

    if len(lgca.props['lab_m']) == len(lgca.props['r_b']) and len(lgca.props['r_b']) == lgca.maxlabel + 1:
        print('---')
    else:
        print('Fehler: len(props) passen nicht!')

    if sum(lgca.props['num_off'][1:]) != lgca.borncells - lgca.diedcells + lgca.maxlabel_init:
        print('num_off falsch!')
    else:
        print('---')

def count_fam(lgca):
    if lgca.maxlabel_init == 0:
        print('ERROR: There are no cells in the lattice!')
    else:
        print('---genealogical research---')
        print('number of ancestors: ', lgca.maxlabel_init)
        print('initial density: ', lgca.maxlabel_init/(lgca.K * lgca.l))
        num = lgca.props['num_off']
        if num[0] != -99:
            print('Etwas stimmt nicht!')
        print('genealogical tree:', num[1:])
        print('max family number is %d with ancestor cell %d' % (max(num[1:]), num.index(max(num[1:]))))
        print('number of ancestors at beginning:', lgca.maxlabel_init)
        print('number of living offsprings:', sum(num[1:]))

        print('number of died cells: ', lgca.diedcells)
        print('number of born cells: ', lgca.borncells)

        return max(num[1:])

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

def entropies_alt(props, order, plot=False, save_plot=False, id=0, off=False):
    time = len(props)
    if off:
        maxlab = len(props[0][1:])

    else:
        maxlab = len(props[0]['num_off'][1:])
    if order == 1:
        # print('Shannon')
        shan_t = np.zeros(time)
        if off:
            for t in range(time):
                if sum(props[t][1:]) != 0:
                    for lab in range(1, maxlab+1):
                        pi = props[t][lab] / sum(props[t][1:])
                        if pi != 0:
                            shan_t[t] -= pi * m.log(pi)
                else:
                    # print('extinct since t= ', t)
                    t_ex = t
                    shan_t[t:] = -1
                    break

        else:
            for t in range(time):
                if sum(props[t]['num_off'][1:]) != 0:
                    for lab in range(1, maxlab+1):
                        pi = props[t]['num_off'][lab] / sum(props[t]['num_off'][1:])
                        if pi != 0:
                            shan_t[t] -= pi * m.log(pi)
                else:
                    # print('extinct since t= ', t)
                    t_ex = t
                    shan_t[t:] = -1
                    break
        shan_max = m.log(maxlab)
        if plot or save_plot:
            plot_entropies(time, shan_t, order, save_plot, id)
        return shan_t, shan_max

    if order == 1.5:
        # print('simpson')
        simpson_t = np.zeros(time) + 1
        if off:
            for t in range(time):
                abs = sum(props[t][1:])
                if abs > 1:
                    for lab in range(1, maxlab + 1):
                        pi = props[t][lab]
                        simpson_t[t] -= (pi * (pi - 1)) / (abs * (abs - 1))
                elif abs == 1:
                    simpson_t[t] = 0
                elif abs == 0:
                    # print('extinct since t= ', t)
                    t_ex = t
                    simpson_t[t:] = -1
                    break
        else:
            for t in range(time):
                abs = sum(props[t]['num_off'][1:])
                if abs > 1:
                    for lab in range(1, maxlab+1):
                        pi = props[t]['num_off'][lab]
                        simpson_t[t] -= (pi * (pi-1)) / (abs * (abs - 1))
                elif abs == 1:
                    simpson_t[t] = 0
                elif abs == 0:
                    # print('extinct since t= ', t)
                    t_ex = t
                    simpson_t[t:] = -1
                    break
        if plot or save_plot:
            plot_entropies(time, simpson_t, order, save_plot, id)
        return simpson_t

    if order == 2:
        # print('ginisimpson')
        ginisimpson_t = np.zeros(time) + 1
        if off:
            for t in range(time):
                if sum(props[t][1:]) != 0:
                    for lab in range(1, maxlab + 1):
                        pi = props[t][lab] / sum(props[t][1:])
                        ginisimpson_t[t] -= pi * pi
                else:
                    # print('extinct since t= ', t)
                    t_ex = t
                    ginisimpson_t[t:] = -1
                    break
        else:
            for t in range(time):
                if sum(props[t]['num_off'][1:]) != 0:
                    for lab in range(1, maxlab+1):
                        pi = props[t]['num_off'][lab] / sum(props[t]['num_off'][1:])
                        ginisimpson_t[t] -= pi * pi
                else:
                    # print('extinct since t= ', t)
                    t_ex = t
                    ginisimpson_t[t:] = -1
                    break
        if plot or save_plot:
            plot_entropies(time, ginisimpson_t, order, save_plot, id)
        return ginisimpson_t

def hillnumber_alt(props, order, plot = False, save_plot = False, id=0, off=False):
    time = len(props)
    if off:
        maxlab = len(props[0][1:])
    else:
        maxlab = len(props[0]['num_off'][1:])

    if order == 1:
        hill_lin = np.zeros(time)
        # print('exp Shannon')
        shan_t, shan_max = entropies(props, order=1, off=off)
        for t in range(time):
            if shan_t[t] != -1:
                hill_lin[t] = np.exp(shan_t[t])
            else:
                # print('extinct since t= ', t)
                t_ex = t
                hill_lin[t:] = -1
                break
        hill_max = np.exp(shan_max)
        if plot or save_plot:
            plot_hill(time, hill_lin, order, save_plot, id)
        return hill_lin, hill_max

    if order >= 2:
        # print('hillnumber order', order)
        hill_quad = np.zeros(time)
        if off:
            for t in range(time):
                if sum(props[t][1:]) != 0:
                    for lab in range(1, maxlab + 1):
                        pi = props[t][lab] / sum(props[t][1:])
                        hill_quad[t] += pi ** order
                    hill_quad[t] = hill_quad[t] ** (1 / (1 - order))
                else:
                    # print('extinct since t= ', t)
                    t_ex = t
                    hill_quad[t:] = -1
                    break
        else:
            for t in range(time):
                if sum(props[t]['num_off'][1:]) != 0:
                    for lab in range(1, maxlab+1):
                        pi = props[t]['num_off'][lab] / sum(props[t]['num_off'][1:])
                        hill_quad[t] += pi ** order
                    hill_quad[t] = hill_quad[t] ** (1/(1 - order))
                else:
                    # print('extinct since t= ', t)
                    t_ex = t
                    hill_quad[t:] = -1
                    break
        if plot or save_plot:
            plot_hill(time, hill_quad, order, save_plot, id)
        return hill_quad

def plot_hill(timesteps, ind, order, save, id):
    time = timesteps
    x = np.arange(0, time, 1)
    y = ind[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='timestep', ylabel='Hillnumber of order {order: d}'.format(order=order))
    plt.xlim(0, time-1)

    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    else:
        plt.xticks(np.arange(0, time, 2))
    plt.yticks(np.arange(0, y.max(), 2))
    # ax.grid()

    plt.show()

    if save:
        save_plot(fig, str(id) + '_hillnumber order ' + str(order) + '.jpg')

def plot_hill_together(props, save=False, id=0):
    time = len(props)
    x = np.arange(0, time, 1)
    o1, hillmax = hillnumber(props, 1)
    o2 = hillnumber(props, 2)
    o3 = hillnumber(props, 3)

    fig, ax = plt.subplots()
    plt.plot(x, o1, 'b-', label='order 1')
    plt.plot(x, o2, 'c--', label='order 2')
    plt.plot(x, o3, 'm:', label='order 3')

    ax.set(xlabel='timesteps', ylabel='Hillnumbers')
    ax.legend()
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    else:
        plt.xticks(np.arange(0, time, 5))
    plt.ylim(1, hillmax*1.1, 10)
    plt.show()

    if save:
        save_plot(fig, str(id) + '_comparing hillnumbers' + '.jpg')

def plot_entropies(timesteps, ind, order, save_plot, id=0):
    time = timesteps
    x = np.arange(0, time, 1)
    y = ind[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    if order == 1:
        ax.set(xlabel='timestep', ylabel='Shannonindex')
    elif order == 2:
        ax.set(xlabel='timestep', ylabel='Ginisimpsonindex')
    elif order == 1.5:
        ax.set(xlabel='timestep', ylabel='Simpsonindex')
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    else:
        plt.xticks(np.arange(0, time, 2))
    plt.ylim(0, y.max()+0.1)
    plt.yticks(np.arange(0, y.max(), 0.2))
    # ax.grid()

    plt.show()

    if save_plot:
        save_plot(fig, str(id) + '_entropy order ' + str(order) + '.jpg')

def plot_entropies_together(props, save=False, id=0):
    time = len(props)
    x = np.arange(0, time, 1)
    shan, shanmax = entropies(props, 1)
    simp = entropies(props, 1.5)
    gini = entropies(props, 2)

    fig, ax = plt.subplots()
    plt.plot(x, shan, 'b-', label='Shannonindex')
    plt.plot(x, simp, 'c--', label='Simpsonindex')
    plt.plot(x, gini, 'm:', label='GiniSimpsonindex')

    ax.set(xlabel='timesteps', ylabel='Index')
    ax.legend()
    plt.xlim(0, time-1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    else:
        plt.xticks(np.arange(0, time, 5))
    plt.ylim(0, shanmax * 1.1, 0.2)
    # ax.grid()
    # plt.axhline(y=0)
    plt.show()

    if save:
        save_plot(fig, str(id) + '_comparing entropies' + '.jpg')

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_sh_gi_hh(props, save=False, id=0, off=False):
    time = len(props)
    x = np.arange(0, time, 1)
    shan, shanmax = entropies(props, order=1, off=off)
    hh = hillnumber(props, order=2, off=off)
    gini = entropies(props, order=2, off=off)

    fig, host = plt.subplots()
    par1 = host.twinx()
    par2 = host.twinx()
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    p1, = host.plot(x, shan, "m", linewidth=0.7, label="Shannonindex")
    p2, = par1.plot(x, gini, "b", linewidth=0.7, label="GiniSimpsonindex")
    p3, = par2.plot(x, hh, "c", linewidth=0.7, label="Hillnumber of order 2")

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

    if save:
        filename = str(id) + '_comparing sh, gi, hh' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename, bbox_inches='tight')


def plot_popsize(props, save=False, id=0, off=False):
    time = len(props)
    x = np.arange(0, time, 1)
    size = np.zeros(time)
    if off:
        for t in range(time):
            size[t] = sum(props[t][1:])
    else:
        for t in range(time):
            size[t] = sum(props[t]['num_off'][1:])
    y = size[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlim(0, time-1)
    plt.ylim(0, size[0] + 0.5)
    # plt.yticks(np.arange(0, size.max() * 1.1, 10))
    ax.set(xlabel='timestep', ylabel='number of living cells')
    # ax.grid(axis='y')

    plt.show()

    if save:
        save_plot(fig, str(id) + '_population size ' + '.jpg')

def spacetime_plot(nodes_t, labels, figindex = None, figsize=None,\
                 cmap='nipy_spectral', tbeg=None, tend=None, save=False, id=0):
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


