import numpy as np
import math as m
import matplotlib.pyplot as plt
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

def bar_stacked(lgca, save = False, id = 0):
    tmax, l, _ = lgca.nodes_t.shape
    # print('tmax', tmax)
    ancs = np.arange(1, lgca.maxlabel_init.astype(int) + 1)

    val = np.zeros((tmax, lgca.maxlabel_init.astype(int) + 1))
    # print('size val', val.shape)
    for t in range(0, tmax):
        for c in ancs:
            # print('c ', c)
            # print('t', t)
            val[t, c] = lgca.props_t[t]['num_off'][c]
    plt.figure(num=None)
    ind = np.arange(0, tmax, 1)
    width = 1
    for c in ancs:
        if c > 1:
            b = np.zeros(tmax)
            for i in range(1, c):
                b = b + val[:, i]
            plt.bar(ind, val[:, c], width, bottom=b, label=c)
        else:
            plt.bar(ind, val[:, c], width, color=['red'], label=c)

    ###plot settings

    plt.ylabel('total number of living cells')
    plt.xlabel('timesteps')
    # plt.title('Ratio of offsprings')
    if len(ind) <= 15:
        plt.xticks(ind)
    else:
        plt.xticks(np.arange(0, len(ind)-1, 5))

    if tmax >= 700:
        plt.xticks(np.arange(0, tmax, 100))
    elif tmax >= 100:
        plt.xticks(np.arange(0, tmax, 50))

    # plt.subplots_adjust(right=0.85)
    # plt.legend(bbox_to_anchor=(1.04, 1))
    plt.tight_layout()
    plt.show()
    if save == True:
        # plt.savefig('pictures/' + str(id) + '  frequency' + str(datetime.now()) +'.jpg')
        # plt.savefig('probe_bar.jpg')
        # filename = str(lgca.r_b) + ', ' + str(id) + ', ' + str(t) + '  frequency' + '.jpg'
        filename = str(lgca.r_b) + ', dens' + str(lgca.maxlabel_init / (lgca.K * lgca.l)) + ', ' \
                   + str(id) + ', ' + str(t) + '  frequency' + '.jpg'

        plt.savefig(pathlib.Path('pictures').resolve() / filename)

def bar_stacked_relative(props_t, save=False, id=0):
    tmax = len(props_t)
    maxlab = sum(props_t[0]['num_off'][1:])
    ancs = np.arange(0, maxlab)
    val = np.zeros((tmax, maxlab))

    for t in range(0, tmax):
        for c in ancs:
            val[t, c] = props_t[t]['num_off'][c + 1]
        c_sum = sum(props_t[t]['num_off'][1:])
        if c_sum != 0:
            val[t] = val[t] / c_sum
    print('erste Schleife fertig, nun plotbar')
    # fig, ax = plt.subplots()
    fig = plt.figure(num=None)
    width = 1
    ind = np.arange(0, tmax)
    b = np.zeros(tmax)
    for c in ancs:
        print('bin schon bei c= ', c)
        plt.bar(ind, val[:, c], width, bottom=b, label=c)
        b = b + val[:, c]
        #TODO: bessere Lösung?
    plt.ylabel(' frequency of families')
    plt.xlabel('timesteps')
    plt.xlim(0, tmax - 0.5)
    plt.ylim(0, 1)
    if tmax <= 15:
        plt.xticks(np.arange(0, tmax, 1))
    elif tmax <= 100:
        plt.xticks(np.arange(0, tmax, 5))
    elif tmax >= 1000:
        plt.xticks(np.arange(0, tmax, 500))
    elif tmax >= 100:
        plt.xticks(np.arange(0, tmax, 50))

    plt.show()

    if save:
        saving_plot(fig, str(id) + '_' + ' rel_frequency' + '.jpg')

# def mullerplot(self, nodes_t=None, props_t=None, figindex=None, figsize=None):
#     if nodes_t is None:
#         nodes_t = self.nodes_t
#     # if figsize is None:
#     #     figsize = estimate_figsize(nodes_t.sum(-1).T, cbar=True)
#     if props_t is None:
#         props_t = self.props_t
#
#     #create values
#     tmax, l, _ = nodes_t.shape
#     ancs = np.arange(1, self.maxlabel_init.astype(int) + 1)
#     # if len(ancs) != lgca.maxlabel_init:
#     #     print('FEHLER: len(ancs) != maxlabel_init!')
#     val = np.zeros((tmax, self.maxlabel_init.astype(int) + 1))
#     for t in range(0, tmax):
#         for c in ancs:
#             val[t, c] = props_t[t]['num_off'][c]
#
#     #write in .txt
#     file = str(datetime.now()) + '.txt'
#     np.savetxt(file, val, fmt="%d")



def entropies(props, order, plot=False, save_plot=False, id=0):
    time = len(props)
    maxlab = len(props[0]['num_off'][1:])
    if order == 1:
        # print('Shannon')
        shan_t = np.zeros(time)
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

def hillnumber(props, order, plot = False, save_plot = False, id=0):
    time = len(props)
    maxlab = len(props[0]['num_off'][1:])

    if order == 1:
        hill_lin = np.zeros(time)
        # print('exp Shannon')
        shan_t, shan_max = entropies(props, 1)
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
        saving_plot(fig, str(id) + '_hillnumber order ' + str(order) + '.jpg')

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
        saving_plot(fig, str(id) + '_comparing hillnumbers' + '.jpg')

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
        saving_plot(fig, str(id) + '_entropy order ' + str(order) + '.jpg')

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
        saving_plot(fig, str(id) + '_comparing entropies' + '.jpg')

def plot_sh_gi_hh(props, save=False, id=0):
    time = len(props)
    x = np.arange(0, time, 1)
    shan, shanmax = entropies(props, 1)
    hh = hillnumber(props, 2)
    gini = entropies(props, 2)

    fig, ax = plt.subplots()
    plt.plot(x, shan, 'b-', label='Shannonindex')
    plt.plot(x, gini, 'm:', label='GiniSimpsonindex')
    plt.plot(x, hh, 'c--', label='Hillnumber of order 2')


    ax.set(xlabel='timesteps', ylabel='Index')
    ax.legend()
    plt.xlim(0, time - 1)
    if time >= 700:
        plt.xticks(np.arange(0, time, 100))
    elif time >= 100:
        plt.xticks(np.arange(0, time, 50))
    else:
        plt.xticks(np.arange(0, time, 5))
    plt.ylim(0, np.exp(shanmax) * 1.1, 0.5)
    plt.show()

    if save:
        saving_plot(fig, str(id) + '_comparing shannon, gini, hill2' + '.jpg')

def plot_popsize(props, save=False, id=0):
    time = len(props)
    x = np.arange(0, time, 1)
    size = np.zeros(time)
    for t in range(time):
        size[t] = sum(props[t]['num_off'][1:])
    y = size[x]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlim(0, time-1)
    plt.yticks(np.arange(0, size.max() * 1.1, 10))
    ax.set(xlabel='timestep', ylabel='number of living cells')
    ax.grid(axis='y')

    plt.show()

    if save:
        saving_plot(fig, str(id) + '_population size ' + '.jpg')

def saving_plot(plot, filename=None):
    if filename is None:
        filename = 'no_name'
    plt.savefig(pathlib.Path('pictures').resolve() / filename)

def aloha(who):
    print('aloha', who)


