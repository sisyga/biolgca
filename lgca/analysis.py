import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from datetime import datetime
import pathlib

from lgca import get_lgca
from lgca.helpers import *

def create_thom(variation, filename, path, rep, save=False):
    thom = np.zeros(rep)
    tmax = 0
    for i in range(rep):
        data = np.load(path + variation + '_' + str(i) + '_' + filename + '_offsprings.npy')
        tend, _ = data.shape
        thom[i] = tend
        if tend > tmax:
            tmax = tend
    if save:
        np.save(path + variation + '_thom', thom)
    print('Max: %d, Min: %d, Mean: %f' %(max(thom), min(thom), thom.mean()))

    return thom

def plot_histogram_thom(thom, int_length, save=False, id=0):
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

#TODO: thom_plt_all

def create_averaged_entropies(variation, filename, path, rep, save=False, plot=False, saveplot=False):
    # thom einlesen
    thom = np.load(path + variation + '_thom.npy')
    tmax = int(max(thom))
    # print('tmax', tmax)

    result_sh = np.zeros(tmax)
    result_gi = np.zeros(tmax)
    result_hill = np.zeros(tmax)

    counter = np.zeros(tmax + 1)

    for i in range(rep):
        data = np.load(path + variation + '_' + str(i) + '_' + filename + '_offsprings.npy')
        t = int(thom[i])
        # print('t', t)
        counter[1:t + 1] += 1

        # 'shannon':
        shannon = calc_shannon(data)
        shannon = np.concatenate((shannon, np.zeros(tmax - t)))
        result_sh = result_sh + shannon

        # 'ginisimpson':
        gini = calc_ginisimpson(data)
        gini = np.concatenate((gini, np.zeros(tmax-t)))
        result_gi = result_gi + gini

        # 'hill order 2':
        hill = calc_hillnumbers(data)
        hill = np.concatenate((hill, np.zeros(tmax-t)))
        result_hill = result_hill + hill

    # Mitteln
    for i in range(tmax):
        result_sh[i] = result_sh[i] / counter[i + 1]
        result_gi[i] = result_gi[i] / counter[i+1]
        result_hill[i] = result_hill[i] / counter[i+1]

    # speichern
    if save:
        np.save(path + variation + '_' + filename + '_' + 'shannon' + '.npy', result_sh)
        np.save(path + variation + '_' + filename + '_' + 'gini' + '.npy', result_gi)
        np.save(path + variation + '_' + filename + '_' + 'hill2' + '.npy', result_hill)

    # plot
    if plot:
        plot_selected_entropies(shannon=result_sh, hill2=result_hill, gini=result_gi,\
                                save=saveplot, id=variation+'_averaged')

def calc_shannon(data):
    time = len(data)
    # print(time)
    maxlab = len(data[0][1:])
    # print(maxlab)

    shannon = np.zeros(time)
    for t in range(time):
        for lab in range(1, maxlab + 1):
            pi = data[t, lab] / sum(data[t, 1:])
            if pi != 0:
                shannon[t] -= pi * m.log(pi)
    return shannon

def calc_ginisimpson(data):
    time = len(data)
    # print(time)
    maxlab = len(data[0][1:])
    # print(maxlab)

    ginisimpson = np.zeros(time) + 1
    for t in range(time):
        for lab in range(1, maxlab + 1):
            pi = data[t, lab] / sum(data[t, 1:])
            if pi != 0:
                ginisimpson[t] -= pi * pi
    return ginisimpson

def calc_simpson(data):
    time = len(data)
    # print(time)
    maxlab = len(data[0][1:])
    # print(maxlab)

    simpson = np.zeros(time) + 1
    for t in range(time):
        n = sum(data[t, 1:])
        if n == 1:
            simpson[t] -= 1
        else:
            for lab in range(1, maxlab + 1):
                ni = data[t, lab]
                simpson[t] -= (ni * (ni - 1))/(n * (n - 1))
    return simpson

def calc_hillnumbers(data, order=2):
    time = len(data)
    maxlab = len(data[0][1:])

    if order ==1:
        hill_lin = np.zeros(time)
        shannon = calc_shannon(data)
        for t in range(time):
            hill_lin[t] = np.exp(shannon[t])
        return hill_lin

    else:
        hill_q = np.zeros(time)
        for t in range(time):
            for lab in range(1, maxlab + 1):
                pi = data[t][lab] / sum(data[t][1:])
                hill_q[t] += pi ** order
            hill_q[t] = hill_q[t] ** (1 / (1 - order))
        return hill_q