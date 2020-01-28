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

def histogram_thom(thom, int_length, save=False, id=0):
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

def create_averaged_entropies(variation, filename, path, rep, save=False, plot=False):
    # thom einlesen
    thom = np.load(path + variation + '_thom.npy')
    tmax = int(max(thom))
    print('tmax', tmax)

    result_sh = np.zeros(tmax)
    result_gi = np.zeros(tmax)
    # result_hill2 = []

    counter = np.zeros(tmax + 1)

    for i in range(rep):
        data = np.load(path + variation + '_' + str(i) + '_' + filename + '_offsprings.npy')
        t = int(thom[i])
        print('t', t)
        counter[1:t + 1] += 1

        # 'shannon':
        shannon = calc_shannon(data)
        shannon = np.concatenate((shannon, np.zeros(tmax - t)))
        result_sh = result_sh + shannon

        # 'ginisimpson':
        gini = calc_ginisimpson(data)
        gini = np.concatenate((gini, np.zeros(tmax-t)))
        result_gi = result_gi + gini
    #     # 'hill2':
    #     hill =  hillnumber(props=data_off, order=2, off=True)
    #     hill = np.concatenate((hill, np.zeros(tmax-t)))
    #     result_hill2 = result_hill2 + hill
    #     print('sh ohne', result_sh[240:246])

    # Mitteln
    for i in range(tmax):
        result_sh[i] = result_sh[i] / counter[i + 1]
        result_gi[i]= result_gi[i] / counter[i+1]
    #     result_hill2[i] = result_hill2[i] / counter[i+1]
    # print('sh mit', result_sh[240:246])
    # print(counter)

    # speichern
    if save:
        print('shannon', result_sh)
        print('gini', result_gi)
        # np.save(path + variation + '_' + filename + '_' + 'shannon' + '.npy', result_sh)
        # np.save('saved_data/' + variation + '_' + 'gini' + '.npy', result_gi)
        # np.save('saved_data/' + variation + '_' + 'hill2' + '.npy', result_hill2)

    # plot
    # if plot:
    # plot_averaged_entropies(sh=result_sh, gi=result_gi, hh=result_hill2, save=False)

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

