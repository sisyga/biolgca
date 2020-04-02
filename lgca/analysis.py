import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import stats, optimize, interpolate
from scipy.stats import binom
from matplotlib import cm
from datetime import datetime
import pathlib

from lgca import get_lgca
from lgca.helpers import *

def create_thom(variation, filename, path, rep, save=False):
    """
    reads offspring-data from path and add length of offsprings (time until homogeneity) to array thom
    :param variation: =(2 * dim + dim * rc) + dim as part of filename of saved offsprings
    :param filename: part of saved filename
    :param path: path to the data
    :param rep: number of repetitions = number of offspring-data
    :param save: saves array of thoms
    :return: array of times until homogeneity has been reached
    """
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

def calc_shannon(data):
    """
    calculate shannon index
    :param data: correct(lgca.offsprings)
    :return: shannon index
    """
    time = len(data)
    # print(time)
    shannon = np.zeros(time)

    for t in range(time):
        for lab in range(0, len(data[t])):
            pi = data[t][lab] / sum(data[t])
            # print('pi', pi)
            if pi != 0:
                shannon[t] -= pi * m.log(pi)
    return shannon

def calc_ginisimpson(data):
    """
    calculate ginisimpson index
    :param data: correct(lgca.offsprings)
    :return: ginisimpson index
    """
    time = len(data)
    ginisimpson = np.zeros(time) + 1

    for t in range(time):
        for lab in range(0, len(data[t])):
            pi = data[t][lab] / sum(data[t])
            # print('pi', pi)
            if pi != 0:
                ginisimpson[t] -= pi * pi
    return ginisimpson

def calc_simpson(data):
    """
    calculate simpson index
    :param data: correct(lgca.offsprings)
    :return: simpson index
    """
    time = len(data)
    # print('time', time)

    simpson = np.zeros(time) + 1
    for t in range(time):
        n = sum(data[t][:])

        for lab in range(0, len(data[t])):
            ni = data[t][lab]
            simpson[t] -= (ni * (ni - 1))/(n * (n - 1))
    return simpson

def calc_hillnumbers(data, order=2):
    """
    calculate hillnumber of order=order
    :param data: correct(lgca.offsprings)
    :param order: desired order
    :return: hillnumber
    """
    time = len(data)

    if order == 1:
        hill_lin = np.zeros(time)
        shannon = calc_shannon(data)
        for t in range(time):
            hill_lin[t] = np.exp(shannon[t])
        return hill_lin

    else:
        hill_q = np.zeros(time)
        for t in range(time):
            for lab in range(0, len(data[t])):
                pi = data[t][lab] / sum(data[t])
                hill_q[t] += pi ** order
            hill_q[t] = hill_q[t] ** (1 / (1 - order))
        return hill_q

def create_averaged_entropies(dic_offs, save=False, plot=False, saveplot=False):
    """
    calculate averaged diversity indices (shannon, ginisimpson, hill 2nd order)
    :param dic_offs: like {'1st data': correct(offsprings), '2nd data': correct(offsprings)}
    :param save: saves indices
    :param plot: calls plot_selected_entropies
    :param saveplot: saves plot
    :return: averaged indices (shannon, ginisimpson, hill 2nd order)
    """
    tmax = len(list(dic_offs.values())[0])
    print('tmax', tmax)
    if '5011' in dic_offs.keys():
        filename = '5011'
    elif "501167" in dic_offs.keys():
        filename = '501167'
    else:
        filename = 'test'
    result_sh = [0] * tmax
    result_gi = [0] * tmax
    result_hill = [0] * tmax

    #falls unterschiedlich lange Einträge np.concatenate((gini, np.zeros(tmax-t)))

    for key in dic_offs:
        print('bin bei file: ', key)
        data = dic_offs[key]

        # 'shannon':
        shannon = calc_shannon(data)
        result_sh += shannon

        # 'ginisimpson':
        gini = calc_ginisimpson(data)
        result_gi += gini

        # 'hill order 2':
        hill = calc_hillnumbers(data)
        result_hill += hill

    # Mitteln
    result_sh = result_sh / len(dic_offs)
    result_gi = result_gi / len(dic_offs)
    result_hill = result_hill / len(dic_offs)

    # speichern
    if save:
        np.save('saved_data/' + filename + '_' + 'averaged_shannon' + '.npy', result_sh)
        np.save('saved_data/' + filename + '_' + 'averaged_gini' + '.npy', result_gi)
        np.save('saved_data/' + filename + '_' + 'averaged_hill2' + '.npy', result_hill)

    # plot
    if plot:
        plot_selected_entropies(shannon=result_sh, hill2=result_hill, gini=result_gi,\
                                save=saveplot, id=filename+'_averaged')

    return result_sh, result_gi, result_hill

def calc_lognormaldistri(thom, int_length):
    """
    calculate lognormal distribution;
    called by plot_lognorm_distribution
    """
    #number of intervalls
    ni = (thom.max() / int_length + 1).astype(int)
    #calc absolue frequency
    count = np.zeros(ni)
    for entry in thom:
        c = (entry / int_length).astype(int)
        count[c] += 1
    if count.sum() != len(thom):
        print('FEHLER!')
    x = np.arange(0, thom.max(), int_length)
    y = count[(x / int_length).astype(int)]
    maxy = y.max()
    #calc function parameters
    thom = (thom / int_length).astype(int)
    param = sp.stats.lognorm.fit(thom)
    xxx = np.arange(0, ni)
    fitted_data = sp.stats.lognorm.pdf(xxx, param[0], loc=0, scale=param[2])
    print('sigma', param[0])
    print('mu', np.log(param[2]))

    return fitted_data, maxy, y


def calc_barerrs(counted_thom):
    """
    calculate error in thom histograms;
    called by plot_lognorm_distribution
    """
    # expect = np.zeros(len(counted_thom))
    # var = np.zeros(len(counted_thom))
    # s = np.zeros(len(counted_thom))
    # b = np.zeros(len(counted_thom))
    # # p = np.zeros(len(expect))
    n = counted_thom.sum()
    # print('n', n)
    err = np.zeros(len(counted_thom))
    for i in range(0, len(err)):
        # p[i] = counted_thom[i]/n
        # if counted_thom[i] != 0:
        #     b[i] = binom.pmf(counted_thom[i], n, p=counted_thom[i]/n)
        # expect[i] = b[i] * n
        # var[i] = expect[i] * (1-b[i])
        # s[i] = (var[i])**0.5
        err[i] = (counted_thom[i]*(1-(counted_thom[i]/n)))**0.5
    print('err', err)
    # print('exp', expect)
    # print('var', var)
    # print('s', s)
    # # print('p', p)
    # print('b', b)
    return err

def calc_quaderr(data, fitted_data): #TODO quad Fehler
    """
    calculate squared error in thom histograms;
    called by plot_lognorm_distribution
    """
    sqd = np.zeros(len(data))
    for i in range(0, len(data)):
        sqd[i] = (data[i] - fitted_data[i])**2
    # print(sqd)
    return sqd

def cond_oneancestor(lgca):
    """
    check if population is pseudo-homogenous;
    means that all cells from only one origin family
    used as exit condition in timeevo_until_pseudohom
    """
    # fi = lgca.maxfamily_init
    nodes = lgca.nodes[lgca.r_int:-lgca.r_int]
    parents = []

    for node in nodes:
        for entry in node[node > 0]:
            fam = lgca.props['lab_m'][entry]
            ori = lgca.tree_manager.tree[fam]['origin']
            # print(p)
            parents.append(ori)

            if len(parents) != 0 and parents.count(parents[0]) != len(parents):
                return False
    #     print('parents node ', parents)
    # print('result ', parents)
    if len(parents) == 0:
        print('ausgestorben')
        return True
    elif parents.count(parents[0]) == len(parents):
        return True
    else:
        return False
