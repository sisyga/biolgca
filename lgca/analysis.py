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
import lgca.helpers

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

def calc_popsize(data):
    """
    :param data: corrected offsprings
    """
    time = len(data)
    size = [0]*time
    for t in range(time):
        size[t] = sum(data[t])
    return size

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

def calc_evenness(data):
    """
    calculate evenness of shannon
    :param data: correct(lgca.offsprings)
    :return: evenness
    """
    time = len(data)
    # print(time)
    shannon = calc_shannon(data)
    evenness = np.zeros(time)

    for t in range(time):
        N = sum(data[t])
        evenness[t] = shannon[t]/m.log(N)
    return evenness

def calc_richness(data):
    """
    calculate species richness per timestep
    :param data: correct(offsprings)
    :return: richness
    """
    rich = []
    for t in range(0, len(data)):
        # print(data[t])
        N = np.array(data[t])[np.nonzero(data[t])]
        # print(N)
        # print(len(N))
        rich.append(len(N))
        # print(rich)
    return rich

def calc_bergerparker(data):
    """
    calculate berger parker index per timestep
    :param data: correct(offsprings)
    :return: berger parker index
    """
    bp = []
    for t in range(0, len(data)):
        m = max(data[t])
        s = sum(data[t])
        bp.append(m/s)
    return bp

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

def create_averaged_entropies(dic_offs, save=True, id=0, plot=False, saveplot=False):
    """
    calculate averaged diversity indices (shannon, ginisimpson, hill 2nd order)
    :param dic_offs: like {'1st data': correct(offsprings), '2nd data': correct(offsprings)}
    :param save: saves indices
    :param plot: calls plot_selected_entropies
    :param saveplot: saves plot
    :return: averaged indices (shannon, ginisimpson, hill 2nd order)
    """
    tmax = len(list(dic_offs.values())[0])
    # print('tmax', tmax)
    filename = str(id)
    result_sh = [0] * tmax
    result_gi = [0] * tmax
    result_hill = [0] * tmax
    result_richness = [0] * tmax
    result_popsize = [0] * tmax

    #falls unterschiedlich lange Einträge np.concatenate((gini, np.zeros(tmax-t)))

    for key in dic_offs:
        print('bin bei file: ', key)
        data = dic_offs[key]

        #popsize
        size = np.array(calc_popsize(data))
        result_popsize += size

        #richness
        rich = np.array(calc_richness(data))
        result_richness += rich

        # 'shannon':
        shannon = calc_shannon(data)
        result_sh += shannon

        # 'ginisimpson':
        gini = calc_ginisimpson(data)
        result_gi += gini

        # 'hill order 2':
        hill = calc_hillnumbers(data)
        result_hill += hill

        # 'hill order 25':
        # hill5 = calc_hillnumbers(data, order=0.5)
        # result_hill5 += hill5

    # Mitteln
    result_sh = result_sh / len(dic_offs)
    result_gi = result_gi / len(dic_offs)
    result_hill = result_hill / len(dic_offs)
    result_richness = result_richness / len(dic_offs)
    result_popsize = result_popsize / len(dic_offs)
    print('gemittelt über ', len(dic_offs))

    # speichern
    if save:
        np.savetxt('saved_data/' + filename + '_' + 'averaged_shannon' + '.csv', result_sh, delimiter=',', fmt='%s')
        np.savetxt('saved_data/' + filename + '_' + 'averaged_gini' + '.csv', result_gi, delimiter=',', fmt='%s')
        np.savetxt('saved_data/' + filename + '_' + 'averaged_hill2' + '.csv', result_hill, delimiter=',', fmt='%s')
        np.savetxt('saved_data/' + filename + '_' + 'averaged_popsize' + '.csv', result_popsize, delimiter=',', fmt='%s')
        np.savetxt('saved_data/' + filename + '_' + 'averaged_richness' + '.csv', result_richness, delimiter=',', fmt='%s')

    # plot
    # if plot:
    #     lgca.helpers.plot_selected_entropies(shannon=result_sh, hill2=result_hill, gini=result_gi,\
    #                             hill_5=result_hill5, save=saveplot, id=filename+'_averaged')
    #     lgca.helpers.plot_sth(data={'gi': result_gi, 'sh': result_sh, 'hill_5': result_hill5,
    #                    'hill_2': result_hill}, save=True, id='averaged_unscaled')
   # return result_sh, result_gi, result_hill

def calc_lognorm(thom, xrange):
    param = sp.stats.lognorm.fit(thom, loc=0)
    new_x = [entry - xrange[1]/2 for entry in xrange[1:]]
    fitted_data = sp.stats.lognorm.pdf(new_x, param[0], loc=param[1], scale=param[2])
    print('sigma', param[0])
    print('?', param[1])
    print('p2', param[2])
    print('mu', np.log(param[2]))

    return fitted_data, new_x

def calc_barerrs(hist_data):
    """
    calculate error in thom histograms;
    called by plot_lognorm_distribution
    """
    print(hist_data)
    n = sum(hist_data)
    print(n)
    err = [(entry * (1 - entry/n))**0.5 for entry in hist_data]
    print(err)

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
