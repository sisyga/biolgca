# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import pandas as pd


def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

def search_names(path):
    files = os.listdir(path)
    names = [entry.replace('_tree.npy', '') for entry in files if 'tree' in entry]
    print(len(names))
    return names

def search_offs(path, name):
    return correct(np.load(path + name + '_offsprings.npy'))

def set_data(path):
    names = search_names(path)
    data = {}
    for name in names:
        data[name] = search_offs(path, name)
    return data

def read_inds(which='si'):
    path = 'saved_data/Indizes_explizit/'
    files = os.listdir(path)
    dataset = {}
    ind = which
    for file in files:
        if ind in file:
            dataset[file[:-(4+len(ind))]] = np.loadtxt(path + file)
    return dataset

def ave_inds(which='shannon', plot=False, save=False, savename=None):
    path = 'saved_data/Indizes_averaged/'
    files = os.listdir(path)
    dataset = {}
    ind = which
    for file in files:
        if ind in file:
            dataset[file[:-(14+len(ind))]] = np.loadtxt(path + file)
    if plot:
        plot_sth(dataset, ylabel=ind, save=save, id=which, savename=savename)
    return dataset

def zahlende(path, steps):
    files = os.listdir(path)
    rel = []
    for file in files:
        if 'offspring' in file:
            rel.append(file)
    print('anz daten', len(rel))
    rel = ['5011_mut_f308e6f3-b126-4189-a736-0de904c67c89_offsprings.npy',
           '5011_mut_f308e6f3-b126-4189-a736-0de904c67c89_offsprings.npy']
    ende = {}
    for i, f in enumerate(rel):
        offs = correct(np.load(path + f)[-steps:])
        fams = []
        for w, e in enumerate(offs[-1]):
            if e > 0:
                fams.append(w+1)
        for s in range(len(offs)):
            offs[s] = [entry for entry in offs[s] if entry > 0]
        ende[str(i + 1)] = offs

    return ende, fams

def mittelende(ende):
    ave = {}
    for key in ende:
        offs = ende[key]
        ave[key] = np.mean(offs, axis=0)
        #todo: mean klappt nicht :D
    return ave

def plotende(ave, fams):
    print(ave)
    for sim in ave:
        # fams = np.arange(0, len(ave[sim][0])) + 1
        print(fams)
        anz = pd.Series(ave[sim][0], index=fams, name='Sim ' + str(sim))
        # explode = [0.1]*len(ave[sim][0])
        anz.plot.pie(figsize=(6, 6))
        # plt.legend()
        plt.show()

    #todo: klappt danna uch nach mittelerde todo mit mehreren steps?



ende, fams = zahlende(path = 'saved_data/5011_ges/', steps=1)
print(ende, fams)
plotende(ende, fams)
# print(mittelende(ende))

"""
    --- Index-Daten einlesen    ---
"""
## data1 = correct(np.load('saved_data/5011_mut_04_01/'
##                         '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))
## data167 = correct(np.load('saved_data/501167_mut_04_02/'
##                           '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))

# si = read_inds(which='si')
# sh = read_inds(which='sh')
# eve = read_inds(which='eve')
# gi = read_inds(which='gi')
# hill1 = read_inds(which='hill1')
# hill2 = read_inds(which='hill2')
# hill3 = read_inds(which='hill3')
# hill_5 = read_inds(which='hill_5')
# hill_25 = read_inds(which='hill_25')
# hill_75 = read_inds(which='hill_75')
#
# ave_sh = ave_inds(which='shannon')
# ave_hill2 = ave_inds(which='hill2')
# ave_hill_5 = ave_inds(which='hill5')
# ave_gi = ave_inds(which='gini')

"""
    --- diverse plots ---
"""
# vars = ['onenode', 'onerc']
# vars = ['onenode']
# for var in vars:
    # plot_sth(data={'onenode': si[var] - gi[var]}, ylabel='diff: simpson - gini', savename='diff_SiGi', save=True)
    # plot_sth(data={'hill_2': hill2[var], 'eve': eve[var]}, save=True, savename=var + '_HhEve')
    # plot_sth(data={'gi': gi[var], 'eve': eve[var]}, save=True, savename=var + '_GiEve')
    # plot_sth(data={'gi': gi[var], 'sh': sh[var], 'eve': eve[var]}, save=True, savename=var + '_GiEveSh')
    # plot_sth(data={'gi': gi[var], 'sh': sh[var]}, save=True, savename=var + '_GiSh')
#     plot_hillnumbers_together(hill2[var],hill_25[var], hill_75[var], save=True, id=var)
#     plot_entropies_together(gini=gi[var], shannon=sh[var])
#     plot_selected_entropies(gini=gi[var], shannon=sh[var], hill2=hill2[var], save=True, id=var + '3')

"""
    --- krasse Indexberechnung  ---  
"""
# data1 = correct(np.load('saved_data/Indizes_explizit/Daten/'
#                         '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))
# data167 = correct(np.load('saved_data/Indizes_explizit/Daten/'
#                           '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))
#
# data = {"1": data1, "167": data167}
#
# def funk(file):
#     print(file)
#     # np.savetxt('saved_data/' + file + 'sh.csv', calc_shannon(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'si.csv', calc_simpson(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'gi.csv', calc_ginisimpson(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill1.csv', calc_hillnumbers(data[file], order=1), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill2.csv', calc_hillnumbers(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill3.csv', calc_hillnumbers(data[file], order=3), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill_25.csv', calc_hillnumbers(data[file], order=0.25), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill_75.csv', calc_hillnumbers(data[file], order=0.75), delimiter=',', fmt='%s')
#     np.savetxt('saved_data/' + file + 'eve.csv', calc_evenness(data[file]), delimiter=',', fmt='%s')
#
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(4)
#     with pool as p:
#         p.map(funk, data)

"""
        ---Test offs---  
"""
# test = [[5], [4, 1], [4, 1, 1], [3, 0, 1], [3, 0, 0, 1]]
# test2 = [[5], [5, 1], [4, 1, 1, 1], [3, 2, 1, 0], [0, 3, 0,	0]]
# data = {'test': test, 'test2': test2}
# #
# for t in data:
#     print(calc_evenness(data[t]))
#     print(calc_ginisimpson(data[t]))
#     print(calc_simpson(data[t]))

"""
        ---differenzen plot---  
"""
# tend = 40001
# x = np.arange(0, tend)
# fig, ax = plt.subplots(figsize=(12, 4))
# sh, = ax.plot(x, ave_sh['onerc'] - ave_sh['onenode'], farben['sh'], label='shannon', linewidth=0.75)
# gi, = ax.plot(x, ave_gi['onerc'] - ave_gi['onenode'], farben['gi'], label='ginisimpson', linewidth=0.75)
# h2, = ax.plot(x, ave_hill2['onerc'] - ave_hill2['onenode'], farben['hill_2'], label='hill order 2', linewidth=0.75)
# h_5, = ax.plot(x, ave_hill_5['onerc'] - ave_hill_5['onenode'], farben['hill_5'], label='hill order 0.5', linewidth=0.75)
# lines = [sh, gi, h2, h_5]
# ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
# ax.set(xlabel='timesteps', ylabel='difference onerc-onenode')
# plt.xlim(0, tend - 1)
# if tend >= 10000:
#     plt.xticks(np.arange(0, tend, 5000))
#
# # plt.ylim(bottom=0)
# plt.axhline(y=0, linewidth=1, linestyle=(0, (1, 10)))
# save_plot(plot=fig, filename='ave_diff.jpg')
# plt.show()

"""
    --- create avereaged entropies  ---
"""
# path167 ="saved_data/501167_ges/"
# path1 ="saved_data/5011_ges/"
# data167 = set_data(path167)
# data1 = set_data(path1)
# datas = {'167': data167, '1': data1}
# for data in datas:
#     create_averaged_entropies(datas[data], save=True, saveplot=True, plot=True)
# #
# names1 = []
# for name in data1:
#     names1.append(name)
# print(names1)
# names167 = []
# for name in data167:
#     names167.append(name)
# print(names167)