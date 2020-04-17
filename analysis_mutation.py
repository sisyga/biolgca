# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing

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

def plot_ind_com(which='si', save=False, plot=True):
    path = 'saved_data/Indizes_explizit/'
    files = os.listdir(path)
    dataset = {}
    ind = which
    for file in files:
        if ind in file:
            dataset[file[:-(4+len(ind))]] = np.loadtxt(path + file)
    if plot:
        plot_sth(dataset, ylabel=ind, save=save, id=which)
    return dataset

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


# data1 = correct(np.load('saved_data/5011_mut_04_01/'
#                         '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))
# data167 = correct(np.load('saved_data/501167_mut_04_02/'
#                           '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))

inds = ['si', 'gi', 'sh', 'hill1', 'hill2', 'hill3', 'hill_5', 'hill_25', 'hill_75']
# inds = ['hill1', 'hill2', 'hill3', 'hill_5']
#
si = plot_ind_com(which='si', save=False, plot=False)
sh = plot_ind_com(which='sh', save=False, plot=False)
eve = plot_ind_com(which='eve', save=False, plot=False)
gi = plot_ind_com(which='gi', save=False, plot=False)
hill1 = plot_ind_com(which='hill1', save=False, plot=False)
hill2 = plot_ind_com(which='hill2', save=False, plot=False)
hill3 = plot_ind_com(which='hill3', save=False, plot=False)
hill_5 = plot_ind_com(which='hill_5', save=False, plot=False)
hill_25 = plot_ind_com(which='hill_25', save=False, plot=False)
hill_75 = plot_ind_com(which='hill_75', save=False, plot=False)

# # # for ind in inds:
# # #     plot_ind_com(which=ind, save=True, plot=True)
# vars = ['onenode', 'onerc']
vars = ['onenode']

# m = max(hill_25['onenode'][22500:30000])
# print(m)
# p = list(hill_25['onenode'][22500:30000]).index(m)
# print(p)
# print(hill2['onenode'][22500+562])
# gr = 0
# gl = 0
# kl = 0
# n = 0
# for i, entry in enumerate(sh['onenode']):
#     if entry > gi['onenode'][i]:
#         gr += 1
#     elif entry == gi['onenode'][i]:
#         gl += 1
#         if entry == 0:
#             n += 1
#     elif entry < gi['onenode'][i]:
#         kl += 1
# print(gr, gl, kl, n)
# print(gr+gl+kl)
# plot_sth(data=gi, save=True, id='gi')
# plot_sth(data=hill2, save=True, id='hill2')
# plot_sth(data=hill_5, save=True, id='hill_5')


for var in vars:
    # plot_sth(data={'onenode': si[var] - gi[var]}, ylabel='diff: simpson - gini', savename='diff_SiGi', save=True)
    # plot_sth(data={'gi': gi[var], 'eve': eve[var]}, save=True, savename=var + '_GiEve')
    # plot_sth(data={'gi': gi[var], 'sh': sh[var], 'eve': eve[var]}, save=True, savename=var + '_GiEveSh')
    # plot_sth(data={'gi': gi[var], 'sh': sh[var]}, save=True, savename=var + '_GiSh')
#     plot_hillnumbers_together(hill2[var],hill_25[var], hill_75[var], save=True, id=var)
#     plot_entropies_together(gini=gi[var], shannon=sh[var])
    plot_selected_entropies(gini=gi[var], shannon=sh[var], hill2=hill2[var], save=False, id=var)
#
#
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


# test = [[5], [4, 1], [4, 1, 1], [3, 0, 1], [3, 0, 0, 1]]
# test2 = [[5], [5, 1], [4, 1, 1, 1], [3, 2, 1, 0], [0, 3, 0,	0]]
# data = {'test': test, 'test2': test2}
# #
# for t in data:
#     print(calc_evenness(data[t]))
#     print(calc_ginisimpson(data[t]))
#     print(calc_simpson(data[t]))
