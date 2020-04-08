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
# path167 ="saved_data/501167_mut_04_02/"
# path1 ="saved_data/5011_mut_04_01/"
# data167 = set_data(path167)
# data1 = set_data(path1)
# # create_averaged_entropies(data, save=True, saveplot=True, plot=True)
#
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
inds = ['si', 'gi', 'sh', 'hill1', 'hill2', 'hill3', 'hill_5']
inds = ['gi', 'sh', 'hill2']
sh = plot_ind_com(which='sh', save=False, plot=False)
gi = plot_ind_com(which='gi', save=False, plot=False)
hill2 = plot_ind_com(which='hill2', save=False, plot=False)
# for ind in inds:
#     plot_ind_com(which=ind, save=True, plot=True)
for var in sh:
    plot_selected_entropies(shannon=sh[var], hill2=hill2[var], gini=gi[var], save=True, id=var)
# plot_ind_com(which=inds[-1], save=True)
# plot_sth(shannon)



# def funk(file):
#     print(file)
#     # np.savetxt('saved_data/' + file + 'sh.csv', calc_shannon(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'si.csv', calc_simpson(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'gi.csv', calc_ginisimpson(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill1.csv', calc_hillnumbers(data[file], order=1), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill2.csv', calc_hillnumbers(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill3.csv', calc_hillnumbers(data[file], order=3), delimiter=',', fmt='%s')
#     np.savetxt('saved_data/' + file + 'hill_5.csv', calc_hillnumbers(data[file], order=0.5), delimiter=',', fmt='%s')
#
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(4)
#     with pool as p:
#         p.map(funk, data)

