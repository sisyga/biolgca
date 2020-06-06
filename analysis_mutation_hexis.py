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

# def read_inds(which='si'):
#     path = 'saved_data/Indizes_explizit/'
#     files = os.listdir(path)
#     dataset = {}
#     ind = which
#     for file in files:
#         if ind in file:
#             dataset[file[:-(4+len(ind))]] = np.loadtxt(path + file)
#     # print('d', dataset)
#     return dataset
#
# def ave_inds(which='shannon', plot=False, save=False, savename=None):
#     path = 'saved_data/Indizes_averaged/'
#     files = os.listdir(path)
#     dataset = {}
#     ind = which
#     for file in files:
#         if ind in file:
#             dataset[file[:-(14+len(ind))]] = np.loadtxt(path + file)
#     if plot:
#         plot_sth(dataset, ylabel=ind, save=save, id=which, savename=savename)
#     return dataset
path = 'saved_data/driver_45sims/'
# path = 'saved_data/hexi_test/'
names = search_names(path)
# for name in names:
#     offs = search_offs(path, name)
#     if len(offs) != 2501:
#         print('!!')
#     if sum(offs[-1]) == 0:
#         print(name, 'ist ausgestorben')

data_driver = {}
for name in names[5:15]:
    print(name)
    data_driver[str(name[:6])] = search_offs(path, name)


def funk(file):
    # print(file)
    np.savetxt(path + file + 'sh.csv', calc_shannon(data_driver[file]), delimiter=',', fmt='%s')
    # np.savetxt(path + file + 'gi.csv', calc_ginisimpson(data_driver[file]), delimiter=',', fmt='%s')
    np.savetxt(path + file + 'hill2.csv', calc_hillnumbers(data_driver[file]), delimiter=',', fmt='%s')
    # np.savetxt(path + file + 'size.csv', calc_popsize(data_driver[file]), delimiter=',', fmt='%s')
    # np.savetxt(path + file + 'rich.csv', calc_richness(data_driver[file]), delimiter=',', fmt='%s')


if __name__ == '__main__':
    pool = multiprocessing.Pool(4)
    with pool as p:
        p.map(funk, data_driver)

# r = [0]*11
# g = [0]*11
# for name in names:
#     r += np.loadtxt(path + name[:6] + 'rich.csv')
#     g += np.loadtxt(path + name[:6] + 'gi.csv')
#
#
# r = r/len(names)
# g = g/len(names)
# print(g)