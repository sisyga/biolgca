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
path = 'C:/Users/Franzi/Downloads/sim123_driver.tar/sim123_driver/'
names = search_names(path)
# np.savetxt(path + '.csv', names, delimiter=',', fmt='%s')
for name in names:
    offs = search_offs(path, name)
    if len(offs) != 2501:
        print('!!')
    if sum(offs[-1]) == 0:
        print(name, 'ist ausgestorben')





