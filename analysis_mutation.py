# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os

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


path ="saved_data/5011_mut_04_01/"
names = search_names(path)
data = {}
# for name in names:
#     data[name] = search_offs(path, name)
# print(data.keys())
test = [[5], [4, 1], [4, 1, 1], [3, 0, 1], [3, 0, 0, 1]]
test2 = [[5], [5, 1], [4, 1, 1, 1], [3, 2, 1, 0], [0, 3, 0,	0]]
data = {'test': test, 'test2': test2}

print(create_averaged_entropies(data, save=True))
# sh = np.load('saved_data/test_averaged_shannon.npy')
# hh = np.load('saved_data/test_averaged_hill2.npy')
# gi = np.load('saved_data/test_averaged_gini.npy')
# plot_selected_entropies(sh, hh, gi)
