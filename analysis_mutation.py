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

def set_data(path):
    names = search_names(path)
    data = {}
    for name in names:
        data[name] = search_offs(path, name)
    return data

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


data1 = correct(np.load('saved_data/5011_mut_04_01/'
                        '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))
data167 = correct(np.load('saved_data/501167_mut_04_02/'
                          '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))
data = {'onenode': data1, 'onerc': data167}

shannon = {}
simpson = {}
gini = {}
hill1 = {}
hill2 = {}
hill3 = {}
hill_5 = {}
for var in data:
    o = data[var]
    shannon[var] = calc_shannon(o)
    simpson[var] = calc_simpson(o)
    gini[var] = calc_ginisimpson(o)
    hill1[var] = calc_hillnumbers(o, 1)
    hill2[var] = calc_hillnumbers(o, 2)
    hill3[var] = calc_hillnumbers(o, 3)
    hill_5[var] = calc_hillnumbers(o, 0.5)

# plot_sth(shannon)