# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd

def read_data(path, dataname, steps):
    ns = ['k_' + str(i) for i in range(0, steps)]

    df = pd.read_table(path + dataname, sep=',', names=ns)
    # print(df)
    if len(df) != 45:
        print('FEHLER!')
    return df

def series(table, which, steps):
    values = []
    for k in range(0, steps):
        values.append(table['k_' + str(k)][which])
    return values

def saving(data, name, w):
    np.savetxt('saved_data/' + w + name, data, delimiter=',', fmt='%s')

def reading(path, gitter, index):
    dataname = gitter + '_' + index + '_501.csv'

    mean = np.loadtxt(path + 'mean' + dataname)
    stdabw = np.loadtxt(path + 'stdabw' + dataname)
    stdmw = np.loadtxt(path + 'stdmw' + dataname)

    return mean, stdabw, stdmw

'''
    einlesen I und II
'''
# path = 'saved_data/'
# steps = 40001
# index = 'hh'
#
# if index == 'hh':
#     ylab = '$D_2(k)$'
#     ymini = 1
# elif index == 'sh':
#     ylab = '$H(k)$'
#     ymini = 0
#
# fig, ax = plt.subplots(figsize=(12, 6))
# size_ticks = 20
# size_legend = 30
#
# for g in [1, 167]:
#     if g ==1:
#         farbe = 'darkred'
#         efarbe = 'darkred'
#         l = 'I'
#     elif g == 167:
#         farbe = 'darkorange'
#         efarbe='orange'
#         l = 'II'
#     mean, stdabw, stdmw = reading(path, str(g), index)
#     print(len(mean), len(stdabw), len(stdmw))
#
#     plt.plot(range(0, steps), mean, color=farbe, linewidth=1.5, label=l)
#     ax.errorbar(range(0, steps), y=mean, yerr=stdmw, linewidth=1.5,
#                 color=efarbe, alpha=0.13, elinewidth=0.07)
# #
# ax.set(xlim=(0, steps-1), ylim=ymini)
# ax.legend(loc='upper left', fontsize=size_ticks)
# plt.xticks(np.arange(0, steps, 10000), fontsize=size_ticks)
# plt.yticks(fontsize=size_ticks)
# #np.arange(0, 2.6, 0.5),
# plt.xlabel('Zeitschritte', fontsize=size_legend)
# plt.ylabel(ylab, fontsize=size_legend)
# # #
# # filename = 'abweichungskrams_' + index + '.jpg'
# # plt.savefig(pathlib.Path('pictures').resolve() / filename)
# # #
# plt.show()

'''
Abweichungen berechnen I und II
'''
# steps = 40001
# path = 'saved_data/'
# names = ['1_hh_501' + '.csv', '167_hh_501' + '.csv',
#          '1_sh_501' + '.csv', '167_sh_501' + '.csv']
#
# for dataname in names:
#     thoms = read_data(path=path, dataname=dataname, steps=steps)
#     means = thoms.mean()
#     saving(means, dataname, 'mean')
#     # print(thoms)
#     anz = len(thoms)
#     print(anz)
#     stdabw = []
#     for i in range(0, steps):
#         stdabw.append(thoms['k_' + str(i)].std())
#     saving(stdabw, dataname, 'stdabw')
#     # print(stdabw)
#     stdmw = []
#     for entry in stdabw:
#         stdmw.append(entry/(anz**0.5))
#     saving(stdmw, dataname, 'stdmw')
