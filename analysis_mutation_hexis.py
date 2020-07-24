# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import pandas as pd

from lgca.helpers2d import plot_density_after


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

def plot_values_hexis(size, mut, akti, save=False, id=0):
    x = np.arange(0, 2501)

    fig, host = plt.subplots(figsize=(8, 4))
    par1 = host.twinx()

    host.set_xlim(0, 2500)
    host.set_ylim(bottom=0, top=610)
    par1.set_ylim(bottom=0, top=60)
    # host.set_ylim(bottom=0, top=230000)
    # par1.set_ylim(bottom=0, top=6000)

    host.set_xlabel("timesteps")
    host.set_ylabel("population size")
    par1.set_ylabel("number of families")

    p1, = host.plot(x, size, 'red', linewidth=1.5, label="population size")
    p2, = par1.plot(x, mut, 'seagreen', linewidth=1.5, linestyle='dotted', label="total")
    p3, = par1.plot(x, akti, 'seagreen', linewidth=1.5, label="active")

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines])
    if save:
        filename = str(id) + '_comparing_values' + '.jpg'
        plt.savefig(pathlib.Path('pictures').resolve() / filename, bbox_inches='tight')
    plt.show()

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
# path = 'saved_data/driver_45sims/'
path = 'C:/Users/Franzi/Downloads/data_nextversion/'
named = 'd785cf8_50x50rc=500_driver'
namep = '46a8f13_50x50rc=500_passenger'

names = [named, namep]

# path = 'saved_data/passenger_45sims/'
# path = 'saved_data/hexi_test/'
# names = search_names(path)
size_d = []
size_p = []
size = []
fam_max_d = []
fam_max_p = []
akti_d = []
akti_p = []
oris = []
rel = []

# for name in names:
#     print(name)
#     fm = 0
#     offs = search_offs(path, name)
#     for z in range(0, len(offs)):
#         print(offs[z])

# name = named
# offs = search_offs(path, name)
# for t in range(0, 2501):
#     fm = 0
#     size_d.append(sum(offs[t]))
#     fam_max_d.append(len(offs[t])-1)
#     fm = [fm + 1 for i in offs[t] if i != 0]
#     akti_d.append(sum(fm))
#
# name = namep
# offs = search_offs(path, name)
# for t in range(0, 2501):
#     fm = 0
#     size_p.append(sum(offs[t]))
#     fam_max_p.append(len(offs[t])-1)
#     fm = [fm + 1 for i in offs[t] if i != 0]
#     akti_p.append(sum(fm))

# plot_values_hexis(size=size_p, mut=fam_max_p, akti=akti_p, save=True, id='passenger')
# plot_values_hexis(size=size_d, mut=fam_max_d, akti=akti_d, save=True, id='driver')
# steps = 1301
# fig, host = plt.subplots(figsize=(8, 4))
# plt.xlim(0, steps-1)
# plt.ylim(0, 2000)
# plt.plot(range(0, steps), size_d[:steps], c='Indigo', label='driver')
# plt.plot(range(0, steps), size_p[:steps], c='MediumSeaGreen', label='passenger')
# plt.legend()
#
# save_plot(fig, 'popsize_until1300')
# plt.show()



#         if offs[t][0] != 0:
#             oris.append(1)
#         else:
#             oris.append(0)
#         rel.append(max(offs[t])/sum(offs[t]))
#         if len(offs) != 2501:
#         print('!!')
#     if sum(offs[-1]) == 0:
#         print(name, 'ist ausgestorben')
# sims = ['']*35
# for i in range(0, 7):
#     sims[i*5] = names[i][:6]
# print('l', len(size), len(fam_max), len(akti), len(oris), len(rel))
# import csv
# toWrite = [['Simulation:'] + sims,
#            ['Popgröße:'] + size,
#            ['Anz. Mutationen:'] + fam_max,
#            ['aktive Familien:'] + akti,
#            ['relativer Anteil maxfam:'] + rel,
#            ['Anfangsfamilie da:'] + oris
#            ]
# # print(toWrite)
#
# file = open(path + 'hexi_nextversion.csv', 'w')
#
# with file as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     for row in toWrite:
#         writer.writerow(row)



# path = 'saved_data/passenger_45sims/'
# names = search_names(path)
# names2 = names
# for name in names2:
#     print(name)
    # data_driver[str(name[:6])] = search_offs(path, name)

# def funk(file):
#     print('file', file)
#     offs = search_offs(path, file)
#     np.savetxt(path + file + 'sh.csv', calc_shannon(offs), delimiter=',', fmt='%s')
#     # # np.savetxt(path + file + 'gi.csv', calc_ginisimpson(data_driver[file]), delimiter=',', fmt='%s')
#     np.savetxt(path + file + 'hill2.csv', calc_hillnumbers(offs), delimiter=',', fmt='%s')
#     # # np.savetxt(path + file + 'size.csv', calc_popsize(data_driver[file]), delimiter=',', fmt='%s')
#     # # np.savetxt(path + file + 'rich.csv', calc_richness(data_driver[file]), delimiter=',', fmt='%s')
#     print('fertig mit ', file)
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(4)
#     with pool as p:
#         p.map(funk, names2)

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
size_d = []
size_p = []
path = 'saved_data/passenger_45sims/'
# path = 'saved_data/driver_45sims/'
names = search_names(path)
for name in names:
    offs = search_offs(path, name)
    size_p.append(sum(offs[-1]))
    # size_d.append(sum(offs[-1]))
#     print(sum(offs[-1]))
fig, ax = plt.subplots(figsize=(13, 8))
plt.xlabel('Simulation', fontsize=30)
plt.ylabel('finale Populationsgröße', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 45.5, 5)
dotsize = [200]*45
# #
# #-----DRIVER-----
# plt.scatter(range(1, 46), size_d, c='black', s=dotsize)
# plt.yticks([0, 300000, 700000], ['0', '3$\cdot$10e5', '7$\cdot$10e5'])
# plt.ylim(0, 700000, 4)
# save_plot(plot=fig, filename='finalsize_driver')
# plt.show()
#
#
# #-----PASSENGER-----
plt.scatter(range(1, 46), size_p, s=dotsize, c='Black')
plt.yticks([0, 1000, 2000], ['0', '1$\cdot$10e3', '2$\cdot$10e3'])
plt.ylim(0, 2000)

save_plot(plot=fig, filename='finalsize_passenger')
plt.show()
#

p_hill = np.loadtxt('saved_data/passenger_45sims/passenger_ave_hill2.csv')
p_sh = np.loadtxt('saved_data/passenger_45sims/passenger_ave_sh.csv')
d_hill = np.loadtxt('saved_data/driver_45sims/driver_ave_hill.csv')
d_sh = np.loadtxt('saved_data/driver_45sims/driver_ave_sh.csv')

# plot_sth(data={'passenger': p_hill[:1400], 'driver': d_hill[:1400]}, ylabel='$D_2(k)$',
#          yrange=[3.5,0.4,3.4], save=True, savename='ave_hill_TP3')
# plot_sth(data={'passenger': p_sh[:1400], 'driver': d_sh[:1400]},
#          ylabel='$H(k)$',yrange=[1.5, 0.2, 1.4], save=False, savename='shannon_d_p_bis1400')

o_d = correct(np.load('saved_data/hexis_mit_bild/d785cf8_50x50rc=500_driver_offsprings.npy'))
o_p = correct(np.load('saved_data/hexis_mit_bild/46a8f13_50x50rc=500_passenger_offsprings.npy'))
sd = calc_bergerparker(o_d)
sp = calc_bergerparker(o_p)
richd = calc_richness(o_d)[:1401]
richp = calc_richness(o_p)[:1401]
sized = calc_popsize(o_d)[:1401]
sizep = calc_popsize(o_p)[:1401]
# print(sd[0], sp[0])
# plot_sth(data={'passenger': sp, 'driver': sd}, ylabel='$d(k)$', yrange=[1.1, 0.25, 1.1])
# plot_sth(data={'passenger': richp, 'driver': richd}, ylabel='$S(k)$', yrange=[19,3, 19])
plot_sth(data={'passenger': sizep, 'driver': sized}, ylabel='$N(k)$', yrange=[4001,1000, 4000])
