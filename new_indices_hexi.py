# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
import os

def load_scenario(sc):
    path = 'saved_data/' + sc + '_45sims'
    files = os.listdir(path)
    sh = [entry for entry in files if 'sh' in entry]
    hh = [entry for entry in files if 'hill2' in entry]
    print(len(sh))
    print(len(hh))
    return sh, hh

def read_scenario(list, sc, time):
    path = 'saved_data/' + sc + '_45sims/'
    data = []
    # i = 0
    for entry in list:
        data.append(np.loadtxt(path + entry)[time])
    return data

def saving(data, name, w):
    np.savetxt('saved_data/' + w + '_45sims/Berechnungen/' + name + '.csv', data, delimiter=',', fmt='%s')


# p_sh_names, p_hill_names = load_scenario('passenger')
# d_sh_names, d_hill_names = load_scenario('driver')
#
# p_means_sh = np.loadtxt('saved_data/passenger_45sims/Berechnungen/passenger_ave_sh.csv')
# p_means_hill = np.loadtxt('saved_data/passenger_45sims/Berechnungen/passenger_ave_hill2.csv')
# d_means_sh = np.loadtxt('saved_data/driver_45sims/Berechnungen/driver_ave_sh.csv')
# d_means_hill = np.loadtxt('saved_data/driver_45sims/Berechnungen/driver_ave_hill.csv')

# steps = 2501

# mut = 'driver'
# mut = 'passenger'
# w = 'sh'
# names = d_sh_names
# names = p_sh_names
# w = 'hill'
# names = d_hill_names
# names = p_hill_names
#
# inds_t = []
# stdab = []
# mwstd = []
# mean = []
# for t in range(0, steps):
#     inds_t = np.array(read_scenario(names, mut, t))
#     # print(len(inds_t))
#     stdab.append(inds_t.std())
#     mwstd.append(inds_t.std()/(45**0.5))
#     mean.append(np.mean(inds_t))
#
# print(stdab[1000], mwstd[1000])
# print(d_means_hill[-5:], mean[-5:])
# saving(stdab, w + mut + 'stdabw', mut)
# saving(mwstd, w + mut + 'mwstd', mut)
# saving(mean, w + mut + 'mean', mut)


path = 'saved_data/passenger_45sims/Berechnungen/'
pass_sh_mean = np.loadtxt(path + 'shpassengermean.csv')
pass_sh_abw = np.loadtxt(path + 'shpassengerstdabw.csv')
pass_sh_std = np.loadtxt(path + 'shpassengermwstd.csv')

pass_hill_mean = np.loadtxt(path + 'hillpassengermean.csv')
pass_hill_abw = np.loadtxt(path + 'hillpassengerstdabw.csv')
pass_hill_std = np.loadtxt(path + 'hillpassengermwstd.csv')

path = 'saved_data/driver_45sims/Berechnungen/'
driv_sh_mean = np.loadtxt(path + 'shdrivermean.csv')
driv_sh_abw = np.loadtxt(path + 'shdriverstdabw.csv')

driv_hill_mean = np.loadtxt(path + 'hilldrivermean.csv')
driv_hill_abw = np.loadtxt(path + 'hilldriverstdabw.csv')
driv_hill_std = np.loadtxt(path + 'hilldrivermwstd.csv')

steps = 2501

index = 'hh'
if index == 'hh':
    ylab = '$D_2(k)$'
    ymini = 1
elif index == 'sh':
    ylab = '$H(k)$'
    ymini = 0

fig, ax = plt.subplots(figsize=(12, 6))
size_ticks = 20
size_legend = 30

fp = 'MediumSeaGreen'
fd = 'Indigo'

if index == 'hh':
    plt.plot(range(0, steps), pass_hill_mean, color=fp, linewidth=1.5, label='Passenger')
    ax.errorbar(range(0, steps), y=pass_hill_mean, yerr=pass_hill_std, linewidth=1.5,
                    color=fp, alpha=0.13, elinewidth=0.07)

    plt.plot(range(0, steps), driv_hill_mean, color=fd, linewidth=1.5, label='Driver')
    ax.errorbar(range(0, steps), y=driv_hill_mean, yerr=driv_hill_std, linewidth=1.5,
                    color=fd, alpha=0.13, elinewidth=0.07)
#
ax.set(xlim=(0, steps-1), ylim=ymini)
ax.legend(loc='upper left', fontsize=size_ticks)
plt.xticks(np.arange(0, steps), fontsize=size_ticks)
plt.yticks(fontsize=size_ticks)
#np.arange(0, 2.6, 0.5),
plt.xlabel('Zeitschritte', fontsize=size_legend)
plt.ylabel(ylab, fontsize=size_legend)
#
# filename = 'abweichungskrams_' + index + '.jpg'
# plt.savefig(pathlib.Path('pictures').resolve() / filename)
# #
plt.show()

print(pass_hill_mean[:10], pass_hill_std[:10], pass_hill_abw[:10])
print(max(pass_hill_std), max(pass_hill_abw))