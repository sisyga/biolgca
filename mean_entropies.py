from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import pandas as pd

cells = 181
name = 'b45955d'
file = '18045'
look = 5500
li = 1000
ids = []
offs0 =  np.load('saved_data/'+ name +'/' + file + '_0_' + name + '_offsprings' +'.npy')
offs1 =  np.load('saved_data/'+ name +'/' + file + '_1_' + name + '_offsprings' +'.npy')
offs2 =  np.load('saved_data/'+ name +'/' + file + '_2_' + name + '_offsprings' +'.npy')
offs3 =  np.load('saved_data/'+ name +'/' + file + '_3_' + name + '_offsprings' +'.npy')
offs4 =  np.load('saved_data/'+ name +'/' + file + '_4_' + name + '_offsprings' +'.npy')
offs5 =  np.load('saved_data/'+ name +'/' + file + '_5_' + name + '_offsprings' +'.npy')
offs6 =  np.load('saved_data/'+ name +'/' + file + '_6_' + name + '_offsprings' +'.npy')
offs7 =  np.load('saved_data/'+ name +'/' + file + '_7_' + name + '_offsprings' +'.npy')
offs8 =  np.load('saved_data/'+ name +'/' + file + '_8_' + name + '_offsprings' +'.npy')
offs9 =  np.load('saved_data/'+ name +'/' + file + '_9_' + name + '_offsprings' +'.npy')
# offs = [offs0, offs1, offs2, offs3, offs4, offs5, offs6, offs7, offs8, offs9]
offs = [offs0, offs1, offs5]
tend = [len(offs0), len(offs1), len(offs2), len(offs3), len(offs4), len(offs5), len(offs6), len(offs7), len(offs8), len(offs9)]
print(tend)
maxt = max(tend)
print(maxt)
sh = np.zeros((maxt, len(offs)))
gi = np.zeros((maxt, len(offs)))
hh = np.zeros((maxt, len(offs)))

for index, entry in enumerate(offs):
    sh_entropy = entropies(entry, order=1, off=True)[0]
    gi_entropy = entropies(entry, order=2, off=True)
    hh_entropy = hillnumber(entry, order=2, off=True)
    if len(sh_entropy) != maxt:
        sh_entropy = np.concatenate((sh_entropy, np.zeros(maxt - len(sh_entropy))))
        gi_entropy = np.concatenate((gi_entropy, np.zeros(maxt - len(gi_entropy))))
        hh_entropy = np.concatenate((hh_entropy, np.zeros(maxt - len(hh_entropy))))
    print(len(sh_entropy))
    sh[:, index] = sh_entropy
    gi[:, index] = gi_entropy
    hh[:, index] = hh_entropy
kontr = [5, 1562, maxt-2]
for entry in kontr:
    print('kontr daten', gi[entry, :], hh[entry, :])
    print('kontr', np.mean(gi[entry, :]), np.mean(hh[entry, :]))

mean_sh = np.zeros(maxt)
mean_gi = np.zeros(maxt)
mean_hh = np.zeros(maxt)
for i in range(0, maxt):
    mean_sh[i] = np.mean(sh[i, :])
    mean_gi[i] = np.mean(gi[i, :])
    mean_hh[i] = np.mean(hh[i, :])

for entry in kontr:
    print(mean_gi[entry], mean_hh[entry])

