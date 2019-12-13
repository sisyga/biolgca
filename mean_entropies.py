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

time = maxt
x = np.arange(0, time, 1)
shan = mean_sh
hh = mean_hh
gini = mean_gi

fig, host = plt.subplots()
par1 = host.twinx()
par2 = host.twinx()
# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

p1, = host.plot(x, shan, "m", linewidth=0.7, label="Shannonindex")
p2, = par1.plot(x, gini, "b", linewidth=0.7, label="GiniSimpsonindex")
p3, = par2.plot(x, hh, "c", linewidth=0.7, label="Hillnumber of order 2")

host.set_xlim(0, time - 1)
host.set_ylim(bottom=0)
par1.set_ylim(bottom=0)
par2.set_ylim(bottom=0)

host.set_xlabel("timesteps")
host.set_ylabel("Shannonindex")
par1.set_ylabel("GiniSimpsonindex")
par2.set_ylabel("Hillnumber of order 2")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines])
plt.show()


