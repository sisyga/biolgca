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


path = 'saved_data/'

# dataname = 'einlesen' + '.csv'
# steps = 5

ylab = '$D_2(k)$'
dataname = '1_hh_501' + '.csv'
# f = 'darkred'

# dataname = '167_hh_501' + '.csv'
f = 'darkred'

# ylab = '$H(k)$'
# dataname = '1_sh_501' + '.csv'
# f = 'darkred'

# dataname = '167_sh_501' + '.csv'
# f = 'darkred'

steps = 40001

save = True

thoms = read_data(path=path, dataname=dataname, steps=steps)
means = thoms.mean()
# print(thoms['k_0'])
err = []
for i in range(0, steps):
    err.append(thoms['k_' + str(i)].std())
print(err)
fig, ax = plt.subplots(figsize=(12, 8))
size_ticks = 20
size_legend = 30

# for v in range(0, len(thoms)):
#     x = range(0, steps)
#     y = series(table=thoms, steps=steps, which=v)
#     plt.plot(x, y, color='black', alpha=0.2, linewidth=0.5)
#
plt.plot(range(0, steps), means, color=f, linewidth=2)
ax.errorbar(range(0, steps), y=means, yerr=err, ls='', capsize=1, capthick=1,
            color='silver', label='Fehler')

ax.set(xlim=(0, steps-1), ylim=(1))
plt.xticks(np.arange(0, steps, 10000), fontsize=size_ticks)
plt.yticks(fontsize=size_ticks)
#np.arange(0, 2.6, 0.5),
plt.xlabel('Zeitschritte', fontsize=size_legend)
plt.ylabel(ylab, fontsize=size_legend)
#
if save:
    filename = 'inds' + str(dataname[:-4]) + '_std.jpg'
    plt.savefig(pathlib.Path('pictures').resolve() / filename)
#
plt.show()