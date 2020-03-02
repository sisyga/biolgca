# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math as m

thom01 = np.load('saved_data/thoms/' + 'rc=178_thom1500.npy')
# plot_lognorm_distribution(thom01, int_length=1000)
# exit(123456)

mini = np.array([1,3,2,3,2,3,4])

thombsp = np.array([1,3,5,7,\
        11,11,13,15,16,12,18,17,\
        22,21,25,27,21,22,\
        35,33])

    # 0 = mini  1 = bsp  2 = thom01
which = 2

if which == 0:
    thom = mini
    int_length = 1
elif which == 1:
    thom = thombsp
    int_length = 10
elif which == 2:
    thom = thom01
    int_length = 1000

max = thom.max().astype(int)
print('max', max)
print('mean', thom.mean())
print('std', thom.std())
fig, ax = plt.subplots()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('thom', fontsize=15)
plt.ylabel('absolute frequency', fontsize=15)
fitted_data, maxy, y = calc_lognormaldistri(thom=thom, int_length=int_length)
maxfit = fitted_data.max()
x = np.arange(0, max, int_length) + int_length/2


plt.xlim(0, x.max() + int_length/2)
plt.ylim(0, maxy + 25)

### plt.hist(thom, bins=np.arange(0, thom.max() + 2*int_length, int_length))
plt.bar(x, y, width=int_length, color='grey', alpha=0.5)


err = calc_barerrs(thom, int_length) / 100 #skaliert
print(err)
#plt.errorbar(x[y > 0], y[y > 0], yerr=err[y > 0], linestyle='') #TODO Käsequäse...?

plt.plot(x, fitted_data * maxy / maxfit)
plt.legend()
print(fitted_data * maxy / maxfit, y)
sqderr = calc_quaderr(fitted_data * maxy / maxfit, y) / 100
print(sqderr)

sqderr = sqderr * err
print(sqderr)
print(fitted_data*maxy/maxfit)
print(x)
plt.errorbar(x, fitted_data*maxy/maxfit, yerr=sqderr, lw=1, capsize=2, capthick=1, color='seagreen')


plt.show()
