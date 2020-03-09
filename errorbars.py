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

plot_lognorm_distribution(thom, int_length)
# x = np.arange(1, max(thom)+1)
# print(x)
# y = np.zeros(max(thom))
# for i in range(0, len(thom)):
#     y[thom[i] - 1] += 1
# print(y)
# plt.bar(x, y)
# # plt.errorbar(x, y, yerr=[1,1,1,1], color='magenta')
# sigma = np.zeros(len(x))
# n = y.sum()
# print(n)
# for i in range(0, len(x)):
#     sigma[i] = (y[i]*(1-(y[i]/n)))**0.5
# print(sigma)
# plt.errorbar(x, y, yerr=sigma, color='magenta')
#
# plt.show()



