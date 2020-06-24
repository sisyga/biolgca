# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math

def whichone(which):
    if which == 0:
        thom = thombsp
        nbars = int(math.ceil(len(thom) ** 0.5))
        print('nbars', nbars)
        int_length = 7

    elif which == 1:
        thom = thom_01
        nbars = int(math.ceil(len(thom) ** 0.5))
        print('nbars', nbars)
        int_length = 4575
    elif which == 2:
        thom = thom_167
        int_length = 4575
    return thom, int_length

thom_01 = np.load('C:/Users/Franzi/PycharmProjects/biolgca/saved_data/thoms501/501thom01_1000.npy')
thom_167 = np.load('C:/Users/Franzi/PycharmProjects/biolgca/saved_data/thoms501/501thom167_1000.npy')

thombsp = np.array([1,3,5,7,\
        11,11,13,15,16,12,18,17,\
        22,21,25,27,21,22,\
        35,33])

    # 0 = bsp  1 = 01  2 = 167
thom, int_length = whichone(1)
plot_lognorm_distribution(thom, int_length)
thom, int_length = whichone(2)
plot_lognorm_distribution(thom, int_length)


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



