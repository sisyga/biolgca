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
        c = None

    elif which == 1:
        thom = thom_01
        nbars = int(math.ceil(len(thom) ** 0.5))
        print('nbars', nbars)
        int_length = 4575
        c = farben['onenode']
    elif which == 2:
        thom = thom_167
        int_length = 4575
        c = farben['onerc']
    return thom, int_length, c

thom_01 = np.load('C:/Users/Franzi/PycharmProjects/biolgca/saved_data/thoms501/501thom01_1000.npy')
thom_167 = np.load('C:/Users/Franzi/PycharmProjects/biolgca/saved_data/thoms501/501thom167_1000.npy')

thombsp = np.array([1,3,5,7,\
        11,11,13,15,16,12,18,17,\
        22,21,25,27,21,22,\
        35,33])

    # 0 = bsp  1 = 01  2 = 167
thom, int_length, c = whichone(1)
# plot_lognorm_distribution(thom, int_length, c=c, id='onenode', save=True)
a = 0
for t in thom:
    if t <= 40000:
        a += 1
print(a)
thom, int_length, c = whichone(2)
# plot_lognorm_distribution(thom, int_length, c=c, id='onerc', save=True)
a = 0
for t in thom:
    if t <= 40000:
        a += 1
print(a)
# plot_all_lognorm({'onenode': thom_01, 'onerc': thom_167}, int_length=4575, save=True)


