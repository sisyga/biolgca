# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

names = ['Test_ohne shuffle', 'Test_rd=0.3', 'Test_normal', 'Test_nochmal']

for i, name in enumerate(names):
    labels = np.load('saved_data/' + name + '_labels.npy')
    nodest = np.load('saved_data/' + name + '_nodest.npy')
    offsprings = np.load('saved_data/' + name + '_offsprings.npy')

    print(len(nodest), len(offsprings))


