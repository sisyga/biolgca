# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

# #choose simulation
# variation = 'Testdaten'
# filename = ''
# path = 'saved_data/'
# rep = 3
#
# selection = [0]
#
# # variation = '18060'
# # filename = '6e8510d'
# # path = 'saved_data/pummelzeugs_60/'
# # rep = 500
#
# for i in selection:
#     data = np.load(path + variation + '_' + str(i) + '_' + filename + '_offsprings.npy')
#     tend, _ = data.shape
#     mullerplot(data, int_length=2, save=True, id=variation + '_' + str(i))

labels = np.load('saved_data/82_0_06c0de0_labels.npy')
nodest = np.load('saved_data/82_0_06c0de0_nodest.npy')

# print(nodest[:20])

spacetime_plot(nodes_t=nodest, labels=labels, save=False, tend=10, id='Testspace')