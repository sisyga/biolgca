# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

# names = ['Test_ohne shuffle', 'Test_rd=0.3', 'Test_normal', 'Test_nochmal']
# names = ['Test_rd=0.3']
# for i, name in enumerate(names):
#     labels = np.load('saved_data/' + name + '_labels.npy')
#     nodest = np.load('saved_data/' + name + '_nodest.npy')
#     offsprings = np.load('saved_data/' + name + '_offsprings.npy')
#
#
#     print(offsprings)
#     print(len(offsprings))
#     # mullerplot(data=offsprings, int_length=3)

# offs = np.array([[-99,2,2,2,2,2], [-99,3,1,2,2,2], [-99,3,0,3,2,2], [-99,4,0,3,1,2], [-99,4,0,3,0,3]\
#                  , [-99,5,0,2,0,3], [-99,6,0,2,0,2], [-99,6,0,1,0,3], [-99,6,0,0,0,4], [-99,7,0,0,0,3]\
#                     , [-99,8,0,0,0,2], [-99,9,0,0,0,1], [-99,10,0,0,0,0]])
# print(offs)
# s = calc_shannon(offs)
# print(s)
#
# plot_index(index_data = s, which='shannon')
# plot_index(index_data=calc_hillnumbers(offs), which='hill2')
# plot_selected_entropies(shannon=s, hill2=calc_hillnumbers(offs), gini=calc_ginisimpson(offs))

offs = np.load('saved_data/' + '123_1_cdb3d6a_offsprings.npy')
print(offs)
