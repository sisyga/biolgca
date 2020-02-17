# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt


# variation01 = '1801'
# path01 = 'saved_data/pummelzeugs_1/'
# thom01 = np.load(path01 + variation01 + '_thom.npy')
#
# variation02 = '1802'
# path02 = 'saved_data/pummelzeugs_2/'
# thom02 = np.load(path02 + variation02 + '_thom.npy')
#
# variation45 = '18045'
# path45 = 'saved_data/pummelzeugs_45/'
# thom45 = np.load(path45 + variation45 + '_thom.npy')
#
# variation60 = '18060'
# path60 = 'saved_data/pummelzeugs_60/'
# thom60 = np.load(path60 + variation60 + '_thom.npy')
#
# variation90 = '18090'
# path90 = 'saved_data/pummelzeugs_90/'
# thom90 = np.load(path90 + variation90 + '_thom.npy')

# data = {'rc=01': thom01, 'rc=02': thom02, 'rc=45': thom45, 'rc=60': thom60, 'rc=90': thom90}
# data = {'rc=01': thom01, 'rc=45': thom45, 'rc=90': thom90}




# create_averaged_entropies(variation='18090', filename='e0797e9', path='saved_data/pummelzeugs_90/',\
#                           rep=500, save=True, plot=True, saveplot=True)

# thom01 = np.load('saved_data/pummelzeugs_1/' + '1801_thom.npy')
# thom60 = np.load('saved_data/pummelzeugs_60/' + '18060_thom.npy')
# thom90 = np.load('saved_data/pummelzeugs_90/' + '18090_thom.npy')

# print(thom01[286])
# print(thom60[81])
# print(thom90[449])
#
data01 = {'offs': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_offsprings.npy'),\
          'labels': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_labels.npy'),\
          'nodes': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_nodest.npy')}

data02 = {'offs': np.load('saved_data/1802_nodes/' + '1802_0_995dedd_offsprings.npy'),\
          'labels': np.load('saved_data/1802_nodes/' + '1802_0_995dedd_labels.npy'),\
          'nodes': np.load('saved_data/1802_nodes/' + '1802_0_995dedd_nodest.npy')}

data03 = {'offs': np.load('saved_data/1803_nodes/' + '1803_1_7fa22bc_offsprings.npy'),\
          'labels': np.load('saved_data/1803_nodes/' + '1803_1_7fa22bc_labels.npy'),\
          'nodes': np.load('saved_data/1803_nodes/' + '1803_1_7fa22bc_nodest.npy')}

data45 = {'offs': np.load('saved_data/18045_nodes/' + '18045_3_fca2b14_offsprings.npy'),\
          'labels': np.load('saved_data/18045_nodes/' + '18045_3_fca2b14_labels.npy'),\
          'nodes': np.load('saved_data/18045_nodes/' + '18045_3_fca2b14_nodest.npy')}

data60 = {'offs': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_offsprings.npy'),\
          'labels': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_labels.npy'),\
          'nodes': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_nodest.npy')}

data90 = {'offs': np.load('saved_data/18090_nodes/' + '18090_1_b8e3972_offsprings.npy'),\
          'labels': np.load('saved_data/18090_nodes/' + '18090_1_b8e3972_labels.npy'),\
          'nodes': np.load('saved_data/18090_nodes/' + '18090_1_b8e3972_nodest.npy')}

saving = True
data = {'03': data03}
# data = {'01': data01, '02': data02, '45': data45, '60': data60, '90': data90}
#
# for entry, name in enumerate(data):
#     print(entry, name)
#
#     spacetime_plot(nodes_t=data[name]['nodes'], labels=data[name]['labels'],\
#                    tend=None, save=saving, id=str(180) + str(name))
#     spacetime_plot(nodes_t=data[name]['nodes'], labels=data[name]['labels'],\
#                    tend=500, save=saving, id=str(180) + str(name))
#     spacetime_plot(nodes_t=data[name]['nodes'], labels=data[name]['labels'],\
#                    tend=100, save=saving, id=str(180) + str(name))


# mullerplot(data=data01['offs'], id='18001_81', save=saving, int_length=250)
# if len(data01['nodes']) != len(data01['offs']):
#     print('NEIIIIIIIIIIIN!')
#
# data60 = {'offs': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_offsprings.npy'),\
#           'labels': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_labels.npy'),\
#           'nodes': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_nodest.npy')}
#
# if len(data60['nodes']) != len(data60['offs']):
#     print('NEIIIIIIIIIIIN!')
#
# data90 = {'offs': np.load('saved_data/pummelzeugs_90/' + '18090_38_e0797e9_offsprings.npy'),\
#           'labels': np.load('saved_data/pummelzeugs_90/' + '18090_38_e0797e9_labels.npy'),\
#           'nodes': np.load('saved_data/pummelzeugs_90/' + '18090_38_e0797e9_nodest.npy')}
# mullerplot(data=data90['offs'], id='18090_38', save=True, int_length=250)
# if len(data90['nodes']) != len(data90['offs']):
#     print('NEIIIIIIIIIIIN!')

# spacetime_plot(nodes_t=data01['nodes'], labels=data01['labels'], \
#                tbeg=None, tend=100, save=False, id='1801')
# spacetime_plot(nodes_t=data60['nodes'], labels=data60['labels'], \
#                tbeg=None, tend=100, save=False, id='18060')
# spacetime_plot(nodes_t=data90['nodes'], labels=data90['labels'], \
#                tbeg=None, tend=100, save=False, id='18090')

# create_averaged_entropies(variation='18090', filename='e0797e9', path='saved_data/pummelzeugs_90/',\
#                           rep=500, save=False, plot=True, saveplot=True)
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
# name = '242_0_80e9ef6ohne reori, birth=0'
name = '242_0_3122dc5ohne reori, death=0'
# name = '242_0_12bc382ohne reori'
labs = np.load('saved_data/checker/' + name + '_labels.npy')
offs = np.load('saved_data/checker/' + name + '_offsprings.npy')
n = np.load('saved_data/checker/' + name + '_nodest.npy')

spacetime_plot(nodes_t=n, labels=labs, figsize=(4,3),\
               tbeg=None, tend=20, save=True, id='ohneR_d0')
