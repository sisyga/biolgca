# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

# offs = np.load('saved_data/passenger_test_offsprings.npy')
# # offs = np.load('saved_data/82_0_93b103e_offsprings.npy')
# nodest = np.load('saved_data/passenger_test_nodest.npy')
# tree = np.load('saved_data/passenger_test_tree.npy')
#
# print(offs)
# print(len(offs[-1])-1)
# print(nodest)
# print(tree)

# mullerplot(offs)
# real = np.load('saved_data/offsspat.npy') #example ohne mutations
# # real = np.load('saved_data/offs.npy')
# print(real)
# families = len(real[-1]) - 1
# print(families)
# print(len(real))
# mullerplot(real)
"""
Beispiele für Datensätze
"""

# data01 = {'offs': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_offsprings.npy'),\
#           'labels': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_labels.npy'),\
#           'nodes': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_nodest.npy')}
# #
# data02 = {'offs': np.load('saved_data/1802_nodes/' + '1802_0_995dedd_offsprings.npy'),\
#           'labels': np.load('saved_data/1802_nodes/' + '1802_0_995dedd_labels.npy'),\
#           'nodes': np.load('saved_data/1802_nodes/' + '1802_0_995dedd_nodest.npy')}
#
# data03 = {'offs': np.load('saved_data/1803_nodes/' + '1803_1_7fa22bc_offsprings.npy'),\
#           'labels': np.load('saved_data/1803_nodes/' + '1803_1_7fa22bc_labels.npy'),\
#           'nodes': np.load('saved_data/1803_nodes/' + '1803_1_7fa22bc_nodest.npy')}
#
# data45 = {'offs': np.load('saved_data/18045_nodes/' + '18045_3_fca2b14_offsprings.npy'),\
#           'labels': np.load('saved_data/18045_nodes/' + '18045_3_fca2b14_labels.npy'),\
#           'nodes': np.load('saved_data/18045_nodes/' + '18045_3_fca2b14_nodest.npy')}
# #
# data60 = {'offs': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_offsprings.npy'),\
#           'labels': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_labels.npy'),\
#           'nodes': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_nodest.npy')}
#
# data90 = {'offs': np.load('saved_data/18090_nodes/' + '18090_1_b8e3972_offsprings.npy'),\
#           'labels': np.load('saved_data/18090_nodes/' + '18090_1_b8e3972_labels.npy'),\
#           'nodes': np.load('saved_data/18090_nodes/' + '18090_1_b8e3972_nodest.npy')}

"""
Einlesen von thoms der Länge 1500
plus color-set
"""
# thom01 = np.load('saved_data/thoms/' + 'rc=178_thom1500.npy')
# thom02 = np.load('saved_data/thoms/' + 'rc=88_thom1500.npy')
# thom45 = np.load('saved_data/thoms/' + 'rc=2_thom1500.npy')
# thom60 = np.load('saved_data/thoms/' + 'rc=1_thom1500.npy')
# data = {'rc=178': thom01, 'rc=88': thom02, 'rc=2': thom45, 'rc=1': thom60}
# # data = {'rc=178': thom01, 'rc=1': thom60}
# colors = ['darkred', 'orange', 'olivedrab', 'darkturquoise']
#
# for var in data:
#     print(var)
#     print('max', max(data[var]))
#     print('min', min(data[var]))
#     print('mean', np.mean(data[var]))

# thom90 = np.load('saved_data/thoms/' + 'rc=178_thom1500.npy')
# mullerplot(data01['offs'], save=True, id='mp18001')
# mullerplot(data01['offs'], int_length=250, save=True, id='mp18001')
# mullerplot(data60['offs'], save=True, id='mp18060')
# mullerplot(data60['offs'], int_length=250, save=True, id='mp18060')
# plot_all_lognorm(thomarray=data, colorarray=colors, int_length=500, save=False)
example = 'rc=178'
# plot_lognorm_distribution(thom=data[example], int_length=1000, save=True, id=example, c=colors[0])

onenode = np.load('saved_data/thoms501/501thom01_1000.npy')
onerc = np.load('saved_data/thoms501/501thom167_1000.npy')
# print(np.load('saved_data/onenode_thom_5.npy'))
# print(np.load('saved_data/onerc_thom_5.npy'))
# print(len(onenode))
# print(len(onerc))
# print(onenode.mean(), onenode.min(), onenode.max())
# print(onerc.mean(), onerc.min(), onerc.max())
# dat = {'onerc': onerc, 'onenode': onenode}
# for var in dat:
#     print(var)
#     print('max', max(dat[var]))
#     print('min', min(dat[var]))
#     print('mean', np.mean(dat[var]))
# plot_all_lognorm(dat, 1000, save=True)
# plot_all_lognorm(dat, colors, 5000, save=False)
# thom_all(dat, 5000, save=False, id='ones, int=5000')
# plot_lognorm_distribution(dat['onenode'], 4575, id='onenode', c=colors[0], save=True)
# plot_lognorm_distribution(dat['onerc'], 4575, id='onerc', c=colors[1], save=True)


# test = [[5], [4, 1], [4, 1, 1], [3, 0, 1], [3, 0, 0, 1]]
# test2 = [[5], [5, 1], [4, 1, 1, 1], [3, 2, 1, 0], [0, 3, 0,	0]]
# data = {'test': test, 'test2': test2}

# gi1 = np.load('saved_data/5011_mut_averaged_gini.npy')
# gi167 = np.load('saved_data/501167_mut_averaged_gini.npy')
#
# sh1 = np.load('saved_data/5011_mut_averaged_shannon.npy')
# sh167 = np.load('saved_data/501167_mut_averaged_shannon.npy')
#
# hill1 = np.load('saved_data/5011_mut_averaged_hill2.npy')
# hill167 = np.load('saved_data/501167_mut_averaged_hill2.npy')
#
# gini = {'onenode': gi1, 'onerc': gi167}
# shannon = {'onenode': sh1, 'onerc': sh167}
# hill2 = {'onenode': hill1, 'onerc': hill167}
#
# plot_sth(gini, id='ginisimpson', save=True, ylabel='ginisimpson')
# plot_sth(shannon, id='shannon', save=True, ylabel='shannon')
# plot_sth(hill2, id='hill2', save=True, ylabel='hill2')

path = 'saved_data/Varianten ohne mut/'
name1 = '5011_2_640e948'
vari1 = {'offs': correct(np.load(path + name1 + '_offsprings.npy')),
         'fams': np.load(path + name1 + '_families.npy'),
         'nodes': np.load(path + name1 + '_nodes.npy'),
         'tree': np.load(path + name1 + '_tree.npy')}

name167 = '501167_4_ac06cfb'
vari167 = {'offs': correct(np.load(path + name167 + '_offsprings.npy')),
         'fams': np.load(path + name167 + '_families.npy'),
         'nodes': np.load(path + name167 + '_nodes.npy'),
         'tree': np.load(path + name167 + '_tree.npy')}

# data = {'onenode': vari1}
data = {'onenode': vari1, 'onerc': vari167}
#
for entry in data:
    print(len(data[entry]['offs'])-1)
    steps = len(data[entry]['offs'])
    p = [0]*steps
    for t in range(0, steps):
        p[t] = sum(data[entry]['offs'][t])
        if p[t]==501:
            print('miiez', t)
    print('kontr', p[0], p[-1])
    print('min', min(p), 'max', max(p), 'mean', np.mean(p))
    # for i in range(0, int(len(data[entry]['nodes'])/1000)):
    #     spacetime_plot(data[entry]['nodes'], data[entry]['fams'], tbeg=1000*i, tend=1000*i+1000, save=True, id=entry)
    # spacetime_plot(data[entry]['nodes'], data[entry]['fams'], tend=100, save=True, id=entry)
