# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt


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

data01 = {'offs': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_offsprings.npy'),\
          'labels': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_labels.npy'),\
          'nodes': np.load('saved_data/pummelzeugs_1/' + '1801_286_21fb7db_nodest.npy')}
#
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
#
data60 = {'offs': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_offsprings.npy'),\
          'labels': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_labels.npy'),\
          'nodes': np.load('saved_data/pummelzeugs_60/' + '18060_81_6e8510d_nodest.npy')}
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
# data = {'rc=178': thom01, 'rc=1': thom60}
# colors = ['darkred', 'orange', 'olivedrab', 'darkturquoise']
# thom90 = np.load('saved_data/thoms/' + 'rc=178_thom1500.npy')
# data = {'rc=178': thom01, 'rc=88': thom02, 'rc=2': thom45, 'rc=1': thom60, 'rc=0': thom90}
# colors = ['darkred', 'orange', 'olivedrab', 'indigo', 'darkturquoise']
# mullerplot(data01['offs'], save=True, id='mp18001')
# mullerplot(data01['offs'], int_length=250, save=True, id='mp18001')
# mullerplot(data60['offs'], save=True, id='mp18060')
# mullerplot(data60['offs'], int_length=250, save=True, id='mp18060')
# plot_all_lognorm(thomarray=data, colorarray=colors, int_length=500, save=False)
example = 'rc=178'
# plot_lognorm_distribution(thom=data[example], int_length=1000, save=True, id=example, c=colors[0])

# onenode = np.load('saved_data/thoms501/501thom01_1000.npy')
# onerc = np.load('saved_data/thoms501/501thom167_1000.npy')
# print(np.load('saved_data/onenode_thom_5.npy'))
# print(np.load('saved_data/onerc_thom_5.npy'))
# print(len(onenode))
# print(len(onerc))
# print(onenode.mean(), onenode.min(), onenode.max())
# print(onerc.mean(), onerc.min(), onerc.max())
# dat = {'onenode': onenode, 'onerc': onerc}
# plot_all_lognorm(dat, colors, 1000, save=True)
# plot_all_lognorm(data, colors, 1000)
# thom_all(dat, 5000, save=False, id='ones, int=5000')
# plot_lognorm_distribution(dat['onenode'], 1000, id='onenode', c=colors[0], save=True)
# plot_lognorm_distribution(dat['onerc'], 1000, id='onerc', c=colors[1], save=True)

dimss = [1, 167]
rcss = [499, 1]

# for i in range(0, 2):
#     dim = dimss[i]
#     rc = rcss[i]
#     name = str(2*dim + dim*rc) + str(dim)
#
#     lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
#            variation=False, density=1, dims=dim, restchannels=rc, r_b=0.5, r_d=0.02)
#     id = name + '_' + str(i) + '_ohne_mut'
#     lgca.timeevo(timesteps=500, record=True)
#
#
#         # np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
#     np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
#     np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
#     np.save('saved_data/' + str(id) + '_nodest', lgca.nodes_t)
#     np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,
#     dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
#
# for i in range(0, 2):
#     dim = dimss[i]
#     rc = rcss[i]
#     name = str(2*dim + dim*rc) + str(dim)
#
#     lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
#            variation=False, density=1, dims=dim, restchannels=rc, r_b=0.5, r_d=0.02, r_m=0.001, pop={1:1})
#     id = name + '_' + str(i) + '_mit_mut'
#     lgca.timeevo(timesteps=500, record=True)
#
#
#     np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
#     np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
#     np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
#     np.save('saved_data/' + str(id) + '_nodest', lgca.nodes_t)
#     np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,
#     dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, rm=lgca.r_m, m=lgca.r_int)

# o=np.load('saved_data/501167_1_mit_mut_offsprings.npy')
# print(len(o))
# o=np.load('saved_data/501167_1_ohne_mut_offsprings.npy')
# print(len(o))

#####spacetimeplot####
files = ['501167_1_mit_mut']
# files = ['5011_0_ohne_mut', '501167_1_mit_mut']
# files = ['5011_0_ohne_mut', '501167_1_ohne_mut', '5011_0_mit_mut', '501167_1_mit_mut']
for entry in files:
    nodes = np.load('saved_data/' + entry + '_nodest.npy')
    fams = np.load('saved_data/' + entry + '_families.npy')
    spacetime_plot(nodes_t=nodes, labels=fams, tbeg=None, tend=None, save=True, id=entry)