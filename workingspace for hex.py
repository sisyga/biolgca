from lgca import get_lgca
from lgca.ib_interactions import driver_mut, passenger_mut
from lgca.helpers import *
from lgca.helpers2d import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from os import environ as env
from uuid import uuid4 as uuid

def create_inh(name, dim, rc, save=False):
    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',
                    variation=False, density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1)
    lgca.timeevo_until_hom(spatial=True)

    if save:
        np.save('saved_data/' + name + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca.offsprings)
        np.save('saved_data/' + name + '_nodes', lgca.nodes_t)

def read_inh(name):
    fams = np.load('saved_data/' + name + '_families.npy')
    offs = np.load('saved_data/' + name + '_offsprings.npy')
    nodes = np.load('saved_data/' + name + '_nodes.npy')
    return fams, offs, nodes

def create_pm(name, dim, rc, save=False):
    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting', density=1, dims=dim, restchannels=rc, r_b=0.8, r_d=0.1, r_m=0.3,
                    pop={1: 1})
    lgca.timeevo(timesteps=20, record=True)

    if save:
        np.save('saved_data/' + name + '_tree', lgca.tree_manager.tree)
        np.save('saved_data/' + name + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca.offsprings)
        np.save('saved_data/' + name + '_nodes', lgca.nodes_t)

def read_pm(name):
    tree = np.load('saved_data/' + name + '_tree.npy')
    fams = np.load('saved_data/' + name + '_families.npy')
    offs = np.load('saved_data/' + name + '_offsprings.npy')
    nodes = np.load('saved_data/' + name + '_nodes.npy')
    return tree, fams, offs, nodes

def create_hex(dim, rc, steps, iiid, driver=False, save=False):
    nodes = np.zeros((dim, dim, 6 + rc))
    for i in range(0, 6 + rc):
        nodes[dim//2, dim//2, i] = i+1
    name = str(dim) + 'x' + str(dim) + '_rc=' + str(rc) + '_steps=' + str(steps) + str(iiid)

    if driver:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                        r_b=0.8, r_d=0.3, r_m=0.5, effect=driver_mut) #r_m=0.01
        name += '_driver'
    else:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                            r_m=0.01, effect=passenger_mut)
        name += '_passenger'

    lgca_hex.timeevo(timesteps=steps, record=True)
    print('maxfam', len(lgca_hex.tree_manager.tree))
    if save:
        np.save('saved_data/' + name + '_tree', lgca_hex.tree_manager.tree)
        np.save('saved_data/' + name + '_families', lgca_hex.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca_hex.offsprings)
        np.save('saved_data/' + name + '_nodes', lgca_hex.nodes_t)
    return lgca_hex


# lgca = create_hex(dim=2, rc=1, steps=10, driver=True, save=True, iiid='mini_kplx')
# create_hex(dim=100, rc=1, steps=500, driver=False, save=True, iiid='_Test')

# for t in range(0, 11):
#     plot_families(nodes_t=lgca.nodes_t[t], lab_m=lgca.props['lab_m'], dims=2)
# for t in range(0, 11):
#     print(lgca.offsprings[t])
# print(lgca.tree_manager.tree)

names_g = ['100x100_rc=1_steps=500_rb1_5_driver']
names_g = ['100x100_rc=1_steps=500_Test_passenger', '100x100_rc=1_steps=500_Test_driver', '100x100_rc=1_steps=500_rb1_5_driver']
# names_g= ['2x2_rc=1_steps=10mini_kplx_driver']

data = {}
# for ind, name in enumerate(names_g):
#     print(ind, name)
#     data[str(ind)] = np.loadtxt('saved_data/' + name + '_sh.csv')
# plot_sth(data, save=True, ylabel='shannon-index', savename='shannon_hex')
tree, fams, offs, nodes = read_pm(name='100x100/'+names_g[0])
x, yp = plot_popsize(offs)
tree, fams, offs, nodes = read_pm(name='100x100/'+names_g[1])
_, yd1 = plot_popsize(offs)
tree, fams, offs, nodes = read_pm(name='100x100/'+names_g[2])
_, yd5 = plot_popsize(offs)



fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(x, yp, farben['0'], label='pass', linewidth=1.5)
plt.plot(x, yd1, farben['1'], label='driver 1,1', linewidth=1.5)
plt.plot(x, yd5, farben['2'], label='driver 1,5', linewidth=1.5)

ax.set(xlabel='timesteps', ylabel='absolute number')
ax.legend(loc='upper left')
tend=501
plt.xlim(0, tend - 1)
if tend >= 10000:
    plt.xticks(np.arange(0, tend, 5000))
elif tend >= 100:
    plt.xticks(np.arange(0, tend, 50))

plt.ylim(0, 100*100*7+10)
# if save:
#     if savename is None:
save_plot(plot=fig, filename='hexis_popsizes.jpg')
#     else:
#         save_plot(plot=fig, filename=savename + '.jpg')

plt.show()
# for name in names_g:
#     tree, fams, offs, nodes = read_pm(name='100x100/'+name)
#     # offs = correct(offs)
#     plot_popsize(offs)
#     print(tree)
#     print(fams)
#     # tree = {1: {'parent': 1, 'origin': 1},
#     #         2: {'parent': 1, 'origin': 1},
#     #         3: {'parent': 1, 'origin': 1},
#     #         4: {'parent': 3, 'origin': 1},
#     #         5: {'parent': 2, 'origin': 1},
#     #         6: {'parent': 3, 'origin': 1},
#     #         7: {'parent': 6, 'origin': 1}}
#     # print(tree)
#     # fams = np.array([0, 1, 4, 3, 2, 5, 7, 6, 7])
#     # print(fams)
#     print(len(offs), len(nodes), len(fams))
#     maxfam = max(fams)
#     gen = [0]*maxfam
#     print(maxfam)
#     for f in range(2, maxfam+1):
#                         # f==1 always generation 0
#         # par = tree.get(f)['parent']
#         par = tree.item().get(f)['parent']
#         gen[f-1] += 1
#
#         while par != 1:
#             # par = tree.get(par)['parent']
#             par = tree.item().get(par)['parent']
#             gen[f - 1] += 1
#
#     generations = [0]*len(fams)
#     for i, f in enumerate(fams):
#         if i != 0:
#             generations[i] = gen[f-1]
#     np.save('saved_data/' + name + '_generations', generations)
    # print(generations)

    # for t in range(0, 101, 100):
    # sh = calc_shannon(offs)
    # np.savetxt('saved_data/' + name + '_sh.csv', sh, delimiter=',', fmt='%s')
    # plot_sth({'sh': sh})
    # plot_families(nodes_t=nodes[0], lab_m=fams, dims=100, save=True, id=name + '_0')
    # plot_families(nodes_t=nodes[50], lab_m=fams, dims=100, save=True, id=name + '_50')
    # plot_families(nodes_t=nodes[100], lab_m=fams, dims=100, save=True, id=name + '_100')

