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
    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting', density=1, dims=dim,
                    restchannels=rc, r_b=0.8, r_d=0.1, r_m=0.3,
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


def create_hex(dim, rc, steps, iiid='None', driver=False, rec=False, rec_off=False, save=False, trange=False):
    nodes = np.zeros((dim, dim, 6 + rc))
    for i in range(0, 6 + rc):
        nodes[dim // 2, dim // 2, i] = i + 1
    name = str(dim) + 'x' + str(dim) + '_rc=' + str(rc) + '_steps=' + str(steps) + str(iiid)

    if driver:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                            r_b=0.09, r_d=0.08, r_m=0.001, effect=driver_mut)
        name += '_driver'
        if trange:
            trange.append('_driver_')
    else:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                            r_b=0.09, r_d=0.08, r_m=0.001, effect=passenger_mut)
        name += '_passenger'
        if trange:
            trange.append('_passenger_')

    lgca_hex.timeevo(timesteps=steps, recordoffs=rec_off, record=rec, trange=trange)
    # print('maxfam', len(lgca_hex.tree_manager.tree))
    # size = sum(lgca_hex.offsprings[-1][1:])
    # print('Popsize final: ', size)
    if save:
        np.save('saved_data/' + name + '_tree', lgca_hex.tree_manager.tree)
        np.save('saved_data/' + name + '_families', lgca_hex.props['lab_m'])
        np.save('saved_data/' + name + '_offsprings', lgca_hex.offsprings)
        if rec:
            np.save('saved_data/' + name + '_nodes', lgca_hex.nodes_t)


def read_mut(name, pod):
    tree = np.load('saved_data/' + name + '_' + pod + '_tree.npy')
    fams = np.load('saved_data/' + name + '_' + pod + '_families.npy')
    offs = np.load('saved_data/' + name + '_' + pod + '_offsprings.npy')
    # nodes = np.load('saved_data/' + name + '_' + pod + '_nodes.npy')
    return tree, fams, offs #, nodes


# create_hex(dim=4, rc=2, steps=6, driver=True, rec_off=True, save=False, trange=[[2, 5]])

# for i in range(0, 2):
#     print('bin bei ', i)
#     lgca = create_hex(dim=50, rc=500, steps=4000, driver=True, rec_off=True, save=True, iiid='_hexi_'+str(i))
#     size_driver[i] = sum(lgca.offsprings[-1][1:])
#     lgca = create_hex(dim=50, rc=500, steps=4000, driver=False, rec_off=True, save=True, iiid='_hexi_'+str(i))
#     size_pass[i] = sum(lgca.offsprings[-1][1:])


# for i in range(0, 10):
#     t, f, o = read_mut('50x50_rc=500_steps=1000_hexi_'+str(i), 'passenger')
#     print('---', i, '---')
#     m += len(o[-1][1:])
#     s += sum(o[-1][1:])
#     print(len(o[-1][1:]))
#     print(sum(o[-1][1:]))
# #
# print(m, (s))
#     plot_popsize(o)
# lgca = create_hex(dim=4, rc=1, steps=10, driver=False, rec=True, save=True, iiid='hexiiii')
# plot_popsize(lgca.offsprings)


# name = '50x50_rc=500_steps=1000hexi_zwei'
# t, f, o, n = read_mut(name, pod='driver')
# plot_popsize(o)
# t, f, o, n = read_mut(name, pod='passenger')
# plot_popsize(o)
# print(o[50])
# print(o[99])
# print(o[100])
# print(o[101])
# plot_popsize(o)
# plot_density_after(nodes_t=n[0], dim=100, rc=100)
# plot_density_after(nodes_t=n[-100], dim=100, rc=100)
# plot_density_after(nodes_t=n[-1], dim=100, rc=100)

# names_g = ['100x100_rc=1_steps=500_rb1_5_driver']
# names_g = ['100x100_rc=1_steps=500_Test_passenger', '100x100_rc=1_steps=500_Test_driver', '100x100_rc=1_steps=500_rb1_5_driver']
# names_g= '2x2_rc=1_steps=10mini_kplx_driver'

# hex = get_lgca(ib=True, geometry='hex', bc='reflecting', dims=4, interaction='mutations',
#                         r_b=0.8, r_d=0.3, r_m=0.5, effect=driver_mut, restchannels=1)
# plot_families(hex.nodes, hex.props['lab_m'], dims=4)
# hex.timeevo(timesteps=10, record=True)
# plot_families(hex.nodes_t[-1], hex.props['lab_m'], dims=4)
#
# hex = get_lgca(ib=True, geometry='lin', bc='reflecting', dims=4, interaction='mutations',
#                         r_b=0.8, r_d=0.3, r_m=0.5, effect=driver_mut, restchannels=1)
# hex.timeevo(timesteps=10, record=True)
#
# spacetime_plot(hex.nodes_t, hex.props['lab_m'])

# tree, fams, offs, nodes = read_pm(name=names_g[0])
# for i in [10, 50, 100, 200]:
#     plot_density_after(nodes[i], save=True, id='pas_' + str(i))
# dens = nodes[-1]
# data = {}
# for ind, name in enumerate(names_g):
#     print(ind, name)
#     data[str(ind)] = np.loadtxt('saved_data/' + name + '_sh.csv')
# plot_sth(data, save=True, ylabel='shannon-index', savename='shannon_hex')
# tree, fams, offs, nodes = read_pm(name='100x100/'+names_g[0])
# # data['0'] = plot_popsize(offs)
# print(sum(correct(offs)[-1]))
# plot_families(nodes[-1], fams, dims=100)
#
# tree, fams, offs, nodes = read_pm(name='100x100/'+names_g[1])
# # data['1'] = plot_popsize(offs)
# print(sum(correct(offs)[-1]))
# plot_families(nodes[-1], fams, dims=100)
# tree, fams, offs, nodes = read_pm(name='100x100/'+names_g[2])
# # data['2'] = plot_popsize(offs)
# print(sum(correct(offs)[-1]))
# plot_families(nodes[-1], fams, dims=100)

# plot_sth(data, ylabel='absolute populationsize', save=True, savename='hexis_popsize')

# fig, ax = plt.subplots(figsize=(10, 5))
# plt.plot(x, yp, farben['0'], label='pass', linewidth=1.5)
# plt.plot(x, yd1, farben['1'], label='driver 1,1', linewidth=1.5)
# plt.plot(x, yd5, farben['2'], label='driver 1,5', linewidth=1.5)
#
# ax.set(xlabel='timesteps', ylabel='absolute number')
# ax.legend(loc='upper left')
# tend=501
# plt.xlim(0, tend - 1)
# if tend >= 10000:
#     plt.xticks(np.arange(0, tend, 5000))
# elif tend >= 100:
#     plt.xticks(np.arange(0, tend, 50))
#
# plt.ylim(0, 100*100*7+10)
# # if save:
# #     if savename is None:
# save_plot(plot=fig, filename='hexis_popsizes.jpg')
# #     else:
# #         save_plot(plot=fig, filename=savename + '.jpg')
#
# plt.show()
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
