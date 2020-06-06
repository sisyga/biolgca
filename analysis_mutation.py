# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import pandas as pd


def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

def search_names(path):
    files = os.listdir(path)
    names = [entry.replace('_tree.npy', '') for entry in files if 'tree' in entry]
    print(len(names))
    return names

def search_offs(path, name):
    return correct(np.load(path + name + '_offsprings.npy'))

def set_data(path):
    names = search_names(path)
    data = {}
    for name in names:
        data[name] = search_offs(path, name)
    return data

def read_inds(which='si'):
    path = 'saved_data/Indizes_explizit/'
    files = os.listdir(path)
    dataset = {}
    ind = which
    for file in files:
        if ind in file:
            dataset[file[:-(4+len(ind))]] = np.loadtxt(path + file)
    # print('d', dataset)
    return dataset

def ave_inds(which='shannon', plot=False, save=False, savename=None):
    path = 'saved_data/Indizes_averaged/'
    files = os.listdir(path)
    dataset = {}
    ind = which
    for file in files:
        if ind in file:
            dataset[file[:-(14+len(ind))]] = np.loadtxt(path + file)
    if plot:
        plot_sth(dataset, ylabel=ind, save=save, id=which, savename=savename)
    return dataset

def zahlende(path, steps):
    files = os.listdir(path)
    rel = []
    for file in files:
        if 'offspring' in file:
            rel.append(file)
    rel = rel[:3]
    # rel = ['Probe_offsprings.npy', 'Probe2_offsprings.npy']
    print('anz daten', len(rel))
    ende = {}
    fams = {}
    muts = {}
    for i, f_rel in enumerate(rel):
        offs = correct(np.load(path + f_rel)[-steps:])
        # print(i+1, offs)
        lmax = len(offs[-1])
        for step in range(1, steps):
            l = len(offs[-1-step])
            if l < lmax:
                offs[-1-step] = offs[-1-step] + [0]*(lmax-l)
        mean_offs = list(np.mean(offs, axis=0))
        nn = []
        nf = []
        for index, entry in enumerate(mean_offs):
            if entry > 0:
                nn.append(entry)
                nf.append(index + 1)

        ende[str(i + 1)] = nn
        fams[str(i + 1)] = nf
        muts[str(i+1)] = mut_number(nf, path, filename=f_rel[:-len('offsprings.npy')] + 'tree.npy')

    return ende, fams, muts

def mut_number(fams, path, filename):
    tree = np.load(path + filename)
    muts = []
    for entry in fams:
        if entry == 1:
            m_anz = 0
        else:
            m_anz = 1
            while tree.item().get(entry)['parent'] != tree.item().get(entry)['origin']:
                # print(entry, ' stammt von ', tree.item().get(entry))
                m_anz += 1
                entry = tree.item().get(entry)['parent']
        muts.append(m_anz)
    return muts

def plotende(ave, fams, muts):
    for sim in ave:
        data = ave[sim]
        fs = fams[sim]
        ms = muts[sim]

        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        def func(pct, allvals):
            absolute = pct / 100. * np.sum(allvals)
            return "{:.1f}%\n(~ {:.1f})".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
        ax.legend(wedges, ms,
                  title="Number of Mutations",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=8, weight="bold")

        ax.set_title("pie of Simulation {:s}".format(sim))

        plt.show()

def max_wert(which, intervallmin=0, intervallmax=40000):
    m = max(which['onenode'][intervallmin:intervallmax])
    print('max bei ', m)
    print('k=', np.where(which['onenode'][intervallmin:intervallmax] == m))

# path = 'saved_data/5011_ges/'
# path = 'saved_data/501167_ges/'
# path = 'saved_data/Indizes_explizit/Daten/'
# files = os.listdir(path)
# print(len(files))
# rel = []
# for file in files:
#     if 'offsprings' in file:
#     # if 'offsprings' in file and '5011_mut' in file:
#         rel.append(file)
# print(len(rel))
# print(rel)
#
# oris = []
# akti = []
# fam_max = []
# max_akti = []
# # rel = rel[:3]
# for r in rel:
#     offs = correct(np.load(path + r)[-1:])
#     fam_max.append(len(offs[0])-1)
#     akti.append(len([entry for entry in offs[0] if entry > 0]))
#     oris.append(int(1*offs[0][0] > 0))
#     max_akti.append('{:.2f}'.format(max(offs[0])/sum(offs[0])))
# print('oris, ati, fammax, maxakti')
# print(oris, akti, fam_max, max_akti)

# import csv
# toWrite = [['Simulation:'] + [i+1 for i in range(0, len(fam_max))],
#            ['Anz. Mutationen:'] + fam_max,
#            ['aktive Familien:'] + akti,
#            ['proz. dominierende:'] + max_akti,
#            ['Anfangsfamilie da:'] + oris
#            ]
# print(toWrite)
#
# file = open(path + '5011_ges.csv', 'w')
#
# with file as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     for row in toWrite:
#         writer.writerow(row)
#
# ende, fams, muts = zahlende(path='saved_data/5011_ges/', steps=10)
# print(ende, fams)
# # plotende(ende, fams, muts)
#
# popdic = {0: [1,2], 1: [3,4], 2: [2,0]}
# xrange = np.arange(0, 2)
# data = pd.DataFrame(popdic, index=xrange)
# data_perc = data.divide(data.sum(axis=1), axis=0)
# plt.stackplot(xrange, *[data_perc[f] for f in range(0, 3)],
#               labels=list(range(0, 3)))
# plt.show()
# o1 = [[-99, 1,3,5,0], [-99, 2,2,4,0,4], [-99, 0,0,4,0,0]]
# np.save('saved_data/testoffs1.npy', o1)
# o2 = [[-99, 0,0,0,1,2], [-99, 0,0,0,2,5], [-99, 0,0,0,0,3,1]]
# np.save('saved_data/testoffs2.npy', o2)


"""
    --- Index-Daten einlesen    ---
"""
data1 = correct(np.load('saved_data/Indizes_explizit/Daten/'
                        '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))

data167 = correct(np.load('saved_data/Indizes_explizit/Daten/'
                          '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))
bp = read_inds(which='bp')
bpinv = {}
for key in bp:
    print(key)
    bpinv[key] = [1/entry for entry in bp[key]]
si = read_inds(which='si')
sh = read_inds(which='sh')
eve = read_inds(which='eve')
gi = read_inds(which='gi')
hill1 = read_inds(which='hill1')
hill2 = read_inds(which='hill2')
hill3 = read_inds(which='hill3')
hill_5 = read_inds(which='hill_5')
hill_25 = read_inds(which='hill_25')
hill_75 = read_inds(which='hill_75')
rich = read_inds(which='rich')
size = {'onenode': calc_popsize(data1), 'onerc': calc_popsize(data167)}


# plot_sth(data={'onenode': size['onenode'], 'onerc': size['onerc']})
whichlist = [hill2, hill1, hill_5, hill_25]
# for w in whichlist:
    # max_wert(which=w)
    # max_wert(which=w, intervallmin=22000, intervallmax=25000)
# steps = [22477, 23062, 37562, 38466]
# for s in steps:
#     print('steps', s)
#     print(hill_25['onenode'][s])
#     print(hill2['onenode'][s])
# max_wert(which=hill2, intervallmin=22000, intervallmax=25000)
# max_wert(which=bpinv, intervallmax=25000)
# # data = correct(np.load('saved_data/2x2_rc=1_steps=5mini_driver_offsprings.npy'))

# print(data)
# bp = {}
# bp['eins'] = calc_bergerparker(data)
# bp['zwei'] = calc_bergerparker(data)
# print('max ', max(gi['onenode'][:30000]))
# print('max ', max(sh['onenode'][:30000]))
# print('max ', max(eve['onenode'][:30000]))
# print(np.where(gi['onenode'] == max(gi['onenode'][:40000])))
# print(np.where(eve['onenode'] == max(eve['onenode'][:30000])))
# print(np.where(sh['onenode'] == max(sh['onenode'][:30000])))
# print(sh['onenode'][19389])
# print(gi['onenode'][19389])
# print(eve['onenode'][19389])
# gleich = 0
# ungleich = 0
# shkleinereve = 0
# gikleinereve = 0
# for i, entry in enumerate(gi['onenode']):
#     if entry == eve['onenode'][i]:
#         if entry == 0:
#             gleich += 1
#         else:
#             ungleich += 1
#
# print(gleich, ungleich)
    # if entry < eve['onenode'][i]:
    #     shkleinereve += 1
# print(shkleinergi, shkleinereve)
# for i, entry in enumerate(gi['onenode']):
#     if entry < eve['onenode'][i]:
#         gikleinereve += 1
# print(gikleinereve)

# plot_sth(data={'bp': bp['onenode'], 'bpinv': bpinv['onenode'], 'si': si['onenode']})


# plot_sth(data={'sh': sh['onenode'], 'gi': gi['onenode'], 'hill_2': hill2['onenode']}, save=True, savename='onenode_ShGiHh')
# plot_sth(data={'sh': sh['onerc'], 'gi': gi['onerc'], 'hill_2': hill2['onerc']}, save=True, savename='onerc_ShGiHh')
# #
ave_sh = ave_inds(which='shannon')
ave_hill2 = ave_inds(which='hill2')
ave_hill_5 = ave_inds(which='hill5')
ave_gi = ave_inds(which='gini')

# plot_sth(data={'sh': ave_sh['onenode'], 'hill_2': ave_hill2['onenode']-1})
# plot_sth(data={'sh': ave_sh['onerc'], 'hill_2': ave_hill2['onerc']-1})
# plot_sth(data={'onenode': ave_sh['onenode'], 'onerc': ave_sh['onerc']}, save=True, savename='sh_vgl', ylabel='shannon-index')
plot_sth(data={'onenode': ave_hill2['onenode'], 'onerc': ave_hill2['onerc']}, save=True, savename='hill2_vgl', ylabel='hill of order 2')
# plot_sth(data={'onenode': ave_sh['onenode'], 'onerc': ave_sh['onerc']}, ylabel='shannon')
# plot_sth(data={'onenode': ave_hill2['onenode'], 'onerc': ave_hill2['onerc']}, ylabel='hill2')
# plot_sth(data={'onenode': ave_hill_5['onenode'], 'onerc': ave_hill_5['onerc']}, ylabel='hill_5')
# plot_sth(data={'onenode': ave_gi['onenode'], 'onerc': ave_gi['onerc']}, ylabel='gini')
"""
    --- diverse plots ---
"""
# vars = ['onenode', 'onerc']
vars = ['onenode']
# for var in vars:
#     plot_sth(data={'hill_2': hill2[var], 'hill_1': hill1[var],
#                    'hill_25': hill_25[var], 'hill_5': hill_5[var]}, save=True, savename='hills_aus')
    # plot_sth(data={'onenode': hill2[var]-gi[var], 'bp': bp[var], 'bpinv': bpinv[var]}, save=True, savename='Hh-Gi_bp_bpinv')
    # plot_sth(data={'bp': bp[var], 'gi': gi[var], 'bpinv': bpinv[var]})
    # plot_sth(data={'bp': bp[var], 'hill_2': hill2[var], 'bpinv': bpinv[var], 'gi': gi[var]}, save=True, savename='GidHh')
#     plot_sth(data={'onenode': si[var] - gi[var]}, ylabel='simpson - gini', savename='diff_SiGi', save=True)
    # plot_sth(data={'hill_2': hill2[var], 'eve': eve[var]}, save=True, savename=var + '_HhEve')
    # plot_sth(data={'gi': gi[var], 'eve': eve[var]}, save=False, savename=var + '_GiEve')
    # plot_sth(data={'gi': gi[var], 'sh': sh[var], 'eve': eve[var]}, save=True, savename=var + '_GiEveSh')
    # plot_sth(data={'gi': gi[var], 'sh': sh[var]}, save=True, savename=var + '_GiSh')
#     plot_hillnumbers_together(hill2[var],hill_25[var], hill_75[var], save=True, id=var)
#     plot_entropies_together(gini=gi[var], shannon=sh[var])
#     plot_selected_entropies(gini=gi[var], shannon=sh[var], hill2=hill2[var], save=True, id=var + '3')

"""
    --- krasse Indexberechnung  ---  
"""
# data1 = correct(np.load('saved_data/Indizes_explizit/Daten/'
#                         '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))
# print('dataaa', sum(data1[38548]))
# data167 = correct(np.load('saved_data/Indizes_explizit/Daten/'
#                           '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))

# np.savetxt('saved_data/' + 'onenode' + 'bp.csv', calc_bergerparker(data1), delimiter=',', fmt='%s')
# np.savetxt('saved_data/' + 'onerc' + 'bp.csv', calc_bergerparker(data167), delimiter=',', fmt='%s')

# data1 = correct(np.load('saved_data/Indizes_explizit/Daten/'
#                         '5011_mut_55186c3c-e01e-4609-8952-d1314b736521_offsprings.npy'))
# data167 = correct(np.load('saved_data/Indizes_explizit/Daten/'
#                           '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d_offsprings.npy'))
#
# np.savetxt('saved_data/' + 'onenode' + 'rich.csv', calc_richness(data1), delimiter=',', fmt='%s')
# np.savetxt('saved_data/' + 'onerc' + 'rich.csv', calc_richness(data167), delimiter=',', fmt='%s')
# data = {"1": data1, "167": data167}
#
# def funk(file):
#     print(file)
#     # np.savetxt('saved_data/' + file + 'sh.csv', calc_shannon(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'si.csv', calc_simpson(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'gi.csv', calc_ginisimpson(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill1.csv', calc_hillnumbers(data[file], order=1), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill2.csv', calc_hillnumbers(data[file]), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill3.csv', calc_hillnumbers(data[file], order=3), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill_25.csv', calc_hillnumbers(data[file], order=0.25), delimiter=',', fmt='%s')
#     # np.savetxt('saved_data/' + file + 'hill_75.csv', calc_hillnumbers(data[file], order=0.75), delimiter=',', fmt='%s')
#     np.savetxt('saved_data/' + file + 'eve.csv', calc_evenness(data[file]), delimiter=',', fmt='%s')
#
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(4)
#     with pool as p:
#         p.map(funk, data)

"""
        ---Test offs---  
"""
# test = [[5], [4, 1], [4, 1, 1], [3, 0, 1], [3, 0, 0, 1]]
# test2 = [[5], [5, 1], [4, 1, 1, 1], [3, 2, 1, 0], [0, 3, 0,	0]]
# # data = {'test': test, 'test2': test2}
# plot_sth(data={'rich': calc_richness(test), 'hill_2': calc_hillnumbers(test, order=2), 'sh': calc_shannon(test)}, save=False, savename='onenode_richHillSh')

# for t in data:
#     print(calc_evenness(data[t]))
#     print(calc_ginisimpson(data[t]))
#     print(calc_simpson(data[t]))

"""
        ---differenzen plot---  
"""
# tend = 40001
# x = np.arange(0, tend)
# fig, ax = plt.subplots(figsize=(12, 4))
# sh, = ax.plot(x, ave_sh['onerc'] - ave_sh['onenode'], farben['sh'], label='shannon', linewidth=0.75)
# gi, = ax.plot(x, ave_gi['onerc'] - ave_gi['onenode'], farben['gi'], label='ginisimpson', linewidth=0.75)
# h2, = ax.plot(x, ave_hill2['onerc'] - ave_hill2['onenode'], farben['hill_2'], label='hill order 2', linewidth=0.75)
# h_5, = ax.plot(x, ave_hill_5['onerc'] - ave_hill_5['onenode'], farben['hill_5'], label='hill order 0.5', linewidth=0.75)
# lines = [sh, gi, h2, h_5]
# ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
# ax.set(xlabel='timesteps', ylabel='difference onerc-onenode')
# plt.xlim(0, tend - 1)
# if tend >= 10000:
#     plt.xticks(np.arange(0, tend, 5000))
#
# # plt.ylim(bottom=0)
# plt.axhline(y=0, linewidth=1, linestyle=(0, (1, 10)))
# save_plot(plot=fig, filename='ave_diff.jpg')
# plt.show()

"""
    --- create avereaged entropies  ---
"""
# path167 ="saved_data/501167_ges/"
# path1 ="saved_data/5011_ges/"
# data167 = set_data(path167)
# data1 = set_data(path1)
# datas = {'167': data167, '1': data1}
# for data in datas:
    # create_averaged_entropies(datas[data], save=True, saveplot=True, plot=True)
# #
# names1 = []
# for name in data1:
#     names1.append(name)
# print(names1)
# names167 = []
# for name in data167:
#     names167.append(name)
# print(names167)