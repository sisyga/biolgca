import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def errors(lgca):
    print('---errors?---')
    inh_l = False
    for i in range(lgca.maxlabel.astype(int) + 1):
        if lgca.props['lab_m'][i] <= lgca.maxlabel_init:
            inh_l = True
        else:
            inh_l = False
    if inh_l:
        print('---')
    else:
        print('Fehler: inheritance label passen nicht!')

    if len(lgca.props['lab_m']) == len(lgca.props['r_b']) and len(lgca.props['r_b']) == lgca.maxlabel + 1:
        print('---')
    else:
        print('Fehler: len(props) passen nicht!')

    if sum(lgca.props['num_off'][1:]) != lgca.borncells - lgca.diedcells + lgca.maxlabel_init:
        print('num_off falsch!')
    else:
        print('---')

def count_fam(lgca):
    if lgca.maxlabel_init == 0:
        print('ERROR: There are no cells in the lattice!')
    else:
        print('---genealogical research---')
        print('number of ancestors: ', lgca.maxlabel_init)
        print('initial density: ', lgca.maxlabel_init/(lgca.K * lgca.l))
        num = lgca.props['num_off']
        if num[0] != -99:
            print('Etwas stimmt nicht!')
        print('genealogical tree:', num[1:])
        print('max family number is %d with ancestor cell %d' % (max(num[1:]), num.index(max(num[1:]))))
        print('number of ancestors at beginning:', lgca.maxlabel_init)
        print('number of living offsprings:', sum(num[1:]))

        print('number of died cells: ', lgca.diedcells)
        print('number of born cells: ', lgca.borncells)

        return max(num[1:])

def bar_stacked(lgca, save = False):
    tmax, l, _ = lgca.nodes_t.shape
    ancs = np.arange(1, lgca.maxlabel_init.astype(int) + 1)
    # if len(ancs) != lgca.maxlabel_init:
    #     print('FEHLER: len(ancs) != maxlabel_init!')
    val = np.zeros((tmax, lgca.maxlabel_init.astype(int) + 1))
    for t in range(0,tmax):
        for c in ancs:
            val[t, c] = lgca.props_t[t]['num_off'][c]
    plt.figure(num=None)
    ind = np.arange(0, tmax, 1)
    width = 1  # the width of the bars: can also be len(x) sequence
    for c in ancs:
        # print('val für c:', val[:,c], c)
        if c > 1:
            b = np.zeros(tmax)
            for i in range(1,c):
                b = b + val[:,i]
            plt.bar(ind, val[:, c], width, bottom=b, label=c)
        else:
            plt.bar(ind, val[:, c], width, color=['red'], label=c)

    ###plot settings

    plt.ylabel('total number of living offsprings')
    plt.xlabel('timesteps')
    plt.title('Ratio of offsprings')
    if len(ind) <= 15:
        plt.xticks(ind)
    else:
        plt.xticks(np.arange(0, len(ind)-1, 5))

    if tmax >= 100:
        plt.xticks(np.arange(0, tmax, 50))

    plt.subplots_adjust(right=0.85)
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.show()
    if save == True:
        # plt.savefig('frequency' + str(datetime.now()) +'.jpg')
        plt.savefig('probe_bar.jpg')

def save_data(lgca):
    #brauche:   rb, rd, dim, restchannel, velocitychannel, dichte, propst
    #nicht:     time, variation

    t = len(lgca.props_t)
    dens = lgca.maxlabel_init/(lgca.K * lgca.l)
    file = open('test.txt', 'w')
    # file = open(str(datetime.now()) + '.txt', 'w')

    file.write("gesetzte Parameter:\n")
    file.write('dimension = {dim:d}, deathrate = {rd:1.5f}, birthrate = {rb:1.5f}, timesteps = {t:d}\n'\
               .format(dim=lgca.l, rd=lgca.r_d, rb=lgca.r_b, t=t-1))
    file.write("velocitychannels = {vc:d}, restchannels = {rc:d}, initial density = {dens:f}\n"\
               .format(vc=lgca.velocitychannels, rc=lgca.restchannels, dens=dens))
    file.write('props_t:\n')
    for i in range(0, t):
        file.write('{i:s}\n'.format(i=str(lgca.props_t[i])))
    file.close()

def ana_si(lgca, p = False, save = False):
    t = len(lgca.props_t)   #timesteps + 1
    for i in range(t):
        if lgca.sim_ind[i] == 0:
            print('Homogeneity since k = ', i)
            break
    if p == True:
        plt.figure(num=None)
        plt.plot(lgca.sim_ind, color='seagreen', linewidth=3)
        plt.ylabel('Simpson-Index')
        plt.xlabel('timesteps')
        plt.yticks(np.arange(0, 1, 0.1))
        if t >= 100:
            plt.xticks(np.arange(0, t, 50))
        plt.tight_layout()
        if save == True:
            # plt.savefig('sim_ind' + str(datetime.now()) +'.jpg')
            plt.savefig('probe_si.jpg')

def aloha(who):
    print('aloha', who)