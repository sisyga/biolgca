import numpy as np
import matplotlib.pyplot as plt


def errors(lgca):
    print('---errors?---')
    inh_l = False
    for i in range(lgca.maxlabel.astype(int) + 1):
        if lgca.props['lab_m'][i] <= lgca.maxlabel_init:
            inh_l = True
        else:
            inh_l = False
    if inh_l:
        print('inheritance label passen')
    else:
        print('Fehler: inheritance label passen nicht!')

    if len(lgca.props['lab_m']) == len(lgca.props['r_b']) and len(lgca.props['r_b']) == lgca.maxlabel + 1:
        print('len(props) passt')
    else:
        print('Fehler: len(props) passen nicht!')


def count_fam(lgca):
    if lgca.maxlabel_init == 0:
        print('ERROR: There are no cells in the lattice!')
    else:
        print('---genealogical research---')
        num = lgca.props['num_off']
        if num[0] != -99:
            print('Etwas stimmt nicht!')
        print('num', num)
        print('genealogical tree:', num[1:])
        print('number of ancestors at beginning:', lgca.maxlabel_init)
        print('number of living offsprings:', sum(num[1:]))
        print('max family number is %d with ancestor cell %d' % (
        max(num[1:]), num.index(max(num[1:]))))

        return max(num[1:])

def bar_stacked(lgca):
    tmax, l, _ = lgca.nodes_t.shape
    ancs = np.arange(1, lgca.maxlabel_init + 1)
    # if len(ancs) != lgca.maxlabel_init:
    #     print('FEHLER: len(ancs) != maxlabel_init!')
    val = np.zeros((tmax, lgca.maxlabel_init + 1))
    for t in range(tmax):
        for c in ancs:
            val[t, c] = lgca.props_t[t]['num_off'][c]

    ind = np.arange(0, tmax, 1)
    width = 0.5  # the width of the bars: can also be len(x) sequence
    for c in ancs:
        print('val für c:', val[:,c], c)
        if c > 1:
            b = np.zeros(tmax)
            for i in range(1,c):
                b = b + val[:,i]
            plt.bar(ind, val[:, c], width, bottom=b)
        else:
            plt.bar(ind, val[:, c], width, color=['red'])

    ###plot settings
    plt.ylabel('total number of living offsprings')
    plt.xlabel('timesteps')
    plt.title('Ratio of offsprings')
    if len(ind) <= 15:
        plt.xticks(ind)
    else:
        plt.xticks(np.arange(0, len(ind)-1, 5))
    ymax = sum(lgca.props['num_off'][1:]) + 1
    if ymax <= 20:
        plt.yticks(np.arange(0, ymax))
    else:
        plt.yticks(np.arange(0, ymax, 5))

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(ancs, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def aloha(who):
    print('aloha', who)