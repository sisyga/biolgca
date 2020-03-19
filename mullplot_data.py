import numpy as np

def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

def create_edges(tree, fams, name):
    edges = [["Parent", "Identity"]]
    np.savetxt('saved_data/' + name + '_edges.csv', edges, delimiter=',', fmt='%s')
    with open('saved_data/' + name + '_edges.csv', "a") as file:
        for entry in fams:
            ori = tree.item().get(entry)['parent']
            if ori == entry:
                file.write(str(0) + ',' + str(entry) + '\n')
            else:
                file.write(str(ori) + ',' + str(entry) + '\n')
    print('edges geschrieben')

def create_pop(offs, times, name):
    nf = len(offs[-1])
    pop = ["Population"]
    for entry in times:
        pop.append(0)
    np.savetxt('saved_data/' + name + '_population.csv', pop, delimiter=',', fmt='%s')
    f = 1
    with open('saved_data/' + name + '_population.csv', "a") as file:
        while f <= nf:
            for t in range(len(times)):
                if f <= len(offs[t]):
                    file.write(str(offs[t][f - 1]) + '\n')
                else:
                    file.write(str(0) + '\n')
            f += 1
    print('pop geschrieben')

def create_input(filename, tbeg=0, tend=None, int_length=1, cutoff=0):
    offs = correct(np.load('saved_data/' + filename + '_offsprings.npy'))
    tree = np.load('saved_data/' + filename + '_tree.npy')
    name = filename + '_int_length=' + str(int_length) + '_cutoff=' + str(cutoff)
    if tend is None:
        tend = len(offs) - 1
    print(offs)
    #TODO tbeg, tend variabel
    steps = tend - tbeg + 1
    nf = len(offs[tend])

    if int_length != 1:
        print('in intervalle einteilen, neue offs')
        offs = create_newoffs(offs, int_length, tbeg, tend)
        #trange
        int_num = (steps // int_length)
        last_int = steps % int_length
        trange = [0]
        for i in range(int_num):
            trange.append(i*int_length + int(int_length/2))
        if last_int != 0:
            trange.append(tend-int(last_int/2))
        trange.append(tend)
        print(trange)
        np.savetxt('saved_data/' + name + '_trange.csv', trange, delimiter=',', fmt='%s')
    else:
        trange = np.arange(tbeg, steps)
        np.savetxt('saved_data/' + name + '_trange.csv', trange, delimiter=',', fmt='%s')

    if cutoff: #TODO kontrollieren
        print('cutoff: edges und pop')
        offs, fams = filter(offs)
        nf = len(offs[-1])
        #edges
        create_edges(tree, fams=fams, name=name)
        #pop unten?

    else:
        print('edges komplett, pop nach offs')
        #edges
        create_edges(tree, fams=np.arange(1, nf+1), name=name)
        #pop unten?
    create_pop(offs, times=trange, name=name)

def create_newoffs(offs, int_length, tbeg, tend):
    steps = tend - tbeg + 1
    print('steps', steps)

    maxfam = len(offs[tend]) - 1
    int_num = (steps // int_length)
    last_int = steps % int_length
    print('intnum, lastint', int_num, last_int)
    offs = np.asarray(offs)
    chunks = np.hsplit(offs[0:steps-last_int], int_num)
    # print(chunks)
    schnappse = [[1] * (len(offs[0])-1)]
    for chunk in chunks:
        sums = [0] * (len(chunk[-1]) - 1)
        for arr in chunk:
            for i, e in enumerate(arr[1:]):
                sums[i] = sums[i] + e
        schnappse.append(sums)
    if last_int != 0:
        sums = [0] * (len(offs[tend]) - 1)
        while last_int != 0:
            for i, e in enumerate(offs[-last_int][1:]):
                sums[i] += e
            last_int -= 1
        schnappse.append(sums)
    schnappse.append(offs[-1][1:])
    print('newoffs', schnappse)

    return schnappse

def filter(offs, cutoff=0.25):   #egal ob original oder new_offs
    if offs[0][0] == -99:
        v = 1
    else:
        v = 0
    rel_offs = [[]] * (len(offs[-1]))
    filtered_offs = [[]] * (len(offs))
    filtered_fams = []
    # abs = []
    # for step in range(0, 2):
    for step in range(len(offs)):
        s = sum(offs[step])
        # abs.append(s)
        for i, e in enumerate(offs[step]):
            rel_offs[i] = np.concatenate((rel_offs[i], [e/s]))
    # print(rel_offs)
    # print(abs)
    for f in range(len(rel_offs)):
        if max(rel_offs[f]) >= cutoff:
            filtered_fams.append(f+v)
    # print(filtered_fams)
    for step in range(len(offs)):
        for entry in filtered_fams:
            if entry < len(offs[step]):
                filtered_offs[step] = np.concatenate((filtered_offs[step], [offs[step][entry]]))

    print('filtered with %.2f:' % cutoff)
    print(filtered_offs)
    return filtered_offs, filtered_fams

# name = 'real180_bsp'
name = 'bsp'
# int_length = 100
create_input(name, int_length=3)
# offs = len(np.load('saved_data/' + name + '_offsprings.npy'))
# print(offs/int_length)
# tree = np.load('saved_data/' + name + '_tree.npy')
# create_input(filename=name, tbeg=0, tend=None, int_length=int_length)
#TODO create_input(filename=name, tbeg=0, tend=None, int_length=int_length, cutoff=0.2)



