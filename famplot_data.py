import numpy as np

def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

def create_edges(gens, name):
    edges = [["Parent", "Identity"]]
    m = max(gens.values())
    # print('max', m)
    np.savetxt('saved_data/' + name + '_gen_edges.csv', edges, delimiter=',', fmt='%s')
    with open('saved_data/' + name + '_gen_edges.csv', "a") as file:
        for f in range(0, m):
            file.write(str(f) + ',' + str(f+1) + '\n')

    print('edges geschrieben')

def create_pop(offs, gens, times, name):
    maxg = max(gens.values())
    # print('maxg', maxg)
    pop = ["Population"]
    np.savetxt('saved_data/' + name + '_gen_population.csv', pop, delimiter=',', fmt='%s')

    for g in range(0, maxg+1):
        fs = [k for k in gens.keys() if gens[k] == g]
        # print(fs)
        for t in range(len(times)):
                sums = 0
                for f in fs:
                    if f <= len(offs[t]):
                        sums += offs[t][f-1]
                # print(sums)
                with open('saved_data/' + name + '_gen_population.csv', "a") as file:
                    file.write(str(sums) + '\n')

    print('pop geschrieben')

def create_generations(tree):
    # tree = {1: {'parent': 1, 'origin': 1},
    #         2: {'parent': 1, 'origin': 1},
    #         3: {'parent': 1, 'origin': 1},
    #         4: {'parent': 3, 'origin': 1},
    #         5: {'parent': 2, 'origin': 1},
    #         6: {'parent': 3, 'origin': 1},
    #         7: {'parent': 6, 'origin': 1}}
    maxfam = list(tree.item().keys())[-1]

    # print(maxfam)
    generations = {1: 0}

    # print(maxfam)
    for f in range(2, maxfam+1):
        generations[f] = 1                            # f==1 always generation 0
        par = tree.item().get(f)['parent']
        while par != 1:
            par = tree.item().get(par)['parent']
            generations[f] += 1

    # np.save('saved_data/' + name + '_generations', generations)
    # print(generations)
    return generations



def create_input(filename, tbeg=0, tend=None, int_length=1):
    tree = np.load('saved_data/' + filename + '_tree.npy')
    gens = create_generations(tree=tree)
    offs = correct(np.load('saved_data/' + filename + '_offsprings.npy'))
    name = filename + '_int_length=' + str(int_length)

    if tend is None:
        tend = len(offs) - 1
    steps = tend - tbeg + 1
    nf = len(offs[tend])

    if int_length != 1:
        offs = create_newoffs(offs, int_length, tbeg, tend)

        #trange
        int_num = (steps // int_length)
        last_int = steps % int_length
        trange = [0]
        for i in range(int_num):
            trange.append(i*int_length + int(int_length/2))
        if last_int != 0:
            if last_int >= 2:
                trange.append(tend-int(last_int/2))
        trange.append(tend)
        # print(trange)
        np.savetxt('saved_data/' + name + '_gen_trange.csv', trange, delimiter=',', fmt='%s')
    else:
        trange = np.arange(tbeg, steps)
        np.savetxt('saved_data/' + name + '_gen_trange.csv', trange, delimiter=',', fmt='%s')


    #edges
    create_edges(gens, name=name)
    create_pop(offs, gens, times=trange, name=name)

def create_newoffs(offs, int_length, tbeg, tend):
    # print('beide', offs)
    steps = tend - tbeg + 1
    # print('steps', steps)

    maxfam = len(offs[tend])
    int_num = (steps // int_length)
    last_int = steps % int_length
    print('intnum, lastint', int_num, last_int)
    offs = np.asarray(offs)
    if last_int != 0:
        chunks = np.split(offs[:-last_int], int_num)
    else:
        chunks = np.split(offs[:], int_num)

    # print('ch', chunks)
    schnappse = [[1] * (len(offs[0]))]
    for chunk in chunks:
        sums = [0] * (len(chunk[-1]))
        for arr in chunk:
            for i, e in enumerate(arr):
                sums[i] = sums[i] + e
        schnappse.append(sums)
    if last_int != 0:
        sums = [0] * (len(offs[tend]))
        z = last_int
        while z != 0:
            for i, e in enumerate(offs[-z]):
                sums[i] += e
            z -= 1
        schnappse.append(sums)  # summe aus zwischenintervall

    schnappse.append(offs[-1])  # Stand tend

    # print('newoffs', schnappse)
    # print(schnappse)
    return schnappse

def create_ori(offs, tree):
    tend = len(offs)
    f = len(offs[0])
    ori = [offs[0]]
    for line in range(1, tend):
        ori.append(offs[line][:f])
        l = len(offs[line])-f
        while l > 0:
            entry = offs[line][f-1+l]
            fam = tree.item().get(f+l)['origin']
            ori[line][fam-1] += entry
            l -= 1
    return ori


named = 'd785cf8_50x50rc=500_driver'
namep = '46a8f13_50x50rc=500_passenger'
names = [namep, named]
for i in names:
# create_input(i, int_length=1)
# tree = np.load('saved_data/' + i + '_tree.npy')
# fams = np.load('saved_data/' + i + '_families.npy')
    create_input(filename=i)

