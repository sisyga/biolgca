import numpy as np

def correct(offs):
    c_offs = []
    for entry in offs:
        c_offs.append(entry[1:])
    return c_offs

def create_edges(tree, fams, name):
    edges = [["Parent", "Identity"]]
    print(fams)
    np.savetxt('saved_data/' + name + '_edges.csv', edges, delimiter=',', fmt='%s')
    with open('saved_data/' + name + '_edges.csv', "a") as file:
        for entry in fams:
            ori = tree.item().get(entry+1)['parent']
            if ori == entry+1:
                file.write(str(0) + ',' + str(entry+1) + '\n')
            else:
                if ori-1 in fams:
                    file.write(str(ori) + ',' + str(entry+1) + '\n')
                else:
                    file.write(str(0) + ',' + str(entry + 1) + '\n')
    print('edges geschrieben')

def create_pop(offs, times, name):
    nf = len(offs[-1])
    fams = np.arange(0, nf)

    pop = ["Population"]
    for entry in times:
        pop.append(0)
    np.savetxt('saved_data/' + name + '_population.csv', pop, delimiter=',', fmt='%s')

    with open('saved_data/' + name + '_population.csv', "a") as file:
        for f in range(len(fams)):
            for t in range(len(times)):
                if f+1 <= len(offs[t]):
                    file.write(str(offs[t][f]) + '\n')
                else:
                    file.write(str(0) + '\n')

    print('pop geschrieben')

def create_input(filename, tbeg=0, tend=None, int_length=1, cutoff=0):
    offs = correct(np.load('saved_data/' + filename + '_offsprings.npy'))
    tree = np.load('saved_data/' + filename + '_tree.npy')
    name = filename + '_int_length=' + str(int_length) + '_cutoff=' + str(cutoff)
    if tend is None:
        tend = len(offs) - 1
                                    #TODO tbeg, tend variabel
    steps = tend - tbeg + 1
    nf = len(offs[tend])

    if int_length != 1:
        offs = create_newoffs(offs, int_length, tbeg, tend)
        # print('int offs', offs)
        #trange
        int_num = (steps // int_length)
        last_int = steps % int_length
        trange = [0]
        for i in range(int_num):
            trange.append(i*int_length + int(int_length/2))
        if last_int != 0:
            if last_int > 2:
                trange.append(tend-int(last_int/2))
        trange.append(tend)
        # print(trange)
        np.savetxt('saved_data/' + name + '_trange.csv', trange, delimiter=',', fmt='%s')
    else:
        trange = np.arange(tbeg, steps)
        np.savetxt('saved_data/' + name + '_trange.csv', trange, delimiter=',', fmt='%s')

    if cutoff:
        offs, fams = filter(offs, cutoff=cutoff)
        # print('filt offs', offs)
        # print('fams', fams)
        nf = len(offs[-1])
        #edges
        create_edges(tree, fams=fams, name=name)
        create_pop(offs, times=trange, name=name)
    else:
        #edges
        create_edges(tree, fams=np.arange(0, nf), name=name)
        create_pop(offs, times=trange, name=name)

def create_newoffs(offs, int_length, tbeg, tend):
    steps = tend - tbeg + 1
    # print('steps', steps)

    maxfam = len(offs[tend])
    int_num = (steps // int_length)
    last_int = steps % int_length
    print('intnum, lastint', int_num, last_int)
    offs = np.asarray(offs)
    chunks = np.hsplit(offs[0:steps-last_int], int_num)
    # print(chunks)
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

def filter(offs, cutoff):   #egal ob original oder new_offs
    print(offs[-1], cutoff)
    rel_offs = [[]] * (len(offs[-1]))
    filtered_offs = [[]] * (len(offs))
    filtered_fams = []

    for step in range(len(offs)):
        s = sum(offs[step])
        for i, e in enumerate(offs[step]):
            rel_offs[i] = np.concatenate((rel_offs[i], [e/s]))
    # print('rel', rel_offs)
    for f in range(len(rel_offs)):
        if max(rel_offs[f]) >= cutoff:
            filtered_fams.append(f)
    # print('filtered fam', filtered_fams)
    for step in range(len(offs)):
        for entry in filtered_fams:
            if entry < len(offs[step]):
                filtered_offs[step] = \
                    np.concatenate((filtered_offs[step], [offs[step][entry]]))

    print('filtered with %.2f:' % cutoff)
    return filtered_offs, filtered_fams

# name = 'bsp'
# name = '42_0_7162808'
name = '5011_0_f8684e7'
print(len(np.load('saved_data/' + name + '_offsprings.npy')))
# print(np.load('saved_data/' + name + '_offsprings.npy'))
# create_input(name, int_length=250)
create_input(name, int_length=250, cutoff=0.004)
# create_input(name, int_length=3, cutoff=0.3)




