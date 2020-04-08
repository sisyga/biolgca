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

def create_input(filename, tbeg=0, tend=None, int_length=1, cutoff=0, ori=None):
    tree = np.load('saved_data/' + filename + '_tree.npy')
    offs = correct(np.load('saved_data/' + filename + '_offsprings.npy'))
    # offs = create_ori(offs, tree)
    name = filename + '_int_length=' + str(int_length) + '_cutoff=' + str(cutoff)

    if ori:
        offs_ori = create_ori(offs, tree)   # offs = offs_ori -> plottet ori
        name = filename + '_int_length=' + str(int_length) + '_cutoff=' + str(cutoff) + '_ori'

    if tend is None:
        tend = len(offs) - 1
                                    #TODO tbeg, tend variabel
    steps = tend - tbeg + 1
    nf = len(offs[tend])

    if int_length != 1:
        offs = create_newoffs(offs, int_length, tbeg, tend)
        if ori:
            offs_ori = create_newoffs(offs_ori, int_length, tbeg, tend)
        # print('int offs', offs)
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
        np.savetxt('saved_data/' + name + '_trange.csv', trange, delimiter=',', fmt='%s')
    else:
        trange = np.arange(tbeg, steps)
        np.savetxt('saved_data/' + name + '_trange.csv', trange, delimiter=',', fmt='%s')

    if cutoff:
        if ori:
            offs, fams = filter_ori(offs, offs_ori, tree, cutoff=cutoff)
        else:
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

def filter(offs, cutoff):   #egal ob original oder new_offs
    print(offs[-1], cutoff)
    rel_offs = [[]] * (len(offs[-1]))
    filtered_offs = [[]] * (len(offs))
    filtered_fams = []

    for step in range(len(offs)):
        s = sum(offs[step])
        for i, e in enumerate(offs[step]):
            rel_offs[i] = np.concatenate((rel_offs[i], [e/s]))
    print('rel', rel_offs)
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

def filter_ori(offsprings, originals, tree, cutoff):
    # print(originals)
    o, f = filter(originals, cutoff)
    # print('o', o)
    # print('f', f)
    # print(tree)
    fams = []
    ori = []

    for entry in f:
        for c in tree.item().keys():
            if tree.item().get(c)['origin'] == entry+1:
                fams.append(c-1)
    # print(fams)

    for step in range(0, len(offsprings)):
        entries = []
        for entry in fams:
            if entry < len(offsprings[step]):
                entries.append(offsprings[step][entry])

        ori.append(entries)
    # print(ori)

    return ori, fams


# name = 'bsp'
# name = '42_0_7162808'
# name = '5011_0_711862f'
# names = ['5011_mut_01d15ca8-03d6-4ca0-985c-777dc41365d8', '5011_mut_062b726c-48ab-4c6a-b2ad-e4cc27cc165a', '5011_mut_498d4c70-5dc8-4f0f-bb52-51820fc66505', '5011_mut_55186c3c-e01e-4609-8952-d1314b736521', '5011_mut_623a24a3-be94-4a90-9141-9ddbafd4f0a8']
#
# for i in names:
#     name = '5011_mut_04_01/' + i
#     print(name)
#     create_input(name, int_length=250)
# names = ['501167_mut_0017b261-6db6-44a2-9b43-1adf27c36267', '501167_mut_085323c9-de7e-4419-904b-34c72ba2aa62', '501167_mut_499d1a96-d0f2-4872-b3db-f949ce1f933d']
# for i in names:
#     name = '501167_mut_04_02/' + i
#     print(name)
#     create_input(name, int_length=250)
name1 = 'Varianten ohne mut/5011_2_640e948'
name167 = 'Varianten ohne mut/501167_4_ac06cfb'

names = [name1, name167]

for i in names:
    print(i)
    create_input(i, int_length=250)
# print(o)
# create_newoffs(o, int_length=3, tbeg=0, tend=len(o) - 1)
# create_input(name, int_length=250)
# create_input(name, int_length=250, ori=True)
# create_input(name, int_length=250, cutoff=0.004)



