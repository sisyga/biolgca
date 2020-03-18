import numpy as np

def create_input(filename, tbeg=0, tend=None, int_length=1, cutoff=None):
    offs = np.load('saved_data/' + filename + '_offsprings.npy')
    if cutoff and int_length == 1:
        #TODO
        exit(999)
    tree = np.load('saved_data/' + filename + '_tree.npy')
    if tend is None:
        tend = len(offs) - 1
    steps = tend - tbeg + 1
    nf = len(offs[tend]) - 1
    # print(nf)
    new_filename = name + '_' + str(tbeg) + '-' + str(tend)

    if cutoff is None:
        edges = [["Parent", "Identity"]]
        np.savetxt('saved_data/' + new_filename + '_edges.csv', edges, delimiter=',', fmt='%s')
        with open('saved_data/' + new_filename + '_edges.csv', "a") as file:
            for i in range(1, nf + 1):
                ori = tree.item().get(i)['parent']
                if ori == i:
                    file.write(str(0) + ',' + str(i) + '\n')
                else:
                    file.write(str(ori) + ',' + str(i) + '\n')

    #TODO edges anpassen für filtered_offs (fams übergeben, dann tree)
    if int_length != 1:
        new_offs = create_newoffs(filename, int_length, tbeg, tend)
        if cutoff:
            new_filename = new_filename + '_filtered %.3f' %cutoff
            new_offs, fams = filter(new_offs)
            nf = len(new_offs[-1])
            edges = [["Parent", "Identity"]]
            np.savetxt('saved_data/' + new_filename + '_edges.csv', edges, delimiter=',', fmt='%s')
            with open('saved_data/' + new_filename + '_edges.csv', "a") as file:
                for i in fams:
                    ori = tree.item().get(i+1)['parent']
                    if ori == i+1:
                        file.write(str(0) + ',' + str(i+1) + '\n')
                    else:
                        file.write(str(ori) + ',' + str(i+1) + '\n')
        last_int = steps % int_length
        mean_i = np.arange(int(int_length / 2), steps - last_int, int_length)
        if last_int != 0:
            mean_li = int((len(offs) + len(offs) - last_int) / 2)   #TODO für tbeg, tend variabel anpassen
            if tend != mean_li:
                times = np.concatenate(([0], mean_i, [mean_li], [len(offs) - 1]))
        else:
            times = np.concatenate(([0], mean_i, [len(offs) - 1]))
        # trange = []
        # trange.append(entry for entry in times)
        np.savetxt('saved_data/' + new_filename + '_summed_timerange.csv', times, delimiter=',', fmt='%s')

        pop = ["Population"]
        # print(new_offs)
        for entry in times:
            pop.append(0)
        np.savetxt('saved_data/' + new_filename + '_summed_population.csv', pop, delimiter=',', fmt='%s')
        f = 1
        with open('saved_data/' + new_filename + '_summed_population.csv', "a") as file:
            while f <= nf:
                for t, _ in enumerate(times):
                    if f <= len(new_offs[t]):
                        file.write(str(new_offs[t][f-1]) + '\n')
                    else:
                        file.write(str(0) + '\n')
                f += 1

    else:
        pop = ["Population"]
        for i in range(tbeg, tend + 1):
            pop.append(0)
        np.savetxt('saved_data/' + new_filename + '_population.csv', pop, delimiter=',', fmt='%s')
        f = 1
        with open('saved_data/' + new_filename + '_population.csv', "a") as file:
            while f <= nf:
                for t in range(tbeg, tend + 1):
                    if f < len(offs[t]):
                        file.write(str(offs[t][f]) + '\n')
                    else:
                        file.write(str(0) + '\n')
                f += 1

def create_newoffs(filename, int_length=1, tbeg=0, tend=None):
    offs = np.load('saved_data/' + filename + '_offsprings.npy')
    if tend is None:
        tend = len(offs) - 1
    steps = tend - tbeg + 1
    print('steps', steps)

    maxfam = len(offs[tend]) - 1
    int_num = (steps // int_length)
    last_int = steps % int_length
    print('intnum, lastint', int_num, last_int)

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
    #TODO -99 bei ori_offs entfernen
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
int_length = 3

offs = np.load('saved_data/' + name + '_offsprings.npy')
tree = np.load('saved_data/' + name + '_tree.npy')

# no = create_newoffs(name, int_length=3)
create_input(name, int_length=int_length, cutoff=0.25)
# tend = None
# tbeg = 0
# if tend is None:
#     tend = len(offs) - 1
# steps = tend - tbeg + 1
# print(steps)
# maxfam = len(offs[tend]) - 1
# int_num = (steps // int_length)
# last_int = steps % int_length
# print(int_num, last_int)




#####nur mutierende timesteps####
# timesteps mit Mutationen:
# mutationsteps = [0]
# for i in range(1, tend):
#     if len(offs[i]) != len(offs[i-1]):
#         mutationsteps.append(i)
# mutationsteps.append(tend-1)
# # print(mutationsteps)
# # edges erstellen
# edges = [["Parent", "Identity"]]
# filename = 'real180_bsp_mut'
# np.savetxt('saved_data/' + filename + 'edges.csv', edges, delimiter=',', fmt='%s')
# with open('saved_data/' + filename + 'edges.csv', "a") as file:
#     for i in range(1, len(offs[-1])):
#         ori = tree.item().get(i)['parent']
#         if ori == i:
#             file.write(str(0) + ',' + str(i) + '\n')
#         else:
#             file.write(str(ori) + ',' + str(i) + '\n')
#
# #pop erstellen
# pop = [["Generation", "Identity", "Population"]]
# for entry in mutationsteps:
#      pop.append([entry, 0, 0])
# np.savetxt('saved_data/' + filename + 'pop.csv', pop, delimiter=',', fmt='%s')
# f = 1
# with open('saved_data/' + filename + 'pop.csv', "a") as file:
#     while f <= maxfam:
#         for t in mutationsteps:
#             if f < len(offs[t]):
#                 file.write(str(t) + ',' + str(f) + ',' + str(offs[t][f]) + '\n')
#             else:
#                 file.write(str(t) + ',' + str(f) + ',' + str(0) + '\n')
#         f += 1
