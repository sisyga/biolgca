import numpy as np
def create_input(filename):
    #fam = np.load('saved_data/' + filename + '_families.npy')
    offs = np.load('saved_data/' + filename + '_offsprings.npy')
    tree = np.load('saved_data/' + filename + '_tree.npy')
    tend = len(offs)  # timesteps = tend-1
    maxfam = len(offs[-1]) - 1

    edges = [["Parent", "Identity"]]
    np.savetxt('saved_data/' + filename + 'edges.csv', edges, delimiter=',', fmt='%s')
    with open('saved_data/' + filename + 'edges.csv', "a") as file:
        for i in range(1, len(offs[-1])):
            ori = tree.item().get(i)['parent']
            if ori == i:
                file.write(str([0, i]) + '\n')
            else:
                file.write(str([ori, i]) + '\n')

    pop = [["Generation", "Identity", "Population"]]
    for i in range(tend):
        pop.append([i, 0, 0])
    np.savetxt('saved_data/' + filename + 'pop.csv', pop, delimiter=',', fmt='%s')
    f = 1
    with open('saved_data/' + filename + 'pop.csv', "a") as file:
        while f <= maxfam:
            for t in range(tend):
                if f < len(offs[t]):
                    file.write(str([t, f, offs[t][f]]) + '\n')
                    # print([t, f, offs[t][f]])
                else:
                    file.write(str([t, f, 0]) + '\n')
            f += 1


#
name = 'real180_bsp'
# create_input(filename)
fam = np.load('saved_data/' + name + '_families.npy')
offs = np.load('saved_data/' + name + '_offsprings.npy')
tree = np.load('saved_data/' + name + '_tree.npy')
tend = len(offs) #timesteps = tend-1
maxfam = len(offs[-1])-1

# timesteps mit Mutationen:
mutationsteps = [0]
for i in range(1, tend):
    if len(offs[i]) != len(offs[i-1]):
        mutationsteps.append(i)
mutationsteps.append(tend-1)
# print(mutationsteps)
# edges erstellen
edges = [["Parent", "Identity"]]
filename = 'real180_bsp_mut'
np.savetxt('saved_data/' + filename + 'edges.csv', edges, delimiter=',', fmt='%s')
with open('saved_data/' + filename + 'edges.csv', "a") as file:
    for i in range(1, len(offs[-1])):
        ori = tree.item().get(i)['parent']
        if ori == i:
            file.write(str(0) + ',' + str(i) + '\n')
        else:
            file.write(str(ori) + ',' + str(i) + '\n')

#pop erstellen
pop = [["Generation", "Identity", "Population"]]
for entry in mutationsteps:
     pop.append([entry, 0, 0])
np.savetxt('saved_data/' + filename + 'pop.csv', pop, delimiter=',', fmt='%s')
f = 1
with open('saved_data/' + filename + 'pop.csv', "a") as file:
    while f <= maxfam:
        for t in mutationsteps:
            if f < len(offs[t]):
                file.write(str(t) + ',' + str(f) + ',' + str(offs[t][f]) + '\n')
            else:
                file.write(str(t) + ',' + str(f) + ',' + str(0) + '\n')
        f += 1
