import numpy as np

name = 'bsp'
fam = np.load('saved_data/' + name + '_families.npy')
offs = np.load('saved_data/' + name + '_offsprings.npy')
tree = np.load('saved_data/' + name + '_tree.npy')
tend = len(offs) #timesteps = tend-1
maxfam = len(offs[-1])-1
print(maxfam)
print(fam)
print(len(offs))
print(offs)
print(tree)

# edges = [["Parent", "Identity"]]
# for i in range(1, len(offs[-1])):
#     ori = tree.item().get(i)['parent']
#     if ori == i:
#         edges.append([0, i])
#     else:
#         edges.append([ori, i])
#
# print(edges)
#
# np.savetxt('saved_data/bsp_edges.csv', edges, delimiter=',', fmt='%s')

pop = [["Generation", "Identity", "Population"]]
for i in range(tend):
    pop.append([i, 0, 0])

f = 1
while f <= maxfam:
    for t in range(tend):
        if f < len(offs[t]):
            pop.append([t, f, offs[t][f]])
            # print([t, f, offs[t][f]])
        else:
            pop.append([t, f, 0])
    f += 1

np.savetxt('saved_data/bsp_pop.csv', pop, delimiter=',', fmt='%s')

print(pop)
