from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid


# dim = 2
# rc = 2
# rep = 1
# steps = 10
dim = int(env['DIMS'])
rc = int(env['RESTCHANNELS'])
rep = int(env['REPETITIONS'])
steps = int(env['STEPS'])


uu = str(uuid())[0:7]
saving_data = True
ausgabe = False

# e = {1: 0.5, 2: 0.3, 3: 0.2}
f = {1: 1}

for i in range(0, rep):
    print('wdh: ', i)
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim) + "_mut"

    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           variation=False, density=1, dims=dim, restchannels=rc,\
                    pop=f)

    id = name + '_mut_' + str(i) + '_' + str(uu)
    lgca.timeevo(timesteps=steps, recordMut=True)
    if saving_data:
        np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
        np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, rm=lgca.r_m, m=lgca.r_int)
    if ausgabe:
        print('tree', lgca.tree_manager.tree)
        print('_families', lgca.props['lab_m'])
        print('_offsprings', lgca.offsprings)
    # print('len offs', len(lgca.offsprings))
    # print('offs', (lgca.offsprings))
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))
