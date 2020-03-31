from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid


dim = 1
rc = 499
rep = 3
# dim = int(env['DIMS'])
# rc = int(env['RESTCHANNELS'])
# rep = int(env['REPETITIONS'])


uu = str(uuid())[0:7]
saving_data = True
ausgabe = False

for i in range(0, rep):
    print('wdh: ', i)
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim)

    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           variation=False, density=1, dims=dim, restchannels=rc, r_b=0.5, r_d=0.02)
    id = name + '_' + str(i) + '_' + str(uu)
    t = lgca.timeevo_until_hom(offsprings=True)
    print(t)
    if saving_data:
        # np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
        np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
    if ausgabe:
        print('tree', lgca.tree_manager.tree)
        print('_families', lgca.props['lab_m'])
        print('_offsprings', lgca.offsprings)
    # print('len offs', len(lgca.offsprings))
    # print('t', t)
    # print('offs', (lgca.offsprings))
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))
