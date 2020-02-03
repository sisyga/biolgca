from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid

dim = int(env['DIMS'])
rc = int(env['RESTCHANNELS'])
rep = int(env['REPETITIONS'])

dens = 1
birthrate = 0.5
deathrate = 0.02

uu = str(uuid())[0:7]
saving_data = True

for i in range(0, rep):
    print('wdh: ', i)
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim)

    lgca= get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           density = dens, dims = dim, r_b = birthrate, variation = False, restchannels = rc ,r_d = deathrate)
    id = name + '_' + str(i) + '_' + str(uu)
    lgca.offsprings.append([-99] + [1]*(rc+2)*dim)
    lgca.timeevo_until_hom(offsprings=True)

    if saving_data:
        np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
    print('len offs', len(lgca.offsprings))
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))
