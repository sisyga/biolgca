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
thom = []

for i in range(0, rep):
    print('bin bei wdh: ', i)
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim)

    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           density = dens, dims = dim, r_b = birthrate, variation = False, restchannels = rc ,r_d = deathrate)
    id = name + '_' + str(i) + '_' + str(uu)
    # id = 'Test_nochmal'

    t = lgca.timeevo_until_hom()
    thom.append(t)

    ende = time.time()
    print('{:5.3f}s'.format(ende-start))

np.save('saved_data/' + str(id) + '_thom_' + str(rep), thom)

