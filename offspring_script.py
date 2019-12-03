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

uu = uuid()[0:7]
saving_data = True

for i in range(0, rep):
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim)

    lgca= get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           density = dens, dims = dim, r_b = birthrate, variation = False, restchannels = rc ,r_d = deathrate)
    id = name + '_' + str(i) + '_'+ str(uu)
    lgca.timeevo_until_hom(record=True)
    offsprings = np.zeros((len(lgca.props_t), len(lgca.props_t[0]['num_off'])))
    for t in range(len(lgca.props_t)):
        for anc in range(len(lgca.props_t[0]['num_off'])):
            offsprings[t, anc] = lgca.props_t[t]['num_off'][anc]
    if saving_data:
        np.save('saved_data/' + str(id) + '_props_t', lgca.props_t)
        np.save('saved_data/' + str(id) + '_offsprings', offsprings)
        np.save('saved_data/' + str(id) + '_nodes_t', lgca.nodes_t)
        np.savez('saved_data/' + str(id) + '_Parameter', density = lgca.density, restchannels = lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))