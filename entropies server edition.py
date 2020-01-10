from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import pandas as pd
from os import environ as env
from uuid import uuid4 as uuid

# lgca erstellen
dim = int(env['DIMS'])
rc = int(env['RESTCHANNELS'])
rep = int(env['REPETITIONS'])

dens = 1
birthrate = 0.5
deathrate = 0.02

uu = str(uuid())[0:7]
saving_data = True

for i in range(0, rep):
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim)

    lgca= get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           density = dens, dims = dim, r_b = birthrate, variation = False, restchannels = rc ,r_d = deathrate)
    id = name + '_' + str(i) + '_'+ str(uu)
    lgca.timeevo_until_hom(offsprings=True)
    # print(lgca.offsprings[-1])

    if saving_data:
        np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        np.savez('saved_data/' + str(id) + '_Parameter', density = lgca.density, restchannels = lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))

# timeevo until hom
    # für jeden Zeitschritt Indexe berechnen, merken -> während thom oder nachträglich?!

# repeat

# zeilenweise aka zeitschrittweise die indexe einlesen, mean()

# plot