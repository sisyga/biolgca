from lgca import get_lgca
from lgca.base import *
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid

# dim = int(env['DIMS'])
# rc = int(env['RESTCHANNELS'])
# rep = int(env['REPETITIONS'])

# dim=1
# rc=10
# which = 'onenode'

dim = 4
rc = 1
which = 'onerc'

rep = 1

dens = 1
birthrate = 0.5
deathrate = 0.02

uu = str(uuid())[0:7]
saving_data = False

for i in range(0, rep):
    print('bin bei wdh: ', i)
    start = time.time()
    name = str(2*dim + dim*rc) + '_' + str(dim) + '_' + str(i)

    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           density = dens, dims = dim, r_b = birthrate, variation = False, restchannels = rc ,r_d = deathrate)
    id = name + '_' + which
    # id = 'Test_nochmal'
    t = lgca.timeevo(timesteps=5, record=True)
    # spacetime_plot(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'],
    #                tend=None, save=saving_data, id=str(180) + str(name))
    spacetime_extra(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'], save=True, id=id)
    # spacetime_plot(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'], \
    #                tend=500, save=saving_data, id=str(180) + str(name))
    # spacetime_plot(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'], \
    #                tend=20, save=saving_data, id=id, figsize=(6,3))
    # print(np.shape(lgca.offsprings))
    # np.save('saved_data/offsspat', lgca.offsprings)
    # o = np.load('saved_data/offsspat.npy')
    # print(o)
    # mullerplot(lgca.offsprings)
    if saving_data:
        np.save('saved_data/spacetime_mini/' + str(id) + '_offsprings', lgca.offsprings)
        np.save('saved_data/spacetime_mini/' + str(id) + '_nodest', lgca.nodes_t)
        np.save('saved_data/spacetime_mini/' + str(id) + '_labels', lgca.props['lab_m'])
        np.savez('saved_data/spacetime_mini/' + str(id) + '_Parameter', density = lgca.density, restchannels = lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))
