from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid

# dim = int(env['DIMS'])
# rc = int(env['RESTCHANNELS'])
# rep = int(env['REPETITIONS'])

dim=2
rc=1
rep=1

dens = 1
birthrate = 0.5
deathrate = 0.2

uu = str(uuid())[0:7]
saving_data = False

for i in range(0, rep):
    print('bin bei wdh: ', i)
    start = time.time()
    name = str(2*dim + dim*rc) + str(dim)

    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',\
           density = dens, dims = dim, r_b = birthrate, variation = False, restchannels = rc ,r_d = deathrate)
    id = name + '_' + str(i) + '_' + str(uu) + 'rc15_mitReori'
    # id = 'Test_nochmal'
    t = lgca.timeevo_until_hom(spatial=True)
    print('thom ', t)
    # spacetime_plot(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'], \
    #                tend=None, save=saving_data, id=str(180) + str(name))
    # spacetime_plot(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'], \
    #                tend=500, save=saving_data, id=str(180) + str(name))
    # spacetime_plot(nodes_t=lgca.nodes_t, labels=lgca.props['lab_m'], \
    #                tend=20, save=saving_data, id=id, figsize=(6,3))
    print(np.shape(lgca.offsprings))
    np.save('saved_data/offsspat', lgca.offsprings)
    o = np.load('saved_data/offsspat.npy')
    print(o)
    # mullerplot(lgca.offsprings)
    if saving_data:
        np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        np.save('saved_data/' + str(id) + '_nodest', lgca.nodes_t)
        np.save('saved_data/' + str(id) + '_labels', lgca.props['lab_m'])
        np.savez('saved_data/' + str(id) + '_Parameter', density = lgca.density, restchannels = lgca.restchannels,\
        dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, m=lgca.r_int)
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))