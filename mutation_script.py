from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from os import environ as env
from uuid import uuid4 as uuid
import multiprocessing

def magie(dim, rc, steps, uu):
    start = time.time()
    # uu = str(uuid())
    saving_data = False

    name = str(2 * dim + dim * rc) + str(dim) + "_mut_"

    lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',
                    variation=False, density=1, dims=dim, restchannels=rc,
                    pop={1: 1})

    id = name + '_' + str(uu)

    lgca.timeevo(timesteps=steps, recordMut=True)

    if saving_data:
        np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
        # np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
        # np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        # np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,
        #          dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, rm=lgca.r_m, m=lgca.r_int)

    ende = time.time()
    print('{:5.3f}min'.format((ende - start)/60))

def zauberei(eins):
    magie(2, 2, 10, eins)


rep = range(5)
zauberei('irgendwas')

with multiprocessing.Pool() as pool:
    pool.map(zauberei, rep)
