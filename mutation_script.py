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
        np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
        np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
        np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,
                 dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, rm=lgca.r_m, m=lgca.r_int)
    print(lgca.tree_manager.tree)
    ende = time.time()
    print('{:5.3f}min'.format((ende - start)/60))

def zauberei(eins):
    magie(167, 1, 40000, eins)

magie(dim=167, rc=1, steps=4, uu=0)

#####___server___####
# from lgca import get_lgca
# from lgca.helpers import *
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# from os import environ as env
# from uuid import uuid4 as uuid
# import multiprocessing
#
# def magie(dim, rc, steps, uu):
#     start = time.time()
#
#     saving_data = True
#
#     id = str(2 * dim + dim * rc) + str(dim) + "_mut_" + str(uu)
#
#     lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',
#                     variation=False, density=1, dims=dim, restchannels=rc,
#                     pop={1: 1})
#
#     #id = name + str(uu)
#
#     lgca.timeevo(timesteps=steps, recordMut=True)
#
#     if saving_data:
#         np.save('saved_data/' + str(id) + '_tree', lgca.tree_manager.tree)
#         np.save('saved_data/' + str(id) + '_families', lgca.props['lab_m'])
#         np.save('saved_data/' + str(id) + '_offsprings', lgca.offsprings)
#         np.savez('saved_data/' + str(id) + '_Parameter', density=lgca.density, restchannels=lgca.restchannels,
#                   dimension=lgca.l, kappa=lgca.K, rb=lgca.r_b, rd=lgca.r_d, rm=lgca.r_m, m=lgca.r_int)
#
#     ende = time.time()
#     print(uu, ': {:5.3f}min'.format((ende - start)/60))
#
# def zauberei(i):
#     magie(1, 499, 40000, i)
#
# print("starting threads")
# simulation_ids = [uuid() for _ in range(4)]
# with multiprocessing.Pool() as pool:
#     pool.map(zauberei, simulation_ids)
#
# print("all threads completed")