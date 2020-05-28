from lgca import get_lgca
from lgca.ib_interactions import driver_mut, passenger_mut
from lgca.helpers import *
from lgca.helpers2d import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from os import environ as env
from uuid import uuid4 as uuid


def create_hex(dim, rc, steps, iiid='None', driver=False, rec=False, rec_off=False,
               save=False, trange=False):
    nodes = np.zeros((dim, dim, 6 + rc))
    for i in range(0, 6 + rc):
        nodes[dim // 2, dim // 2, i] = i + 1
    name = '_' + str(dim) + 'x' + str(dim) + '_rc=' + str(rc) + '_steps=' + str(steps) + str(iiid)

    if driver:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                            r_b=0.09, r_d=0.08, r_m=0.001, effect=driver_mut)
        name += '_driver'
        if trange:
            trange.append('_driver')
    else:
        lgca_hex = get_lgca(ib=True, geometry='hex', bc='reflecting', nodes=nodes, interaction='mutations',
                            r_b=0.09, r_d=0.08, r_m=0.001, effect=passenger_mut)
        name += '_passenger'
        if trange:
            trange.append('_passenger')
    uu = str(uuid())[0:5]
    trange.append(str(uu))
    lgca_hex.timeevo(timesteps=steps, recordoffs=rec_off, record=rec, trange=trange)
    if save:
        np.save('saved_data/uuid=' + str(uu) + name + '_tree', lgca_hex.tree_manager.tree)
        np.save('saved_data/uuid=' + str(uu) + name + '_families', lgca_hex.props['lab_m'])
        np.save('saved_data/uuid=' + str(uu) + name + '_offsprings', lgca_hex.offsprings)
        if rec:
            np.save('saved_data/uuid=' + str(uu) + name + '_nodes', lgca_hex.nodes_t)
#TEST
rep = 1
dim = 4
rc = 2
steps = 10
t_plots = [2, 5]

#ORIGINAL
# rep = 2
# dim = 50
# rc = 500
# steps = 2500
# t_plots = [500, 1000, 1500, 2000, 2500]

for i in range(0, rep):
    print('bin bei ', i)
    trange = [t_plots]
    create_hex(dim=dim, rc=rc, steps=steps, driver=True, trange=trange,
               rec_off=True, save=True, iiid='_hexi_'+str(i))
    trange = [t_plots]
    create_hex(dim=dim, rc=rc, steps=steps, driver=False, trange=trange,
               rec_off=True, save=True, iiid='_hexi_'+str(i))



