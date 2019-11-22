# %%
from lgca import get_lgca
from lgca.helpers import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Paraliste:
# timesteps = 2
dim = 60
dens = 1
rc = 1
birthrate = 0.5
deathrate = 0.02
rep = 100
thom = np.zeros(rep)
t0 = time.time()
# Speichervorbereitungen:
saving = True
id = str(2 * dim + dim * rc) + str(dim)
# print(id)
# %%
# Durchlauf

for wdh in range(0, rep):
    start = time.time()
    print('bin bei wdh', wdh)
    lgca = get_lgca(ib=True, geometry='lin', interaction='inheritance', bc='reflecting',
                    density=dens, dims=dim, r_b=birthrate, variation=False, restchannels=rc, r_d=deathrate)
    lgca.timeevo_until_hom(record=True)
    thom[wdh - 1] = len(lgca.props_t)
    ende = time.time()
    print(len(lgca.props_t), '{:5.3f}min'.format((ende-start)/60))

tend = time.time()
print('total: ', '{:5.3f}min'.format((tend-t0)/60))
# %%
# Speichern
if saving:
    np.save('saved_data/' + str(id) + '_th', thom)
