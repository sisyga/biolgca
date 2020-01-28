# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt

#choose simulation
variation = 'Testdaten'
filename = ''
path = 'saved_data/'
rep = 3

#create thom
create = False

#create histogram
histogram = False
intervall = 2
save_hist = True

#calculate averaged entropies
entropies = True

if create:
#create thom
    create_thom(variation=variation, filename=filename, path=path, rep=rep, save=True)

#read thom
thom = np.load(path + variation + '_thom.npy')

if histogram:
#plot thom histogram
    histogram_thom(thom=thom, int_length=intervall, save=save_hist, id=variation + '_' + filename)

# for i in range(rep):
#     data = np.load(path + variation + '_' + str(i) + '_' + filename + '_offsprings.npy')
    # tend, _ = data.shape
    # print(calc_shannon(data))
    # print(calc_ginisimpson(data))

if entropies:
    create_averaged_entropies(filename=filename, variation=variation, path=path, rep=rep, save=True)
