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
entropies = False

if create:
#create thom
    create_thom(variation=variation, filename=filename, path=path, rep=rep, save=True)

#read thom
thom = np.load(path + variation + '_thom.npy')

if histogram:
#plot thom histogram
    histogram_thom(thom=thom, int_length=intervall, save=save_hist, id=variation + '_' + filename)

for i in range(1):
    data = np.load(path + variation + '_' + str(i) + '_' + filename + '_offsprings.npy')
    tend, _ = data.shape
    plot_index(calc_hillnumbers(data), save=True, id=999, which='hill2')
    plot_hillnumbers_together(hill_1=calc_hillnumbers(data, order=1), hill_2=calc_hillnumbers(data, order=2),\
                              hill_3=calc_hillnumbers(data, order=3))
    plot_entropies_together(simpson=calc_simpson(data), shannon=calc_shannon(data), gini=calc_ginisimpson(data), save=True, id=999)
    plot_selected_entropies(shannon=calc_shannon(data), gini=calc_ginisimpson(data), hill2=calc_hillnumbers(data), save=True)
if entropies:
    create_averaged_entropies(filename=filename, variation=variation, path=path, rep=rep, save=True)
