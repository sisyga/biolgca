import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from multiprocess import PATH

parameters = np.load(PATH+'params.npz', allow_pickle=True)

constparams = parameters['constparams']#
r_ds = parameters['r_ds']
thetas = parameters['thetas']

data = np.load(PATH+'n_pr.npy', allow_pickle=True)

# create a grid of figures with size of 'data'
fig, axes = plt.subplots(data.shape[0], data.shape[1], figsize=(10, 10), sharex=True, sharey=True)

# iterate over the grid
for ax, lgca in zip(axes.flat, data.flat):
    # plot the data
    lgca.plot_prop_spatial(propname='kappa', ax=ax)







