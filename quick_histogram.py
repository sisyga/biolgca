import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
mpl.use('Agg')

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""

"""
Script to plot a histogram to disk without showing it in a popup window
Histogram for a summary measure for my lgca
"""

filename="dens_betaana_old_110_1__70_dens_beta_1000_100"

entropy = pd.read_pickle("./images/" + filename + ".pkl")
filename = filename + "_hist_"

plt.rcParams["figure.figsize"] = [7, 3]
fig = plt.hist(entropy[0,0,2], bins=40, range=(0,1)) #in the examples there is only one value for dens and beta each
# 2 polar alignment parameter
# 3 mean alignment
#fig.set_figheight(5)
plt.title("Polar alignment parameter histogram")
plt.xlim([0,1])
suffix = "polal"
plt.savefig('./images/' + filename + suffix + '.png')

plt.cla()
plt.xlim([-1,1])
plt.title("Mean alignment histogram")
suffix = "meanal"
fig = plt.hist(entropy[0,0,3], bins=40, range=(-1,1))
#fig.set_figheight(5)
plt.savefig('./images/' + filename + suffix + '.png')