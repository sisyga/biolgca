"""from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np

#nodes = np.array([[0,0],[2,0],[0,0],[4,0],[1,0],[0,0],[0,0],[0,0],[0,0],[30,0],[14,0],[0,0]])
#nodes = np.array([[0,0],[0,0],[0,0],[0,0],[0,1],[1,0],[3,0],[0,0]])

#Setup
beta = 5
density = 3
dims = 100
timesteps = 100

#INITIAL STATE
lgca = get_lgca(interaction='di_alignment', ve=False, bc='periodic', density=density, geometry='lin', dims=dims, beta=beta)

#TIMESTEPPING
lgca.timeevo(timesteps=timesteps, record=True)
ani = lgca.plot_density()
#ani2 = lgca.plot_flux_fancy()
ani3 = lgca.plot_flux()
plt.show()

#Todo:
# plot title reflect parameters - done
# flux plot title, too
# and for Simon's plots also
# list of parameters - done
# dd/di for ve? what did Simon implement?
# way to store initial conditions permanently for repeating experiments - done
# fancy flux plotting
# how do I characterize clusters?
# /home/bianca/
# repos/biolgca/images/
"""

from lgca import get_lgca
import numpy as np
import pandas as pd

"""Setup"""
dims = 50
timesteps = 100

# density is outer loop
# beta to be set

#densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
densities = [0.3]
betas = [0.1, 0.5, 1, 1.5, 2, 2.5, 3]
#betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3]

entropy = np.empty((len(densities), len(betas), 6)) #start_entr, end_entr, ratio; same for normalized
trials = 5
start_entr = np.empty((trials,))
end_entr = np.empty((trials,))
ratio_entr = np.empty((trials,))
start_norm_entr = np.empty((trials,))
end_norm_entr = np.empty((trials,))
ratio_norm_entr = np.empty((trials,))

for d in range(len(densities)):
    print("Density: ")
    print(densities[d])
    for b in range(len(betas)):
        print("Beta:")
        print(betas[b])
        for i in range(trials):
            lgca1 = get_lgca(interaction='di_alignment', ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b])
            start_entr[i] = lgca1.calc_entropy()
            start_norm_entr[i] = lgca1.calc_normalized_entropy()
            lgca1.timeevo(timesteps=timesteps, record=True)
            end_entr[i] = lgca1.calc_entropy()
            end_norm_entr[i] = lgca1.calc_normalized_entropy()
            # ratio_entr[i]= # figure this out!
            # ratio_norm_entr[i]=
        entropy[d][b][0] = start_entr.sum() / trials
        entropy[d][b][1] = end_entr.sum() / trials
        entropy[d][b][3] = start_norm_entr.sum() / trials
        entropy[d][b][4] = end_norm_entr.sum() / trials

pd.to_pickle(entropy, "./images/test1.pkl")