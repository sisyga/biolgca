from lgca import get_lgca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""
"""Setup"""
dims = 50
timesteps = 100

# density is outer loop
# beta to be set

#densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#densities = [0.2]
densities = np.ones(1)*0.2
#betas = [0.1, 0.12]
#betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3]
betas=np.arange(0, 5, 0.1)

entropy = np.empty((len(densities), len(betas), 8)) #start_entr, end_entr, ratio; same for normalized; polar alignment param
trials = 10
start_entr = np.empty((trials,))
end_entr = np.empty((trials,))
ratio_entr = np.empty((trials,))
start_norm_entr = np.empty((trials,))
end_norm_entr = np.empty((trials,))
ratio_norm_entr = np.empty((trials,))
end_polar_alignment = np.empty((trials,))
fig, ax = plt.subplots(nrows=1, ncols=1)

for d in range(len(densities)):
    #print("Density: ")
    #print(densities[d])
    for b in range(len(betas)):
        #print("Beta:")
        #print(betas[b])
        for i in range(trials):
            lgca1 = get_lgca(interaction='di_alignment', ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b]) #density=densities[d]
            start_entr[i] = lgca1.calc_entropy()
            start_norm_entr[i] = lgca1.calc_normalized_entropy()
            lgca1.timeevo(timesteps=timesteps, record=True)
            end_entr[i] = lgca1.calc_entropy()
            end_norm_entr[i] = lgca1.calc_normalized_entropy()
            end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            # ratio_entr[i]= # figure this out!
            # ratio_norm_entr[i]=
        entropy[d][b][0] = start_entr.sum() / trials
        entropy[d][b][1] = end_entr.sum() / trials
        entropy[d][b][3] = start_norm_entr.sum() / trials
        entropy[d][b][4] = end_norm_entr.sum() / trials
        entropy[d][b][6] = end_polar_alignment.sum()/trials
    label = "Density: " + str(densities[d])
    ax.plot(betas, entropy[d,:,6], label=label) #4

#ax.set_title("Normalized Entropy")
ax.set_title("Polar alignment parameter")
plt.xlabel("Beta")
#plt.ylabel("S_norm")
plt.legend()
pd.to_pickle(entropy, "./images/dens_beta_analysis_di_50tr.pkl")

#lgca1.plot_density()
#lgca1.plot_flux()