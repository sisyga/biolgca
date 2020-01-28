from lgca import get_lgca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""
"""Setup"""
dims = 70
timesteps = 101 #300 dd, >500 di
trials = 2 #50

# density is outer loop
# beta to be set

#densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
densities = np.array([0.2, 0.8])
#densities = np.ones(1)*0.2
betas = np.array([0.1, 0.2, 0.25, 0.3, 0.35])
mode = "dd"

savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_BETA"
pd.to_pickle(betas, "./pickles/" + savestr + ".pkl")
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
pd.to_pickle(densities, "./pickles/" + savestr + ".pkl")
#betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3]
#betas=np.arange(0, 5, 0.1)

measures = np.empty((len(densities), len(betas), 11))
start_entr = np.empty((trials,))            #0
end_entr = np.empty((trials,))              #1
diff_entr = np.empty((trials,))             #2
start_norm_entr = np.empty((trials,))       #3
end_norm_entr = np.empty((trials,))         #4
start_polar_alignment = np.empty((trials,)) #5
end_polar_alignment = np.empty((trials,))   #6
ratio_polar_alignment = np.empty((trials,)) #7
start_mean_alignment = np.empty((trials,))  #8
end_mean_alignment = np.empty((trials,))    #9
ratio_mean_alignment = np.empty((trials,))  #10

#fig, ax = plt.subplots(nrows=1, ncols=1)

for d in range(len(densities)):
    #print("Density: ")
    #print(densities[d])
    for b in range(len(betas)):
        #print("Beta:")
        #print(betas[b])
        for i in range(trials):
            lgca1 = get_lgca(interaction=(mode + '_alignment'), ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b]) #density=densities[d]
            start_entr[i] = lgca1.calc_entropy()
            start_norm_entr[i] = lgca1.calc_normalized_entropy()
            start_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            start_mean_alignment[i] = lgca1.calc_mean_alignment()
            lgca1.timeevo(timesteps=timesteps, record=True)
            end_entr[i] = lgca1.calc_entropy()
            end_norm_entr[i] = lgca1.calc_normalized_entropy()
            end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            end_mean_alignment[i] = lgca1.calc_mean_alignment()
            diff_entr[i] = end_entr[i] - start_entr[i]
            ratio_polar_alignment[i] = end_polar_alignment[i]/start_polar_alignment[i]
            ratio_mean_alignment = end_mean_alignment[i]/start_mean_alignment[i]
            # ratio_entr[i]= # figure this out!
            # ratio_norm_entr[i]=
        measures[d][b][0] = start_entr.sum() / trials
        measures[d][b][1] = end_entr.sum() / trials
        measures[d][b][2] = diff_entr.sum() / trials
        measures[d][b][3] = start_norm_entr.sum() / trials
        measures[d][b][4] = end_norm_entr.sum() / trials
        measures[d][b][5] = start_polar_alignment.sum() / trials
        measures[d][b][6] = end_polar_alignment.sum() / trials
        measures[d][b][7] = ratio_polar_alignment.sum() / trials
        measures[d][b][8] = start_mean_alignment.sum() / trials
        measures[d][b][9] = end_mean_alignment.sum() / trials
        measures[d][b][10] = ratio_mean_alignment.sum() / trials
    #label = "Density: " + str(densities[d])
    savestr = "110_" + str(dims) + "_" + '{0:.6f}'.format(densities[d]) + "_beta_" + str(timesteps) + "_" + str(trials)
    pd.to_pickle(measures[d], "./pickles/" + savestr + "_" + mode + ".pkl")

savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
pd.to_pickle(measures, "./pickles/" + savestr + "_" + mode + ".pkl")
    #ax.plot(betas, entropy[d,:,6], label=label) #4

#ax.set_title("Normalized Entropy")
#ax.set_title("Polar alignment parameter")
#plt.xlabel("Beta")
#plt.ylabel("S_norm")
#plt.legend()
"""
#check if pickling does what I think it does
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
densities_new = pd.read_pickle("./pickles/" + savestr + ".pkl")
print(np.all(densities_new == densities))
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
measures_new = pd.read_pickle("./pickles/" + savestr + "_" + mode + ".pkl")
print(np.all(measures_new == measures))
"""
#lgca1.plot_density()
#lgca1.plot_flux()