from lgca import get_lgca
import numpy as np
import pandas as pd

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""

"""
Script to obtain summary statistics/measures from a specified number of lgca runs
for all combinations of all given densities and sensitivities
"""

"""Setup"""
# number of lattice sites in x-direction
dims = 70
# number of sample runs per lgca configuration
trials = 2 #50
# length of each sample run
timesteps = 101 #300 dd, >500 di

# Set varying parameters as ndarrays
# density outer loop
# beta inner loop
densities = np.array([0.2, 0.8])
betas = np.array([0.1, 0.2, 0.25, 0.3, 0.35])
# choice of transition probability model
# density-dependent (dd) or density-independent (di)
mode = "dd"
# set prefix for boundary conditions etc. further down (better in parallel script)

# save configuration (densities and sensitivities)
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_BETA"
pd.to_pickle(betas, "./pickles/" + savestr + ".pkl")
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
pd.to_pickle(densities, "./pickles/" + savestr + ".pkl")

# prepare for saving the measures from lgca runs
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

"""Run"""
# loop through densities
for d in range(len(densities)):
    #print("Density: ")
    #print(densities[d])
    # loop through sensitivities
    for b in range(len(betas)):
        #print("Beta:")
        #print(betas[b])
        # run <trials> samples of this lgca configuration
        for i in range(trials):
            # initiate lgca
            lgca1 = get_lgca(interaction=(mode + '_alignment'), ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b])
            # compute statistics of initial state
            start_entr[i] = lgca1.calc_entropy()
            start_norm_entr[i] = lgca1.calc_normalized_entropy()
            start_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            start_mean_alignment[i] = lgca1.calc_mean_alignment()
            # run the lgca
            lgca1.timeevo(timesteps=timesteps, record=True)
            # compute statistics of final state
            end_entr[i] = lgca1.calc_entropy()
            end_norm_entr[i] = lgca1.calc_normalized_entropy()
            end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            end_mean_alignment[i] = lgca1.calc_mean_alignment()
            # compute desired comparisons between initial and final state
            diff_entr[i] = end_entr[i] - start_entr[i]
            ratio_polar_alignment[i] = end_polar_alignment[i]/start_polar_alignment[i]
            ratio_mean_alignment = end_mean_alignment[i]/start_mean_alignment[i]
        # summarize the measures from all trials
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
    # save result for all sensitivities at one density to protect against crashes from the outside
    savestr = "110_" + str(dims) + "_" + '{0:.6f}'.format(densities[d]) + "_beta_" + str(timesteps) + "_" + str(trials)
    pd.to_pickle(measures[d], "./pickles/" + savestr + "_" + mode + ".pkl")

# save full result
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
pd.to_pickle(measures, "./pickles/" + savestr + "_" + mode + ".pkl")

"""
#check if pickling does what I think it does
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
densities_new = pd.read_pickle("./pickles/" + savestr + ".pkl")
print(np.all(densities_new == densities))
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
measures_new = pd.read_pickle("./pickles/" + savestr + "_" + mode + ".pkl")
print(np.all(measures_new == measures))
"""