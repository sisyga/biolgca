from lgca import get_lgca
import numpy as np
import pandas as pd
from multiprocessing import Pool

"""
Script to obtain summary statistics/measures from a specified number of 1D lgca runs
for all combinations of all given densities and sensitivities
The workload can be parallelized
"""

"""Parallelization Setup"""
# number of processes for parallel processing
# if sequential execution is desired, set to 1
nprocesses = 3

"""Setup"""
# number of lattice sites in x-direction
dims = 70
# number of sample runs per lgca configuration
trials = 2 #50
# length of each sample run
timesteps = 101 #300 dd, >500 di

# Set varying parameters as ndarrays
densities = np.arange(0, 1.1, 0.1)
betas = np.append(np.arange(0,0.5,0.1), np.arange(0.5,3.3, 0.05))
# choice of transition probability model
# density-dependent (dd) or density-independent (di)
mode = "di"
# set a prefix for saving result files, e.g. to identify boundary conditions
prefix = "par_110_12_"

# save configuration (densities and sensitivities)
savestr = prefix + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_BETA"
pd.to_pickle(betas, "./pickles/" + savestr + ".pkl")
savestr = prefix + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
pd.to_pickle(densities, "./pickles/" + savestr + ".pkl")

# prepare for saving the measures from lgca runs
# global ndarray to collect results from all processes has to provide space for all results
measures = np.empty((len(densities), len(betas), 15))

"""Workload for each process"""
def job(d):
    #print("Density: ")
    #print(d)

    # prepare for saving the measures from lgca runs
    # local result ndarray for this process only needs space for results with 1 density
    measures_t = np.empty((len(betas), 15))
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
    # for completeness of measure list:
    # std error of the mean polar alignment     #11
    # std error of the mean mean alignment      #12
    # variance of the polar alignment           #13
    # variance of the mean alignment            #14

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
            lgca1.timeevo(timesteps=timesteps, record=True, showprogress=False)
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
        measures_t[b][0] = start_entr.sum() / trials
        measures_t[b][1] = end_entr.sum() / trials
        measures_t[b][2] = diff_entr.sum() / trials
        measures_t[b][3] = start_norm_entr.sum() / trials
        measures_t[b][4] = end_norm_entr.sum() / trials
        measures_t[b][5] = start_polar_alignment.sum() / trials
        measures_t[b][6] = end_polar_alignment.sum() / trials
        measures_t[b][7] = ratio_polar_alignment.sum() / trials
        measures_t[b][8] = start_mean_alignment.sum() / trials
        measures_t[b][9] = end_mean_alignment.sum() / trials
        measures_t[b][10] = ratio_mean_alignment.sum() / trials
        measures_t[b][11] = np.std(end_polar_alignment) / np.sqrt(trials)
        measures_t[b][12] = np.std(end_mean_alignment) / np.sqrt(trials)
        measures_t[b][13] = np.var(end_polar_alignment)
        measures_t[b][14] = np.var(end_mean_alignment)
    # save result for all sensitivities at one density to protect against crashes from the outside
    savestr = prefix + str(dims) + "_" + '{0:.6f}'.format(densities[d]) + "_beta_" + str(timesteps) + "_" + str(trials)
    pd.to_pickle(measures_t, "./pickles/" + savestr + "_" + mode + ".pkl")
    # return results to parent process
    return d, measures_t


"""Parallelization"""
# send the loops to different processes
if __name__ == '__main__':
    p = Pool(processes=nprocesses)
    # iterate job over a list or range, pass the returned values into a list called data
    # thereby loop through densities
    data = p.map(job, range(len(densities)))
    p.close()
    # collect results from the separate processes
    for i in range(len(data)):
        measures[data[i][0]] = data[i][1]
    # save full result
    savestr = prefix + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
    pd.to_pickle(measures, "./pickles/" + savestr + "_" + mode + ".pkl")

