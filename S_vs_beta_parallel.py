from lgca import get_lgca
import numpy as np
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""
"""is different to single now! (2 more measures)"""

"""Parallel Setup"""
#number of processes for parallel processing
nprocesses = 3

"""Setup"""
dims = 70
timesteps = 1000 #500 dd >300, >500 di: 1000
trials = 100 #50

# density is outer loop
# beta to be set

#densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#densities = np.append(np.arange(0,1,0.05), np.arange(1,3,0.25)) # interesting ranges
densities = np.arange(0, 1.1, 0.1)

#densities = np.ones(1)*0.2
betas = np.append(np.arange(0,0.5,0.1), np.arange(0.5,3.3, 0.05)) # interesting ranges
#betas = np.arange(0, 0.5, 0.005)

#betas=np.array([4.1, 5.1, 6.1, 7.1, 8.1, 9.1])
#betas = np.append(betas1, np.arange(1, 3, 0.25))

mode = "di"
prefix = "par_110_12_"

savestr = prefix + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_BETA"
pd.to_pickle(betas, "./pickles/" + savestr + ".pkl")
savestr = prefix + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
pd.to_pickle(densities, "./pickles/" + savestr + ".pkl")
#betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3]
#betas=np.arange(0, 5, 0.1)

measures = np.empty((len(densities), len(betas), 15))
#fig, ax = plt.subplots(nrows=1, ncols=1)

def job(d):
#for d in range(len(densities)):
    #print("Density: ")
    #print(densities[d])
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
    #std_error_mean_polal  #11
    #std_error_mean_meanal #12
    #variance_polal       #13
    #variance_meanal      #14
    for b in range(len(betas)):
        #print("Beta:")
        #print(betas[b])
        for i in range(trials):
            lgca1 = get_lgca(interaction=(mode + '_alignment'), ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b]) #density=densities[d]
            start_entr[i] = lgca1.calc_entropy()
            start_norm_entr[i] = lgca1.calc_normalized_entropy()
            start_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            start_mean_alignment[i] = lgca1.calc_mean_alignment()
            lgca1.timeevo(timesteps=timesteps, record=True, showprogress=False)
            end_entr[i] = lgca1.calc_entropy()
            end_norm_entr[i] = lgca1.calc_normalized_entropy()
            end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            end_mean_alignment[i] = lgca1.calc_mean_alignment()
            diff_entr[i] = end_entr[i] - start_entr[i]
            ratio_polar_alignment[i] = end_polar_alignment[i]/start_polar_alignment[i]
            ratio_mean_alignment = end_mean_alignment[i]/start_mean_alignment[i]
            # ratio_entr[i]= # figure this out!
            # ratio_norm_entr[i]=
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
    #label = "Density: " + str(densities[d])
    savestr = prefix + str(dims) + "_" + '{0:.6f}'.format(densities[d]) + "_beta_" + str(timesteps) + "_" + str(trials)
    pd.to_pickle(measures_t, "./pickles/" + savestr + "_" + mode + ".pkl")
    return d, measures_t


#sending the loops to different processes
if __name__ == '__main__':
    p= Pool(processes = nprocesses)
    data = p.map(job, range(len(densities))) #iterate job over a list or range, pass the returns into a list called data
    p.close()
    #datalength = len(data)
    #collecting results from the seperate processes
    for i in range(len(data)):
        measures[data[i][0]] = data[i][1]
    savestr = prefix + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
    pd.to_pickle(measures, "./pickles/" + savestr + "_" + mode + ".pkl")
    #ax.plot(betas, entropy[d,:,6], label=label) #4

#ax.set_title("Normalized Entropy")
#ax.set_title("Polar alignment parameter")
#plt.xlabel("Beta")
#plt.ylabel("S_norm")
#plt.legend()
"""
#check if pickling does what I think it does
#savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
#densities_new = pd.read_pickle("./pickles/" + savestr + ".pkl")
#print(np.all(densities_new == densities))

savestr = "par_110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
measures_new2 = pd.read_pickle("./pickles/" + savestr + "_" + mode + ".pkl")
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
measures_new = pd.read_pickle("./pickles/" + savestr + "_" + mode + ".pkl")
#print(np.all(measures_new == measures_new2))
fig=plt.figure()
plt.plot(betas, measures_new[1,:,6], label='serial')
plt.plot(betas, measures_new2[1,:,6], label='parallel')
plt.legend()
plt.xlabel('beta')
plt.savefig('./images/serpar_polal.png')
"""
#lgca1.plot_density()
#lgca1.plot_flux()