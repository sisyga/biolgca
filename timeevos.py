from lgca import get_lgca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""
"""Setup"""
dims = 70
timesteps = 2000
trials = 50
time = range(timesteps + 1)

plt.rcParams['figure.figsize'] = (9, 5)
fig_ent = plt.figure("Entropy")
fig_ent.suptitle("Entropy")
plt.xlabel("timestep")
fig_norment = plt.figure("Normalized Entropy")
fig_norment.suptitle("Normalized Entropy")
plt.xlabel("timestep")
fig_polAlParam = plt.figure("Polar Alignment Parameter")
fig_polAlParam.suptitle("Polar Alignment Parameter")
plt.xlabel("timestep")
fig_meanAlign = plt.figure("Mean Alignment")
fig_meanAlign.suptitle("Mean Alignment")
plt.xlabel("timestep")
plt.rcdefaults()
# density is outer loop
# beta to be set

#densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#densities = [0.2]
densities = np.ones(1)*0.8
#betas = [0.1, 0.12]
#betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3]
#betas=np.arange(0, 5, 0.1)
betas = np.ones(1)*0.00001

#entropy = np.empty((len(densities), len(betas), 8)) #start_entr, end_entr, ratio; same for normalized; polar alignment param

#start_entr = np.empty((trials,))
#end_entr = np.empty((trials,))
#ratio_entr = np.empty((trials,))
#start_norm_entr = np.empty((trials,))
#end_norm_entr = np.empty((trials,))
#ratio_norm_entr = np.empty((trials,))
#end_polar_alignment = np.empty((trials,))
#fig, ax = plt.subplots(nrows=1, ncols=1)

for d in range(len(densities)):
    #print("Density: ")
    #print(densities[d])
    for b in range(len(betas)):
        for i in range(trials):
            #print("Beta:")
            #print(betas[b])
            lgca1 = get_lgca(interaction='dd_alignment', ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b]) #density=densities[d]
            #start_entr[i] = lgca1.calc_entropy()
            #start_norm_entr[i] = lgca1.calc_normalized_entropy()
            lgca1.timeevo(timesteps=timesteps, record=True, recordnove=True)
            plt.figure(fig_ent.number)
            plt.plot(time, lgca1.ent_t, linewidth=0.75)
            plt.figure(fig_norment.number)
            plt.plot(time, lgca1.normEnt_t, linewidth=0.75)
            plt.figure(fig_polAlParam.number)
            plt.plot(time, lgca1.polAlParam_t, linewidth=0.75)
            plt.figure(fig_meanAlign.number)
            plt.plot(time, lgca1.meanAlign_t, linewidth=0.75)
            #end_entr[i] = lgca1.calc_entropy()
            #end_norm_entr[i] = lgca1.calc_normalized_entropy()
            #end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            # ratio_entr[i]= # figure this out!
            # ratio_norm_entr[i]=
            #entropy[d][b][0] = start_entr.sum() / trials
            #entropy[d][b][1] = end_entr.sum() / trials
            #entropy[d][b][3] = start_norm_entr.sum() / trials
            #entropy[d][b][4] = end_norm_entr.sum() / trials
            #entropy[d][b][6] = end_polar_alignment.sum()/trials
    #label = "Density: " + str(densities[d])
    #ax.plot(betas, entropy[d,:,6], label=label) #4


#ax.set_title("Normalized Entropy")
#ax.set_title("Polar alignment parameter")
#plt.xlabel("Beta")
#plt.ylabel("S_norm")
#plt.legend()
#pd.to_pickle(entropy, "./images/dens_beta_analysis_di_50tr.pkl")
paramstr ="Density: "+ str(densities[0]) + ", Beta: " + str(betas[0])
savestr = "100_" + str(dims) + "_" + '{0:.6f}'.format(densities[0]) + "_" + '{0:.6f}'.format(betas[0]) + "_" + str(timesteps) + "_" + str(trials)

plt.figure(fig_ent.number)
plt.plot(time, np.ones(len(time))*lgca1.smax, label="Maximal Entropy", linewidth=0.75)
plt.ylim([-0.01, lgca1.smax + 0.1])
plt.title(paramstr)
plt.legend()
plt.savefig('./images/'+savestr+"_entropy.png")
plt.figure(fig_norment.number)
plt.ylim([-0.01,1.01])
plt.title(paramstr)
plt.savefig('./images/'+savestr+"_normentropy.png")
plt.figure(fig_polAlParam.number)
plt.ylim([-0.01,1.01])
plt.title(paramstr)
plt.savefig('./images/'+savestr+"_polal.png")
plt.figure(fig_meanAlign.number)
plt.ylim([-1.02,1.02])
plt.title(paramstr)
plt.savefig('./images/'+savestr+"_meanal.png")

#lgca1.plot_density()
#lgca1.plot_flux()
#plt.show()