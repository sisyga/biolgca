from lgca import get_lgca
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#c:\Users\Nr.12\Anaconda3\Lib\site-packages\
mpl.use('Agg')
"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""

"""
Script to plot the time evolution of 4 summary statistics for the lgca
For each parameter combination and measure all trials are plotted into one figure
"""

"""Setup"""
dims = 10
timesteps = 10
trials = 1 #50 for talk
time = range(timesteps + 1)
mode = 'di'
mode_num = 'CRAP_TEST'

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


densities = np.ones(2)*0.6
betas = np.ones(2)*3.5

entropy = np.empty((len(densities), len(betas), 4, trials)) #end_entr, end_normalized_entropy, end_polar alignment param, end_mean_alignment

#start_entr = np.empty((trials,))
end_entr = np.empty((trials,))
#ratio_entr = np.empty((trials,))
#start_norm_entr = np.empty((trials,))
end_norm_entr = np.empty((trials,))
#ratio_norm_entr = np.empty((trials,))
end_polar_alignment = np.empty((trials,))
end_mean_alignment = np.empty((trials,))
#fig, ax = plt.subplots(nrows=1, ncols=1)

for d in range(len(densities)):
    #print("Density: ")
    #print(densities[d])
    for b in range(len(betas)):
        for i in range(trials):
            #print("Beta:")
            #print(betas[b])
            lgca1 = get_lgca(interaction=(mode + '_alignment'), ve=False, bc='periodic', density=densities[d], geometry='lin', dims=dims, beta=betas[b], exclude_center=True) #exclude_center=True
            #start_entr[i] = lgca1.calc_entropy()
            #start_norm_entr[i] = lgca1.calc_normalized_entropy()
            lgca1.timeevo(timesteps=timesteps, record=True, recordnove=True, showprogress=False)
            plt.figure(fig_ent.number)
            plt.plot(time, lgca1.ent_t, linewidth=0.75)
            plt.figure(fig_norment.number)
            plt.plot(time, lgca1.normEnt_t, linewidth=0.75)
            plt.figure(fig_polAlParam.number)
            plt.plot(time, lgca1.polAlParam_t, linewidth=0.75)
            plt.figure(fig_meanAlign.number)
            plt.plot(time, lgca1.meanAlign_t, linewidth=0.75)
            end_entr[i] = lgca1.calc_entropy()
            end_norm_entr[i] = lgca1.calc_normalized_entropy()
            end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            end_mean_alignment[i] = lgca1.calc_mean_alignment()
            # ratio_entr[i]= # figure this out!
            # ratio_norm_entr[i]=
        #entropy[d][b][0] = start_entr.sum() / trials
        entropy[d][b][0] = end_entr
        #entropy[d][b][3] = start_norm_entr.sum() / trials
        entropy[d][b][1] = end_norm_entr
        entropy[d][b][2] = end_polar_alignment
        entropy[d][b][3] = end_mean_alignment #MISTAKE!!!!
        paramstr = "Density: " + '{0:.1f}'.format(densities[d]) + ", Beta: " + '{0:.1f}'.format(betas[b])
        savestr = mode_num + "_" + str(dims) + "_" + '{0:.6f}'.format(densities[d]) + "_" + '{0:.6f}'.format(\
            betas[b]) + "_" + str(timesteps) + "_" + str(trials)

        plt.figure(fig_ent.number)
        plt.plot(time, np.ones(len(time)) * lgca1.smax, label="Maximal Entropy", linewidth=0.75)
        plt.ylim([-0.01, lgca1.smax + 0.1])
        plt.title(paramstr)
        plt.legend()
        plt.savefig('./images/' + savestr + "_entropy.png")
        plt.cla()
        plt.figure(fig_norment.number)
        plt.ylim([-0.01, 1.01])
        plt.title(paramstr)
        plt.savefig('./images/' + savestr + "_normentropy.png")
        plt.cla()
        plt.figure(fig_polAlParam.number)
        plt.ylim([-0.01, 1.01])
        plt.title(paramstr)
        plt.savefig('./images/' + savestr + "_polal.png")
        plt.cla()
        plt.figure(fig_meanAlign.number)
        plt.ylim([-1.02, 1.02])
        plt.title(paramstr)
        plt.savefig('./images/' + savestr + "_meanal.png")
        plt.cla()
    #label = "Density: " + str(densities[d])
    #ax.plot(betas, entropy[d,:,6], label=label) #4


savestr = mode_num + "_" + str(dims) + "_" + 'dens' + "_" + 'beta' + "_" + str(timesteps) + "_" + str(trials)
pd.to_pickle(entropy, "./images/dens_beta" + savestr + ".pkl")

#entropy_longlong = pd.read_pickle("./images/dens_betaana_old_100_3__210_dens_beta_500_100.pkl")
#plt.cla()
#fig2 = plt.hist(entropy_longlong[0,0,2])
#plt.title("Polar alignment parameter histogram, L=210")
#Out[26]: Text(0.5, 1.0, 'Polar alignment parameter histogram, L=210')