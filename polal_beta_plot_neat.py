import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

"""CAREFUL WITH RUNNING! CHANGE FILENAME FIRST!"""

"""
Script to plot several measures for one parameter combination
can combine several measure matrices if they match in format (careful with matching content)
"""

"""
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

"""
#filename = "par_100_8_70_dens_beta_500_100_dd"
filename = "par_110_12_70_dens_beta_1000_100_di" # relevant
#filename= "par_110_2_70_dens_beta_1000_50_di" #without type!
#filename= "par_100_6_70_dens_beta_500_50_dd"
#suffix = "_meanal"
suffix = ""

measures = pd.read_pickle("./pickles/" + filename + ".pkl")
betas = pd.read_pickle("./pickles/" + filename + "_BETA.pkl")
densities = pd.read_pickle("./pickles/" + filename + "_DENS.pkl")

filenames2=[]
#filenames2=["par_110_8_70_dens_beta_1000_50_di"]
#filenames2= ["par_110_3_70_dens_beta_1000_50_di", "par_110_4_70_dens_beta_1000_50_di", "par_110_5_70_dens_beta_1000_50_di", "par_110_6_70_dens_beta_1000_50_di", "par_110_7_70_dens_beta_1000_50_di", "par_110_8_70_dens_beta_1000_50_di"] #without type!
for i in range(len(filenames2)):
    measures2 = pd.read_pickle("./pickles/" + filenames2[i] + ".pkl")
    betas2 = pd.read_pickle("./pickles/" + filenames2[i] + "_BETA.pkl")
    densities2 = pd.read_pickle("./pickles/" + filenames2[i] + "_DENS.pkl")
    if np.all(np.equal(densities, densities2)):
        betas = np.append(betas, betas2)
        measures = np.append(measures, measures2, axis=1)

#betas = np.append(betas, betas2)
#densities = np.append(densities, densities2)
#measures = np.append(measures, measures2, axis=1)
print("Beta:")
print(betas)
print("Density:")
print(densities)


betas2 = np.sort(betas)
indices = np.arange(1, len(betas)+1)
measures2 = np.empty(measures.shape)
for i in range(len(betas2)):
    indices[i] = np.argwhere(betas==betas2[i]) #contains source index, i is target index
    measures2[:,i,:] = measures[:,indices[i],:]
#print(np.all(measures_new == measures_new2))
fig, ax=plt.subplots(nrows=2, ncols=1, sharex=True)

for d in range(len(densities)):
#Mean alignment in blue, polar alignment in black, normalized entropy in green
    #plt.plot(betas2, np.divide(measures2[d,:,9], measures2[d,len(betas2)-1,9]), label=("Density: "+ str(densities[d])))
    ax[0].plot(betas2, measures2[d,:,6], label=("Density: "+ '{0:.1f}'.format(densities[d])))


    ax[1].plot(betas2, measures2[d,:,9], label=("Density: "+ '{0:.1f}'.format(densities[d])))
    #plt.plot(betas2, measures2[d,:,9]+measures2[d,:,12], c='grey', ls='dashed')
    #plt.plot(betas2, measures2[d,:,9]-measures2[d,:,12], c='grey', ls='dashed')
    if d in [1, len(densities)-1]:
        ax[0].plot(betas2, measures2[d,:,6]+measures2[d,:,11], c='grey', ls='dotted') #std error
        ax[0].plot(betas2, measures2[d,:,6]-measures2[d,:,11], c='grey', ls='dotted')
        ax[0].plot(betas2, measures2[d,:,6]+np.divide(1.9842 * np.sqrt(measures2[d,:,13]), 10), c='grey', ls='dashed') #confidence interval
        ax[0].plot(betas2, measures2[d,:,6]-np.divide(1.9842 * np.sqrt(measures2[d,:,13]), 10), c='grey', ls='dashed') #confidence interval
        #1.9842
        ax[1].plot(betas2, measures2[d,:,9]+measures2[d,:,12], c='grey', ls='dotted')
        ax[1].plot(betas2, measures2[d,:,9]-measures2[d,:,12], c='grey', ls='dotted')
        ax[1].plot(betas2, measures2[d,:,9]+np.divide(1.9842 * np.sqrt(measures2[d,:,14]), 10), c='grey', ls='dashed') #confidence interval
        ax[1].plot(betas2, measures2[d,:,9]-np.divide(1.9842 * np.sqrt(measures2[d,:,14]), 10), c='grey', ls='dashed') #confidence interval
    #print(measures2[d,len(betas2)-1,9])
    #print(np.divide(measures2[d,:,9], measures2[d,len(betas2)-1,9]))
    #print(measures[d,:,9])
#plt.plot(betas2, measures2[1,:,6], label=("Density: "+ str(densities[1])))
ax[0].cla()
ax[1].cla()
ax[1].set_title("Mean alignment")
ax[0].set_title("Polar alignment parameter")
ax[0].plot(betas2, measures2[3, :, 6], label=("Density: " + '{0:.1f}'.format(densities[3])))
ax[1].plot(betas2, measures2[3, :, 9], label=("Density: " + '{0:.1f}'.format(densities[3])))

fontP = FontProperties()
fontP.set_size('small')
#legend([plot1], "title", prop=fontP)
ax[0].legend(prop=fontP)
ax[1].legend(prop=fontP)
#plt.xlim([0, 4.2])
ax[0].set_ylim([-0.01, 1.01])
ax[1].set_ylim([-0.01, 1.01])
#plt.ylim([-1.01, 1.01])
#plt.yticks(np.arange(0,1,0.1))
#plt.xticks(np.arange(0.05,0.35,0.005))
ax[0].grid()
ax[1].grid()
plt.xlabel('Beta')
#plt.savefig('./images/' + filename + suffix + '.png')
plt.show()