from lgca import get_lgca
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

"""
Script to obtain summary statistics/measures from a specified number of lgca runs
for all combinations of all given densities and sensitivities
"""

"""Setup"""
# number of lattice sites in x-direction
#dims = 70
# number of sample runs per lgca configuration
trials = 30 #50
# length of each sample run
timesteps = 200 #300 dd, >500 di


# Set varying parameters as ndarrays
# density outer loop
# beta inner loop
densities = np.array([0.2, 0.4, 0.6, 1, 1.3, 1.5, 2, 2.5])
betas = np.array([0, 0.1, 0.2, 0.35, 0.5, 0.65 , 0.8, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5, 3, 5])
# choice of transition probability model
# density-dependent (dd) or density-independent (di)
mode = "dd"
# set prefix for boundary conditions etc. further down (better in parallel script)

# save configuration (densities and sensitivities)
#savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_BETA"
#pd.to_pickle(betas, "./pickles/" + savestr + ".pkl")
#savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
#pd.to_pickle(densities, "./pickles/" + savestr + ".pkl")

"""
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

"""


entropy = []
normentropy = []
palignment = []
mpalignment = []

fentropy = []
fnormentropy = []
fpalignment = []
fmpalignment = []



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
            lgca1 = get_lgca(interaction=(mode + '_alignment'), ve=False, bc='refl', density=densities[d], geometry='hex', nodes=None, beta=betas[b])
            # compute statistics of initial state
            #start_entr[i] = lgca1.calc_entropy()
            entropy.append(lgca1.calc_entropy())
            #start_norm_entr[i] = lgca1.calc_normalized_entropy()
            normentropy.append(lgca1.calc_normalized_entropy())
            #start_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            palignment.append(lgca1.calc_polar_alignment_parameter())
            #start_mean_alignment[i] = lgca1.calc_mean_alignment
            mpalignment.append(lgca1.calc_mean_alignment())
            # run the lgca
            lgca1.timeevo(timesteps=timesteps, record=True)
            # compute statistics of final state
            #end_entr[i] = lgca1.calc_entropy()
            fentropy.append(lgca1.calc_entropy())
            #end_norm_entr[i] = lgca1.calc_normalized_entropy()
            fnormentropy.append(lgca1.calc_normalized_entropy())
            #end_polar_alignment[i] = lgca1.calc_polar_alignment_parameter()
            fpalignment.append(lgca1.calc_polar_alignment_parameter())
            #end_mean_alignment[i] = lgca1.calc_mean_alignment
            fmpalignment.append(lgca1.calc_mean_alignment())
            # compute desired comparisons between initial and final state
            #diff_entr[i] = end_entr[i] - start_entr[i]
            #ratio_polar_alignment[i] = end_polar_alignment[i]/start_polar_alignment[i]
            #ratio_mean_alignment = end_mean_alignment[i]/start_mean_alignment[i]
        # summarize the measures from all trials
        #measures[d][b][0] = start_entr.sum() / trials
        #measures[d][b][1] = end_entr.sum() / trials
        #measures[d][b][2] = diff_entr.sum() / trials
        #measures[d][b][3] = start_norm_entr.sum() / trials
        #measures[d][b][4] = end_norm_entr.sum() / trials
        #measures[d][b][5] = start_polar_alignment.sum() / trials
        #measures[d][b][6] = end_polar_alignment.sum() / trials
        #measures[d][b][7] = ratio_polar_alignment.sum() / trials
        #measures[d][b][8] = start_mean_alignment.sum() / trials
        #measures[d][b][9] = end_mean_alignment.sum() / trials
        #measures[d][b][10] = ratio_mean_alignment.sum() / trials





    # save result for all sensitivities at one density to protect against crashes from the outside
    #savestr = "110_" + str(dims) + "_" + '{0:.6f}'.format(densities[d]) + "_beta_" + str(timesteps) + "_" + str(trials)
    #pd.to_pickle(measures[d], "./pickles/" + savestr + "_" + mode + ".pkl")

# save full result
#savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
#pd.to_pickle(measures, "./pickles/" + savestr + "_" + mode + ".pkl")


entropyavg = []
normentropyavg = []
palignmentavg = []
mpalignmentavg = []

fentropyavg = []
fnormentropyavg = []
fpalignmentavg = []
fmpalignmentavg = []



ii = 0

for x in range(len(betas) * len(densities)):

    ent = 0
    norment = 0
    palign = 0
    mpalign = 0
    fent = 0
    fnorment = 0
    fpalign = 0
    fmpalign = 0

    for i in range(trials):

        ent = ent + entropy[ii]

        norment = norment + normentropy[ii]

        palign = palign + palignment[ii]

        mpalign = mpalign + mpalignment[ii]

        fent = fent + fentropy[ii]

        fnorment = fnorment + fnormentropy[ii]

        fpalign = fpalign + fpalignment[ii]

        fmpalign = fmpalign + fmpalignment[ii]



        ii = ii + 1

    entropyavg.append(ent / trials)
    normentropyavg.append(norment / trials)
    palignmentavg.append(palign / trials)
    mpalignmentavg.append(mpalign / trials)

    fentropyavg.append(fent / trials)
    fnormentropyavg.append(fnorment / trials)
    fpalignmentavg.append(fpalign / trials)
    fmpalignmentavg.append(fmpalign / trials)


print(len(entropyavg))
print(len(fentropyavg))
print(len(palignmentavg))
print(len(fnormentropyavg))


print("")

print("trials:")
print(trials)

print("")

print("betas:")
print(betas)

print("")

print("densities:")
print(densities)

print("")

print("final entropies")
print(fentropy)
print("final total alignments")
print(fpalignment)
print("final local alignments")
print(fmpalignment)
print("final normalized entropies")
print(fnormentropy)

print("")



# For density = 1,
#Initial states
k = 0

for j in range(len(densities)):

    plotentropy1 =[]
    plotnormentropy1 = []
    plotalignment1 = []
    plotmalignment1 = []
    print(k)

    for i in range(len(betas)):
        plotentropy1.append(entropyavg[k + i])

    for i in range(len(betas)):
        plotnormentropy1.append(normentropyavg[k + i])

    for i in range(len(betas)):
        plotalignment1.append(palignmentavg[k + i])

    for i in range(len(betas)):
        plotmalignment1.append(mpalignmentavg[k + i])

    plt.plot(betas, plotentropy1)
    plt.title("initial entropy density =  " + str(densities[j]))
    plt.show()

    plt.plot(betas, plotalignment1)
    plt.title("initial palignment density = " + str(densities[j]))
    plt.show()


    plt.plot(betas, plotnormentropy1)
    plt.title("initial norm entropy density = " + str(densities[j]))
    plt.show()

    plt.plot(betas, plotmalignment1)
    plt.title("initial mean alignment (local) = " + str(densities[j]))
    plt.show()

    #Final states

    plotentropy2 =[]
    plotnormentropy2 = []
    plotalignment2 = []
    plotmalignment2 = []


    for i in range(len(betas)):
        plotentropy2.append(fentropyavg[k + i])

    for i in range(len(betas)):
        plotnormentropy2.append(fnormentropyavg[k + i])

    for i in range(len(betas)):
        plotalignment2.append(fpalignmentavg[k + i])

    for i in range(len(betas)):
        plotmalignment2.append(fmpalignmentavg[k + i])

    plt.plot(betas, plotentropy2)
    plt.title("final entropy, density = " + str(densities[j]))
    plt.show()

    plt.plot(betas, plotalignment2)
    plt.title("final palignment density = " + str(densities[j]))
    plt.show()


    plt.plot(betas, plotnormentropy2)
    plt.title("final norm entropy density = " + str(densities[j]))
    plt.show()

    plt.plot(betas, plotmalignment2)
    plt.title("final mean alignment (local) = " + str(densities[j]))
    plt.show()


    k = k + len(betas)


input()





"""
plt.plot(betas, entropyavg)
plt.plot(betas, fentropyavg, betas)

plt.show()


plt.plot(betas, normentropyavg)
plt.plot(betas, fnormentropyavg)

plt.show()


plt.plot(betas, palignmentavg)
plt.plot(betas, fpalignmentavg)

plt.show()


plt.plot(densities, entropyavg)
plt.plot(densities, fentropyavg)

plt.show()


plt.plot(densities, normentropyavg)
plt.plot(densities, fnormentropyavg)

plt.show()


plt.plot(densities, palignmentavg)
plt.plot(densities, fpalignmentavg)

plt.show()
"""


"""
#check if pickling does what I think it does
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials) + "_" + mode + "_DENS"
densities_new = pd.read_pickle("./pickles/" + savestr + ".pkl")
print(np.all(densities_new == densities))
savestr = "110_" + str(dims) + "_dens_beta_" + str(timesteps) + "_" + str(trials)
measures_new = pd.read_pickle("./pickles/" + savestr + "_" + mode + ".pkl")
print(np.all(measures_new == measures))
"""