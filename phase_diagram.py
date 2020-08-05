import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.font_manager import FontProperties

def hyperbola(x, A, b, c):
    return np.divide(A, c*x)+b

def linear(x, A, b, c):
    return A*(x-3) + c

dens = np.arange(0.1, 1.1, 0.1)
beta_crit_di = [1.83, 1.735, 1.7, 1.63, 1.57, 1.52, 1.5, 1.485, 1.47, 1.45]
beta_init_di = [0.56, 0.45, 0.46, 0.43, 0.49, 0.44, 0.48, 0.47, 0.51, 0.52]
beta_crit_dd = [0.307, 0.207, 0.16, 0.133, 0.115, 0.1, 0.09, 0.082, 0.077, 0.071]
beta_init_dd = [0.224, 0.146, 0.123, 0.102, 0.093, 0.082, 0.072, 0.065, 0.063, 0.057]

#print(beta_crit_dd/beta_crit_dd[4])
#print(beta_init_dd/beta_crit_dd[4])
#print(beta_crit_di/beta_crit_di[4])
#print(beta_init_dd/beta_crit_dd[4])
#plt.scatter(dens, np.array(beta_crit_dd)/beta_crit_dd[4], c='green', label=("Critical sensitivity (dd)"))
#plt.scatter(dens, np.array(beta_init_dd)/beta_crit_dd[4], c='olive', label=("Initiating sensitivity (dd)"))
#plt.scatter(dens, np.array(beta_crit_di)/beta_crit_di[4], c='red', label=("Critical sensitivity (di)"))
#plt.scatter(dens, np.array(beta_init_dd)/beta_crit_di[4], c='orange', label=("Initiating sensitivity (di)"))
#plt.scatter(dens, (np.array(beta_crit_dd)-np.array(beta_init_dd))/np.array(beta_crit_dd), c='green', label=("Density-dependent"))
#plt.scatter(dens, np.array(beta_init_dd)/beta_crit_dd[4], c='olive', label=("Initiating sensitivity (dd)"))
#plt.scatter(dens, (np.array(beta_crit_di)-np.array(beta_init_di))/np.array(beta_crit_di), c='red', label=("Density-independent"))
#plt.scatter(dens, np.array(beta_init_dd)/beta_crit_di[4], c='orange', label=("Initiating sensitivity (di)"))
ydata = beta_crit_dd
params, params_covariance = optimize.curve_fit(hyperbola, dens, ydata)
ydata_1 = beta_crit_di
params1, params_covariance1 = optimize.curve_fit(hyperbola, dens, ydata_1)
params2, params_covariance2 = optimize.curve_fit(linear, dens, ydata_1)
estimates = hyperbola(dens, params[0], params[1], params[2])
estimates1 = hyperbola(dens, params1[0], params1[1], params1[2])
estimates2 = linear(dens, params2[0], params2[1], params2[2])
#residuals = ydata - estimates
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((ydata - np.mean(ydata))**2)
#r_squared = 1 - (ss_res/ss_tot) #dd: 0.9756656890315343; di: 0.8237794103875226 for hyperbola, 0.9268236296894914 for linear
#print(r_squared)
#plt.plot(dens, estimates, color='grey') #params[0], params[1], params[2]
#plt.scatter(dens, beta_crit_dd, c='tab:green', label=("Critical sensitivity"))
#plt.scatter(dens, beta_init_dd, c='tab:olive', label=("Initiating sensitivity"))
#plt.scatter(dens, beta_crit_di, c='tab:red', label=("Critical sensitivity"))
#plt.scatter(dens, beta_init_di, c='tab:orange', label=("Initiating sensitivity"))
#plt.ylim([1.4, 2])

fig, ax1 = plt.subplots(figsize=(5,9))
ax1.set_xlabel("Density")
ax1.set_ylabel("Sensitivity (dd)", color='green')
plt.plot(dens, estimates, color='grey')
plt.plot(dens, estimates, color='green', ls='dotted')
l1=plt.scatter(dens, beta_crit_dd, c='tab:green', label=("Critical sensitivity (dd)"))
l2=plt.scatter(dens, beta_init_dd, c='tab:olive', label=("Initiating sensitivity (dd)"))
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_ylim([0.05, 0.7])

ax2 = ax1.twinx()
ax2.set_ylabel("Sensitivity (di)", color='red')
plt.plot(dens, estimates1, color='grey')
plt.plot(dens, estimates2, color='grey')
plt.plot(dens, estimates1, color='red', ls='dotted')
plt.plot(dens, estimates2, color='red', ls='dotted')
l01=plt.scatter(dens, beta_crit_di, c='tab:red', label=("Critical sensitivity (di)"))
l02=plt.scatter(dens, beta_init_di, c='tab:orange', label=("Initiating sensitivity (di)"))
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim([0, 1.9])
fig.tight_layout()

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fontP = FontProperties()
fontP.set_size('small')
plt.legend(lines +  lines2, labels+labels2, loc="center", prop=fontP)
#fig.legend() # works easily, but the placement is shitty
plt.title("Phase diagram of sensitivity and density")
plt.grid()

plt.savefig("./images/phase_combined.png", fig=fig)
plt.show()