import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dens = [0.2, 0.5, 1]
beta_di = [1.75, 1.5, 1.4]
beta_dd = [0.205, 0.12, 0.07]

plt.scatter(dens, beta_dd, label=("0.5 treshold"))
#plt.scatter(dens, beta_di, label=("0.5 treshold"))
plt.legend()
#plt.xlim([0, 4.2])
#plt.ylim([-0.01, 1.01])
#plt.ylim([-1.01, 1.01])
plt.title("Density-dependent model")
#plt.title("Density-independent model")
#plt.grid()
plt.xlabel('Density')
plt.ylabel('Sensitivity')
#plt.savefig('./images/' + filename + suffix + '.png')
plt.show()