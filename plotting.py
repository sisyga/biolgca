from lgca import get_lgca
import numpy as np
import matplotlib.pyplot as plt

path = '/home/simon/Dokumente/projects/leup_go_or_grow/param_scan/'
s = np.load(path + 'entropy_cluster.npy')

plt.imshow(s.T, origin='lower')
plt.colorbar()
plt.xlabel('beta')
plt.ylabel('rho')
plt.show()