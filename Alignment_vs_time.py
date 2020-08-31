import lgca
from lgca import get_lgca
import matplotlib.pyplot as plt
import numpy as np


lgca2 = get_lgca(density=2, ve=False, geometry='hex', bc='refl', interaction='di_alignment', nodes=None, beta=5)

lgca2.timeevo(timesteps=100, record=True)
ex = 100

alignment = []


for i in range(ex):

        sumx = 0
        sumy = 0

        abb = lgca2.calc_flux(lgca2.nodes_t[i]) #[lgca2.nonborder] #nonborder?
        x = len(abb)
        y = len(abb[0])
        z = len(abb[0][0])



        for a in range(0, x):
            for b in range(0, y):
                for c in range(0, z):
                    if c == 0:
                        sumx = sumx + abb[a][b][c]
                    if c == 1:
                        sumy = sumy + abb[a][b][c]

        cells = lgca2.nodes_t[i].sum()                #tnodes[i][lgca2.nonborder].sum()

        print("sumy")
        print(sumy)
        print("cells")
        print(cells)

        sumy = sumy / cells  # / self.nodes[self.nonborder].sum()

        sumx = sumx / cells

        print("count")
        print(i)
        magnitude = np.sqrt(sumx**2 + sumy**2)

        alignment.append(magnitude)



print(alignment)

xax = [time for time in range(ex)]
plt.plot(xax, alignment)


plt.show()