from lgca import get_lgca
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


lgca1 = get_lgca(interaction='di', ve=False, bc='refl', density=2, geometry='hex', nodes=None, beta=10)

lgca1.timeevo(timesteps=20, record=True)

ani = lgca1.animate_flow()
ani2 = lgca1.animate_flux()
ani3 = lgca1.animate_density()

fentropy = lgca1.calc_entropy()
fnormentropy = lgca1.calc_normalized_entropy()
fpalignment = lgca1.calc_polar_alignment_parameter()
fmpalignment = lgca1.calc_mean_alignment()

print("final entropies")
print(fentropy)
print("final total alignments")
print(fpalignment)
print("final local alignments")
print(fmpalignment)
print("final normalized entropies")
print(fnormentropy)

plt.show()