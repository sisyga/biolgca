from lgca import get_lgca
from matplotlib import pyplot as plt

lgca = get_lgca(dims=(10, 10), bc='refl', ib=False, ve=False, geometry='square', interaction='go_or_grow', capacity=200,
                kappa=-1, restchannels=100)
print(lgca)

#%%
