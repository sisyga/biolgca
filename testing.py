from lgca import get_lgca
from matplotlib import pyplot as plt

lgca = get_lgca(dims=(20, 20, 20), bc='refl', ib=False, ve=True, geometry='cubic', interaction='alignment',
                beta=1, restchannels=0)
print(lgca)
lgca.timeevo(100, record=True)
lgca.animate_density()

#%%
