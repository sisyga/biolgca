from matplotlib import pyplot as plt

from lgca import get_lgca

lgca = get_lgca(dims=(100, 100), bc='refl', ib=False, ve=True, geometry='hex', interaction='alignment',
                beta=3, restchannels=0)
print(lgca)
lgca.timeevo(100, record=True)
anim = lgca.animate_flux()
plt.show()

#%%
