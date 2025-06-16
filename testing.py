from matplotlib import pyplot as plt

from lgca import get_lgca

lgca = get_lgca(dims=100, bc='refl', ib=False, ve=True, geometry='hex', interaction='nematic',
                beta=4, restchannels=1)
print(lgca)
lgca.timeevo(100, record=True)
anim = lgca.animate_flux()
plt.show()

#%%
