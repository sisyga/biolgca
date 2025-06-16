# from matplotlib import pyplot as plt
from mayavi import mlab
from lgca import get_lgca


lgca = get_lgca(dims=10, bc='refl', ib=False, ve=False, geometry='moore', interaction='di_alignment', density=0.02,
                beta=3, restchannels=0, nb_include_center=False)
print(lgca)
lgca.timeevo(10, record=True)
anim = lgca.animate_flux()
# plt.show()
mlab.show()

#%%
