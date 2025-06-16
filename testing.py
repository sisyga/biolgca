import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'
from matplotlib import pyplot as plt
from mayavi import mlab
from lgca import get_lgca

lgca = get_lgca(dims=20, bc='refl', ib=False, ve=True, geometry='moore', interaction='di_alignment', density=0.02,
                beta=2, restchannels=0, nb_include_center=False)
print(lgca)
lgca.timeevo(100, record=True)
anim = lgca.animate_flux()
# plt.show()
mlab.show()

#%%
