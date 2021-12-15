import numpy as np
from lgca import get_lgca

lgca=get_lgca(geometry='hex', ib=True, interaction='go_and_grow_mutations', effect='driver_mutation', dims=2, density=0.5, restchannels=2, r_m=0.5, r_d=0.1)
lgca.timeevo(timesteps=30)

print("goodbye!")