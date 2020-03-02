from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math as m

dens = 1
# birthrate = 0.5
# deathrate = 0.02

dim = 2
rc = 1

lgca = get_lgca(ib=True, geometry='lin', interaction='passenger_mutations', bc='reflecting',\
           density=dens, dims=dim, restchannels=rc)

lgca.timeevo_until_hom()