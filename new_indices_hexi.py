# from lgca import get_lgca
from lgca.helpers import *
from lgca.analysis import *
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
import os

def load_scenario(sc):
    path = 'saved_data/' + sc + '_45sims'
    files = os.listdir(path)
    sh = [entry for entry in files if 'sh' in entry]
    hh = [entry for entry in files if 'hill2' in entry]
    print(len(sh))
    print(len(hh))
    return sh, hh

def read_scenario(list, sc):
    path = 'saved_data/' + sc + '_45sims/'
    data = [[]]
    i = 0
    for entry in list:
        data[i][:] = np.loadtxt(path + entry)
    return data


p_sh_names, p_hill_names = load_scenario('passenger')
d_sh_names, d_hill_names = load_scenario('driver')

p_sh = read_scenario(p_sh_names, 'passenger')
print(p_sh)
print(np.mean(p_sh))
p_means_hill = np.loadtxt('saved_data/passenger_45sims/Berechnungen/passenger_ave_hill2.csv')
print(len(p_means_hill))

p_means_sh = np.loadtxt('saved_data/passenger_45sims/Berechnungen/passenger_ave_sh.csv')
print(len(p_means_sh))
print(p_means_sh[0])

d_means_hill = np.loadtxt('saved_data/passenger_45sims/Berechnungen/passenger_ave_hill2.csv')
print(len(d_means_hill))
d_means_sh = np.loadtxt('saved_data/passenger_45sims/Berechnungen/passenger_ave_sh.csv')
print(len(d_means_sh))