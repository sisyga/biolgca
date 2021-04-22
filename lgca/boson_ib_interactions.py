from random import choices
import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm
from copy import deepcopy

try:
    from .interactions import tanh_switch
except ImportError:
    from interactions import tanh_switch
    
def trunc_gauss(lower, upper, mu, sigma=.1, size=1):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)

def randomwalk(lgca):
    temp = 1
    
def birth(lgca):
    temp = 1
    
def birthdeath(lgca):
    temp = 1

def go_or_growold(lgca):
    relevant = (lgca.density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = 0
        for channel in node:
            for cell in channel:
                if(npr.random() <= lgca.r_d):
                    channel.remove(cell)
            density += len(channel)
        rho = density/lgca.capacity

        #lgca.maxlabel=len(lgca.props["kappa"])       
        velcells = []
        restcells = []
        ch_counter = 0
        for channel in node:
            for cell in channel:
                if(npr.random()<=tanh_switch(rho=rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell])):
                    restcells.append(cell)
                else:
                    velcells.append(cell)
                if ch_counter == 2:
                    if(npr.random() <= lgca.r_b*(1-rho)):
                        restcells.append(len(lgca.props["kappa"]))
                        lgca.props['kappa'].append(npr.normal(loc=lgca.props['kappa'][cell], scale=lgca.kappa_std))
                        lgca.props['theta'].append(trunc_gauss(0,1,mu=lgca.props['theta'][cell],sigma=lgca.theta_std))
            ch_counter +=1
            
        node = [[],[],restcells]
        for cell in velcells:
            node[npr.randint(lgca.K-1)].append(cell)
        lgca.nodes[coord] = deepcopy(node)   

def go_or_grow(lgca):
    relevant = (lgca.density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = 0
        for channel in node:
            for cell in channel:
                if(npr.random() <= lgca.r_d):
                    channel.remove(cell)
            density += len(channel)
        rho = density/lgca.capacity

        #lgca.maxlabel=len(lgca.props["kappa"])       
        velcells = []
        restcells = []
        ch_counter = 0
        for channel in node:
            for cell in channel:
                
                if ch_counter == 2:
                    if(npr.random()<=(tanh_switch(rho=rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell])**(1/3.0))):#increasing resting cells probability to rest
                        restcells.append(cell)
                    else:
                        velcells.append(cell)
                    if(npr.random() <= lgca.r_b*(1-rho)):
                        restcells.append(len(lgca.props["kappa"]))
                        lgca.props['kappa'].append(npr.normal(loc=lgca.props['kappa'][cell], scale=lgca.kappa_std))
                        lgca.props['theta'].append(trunc_gauss(0,1,mu=lgca.props['theta'][cell],sigma=lgca.theta_std))
                else:
                    if(npr.random()<=(tanh_switch(rho=rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell])**(3.0))): #reducing moving cells probability to rest
                        restcells.append(cell)
                    else:
                        velcells.append(cell)
            ch_counter +=1
            
        node = [[],[],restcells]
        for cell in velcells:
            node[npr.randint(lgca.K-1)].append(cell)
        lgca.nodes[coord] = deepcopy(node)   
    
"""   
        for cell in node[-1]:
            if(npr.random() <= lgca.r_b*(1-rho)):
                node[-1].append(len(lgca.props["kappa"]))
                lgca.props['kappa'].append(npr.normal(loc=lgca.props['kappa'][cell], scale=lgca.kappa_std))
                lgca.props['theta'].append(trunc_gauss(0,1,mu=lgca.props['theta'][cell],sigma=lgca.theta_std))
                density+=1
                
        for channel in node:
            for cell in channel:
                if(npr.random()<=tanh_switch(rho=rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell])):
                    node[-1].append(cell)
                    channel.remove(cell)
                else:
                    node[npr.randint(0,lgca.K-1)].append(cell)
                    channel.remove(cell)

"""