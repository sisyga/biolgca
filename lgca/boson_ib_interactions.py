from random import choices
import numpy as np
from numpy import random as npr
from scipy.stats import truncnorm

try:
    from .interactions import tanh_switch
except ImportError:
    from interactions import tanh_switch
    
    
def randomwalk(lgca):
    temp = 1
    
def birth(lgca):
    temp = 1
    
def birthdeath(lgca):
    temp = 1
"""    
def go_or_grow_lame(lgca):
    for node in lgca.nodes:
        channel_counter = 1
        for channel in node:
            for cell in channel:
                if(np.random.random() < lgca.r_d):
                    channel.remove(cell) 
                if(cell in channel):
                    if(npr.random() < tanh_switch(rho=(len(channel)/lgca.capacity), kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell])):    #should we calculate rho after or before death?
                        #switch to rest
                        node[lgca.K-1].append(cell) #add cell to rest channel
                        channel.remove(cell)
                    else:
                        #switch to moving + reorientation in one step
                        node[np.random.randint(lgca.K-1)].append(cell)  #add cell to a random velocity channel
                        channel.remove(cell)
            if(channel_counter%lgca.K == 0):  #detect if channel is rest channel
                rho = len(channel)/lgca.capacity
                for cell in channel:
                    if(npr.random()<lgca.r_b*(1-rho)):
                        lgca.calc_max_label()
                        lgca.props['kappa'].append(np.random.normal(loc=lgca.props['kappa'][cell], scale = 0.2))
                        lgca.props['theta'].append(np.random.normal(loc=lgca.props['theta'][cell], scale = 0.2))
                        channel.append(lgca.maxlabel)
            channel_counter += 1
def go_or_grow3(lgca):
    for node in lgca.nodes:
        node_pop = 0
        for channel in node:
            print(channel)
            for cell in channel:
                if(np.random.random() < 0.1):#why the hell is this the death rate?
                    channel.remove(cell) 
            node_pop+=len(channel)
            print(node_pop)
        rho = node_pop/lgca.capacity
        for channel in node:
            for cell in channel:
                if(np.random.random() < tanh_switch(rho=rho, kappa=lgca.props['kappa'][cell], theta=lgca.props['theta'][cell])):
                    #switch to rest
                    node[2].append(cell)
                    channel.remove(cell)
                else:
                    #switch to moving + reorientation 
                    node[np.random.randint(2)].append(cell)
                    channel.remove(cell)
        for cell in node[2]:
            if(np.random.random()<0.2*(1-rho)):
                lgca.calc_max_label()
                lgca.props['kappa'].append(np.random.normal(loc=lgca.props['kappa'][cell], scale = 0.2))
                lgca.props['theta'].append(np.random.normal(loc=lgca.props['theta'][cell], scale = 0.2))
                #channel.append(lgca.maxlabel)
"""
def go_or_grow(lgca):
    for node in lgca.nodes:
        node_pop = 0
        for channel in node:
            for cell in channel:
                if(npr.random() <= lgca.r_d):
                    channel.remove(cell)
            node_pop += len(channel)
        rho = node_pop/lgca.capacity
        for channel in node:
            for cell in channel:
                if(npr.random() <= tanh_switch(rho = rho, kappa = lgca.props['kappa'][cell], theta = lgca.props['theta'][cell])):
                    node[2].extend([cell])
                    channel.remove(cell)
                else:
                    node[npr.randint(2)].extend([cell])
                    channel.remove(cell)
        for cell in node[2]:
            if(npr.random()<=lgca.r_b*(1-rho)):
                lgca.calc_max_label()
                node[2].append(lgca.maxlabel)
                lgca.props['kappa'].append(np.random.normal(loc=lgca.props['kappa'][cell], scale = 0.05))
                lgca.props['theta'].append(np.random.normal(loc=lgca.props['theta'][cell], scale = 0.01))