from random import choices, random, shuffle
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
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        cells = node.sum()

        channeldist = npr.multinomial(len(cells), [1./lgca.K] * lgca.K).cumsum()
        shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)
    
def birth(lgca):
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.capacity
        cells = node.sum()

        for cell in cells:
            r_b = lgca.props['r_b'][cell]
            if random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                cells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        channeldist = npr.multinomial(len(cells), [1./lgca.K] * lgca.K).cumsum()
        shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)
    
def birthdeath(lgca):
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.capacity
        cells = node.sum()

        for cell in cells:
            if random() < lgca.r_d:
                cells.remove(cell)

            r_b = lgca.props['r_b'][cell]
            if random() < r_b * (1 - rho):
                lgca.maxlabel += 1
                cells.append(lgca.maxlabel)
                lgca.props['r_b'].append(float(trunc_gauss(0, lgca.a_max, r_b, sigma=lgca.std)))

        channeldist = npr.multinomial(len(cells), [1./lgca.K] * lgca.K).cumsum()
        shuffle(cells)
        newnode = [cells[:channeldist[0]]] + [cells[i:j] for i, j in zip(channeldist[:-1], channeldist[1:])]

        lgca.nodes[coord] = deepcopy(newnode)

# continue here!
def go_or_grow(lgca):
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = deepcopy(lgca.nodes[coord])
        density = lgca.cell_density[coord]
        rho = density / lgca.capacity
        cells = node.sum()
        for cell in cells:
            if random() < lgca.r_d:
                channel.remove(cell)

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
            ch_counter += 1
            
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
def go_or_grow4(lgca):
    #Death -> Birth -> Switch+Reorientation
    #the difference compared to iblgca interaction is that the birth operator is applied before phenotypic switch for performance reasons
    relevant = (lgca.density[lgca.nonborder] > 0)
    coords = [a[relevant] for a in lgca.nonborder]
    for coord in zip(*coords):
        node = lgca.nodes[coord]
        density = 0
        for channel in node:
            for cell in channel:
                if(npr.random() <= lgca.r_d):
                    channel.remove(cell)
            density += len(channel)
        rho = density/lgca.capacity
        
        for cell in node[-1]:
            if(npr.random()<=lgca.r_b*(1-rho)):
                lgca.maxlabel = len(lgca.props["kappa"])#for some unknown reason I have to do it every time. I can't just lgca.maxlabel+=1 (i get an error) 
                node[-1].append(lgca.maxlabel)
                lgca.props['kappa'].append(npr.normal(loc=lgca.props['kappa'][cell], scale=lgca.kappa_std))
                #lgca.props['kappa'].append(list(np.random.normal(loc=lgca.props['kappa'][cell], scale = lgca.kappa_std)))
                lgca.props['theta'].append(trunc_gauss(0,1,lgca.props['theta'][cell], lgca.theta_std))
                density += 1
                
        rho = density/lgca.capacity

        for channel in node:
            for cell in channel:
                if(npr.random() <= tanh_switch(rho = rho, kappa = lgca.props['kappa'][cell], theta = lgca.props['theta'][cell])):
                    node[-1].append(cell)
                    channel.remove(cell)
                else:
                    node[np.random.randint(lgca.K-1)].append(cell)
                    channel.remove(cell)
"""
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