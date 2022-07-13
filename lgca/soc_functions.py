# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:07:38 2021

@author: Ashish Kangen
"""

from lgca import get_lgca
import numpy as np

def c_min(coord,nodes,nb_sum,table,c):
    current_config = nodes[coord]
    current_config = np.multiply(current_config, 1)
    current_sum = nb_sum[coord]
    [index1] = np.where(current_config==1)
    [index2] = np.where(current_sum!=0)
    if len(index2)==0:
        c_theta_max=np.array([0, 0]).reshape((2,1))
        min_dot=np.array([0])
        
    else:             
        c_phi1=table[index1]
        c_phi2=table[index1][:,index2]
        min_dot=np.min(c_phi2,axis=1)

        c_min_ind=[np.where(c_phi1[i]==min_dot[i]) for i in range(len(min_dot))]
        c_min_in=[c[:,c_min_ind[i][0][:]] for i in range(len(c_min_ind))]
        c_min1=[[i,c_min_in[i]] for i in range(len(c_min_in)) if c_min_in[i].shape[1]==1]
        c_min2=[[i,c_min_in[i]] for i in range(len(c_min_in)) if c_min_in[i].shape[1]>1]
        c_min2=[[c_min2[i][0],c_min2[i][1][:,np.random.choice(c_min2[i][1].shape[1])].reshape(2,1)] for i in range(len(c_min2))]
        c_min_fin=c_min1+c_min2
        c_min_fin.sort()

        c_min_fin=[i[1] for i in c_min_fin]
        c_theta_max=np.asanyarray(c_min_fin)[...,0].T
    return c_theta_max,min_dot
        
def table():
    table=np.array([[1,	0.5,	-0.5,	-1,	-0.5,	0.5],
    [0.5,	1,	0.5,	-0.5,	-1,	-0.5],
    [-0.5,	0.5,	1,	0.5,	-0.5,	-1],
    [-1,	-0.5,	0.5,	1,	0.5,	-0.5],
    [-0.5,	-1,	-0.5,	0.5,	1,	0.5],
    [0.5,	-0.5,	-1,	-0.5,	0.5,	1]])
    return table

# def table_square():
    
    
def weightedBinary(bias):
    x=np.random.uniform(0,1)
    if x<=bias:
        x=0
    else:
        x=1
    return x   
    
def weighted_flux(threshold,ng,coord):
    if (np.linalg.norm(ng[coord])/6)>=threshold:
        x=1
    else:
        x=0
    return x    

def SOC_activation(threshold,g,ng,coord):
    if 0.92>(np.dot(g[coord],(ng[coord])/6))>=threshold:
        x=1
    else:
        x=0
    return x
      
def H_1(min_dot,gamma):
    H=np.where(np.abs(min_dot)<=gamma,0,1)
    return H
            
def H_2(min_dot,gamma):
    H=np.where(gamma-min_dot<=0,0,1)
    return H

def average_flux(g): 
    g_avg=np.average(g,axis=0)
    g_avg=np.average(g_avg,axis=0)
    return g_avg

def cell_count(nodes):
    node_cell_sum=np.sum(nodes,axis=(0,1))

    return node_cell_sum

def cell_count_total(nodes):
    node_cell_sum=np.sum(nodes,axis=(0,1,2))

    return node_cell_sum

def cell_count_t(nodes):
    node_cell_sum=np.sum(nodes,axis=0)
    return node_cell_sum

def orderParameternorm(g,L):
    g_sum=np.sum(g,axis=(0,1))
    # g_sum=np.sum(g_sum,axis=0)
    o=np.linalg.norm(g_sum)/(2*L**2)
    return o

def orderParameterXY(g,L):
    g_sum=np.sum(g,axis=(0,1))
    # g_sum=np.sum(g_sum,axis=0)
    o=g_sum/(2*L**2)
    return o

def orderParameter(g):
    g_avg=np.average(g,axis=0)
    g_avg=np.average(g_avg,axis=0)
    o=np.linalg.norm(g_avg)
    return o

def orderParameter2(g,ng):
    o=np.zeros(np.shape(g[...,0]))
    for i in range(np.shape(g)[0]):
        for j in range(np.shape(g)[1]):
            o[i][j]=np.dot(g[i][j],ng[i][j]/6)
            
    o_avg=np.average(o,axis=0)
    o_avg=np.average(o_avg,axis=0)
    return o_avg

def orderParameter2norm(g,ng):
    o=np.zeros(np.shape(g[...,0]))
    for i in range(np.shape(g)[0]):
        for j in range(np.shape(g)[1]):
            if np.linalg.norm(g[i][j])!=0 and np.linalg.norm(ng[i][j])!=0:
              o[i][j]=np.dot(g[i][j]/np.linalg.norm(g[i][j]),ng[i][j]/np.linalg.norm(ng[i][j]))
            
    o_avg=np.average(o,axis=(0,1))
    # o_avg=np.average(o_avg,axis=0)
    return o_avg
