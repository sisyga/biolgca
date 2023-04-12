import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

startTime = datetime.now()


rho, dt, L = 1000, 0.02, 2  
N = int(rho * L ** 2)
r, eta = 0.01, 0.6
gamma, eps = -0.5, 0.5

xs = np.random.rand(N) * L
ys = np.random.rand(N) * L
ths = np.random.uniform(-np.pi, np.pi, size=N)

Lc = int(L/r)
sqL = Lc*Lc


@jit
def op(ths):
    return np.sqrt((np.sum(np.cos(ths))) ** 2 +
                   (np.sum(np.sin(ths))) ** 2) / len(ths)


# @jit
# def update_soc(xs, ys, ths, bg):
#     sin_avg = np.zeros(N)
#     cos_avg = np.zeros(N)
#     Dcos = np.zeros(N)
#     Dsin = np.zeros(N)
#     Dc = np.zeros(N)
#     count = np.zeros(N)

#     cos = np.cos(ths)
#     sin = np.sin(ths)
#     ct = 1  # shouldn't this be 1 in the beginning for every particle?
#     distances = [[np.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(N)] for i in range(N)]

#     for i in range(N):
#         "--------------average neighbourhood velocity-------------"  # should this really count particle i as well?
#         for j in range(N):
#             # if np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) < r:
#             if distances[i][j] < r:
#                 Dcos[i] += cos[j]
#                 Dsin[i] += sin[j]
#                 Dc[i] += 1
#     Dcos = Dcos / Dc
#     Dsin = Dsin / Dc

#     for i in range(N):
#         #array of all particles within range
#         js = np

        
        
#         flag = 0
#         ct = 1
#         "----------------minority interaction-------------------"
#         for j in js:  # check all other particles
#             # if np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) < r:  # within radius?
#             if distances[i][j] < r:
#                 if Dcos[i] * cos[j] + Dsin[i] * sin[j] < ct:  # is neighbor maximum defector?
#                     ct = Dcos[i] * cos[j] + Dsin[i] * sin[j]  # set new criterion for defector
#                     if ct < gamma:  # is defector going against majority?
#                         cc = Dcos[i] * cos[i] + Dsin[i] * sin[i]
#                         if cc > eps:  # is poi going with flock?
#                             flag = 1  # poi is affected by minority ia
#                             sin_avg[i] = sin[j]  # set director to defector
#                             cos_avg[i] = cos[j]
#         if flag == 1:
#             ths[i] = np.arctan2(sin_avg[i], cos_avg[i]) + \
#                       np.random.uniform(-eta / 2, eta / 2)

#         else: # flag == 0:
#             "------------polar alignment-----------------"
#             for j in js:
#                 if distances[i][j] < r:
#                 # if np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) < r:
#                     sin_avg[i] += sin[j]
#                     cos_avg[i] += cos[j]
#                     count[i] += 1

#             cos_avg[i] = cos_avg[i] / count[i]
#             sin_avg[i] = sin_avg[i] / count[i]
#             ths[i] = np.arctan2(sin_avg[i], cos_avg[i]) + \
#                      np.random.uniform(-eta / 2, eta / 2)

#     "----------------position update-------------------"
#     xs += cos * dt
#     ys += sin * dt
#     xs = xs % L
#     ys = ys % L

#     return xs, ys, ths

@jit
def update_vicsek(xs, ys, ths):
    bg = bg_update(xs,ys)
    sin_avg = np.zeros(N)
    cos_avg = np.zeros(N)
    count = np.zeros(N)
    cos = np.cos(ths)
    sin = np.sin(ths)
    distances = [[np.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(N)] for i in range(N)]

    for i in range(N):
        "------------polar alignment-----------------"
        k = int(xs[i]*Lc/L) + Lc * int(ys[i]*Lc/L)
        # print(k)
        js =  np.where((bg == k) | (bg == nbr[k,0] ) | (bg == nbr[k,1] ) |
                    (bg == nbr[k,2] ) | (bg == nbr[k,3]) | (bg == nbr[k,4]) |
                    (bg == nbr[k,5])| (bg == nbr[k,6])| (bg == nbr[k,7]))[0]
        for j in js:
        # for j in range(N):
            if distances[i][j] < r:
            # if np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) < r:
                sin_avg[i] += sin[j]
                cos_avg[i] += cos[j]
                count[i] += 1

        cos_avg[i] = cos_avg[i] / count[i]
        sin_avg[i] = sin_avg[i] / count[i]
        ths[i] = np.arctan2(sin_avg[i], cos_avg[i]) + \
                 np.random.uniform(-eta / 2, eta / 2)

    "----------------position update-------------------"
    xs += cos * dt
    ys += sin * dt
    xs = xs % L
    ys = ys % L

    return xs, ys, ths


@jit
def nbr2d(k,L):    
    nbr=np.zeros((L*L,8)) 
    for i in range(L):
            for j in range(L):
                k =  j*L +i
                
                nbr[k,0]=   j*L + ((i+1)%L)    
                nbr[k,1]=  i + L*((j+1)%L)       
                nbr[k,2]= ((i-1+L)%L) +j*L    
                nbr[k,3]= ((j-1+L)%L)*L+i 
                
                nbr[k,4]=   ((j+1)%L)*L + ((i+1)%L)    
                nbr[k,5]=   (i+1)%L + L*((j-1)%L)       
                nbr[k,6]=   ((i-1+L)%L) +((j+1)%L)*L    
                nbr[k,7]=   ((j-1+L)%L)*L+(i-1+L)%L 
    return nbr 


@jit
def bg_update(xs,ys):    
    bg = np.zeros(sqL)
    for i in range(N):
        bg[i] = int(xs[i]*Lc/L)  + Lc*int(ys[i]*Lc/L)
    return bg



for k in range(sqL):
    nbr=nbr2d(k,Lc)
nbr = nbr.astype('int16')


time = 1000

ops = np.zeros(time+1)
arr = np.empty((time+1, 3, N))
ops[0] = op(ths)

x2=np.copy(xs)
y2=np.copy(ys)
th2=np.copy(ths)
op2 = np.zeros(time+1)
op2[0] = op(th2)

for t in range(time):
    # if t>0 and t%100 == 0:                
    #     fig = plt.figure()
    #     fig = plt.figure(figsize = (25,25))
    #     ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1.0)
    #     # ax2 = fig.add_subplot(1,2,2, adjustable='box',aspect=2800)
    #     ax1.quiver(x2, y2, np.cos(th2),np.sin(th2))
    #     # ax2.plot(np.arange(time+1),ops)
    #     ax1.set_xlim(0,L)
    #     ax1.set_ylim(0,L)
    #     # ax2.set_ylim(0,1)
    #     plt.show()   
    
    x2, y2, th2 = update_vicsek(x2, y2, th2)
    op2[t+1] = op(th2)



plt.plot(op2)



print("Execution time:",datetime.now() - startTime)