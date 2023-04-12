import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime
import scipy.stats as ss
import matplotlib.animation as animation
from time import sleep

startTime = datetime.now()

@jit
def op(ths):
    return np.sqrt((np.sum(np.cos(ths))) ** 2 +
                   (np.sum(np.sin(ths))) ** 2) / len(ths)


@jit
def update_soc(xs, ys, ths):
    sin_avg = np.zeros(N)
    cos_avg = np.zeros(N)
    Dcos = np.zeros(N)
    Dsin = np.zeros(N)
    Dc = np.zeros(N)
    count = np.zeros(N)

    cos = np.cos(ths)
    sin = np.sin(ths)
    ct = 1  # shouldn't this be 1 in the beginning for every particle?
    distances = [[np.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(N)] for i in range(N)]

    for i in range(N):
        "--------------average neighbourhood velocity-------------"  # should this really count particle i as well?
        for j in range(N):
            # if np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) < r:
            if distances[i][j] < r:
                Dcos[i] += cos[j]
                Dsin[i] += sin[j]
                Dc[i] += 1
    Dcos = Dcos / Dc
    Dsin = Dsin / Dc

    for i in range(N):
        flag = 0
        ct = 1
        "----------------minority interaction-------------------"
        for j in range(N):  # check all other particles
            # if np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) < r:  # within radius?
            if distances[i][j] < r:
                if Dcos[i] * cos[j] + Dsin[i] * sin[j] < ct:  # is neighbor maximum defector?
                    ct = Dcos[i] * cos[j] + Dsin[i] * sin[j]  # set new criterion for defector
                    if ct < gamma:  # is defector going against majority?
                        cc = Dcos[i] * cos[i] + Dsin[i] * sin[i]
                        if cc > eps:  # is poi going with flock?
                            flag = 1  # poi is affected by minority ia
                            sin_avg[i] = sin[j]  # set director to defector
                            cos_avg[i] = cos[j]
        if flag == 1:
            ths[i] = np.arctan2(sin_avg[i], cos_avg[i]) + \
                      np.random.uniform(-eta / 2, eta / 2)

        else: # flag == 0:
            "------------polar alignment-----------------"
            for j in range(N):
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
def update_vicsek(xs, ys, ths):
    
    sin_avg = np.zeros(N)
    cos_avg = np.zeros(N)
    count = np.zeros(N)
    cos = np.cos(ths)
    sin = np.sin(ths)
    distances = [[np.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(N)] for i in range(N)]

    for i in range(N):
        "------------polar alignment-----------------"
        for j in range(N):
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


rho, dt, L = 1000, 0.02, 2
N = int(rho * L ** 2)
r, eta = 0.01, 0.6
gamma, eps = -0.5, 0.5

xs = np.random.rand(N) * L
ys = np.random.rand(N) * L
ths = np.random.uniform(-np.pi, np.pi, size=N)
time = 1000
ops = np.zeros(time+1)
arr = np.empty((time+1, 3, N))
arr[0] = xs, ys, ths
ops[0] = op(ths)

x2=np.copy(xs)
y2=np.copy(ys)
th2=np.copy(ths)
op2 = np.zeros(time+1)
op2[0] = op(th2)

for t in range(time):
    # if t>0 and t%1 == 0:                
    #     fig = plt.figure()
    #     fig = plt.figure(figsize = (25,25))
    #     ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1.0)
    #     ax2 = fig.add_subplot(1,2,2, adjustable='box',aspect=2800)
    #     ax1.quiver(xs, ys, np.cos(ths),np.sin(ths))
    #     ax2.plot(np.arange(time+1),ops)
    #     ax1.set_xlim(0,L)
    #     ax1.set_ylim(0,L)
    #     ax2.set_ylim(0,1)
    #     plt.show()   
        
    # xs, ys, ths = update_soc(xs, ys, ths)
    # arr[t+1] = xs, ys, ths
    # ops[t+1] = op(ths)
    
    x2, y2, th2 = update_vicsek(x2, y2, th2)
    op2[t+1] = op(th2)



plt.plot(op2)

# x, y, theta = arr[:, 0], arr[:, 1], arr[:, 2]
# vx, vy = np.cos(theta), np.sin(theta)

# fig = plt.figure()
# i = 0
# im = plt.quiver(x[0], y[0], vx[0], vy[0])

# def updatefig(*args):
#     global i, im
#     if i < len(arr) - 1:
#         i += 1
#     else:
#         i = 0
#     im.set_UVC(vx[i], vy[i])
#     im.set_offsets(arr[i, :2].T)  # np.array([arr[i,0],arr[i,1]]).reshape(N,2))
#     return im,


# ani = animation.FuncAnimation(fig, updatefig, frames=len(arr), blit=False)
# plt.xlim(0, L)
# plt.ylim(0, L)
# plt.tight_layout()
# writervideo = animation.FFMpegWriter(fps=4)
# ani.save("asd.mp4", dpi=300)
# plt.close()
# plt.show()


# def autocorr(data):
#     mean = np.mean(data)
#     var = np.var(data)
#     ndata = ops - mean
#     acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
#     acorr = acorr / var / len(ndata)
#     return acorr


# """Observations: 'checkerboard' artifacts (particles flipping between opposing directions), can this be prevented? is it a problem? """

# np.save("data/SOC_op_L="+str(L)+"_gamma="+
#         str(gamma)+"_eps="+str(eps)+".npy",np.vstack((np.arange(time+1),ops)))

# op_var = np.var(ops)
# op_avg = np.average(ops)
# op2_avg = np.mean(op2)

# print("Variance of OP = ",op_var)
# print("Average of OP for Vicsek = ",op2_avg)

# "-------------------------block minimiaztion----------------------"

# bl = 300
# ms_soc = []
# ms_vic = []

# for i in range(0,len(ops),bl):
#     ms_soc.append(min(ops[i:i+bl]))
#     ms_vic.append(min(op2[i:i+bl]))
    
# ms_soc = np.array(ms_soc)
# ms_vic = np.array(ms_vic)
# print("(block min avg - usual avg) of SOC =",op_avg - np.average(ms_soc))
# print("(block min avg - usual avg) of Vicsek =",op2_avg - np.average(ms_vic))

# "-----------------------------------------------------------------"



# plt.plot(ops)
# plt.plot(np.ones_like(ops)*op2_avg,'--',color='r',\
#           label='order parameter average for Vicsek')
# plt.ylim(0,1)
# plt.legend()
# plt.title("SOC model - Order paramter")
# plt.savefig("SOC_OP_L="+str(L)+".png",dpi=400)
# plt.show()


# plt.plot(op2)
# plt.plot(np.ones_like(op2)*op2_avg,'--',color='r',\
#           label='order parameter average for Vicsek')
# plt.ylim(0,1)
# plt.legend()
# plt.title("Vicsek model - Order paramter")
# plt.savefig("Vicsek_OP_L="+str(L)+".png",dpi=400)
# plt.show()

# tc = 1000


# "------------------distribution of order parameter---------------------"

# plt.hist(ops[tc:])
# plt.loglog()
# plt.title("SOC")
# plt.show()

# plt.hist(op2[tc:])
# plt.loglog()
# plt.title("Vicsek")
# plt.show()



"---------------------------avalanches-------------------------------"
# i don't understand this part

# avl = []

# for i in range(tc, len(ops)-1):
#     if ops[i] < ops[i-1] and ops[i] < ops[i+1]:
#         avl.append(1-ops[i])

# avl = np.array(avl)
# print(avl)
# no_bins = 10
# freq, edges = np.histogram(avl,  \
#                     bins=np.logspace(np.log10(max(min(avl),1e-5)),\
#                     np.log10(max(avl)), no_bins),density=True)
# r_time = (edges[1:]+edges[:-1])/2
# plt.plot(r_time,freq,'o-',color='blue')
# plt.loglog()
# plt.title("SOC model - avalanches")
# # plt.savefig("SOC_avalanche.png",dpi=400)
# plt.show()

"---------------------------return time-----------------------------"

return_time = []
start = tc
finish = time

return_time_vic = []
start_vic = tc
fin_vic = time


for t in range(tc,time-1):  # is this correct??? shouldn't you set the start time in the moment when you first go below the median?
    if (ops[t]<op2_avg and ops[t-1]>=op2_avg):
        start = t
    if (ops[t]<op2_avg and ops[t+1]>=op2_avg):
        finish = t
        return_time.append(finish-start+1)


for t in range(tc,time-1):  
    if (op2[t]<op2_avg and op2[t-1]>=op2_avg):
        start_vic = t
    if (op2[t]<op2_avg and op2[t+1]>=op2_avg):
        fin_vic = t
        return_time_vic.append(fin_vic-start_vic+1)


return_time = np.array(return_time)
no_bins = 10
freq, edges = np.histogram(return_time, \
                    bins=np.logspace(np.log10(min(return_time)),\
                    np.log10(max(return_time)), no_bins), density=True)
r_time = (edges[1:]+edges[:-1])/2
freq0, edges0 = np.histogram(return_time,\
                              bins=np.logspace(np.log10(min(return_time)),\
                              np.log10(max(return_time)), no_bins))

fact = freq0[0]/freq[0]*(edges[1:]-edges[:-1])/(edges[1]-edges[0])
count = len(return_time)
err = np.sqrt(freq*(1-freq/count)/fact)
x_m = min(return_time)
alpha = len(return_time)/(np.sum(np.log(return_time/x_m)))
r_times = np.linspace(min(r_time),max(r_time),101)
aa = '%.5f'%(-alpha-1)
print('Powerlaw exponent =',-1-alpha)
plt.errorbar(r_time,freq,err,fmt='o',capsize=5,color='black')
plt.plot(r_time,freq0/fact,'o-',color='blue')
plt.plot(r_times,alpha*x_m**(alpha)/r_times**(alpha+1), \
          color='red',label=r'MLE of powerlaw for return time with $\alpha=$'+str(aa))
plt.loglog()
plt.title("SOC return time pdf; power="+str(aa))
plt.xlabel("return time")
plt.ylabel("PDF of return time")
plt.legend()
plt.savefig("SOC_returntime_L="+str(L)+".png",dpi=400)
plt.show()


# return_time_vic = np.array(return_time_vic)
# freq, edges = np.histogram(return_time_vic, \
#                     bins=np.logspace(np.log10(min(return_time_vic)),\
#                     np.log10(max(return_time_vic)), no_bins), density=True)
# r_time = (edges[1:]+edges[:-1])/2
# freq0, edges0 = np.histogram(return_time_vic,\
#                               bins=np.logspace(np.log10(min(return_time_vic)),\
#                               np.log10(max(return_time_vic)), no_bins))

# fact = freq0[0]/freq[0]*(edges[1:]-edges[:-1])/(edges[1]-edges[0])
# count = len(return_time_vic)
# err = np.sqrt(freq*(1-freq/count)/fact)
# x_m = min(return_time_vic)
# alpha = len(return_time_vic)/(np.sum(np.log(return_time_vic/x_m)))
# r_times = np.linspace(min(r_time),max(r_time),101)
# aa = '%.5f'%(-alpha-1)
# print('Powerlaw exponent =', -1-alpha)
# plt.errorbar(r_time,freq,err,fmt='o',capsize=5,color='black')
# plt.plot(r_time,freq0/fact,'o-',color='blue')
# plt.plot(r_times,alpha*x_m**(alpha)/r_times**(alpha+1), \
#           color='red',label=r'MLE of powerlaw for return time with $\alpha=$'+str(aa))
# plt.loglog()
# plt.title("Vicsek return time pdf; power="+str(aa))
# plt.xlabel("return time")
# plt.ylabel("PDF of return time")
# plt.legend()
# plt.savefig("Vic_returntime_L="+str(L)+".png",dpi=400)
# plt.show()




"-------------------------kurtosis----------------------"

# kurt = ss.kurtosis(return_time)
# print("Kurtosis of SOC =",kurt)

# kurt_vic = ss.kurtosis(return_time_vic)
# print("Kurtosis of Vicsek =",kurt_vic)



"--------------------------velocity correlations-----------------------------"

@jit
def vel_corr(xs,ys,ths,rs):
    Cs = np.zeros_like(rs)
    norm = np.zeros_like(rs)
    v_avg = op(ths)

    for i in range(N):
        for j in range(i,N):
            dist = np.sqrt(min( (xs[i]-xs[j])**2, (L - \
                                np.abs(xs[i] - xs[j]) )**2) + \
                            min( (ys[i]-ys[j])**2, \
                                (L - np.abs(ys[i] - ys[j]) )**2))
            ss = np.abs(rs-dist*np.ones_like(rs))
            idx = np.where(ss==min(ss))[0]
            Cs[idx] += np.cos(ths[i])*np.cos(ths[j])+\
                        np.sin(ths[i])*np.sin(ths[j])
            norm[idx] += 1

    for i in range(len(rs)):
        if norm[i] > 0:
            Cs[i] = Cs[i]/norm[i] - v_avg**2

    Cs = np.ma.masked_where(norm == 0, Cs)

    return Cs

@jit
def corr_avg(rs):
    corr = np.zeros_like(rs)
    count = 0
    for t in range(tc, time-1):
        Cs = vel_corr(arr[t,0],arr[t,1],arr[t,2],rs)
        corr += Cs
        count += 1

    return corr/count


del_r = r
rs = np.arange(0,L/np.sqrt(2),del_r)
corr = corr_avg(rs)

np.save("data/soc_velcorr_L="+str(L)+"_gamma="+
        str(gamma)+"_eps="+str(eps)+".npy",np.vstack((rs,corr)))

f = interpolate.UnivariateSpline(rs,corr, s=0)
yToFind = 0
yreduced = np.array(corr) - yToFind
freduced = interpolate.UnivariateSpline(rs, yreduced, s=0)
xi = freduced.roots()[0]

plt.plot(rs,corr,'.-',label="velocity fluctuation correlation")
plt.plot(rs,np.zeros_like(rs),'--',color='red')
plt.legend()
plt.xlabel("Spatial distance")
plt.title("SOC: Vel corr")
plt.savefig("SOC_velcorr_L="+str(L)+".png",dpi=400)
plt.show()

print("Correlation length=",xi)

"vel corr"
# Ls = [2, 1.8, 1.5, 1.2, 1, 0.8, 0.6, 0.4 ]
# xis = [0.38826, 0.35183, 0.26597, 0.23280 , 0.19298, 0.15671,  0.12937,0.09271]

"susceptibility"
# Ls = [2, 1.8, 1.5, 1.2, 1, 0.8, 0.6, 0.4 ]
# sus = [0.03601, 0.03880, 0.041069 , 0.03712 ,0.0328, 0.02220, 0.01689, 0.01368]

print("Execution time:",datetime.now() - startTime)