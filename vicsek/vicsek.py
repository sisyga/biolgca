import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime
import matplotlib.animation as animation
from scipy import interpolate

startTime = datetime.now()

@jit
def op(ths):
    return np.sqrt((np.sum(np.cos(ths)))**2 + 
                   (np.sum(np.sin(ths)))**2)/len(ths)

@jit
def update(xs,ys,ths):
    sin_avg = np.zeros(N)
    cos_avg = np.zeros(N)
    count = np.zeros(N)
    cos = np.cos(ths)
    sin = np.sin(ths)
    xs += cos*dt
    ys += sin*dt
    xs = xs%L
    ys = ys%L
    for i in range(N):
        for j in range(N):
            if np.sqrt((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)<r:
                sin_avg[i] += sin[j]
                cos_avg[i] += cos[j]
                count[i] += 1
    cos = cos_avg/count
    sin = sin_avg/count
    ths = np.arctan2(sin,cos) + np.random.uniform(-eta/2,eta/2,N)

    return xs, ys, ths

N, dt, L = 40, 0.03, 0.4
r, eta = 0.05, 0.5

xs = np.random.rand(N)*L
ys = np.random.rand(N)*L
ths = np.random.uniform(-np.pi,np.pi,size=N)
time = 30000
ops = np.zeros(time)
arr=[]


for t in range(time):
    if t%1 == 0:
        arr.append([xs,ys,ths])
        # plt.quiver(xs, ys, np.cos(ths),np.sin(ths))
        # plt.xlim(0,L)
        # plt.ylim(0,L)
        # plt.show()
    ops[t] = op(ths)
    xs, ys, ths = update(xs,ys,ths)


arr = np.array(arr)


# fig = plt.figure()
# i=0
# im = plt.quiver(arr[0,0],arr[0,1],arr[0,2],arr[0,3])
# def updatefig(*args):
#     global i
#     if (i<len(arr)-1):
#         i += 1
#     else:
#         i=0
#     im.set_UVC(arr[i,2],arr[i,3])
#     im.set_offsets(np.array([arr[i,0],arr[i,1]]).reshape(N,2))
#     return im,

# ani = animation.FuncAnimation(fig, updatefig, frames=len(arr), blit=False)
# writervideo = animation.FFMpegWriter(fps=4)
# ani.save("asd.mp4", writer=writervideo,dpi=300)
# plt.close()

# Points must be Nx2 array, got 2x10
    
# np.save("data/vicsek_op_dens="+str(N/L**2)+"_eta="+
#         str(eta)+"_r="+str(r)+".npy",np.vstack((np.arange(time),op)))  


op_var = np.var(ops)
op_med = np.median(ops)
plt.plot(np.arange(time),ops)
plt.plot(np.arange(time),np.ones_like(ops)*op_med,'--',color='r')
plt.ylim(0,1)
plt.title("Vicsek")
# plt.savefig("Vicsek_OP.png",dpi=400)
plt.show()

print("Variance of OP = ",op_var)

tc = 400

"---------------------------avalanches-------------------------------"

avl = []
start = tc
finish = time

for i in range(tc,time-1):
    if ops[i]<ops[i-1] and ops[i]<ops[i+1]:
        avl.append(1-ops[i])

avl=np.array(avl)
no_bins=10
freq, edges = np.histogram(avl,  \
                    bins=np.logspace(np.log10(min(avl)),\
                    np.log10(max(avl)), no_bins), density=True)
r_time = (edges[1:]+edges[:-1])/2
plt.plot(r_time,freq,'o-',color='blue')
plt.loglog()
plt.title("Vicsek")
# plt.savefig("Vicsek_avalanche.png",dpi=400)
plt.show()

"---------------------------return time-----------------------------"

return_time = []
start = tc
finish = time

for t in range(tc,time-1):
    if (ops[t]<op_med and ops[t+1]>=op_med):
        finish = t
        return_time.append(finish-start+1)
        start = finish

return_time = np.array(return_time)
no_bins = 10
freq, edges = np.histogram(return_time, \
                    bins=np.logspace(np.log10(min(return_time)),\
                    np.log10(max(return_time)), no_bins), density=True) 
r_time = (edges[1:]+edges[:-1])/2

def power_law(x,a,b):
    return a*x**b

freq0, edges0 = np.histogram(return_time,\
                              bins=np.logspace(np.log10(max(min(return_time),1e-5)),\
                              np.log10(max(return_time)), no_bins))

fact = freq0[0]/freq[0]*(edges[1:]-edges[:-1])/(edges[1]-edges[0])
count = len(return_time)
err = np.sqrt(freq*(1-freq/count)/fact)
x_m = min(return_time)
alpha = len(return_time)/(np.sum(np.log(return_time/x_m)))
r_times = np.linspace(min(r_time),max(r_time),101)
aa = '%.5f'%(-alpha-1)

plt.errorbar(r_time,freq,err,fmt='o',capsize=5,color='black')
plt.plot(r_time,freq0/fact,'o-',color='blue')
plt.plot(r_times,alpha*x_m**(alpha)/r_times**(alpha+1), \
          color='red',label='MLE of powerlaw for return time')
plt.loglog()
plt.title("power="+str(aa))
plt.xlabel("return time")
plt.ylabel("PDF of return time")
plt.legend()
plt.title("Vicsek")
# plt.savefig("Vicsek_returntime.png",dpi=400)
plt.show()


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
    
    return Cs    

@jit
def corr_avg(rs):
    corr = np.zeros_like(rs)
    count = 0
    for t in range(tc,time-1):
        Cs = vel_corr(arr[t,0],arr[t,1],arr[t,2],rs)
        corr += Cs
        count += 1
        
    return corr/count


del_r = 0.01
rs = np.arange(0,L/np.sqrt(2),del_r)    
corr = corr_avg(rs) 

# f = interpolate.UnivariateSpline(rs,corr, s=0)
# yToFind = 0
# yreduced = np.array(corr) - yToFind
# freduced = interpolate.UnivariateSpline(rs, yreduced, s=0)
# xi = freduced.roots()[0]

plt.plot(rs,corr,'.-',label="velocity fluctuation correlation")
plt.plot(rs,np.zeros_like(rs),'--',color='red')
plt.legend()
plt.xlabel("Spatial distance")
plt.title("Vicsek: Vel corr")
plt.savefig("Vicsek_velcorr.png",dpi=400)
plt.show()

# print("Correlation length=",xi)


print("Execution time:",datetime.now() - startTime)


# Ls = [1,2,3,4,5]
# sus = [0.03351,0.03351,]
# xis = [0.137435,]
