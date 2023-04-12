import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

startTime = datetime.now()

@jit
def op(ths):
    return np.sqrt((np.sum(np.cos(ths))) ** 2 +
                   (np.sum(np.sin(ths))) ** 2) / len(ths)


@jit
def update_soc(xs, ys, ths, eps, gamma):
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

rho, dt, L = 100, 0.02, 1.0
N = int(rho * L ** 2)
r, eta = 0.1, 1
time = 20000

def soc_measure(r,eta,gamma,eps):    
    xs = np.random.rand(N) * L
    ys = np.random.rand(N) * L
    ths = np.random.uniform(-np.pi, np.pi, size=N)
    ops = np.zeros(time+1)
    ops[0] = op(ths)
    for t in range(time):
        xs, ys, ths = update_soc(xs, ys, ths, eps, gamma)
        ops[t+1] = op(ths)
    bl = 300
    ms_soc = []
    for i in range(0,len(ops),bl):
        ms_soc.append(min(ops[i:i+bl]))
    ms_soc = np.array(ms_soc)
    op_avg = np.average(ops)
    print(eps,gamma)
    print("(block min avg - usual avg) of SOC =",op_avg - np.average(ms_soc))
    return op_avg - np.average(ms_soc)

gs = np.linspace(-1,0,15)
es = np.linspace(0,1,15)
meas = np.zeros((len(es),len(gs)))
i,j=0,0
for eps in es:
    for gamma in gs:    
        meas[i,j] = soc_measure(r,eta,gamma,eps)
        j+=1
    i+=1
    j=0
    
plt.imshow(meas-0.11,extent=[-1,0,0,1]) #vicsek = 0.11
plt.colorbar()
plt.ylabel("Epsilon")
plt.xlabel("Gamma")
plt.savefig("param_scan.png",dpi=400)
plt.show() 

print("Execution time:",datetime.now() - startTime)