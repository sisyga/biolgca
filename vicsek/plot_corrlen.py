import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Ls = np.array([2, 1.8, 1.5, 1.2, 1, 0.8, 0.6, 0.4 ])
# xis = [0.38826, 0.35183, 0.26597, 0.23280 , 0.19298, 0.15671,  0.12937,0.09271]
# sus = [0.03601, 0.03880, 0.041069 , 0.03712 ,0.0328, 0.02220, 0.01689, 0.01368]

Ls = np.array([0.4, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2 ])
ts1, o1 = np.load("data/soc_op_L=0.4_gamma=-0.6_eps=0.6.npy")
ts2, o2 = np.load("data/soc_op_L=0.6_gamma=-0.6_eps=0.6.npy")
ts3, o3 = np.load("data/soc_op_L=0.8_gamma=-0.6_eps=0.6.npy")
ts4, o4 = np.load("data/soc_op_L=1_gamma=-0.6_eps=0.6.npy")
ts5, o5 = np.load("data/soc_op_L=1.2_gamma=-0.6_eps=0.6.npy")
ts6, o6 = np.load("data/soc_op_L=1.5_gamma=-0.6_eps=0.6.npy")
ts7, o7 = np.load("data/soc_op_L=1.8_gamma=-0.6_eps=0.6.npy")
ts8, o8 = np.load("data/soc_op_L=2_gamma=-0.6_eps=0.6.npy")

var = [np.var(o1),np.var(o2),np.var(o3),np.var(o4),np.var(o5),np.var(o6), \
       np.var(o7), np.var(o8)]

err = [np.sqrt(2*np.var(o1)**4/(len(o1))), np.sqrt(2*np.var(o2)**4/(len(o2))), \
       np.sqrt(2*np.var(o3)**4/(len(o3))), np.sqrt(2*np.var(o4)**4/(len(o4))), \
       np.sqrt(2*np.var(o5)**4/(len(o5))), np.sqrt(2*np.var(o6)**4/(len(o6))), \
       np.sqrt(2*np.var(o7)**4/(len(o7))), np.sqrt(2*np.var(o8)**4/(len(o8))), ]


rho, dt, L = 100, 0.02, 2
N = int(rho * L ** 2)
r, eta = 0.05, 1.
gamma, eps = -0.6, .6

def st_line(x,m,c):
    return m*x + c

# m1, c1 = curve_fit(st_line, Ls, xis)[0]

plt.plot(Ls,var,'o-',label='Susceptibility')
plt.errorbar(Ls,var,err,fmt='o',capsize=5,color='black')
# plt.plot(Ls, m1*Ls + c1, '--',label='St line fit',color='red')
plt.xlabel("System size (L)")
plt.ylabel("Correlation length")
plt.legend()
plt.savefig("suscep_vs_size.png",dpi=500)
plt.show()