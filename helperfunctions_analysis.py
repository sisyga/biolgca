import numpy as np
import matplotlib.pyplot as plt
from lgca import get_lgca
from scipy.integrate import odeint, solve_ivp
from scipy.ndimage.filters import laplace
from scipy.special import erf
from math import sqrt


def gaussian(x):
    y = np.exp(-0.5 * x**2) / sqrt(2 * np.pi)
    return y

def cdf_gaussian(x):
    y = 0.5 * (1 + erf(x / sqrt(2)))
    return y

def trunc_gaussian(x, mu, sigma, a=0, b=1):
    xi = (x - mu) / sigma
    beta = (b - mu) / sigma
    alpha = (a - mu) / sigma
    y = gaussian(xi) / sigma
    y /= cdf_gaussian(beta) - cdf_gaussian(alpha)
    return y
    

def dydt_int(t, y, alpha, r_d, var, a_min, a_max):
    dalpha = alpha[1]-alpha[0]
#     rho = y.sum() * dalpha
    rho = np.trapz(y, dx=dalpha)
    dy = np.empty_like(y)
    for i, a in enumerate(alpha):
        dy[i] = np.trapz(alpha * y * trunc_gaussian(a - alpha, 0., sqrt(var), a=a_min-alpha, b=a_max-alpha), dx=dalpha)
    
#     dy = alpha * y + var * np.gradient(y, dalpha) + 0.5 * alpha * var * laplace(y) / dalpha**2
    dy *= 1 - rho
    dy -= r_d * y
    return dy


def dydt(t, y, alpha, r_d, var):
    dalpha = alpha[1]-alpha[0]
    rho = np.trapz(y, dx=dalpha)
    dy = alpha * y + var * np.gradient(y, dalpha) + 0.5 * alpha * var * laplace(y) / dalpha**2
    dy[0] = alpha[0] * y[0] + alpha[0] * var * (y[1] - y[0]) / dalpha**2
    dy[-1] = alpha[-1] * y[-1] + alpha[-1] * var * (y[-2] - y[-1]) / dalpha**2
    dy *= 1 - rho
    dy -= r_d * y
    return dy