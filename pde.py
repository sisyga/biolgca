import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import laplace, convolve
from matplotlib.animation import FuncAnimation as animation

def gradient_pbc(y, dx=1):
    grad = convolve(y, [1., 0., -1.], mode='wrap') * 0.5 / dx
    return grad

def lapl_2nd(y, dx=1., mode='wrap'):
    stencil = (1.0 / (12.0 * dx * dx)) * np.array([-1, 16, -30, 16, -1], dtype=float)
    lapl = convolve(y, stencil, mode=mode)
    return lapl


def e0(rho0, rho1, a, b):
    br = (rho0 * (rho1 - a) / (rho1 * (rho0 - a)))**b
    e0 = (rho1 - rho0 * br) / (1 + br)
    return e0


def dydt(t, y, dx, b=1., a=0.01, r=0., d=0., D=20):
    r0, r1 = np.hsplit(y, 2)

    e = e0(r0, r1, a, b)

    c = r0 + r1

    dr0 = e
    dr0 += r * r0 * (1 - c)
    dr0 += lapl_2nd(r0, dx=dx)
    # dr0 -= d * r0


    dr1 = -e
    # dr1 += r * r0 * (1 - c)
    dr1 += lapl_2nd(r1, dx=dx) * D
    # dr1 -= d * r1
    dy = np.hstack((dr0, dr1))
    return dy


def dydt_ve(t, y, dx, b=1., a=0.05, r=0., d=0., D=20.):
    r0, r1 = np.hsplit(y, 2)

    e = e0(r0, r1, a, b)

    c = r0 + r1
    # laplc = laplace(c, mode='wrap') / dx ** 2
    # laplr0 = laplace(r0, mode='wrap') / dx ** 2
    # laplr1 = laplace(r1, mode='wrap') / dx ** 2
    laplc = lapl_2nd(c, dx=dx)
    laplr0 = lapl_2nd(r0, dx=dx)
    laplr1 = lapl_2nd(r1, dx=dx)


    dr0 = e
    growth = r * r0 * (1 - c)
    # print(e, growth, (1 - c) * laplr0 + r0 * laplc)
    dr0 += growth
    dr0 += (1 - c) * laplr0 + r0 * laplc
    # dr0 -= d * r0

    dr1 = -e
    # dr1 += r * r0 * (1 - 2 * r1) / 2
    dr1 += D * ((1 - c) * laplr1 + r1 * laplc)
    # dr1 -= d * r1
    dy = np.hstack((dr0, dr1))
    return dy

def dydt2(t, y, dx, b=1., a=0.01, r=0., d=0., D=20):
    r0, r1 = np.hsplit(y, 2)

    p_to_rest = p_mov_to_rest(r0, r1, a, b)
    p_to_move = p_rest_to_move(r0, r1, a, b)

    # dr0 = laplace(r0, mode='wrap') / dx**2
    dr0 = lapl_2nd(r0, dx=dx)
    dr0 += r * r0 * (1 - r0 - r1)
    dr0 += p_to_rest * r1 * (1 - r0)
    dr0 -= p_to_move * r0 * (1 - r1)
    dr0 -= d * r0 * (r0 + r1) / 2

    # dr1 = laplace(r1, mode='wrap') / dx**2 * D
    dr1 = lapl_2nd(r1, dx=dx) * D
    dr1 += r * r0 * (1 - r1 - r0)
    dr1 += p_to_move * r0 * (1 - r1)
    dr1 -= p_to_rest * r1 * (1 - r0)
    dr1 -= d * r1 * (r0 + r1) / 2
    dy = np.hstack((dr0, dr1))
    return dy

def dydt_taylor(t, y, dx, b=1., a=0.01, r=0., d=0., D=20):
    r0, r1 = np.hsplit(y, 2)
    p_to_rest = p_change_approx(r0, r1, b)

    dr0 = laplace(r0, mode='wrap') / dx**2
    dr0 += r * r0 * (1 - r0)
    dr0 += p_to_rest * r1 * (1 - r0)
    dr0 -= (1 - p_to_rest) * r0 * (1 - r1)
    dr0 -= d * r0

    dr1 = laplace(r1, mode="wrap") / dx**2 * D
    dr1 += r * r0 * (1 - r1)
    dr1 += (1 - p_to_rest) * r0 * (1 - r1)
    dr1 -= p_to_rest * r1 * (1 - r0)
    dr1 -= d * r1
    dy = np.hstack((dr0, dr1))
    return dy


def p_mov_to_rest(r0, r1, a, b):
    p = (r1 * r0)**b / ((r1 * r0)**b + ((r0 + a) * (r1 - a))**b)
    return p


def p_rest_to_move(r0, r1, a, b):
    p = (r1 * r0)**b / ((r1 * r0)**b + ((r1 + a) * (r0 - a))**b)
    return p


def p_change_approx(r0, r1, b):
    return 0.5 + b * (r0 - r1) / r0 / r1


def update(n):
    title.set_text('Time t = {:.1f}'.format(t[n]))
    for i, line in enumerate(lines):
        line.set_ydata(data[n, i])

    return lines + [title]


b = 40
a = 0.05
# a = 1. / 12
r = .05
d = .0
D = 20.
r00 = 0.3
r10 = 0.3
xmax = 1000.
tmax = 1000.
points = 200
x, dx = np.linspace(0, xmax, points, endpoint=False, retstep=True)
# initial config
r00 = np.ones_like(x) * r00
# r00 += np.sin(r00.shape) * 0.1
# r00 = 0.25 * (1 + np.sin(x / 10))
r10 = np.ones_like(x) * r10
# r10 = 0.25 * (1 + np.cos(x / 10))
y0 = np.hstack((r00, r10))
y0 *= (np.random.standard_normal(y0.shape) * 0.1 + 1)  # add noise

sol = solve_ivp(fun=lambda t, y: dydt2(t, y, dx, b=b, a=a, r=r, d=d, D=D), t_span=(0, tmax), y0=y0,
                t_eval=np.linspace(0, tmax, 101), method='Radau')
print(sol.status)
fig = plt.figure()

t = sol.t
r0, r1 = np.split(sol.y, 2)
r0 = r0.T
r1 = r1.T
lines = []
line0 = plt.plot(x, r0[0], label='$\\rho_0$')[0]
line1 = plt.plot(x, r1[0], label='$\\rho_1$')[0]
data = np.array([r0, r1])
data = np.moveaxis(data, 0, 1)
lines.append(line0)
lines.append(line1)
title = plt.title('Time t = 0')
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, xmax)
ani = animation(fig, update, blit=False, frames=len(t), repeat=False, interval=50)
# ani.save('pde_pattern.mp4')
plt.xlabel('$x$')
print(data[0].sum(), data[-1].sum())
plt.show()

# plt.plot(x, r0[-1], label='$\\rho_0$')
# plt.plot(x, e0(r0[-1], r1[1], a, b), label='$\\E_0$')
# plt.show()

# plt.figure()
# plt.imshow(r0)
# plt.colorbar()
# plt.show()