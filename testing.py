# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np
#
# # Example data
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# # Create a GridSpec object with 2 rows and 2 columns
# gs = gridspec.GridSpec(2, 2)
#
# # Create a figure
# fig = plt.figure()
#
# # Add subplots to the grid
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 0])
# ax4 = fig.add_subplot(gs[1, 1])
#
# # Plot the data on the subplots
# ax1.plot(x, y1)
# ax2.plot(x, y2)
# ax3.plot(x, y1 * y2)
# ax4.plot(x, y1 + y2)
#
# # Adjust layout and display the figure
# fig.tight_layout()
# plt.show()
#

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
m = 1.0
s = 1.0
x0 = 0.0  # location of the delta peak
N = 100  # number of grid points
t_span = (0.0, 1.0)  # time interval to solve on

# Discretize the spatial domain
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

# Initial condition
u0 = np.zeros(N)
u0[N // 2] = 1 / dx  # a "delta function" at x0


# Discretize the PDE
def f(t, u):
    # Allocate arrays for derivatives
    du_dx = np.zeros_like(u)
    d2u_dx2 = np.zeros_like(u)

    # Compute derivatives in the interior using centered differences
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2

    # Compute derivatives at the boundaries using no-flux conditions
    # Here we use the no-flux boundary condition
    v = m * x - s / 2
    D = x * s / 2

    # For the left boundary:
    # du_dx[0] = (v[0] * u[0] + D[0] * (u[1] - u[0]) / dx) / D[0]
    # d2u_dx2[0] = (u[1] - u[0]) / dx ** 2  # one-sided second derivative

    # For the right boundary:
    du_dx[-1] = (-v[-1] * u[-1] + D[-1] * (u[-2] - u[-1]) / dx) / D[-1]
    d2u_dx2[-1] = (u[-2] - u[-1]) / dx ** 2  # one-sided second derivative

    return D * d2u_dx2 - v * du_dx  # d/dt u(x,t) = D * d^2/dx^2 u - v * du/dx


# Solve the system of ODEs
sol = solve_ivp(f, t_span, u0)

# plt.plot(x, sol.y[-1])

plt.show()
# sol.y[-1] is the solution at the final time
