import numpy as np
import matplotlib.pyplot as plt

# Set simulation parameters
N = 100 # Number of particles
L = 10.0 # Size of the box
v0 = 0.1 # Magnitude of the particle velocity
dt = 0.1 # Time step
eta = 1.0 # Noise strength
rc = 1.0 # Interaction radius
alpha = 1 # Alignment strength

# Initialize particle positions and velocities
pos = L * np.random.rand(N, 2)
vel = v0 * np.random.rand(N, 2)

# Define function to compute distance between particles with periodic boundary conditions
def distance(x1, y1, x2, y2, L):
    dx = x2 - x1
    dy = y2 - y1
    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)
    return np.sqrt(dx**2 + dy**2)

# Define function to compute the average velocity of neighboring particles
def get_neighborhood_avg_vel(pos, vel, rc, alpha):
    N = pos.shape[0]
    avg_vel = np.zeros((N, 2))
    for i in range(N):
        d = distance(pos[i, 0], pos[i, 1], pos[:, 0], pos[:, 1], L)
        idx = np.where((d > 0) & (d < rc))[0]
        if len(idx) > 0:
            avg_vel[i, :] = np.mean(vel[idx, :], axis=0)
    return v0 * (avg_vel / np.linalg.norm(avg_vel, axis=1)[:, np.newaxis])**alpha

# Run the simulation
fig, ax = plt.subplots(figsize=(6, 6))
for t in range(100):
    # Update particle velocities
    vel += dt * get_neighborhood_avg_vel(pos, vel, rc, alpha)
    # Add noise to particle velocities
    vel += eta * np.random.randn(N, 2)
    # Normalize particle velocities
    vel = v0 * (vel / np.linalg.norm(vel, axis=1)[:, np.newaxis])
    # Update particle positions
    pos += dt * vel
    # Apply periodic boundary conditions
    pos = pos - L * np.floor(pos / L)
    # Plot particle positions
    ax.clear()
    ax.scatter(pos[:, 0], pos[:, 1], s=10)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    plt.pause(0.01)