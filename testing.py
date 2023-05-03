import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a GridSpec object with 2 rows and 2 columns
gs = gridspec.GridSpec(2, 2)

# Create a figure
fig = plt.figure()

# Add subplots to the grid
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot the data on the subplots
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y1 * y2)
ax4.plot(x, y1 + y2)

# Adjust layout and display the figure
fig.tight_layout()
plt.show()

