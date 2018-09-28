import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d

mask = np.ones((3, 3), dtype=int)
mask[1, 1] = 0


def gol(config, bc='fill'):
    n_nb = convolve2d(config, mask, mode='same', boundary=bc)
    underpop = n_nb < 2
    overpop = n_nb > 3
    birth = n_nb == 3
    config[underpop | overpop] = 0
    config[birth] = 1
    return config


def init_func():
    global plot
    plot.set_data(config)
    return plot,


def update(t):
    plot.set_data(gol(config))
    return plot,


if __name__ == '__main__':
    lx = 200
    ly = lx
    from numpy.random import randint

    global config, plot
    config = randint(0, 2, size=(lx, ly), dtype=np.bool)
    import seaborn as sns

    sns.set_style('white')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.xlabel(r"\textbf{Conway's Game of Life}", fontsize='x-large', fontstyle='oblique')
    ax.xaxis.set_label_position('top')
    plt.xticks([], [])
    plt.yticks([], [])
    plot = ax.matshow(config, cmap='viridis', vmin=0, vmax=1, interpolation='None')
    plt.tight_layout(pad=1.)
    animation = FuncAnimation(fig, update, blit=True, init_func=init_func)
    plt.show()
