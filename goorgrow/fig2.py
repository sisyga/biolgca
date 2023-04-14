import numpy as np
import matplotlib.pyplot as plt
import string
from itertools import cycle

def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_uppercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (-0.05, 1.15)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc, ha='right', weight='bold', size=10,
                    xycoords='axes fraction',
                    **kwargs)

def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))

if __name__ == '__main__':
    plt.style.use('../frontiers_style.mplstyle')
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    plt.sca(axes[0])
    rho = np.linspace(0, 1, num=100)
    kappa = 4.
    theta = .3
    switch = tanh_switch(rho, kappa=kappa, theta=theta)
    # plot the tanh_switch function and fill the area under the curve with blue color and the one above with red
    plt.fill_between(rho, switch, color='blue', alpha=0.5)
    plt.fill_between(rho, switch, 1, color='red', alpha=.5)
    # plot the tanh_switch function
    plt.plot(rho, switch, color='black')
    # add a dottet line at theta up to the tanh_switch function
    plt.plot([theta, theta], [0, tanh_switch(theta, kappa=kappa, theta=theta)], color='black', linestyle='dotted')
    # add a dotted horizontal line at 0.5 up to the tanh_switch function
    plt.plot([0, theta], [0.5, 0.5], color='black', linestyle='dotted')
    # add a text label at the point (theta, 0.5)
    plt.text(theta+0.025, 0.5, r'$\kappa > 0$', fontsize=10)
    # place text box in upper left corner
    plt.text(0.05, 0.95, r'"Go"', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.95, 0.05, r'"Grow"', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
             horizontalalignment='right')

    # add a straight line indicating the gradient of the tanh_switch function at theta
    # calculate the gradient of the tanh_switch function at theta
    # to be reworked
    # grad = kappa * (1 - switch(theta, kappa=kappa, theta=theta) ** 2)
    # plt.plot([theta - .1, theta + 0.1], [switch, switch(theta + 0.01, kappa=kappa, theta=theta)],
    #             color='black', linestyle='dashed')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # set y ticks to 0, 0.5, 1
    plt.yticks([0, 0.5, 1])
    # set x ticks to 0, theta, 0.5, 1
    plt.xticks([0, theta, 0.5, 1], [0, r'$\theta$', .5, 1])
    plt.ylabel(r'Phenotypic switch $r_\kappa (\rho_{\mathcal{N}})$', fontsize=10)

    plt.sca(axes[1])
    kappa = -4.
    switch = tanh_switch(rho, kappa=kappa, theta=theta)
    # plot the tanh_switch function and fill the area under the curve with blue color and the one above with red
    plt.fill_between(rho, switch, color='blue', alpha=0.5)
    plt.fill_between(rho, switch, 1, color='red', alpha=.5)
    # plot the tanh_switch function
    plt.plot(rho, switch, color='black')
    # add a dottet line at theta up to the tanh_switch function
    plt.plot([theta, theta], [0, tanh_switch(theta, kappa=kappa, theta=theta)], color='black', linestyle='dotted')
    # add a dotted horizontal line at 0.5 up to the tanh_switch function
    plt.plot([0, theta], [0.5, 0.5], color='black', linestyle='dotted')
    # add a text label at the point (theta, 0.5)

    plt.text(theta+0.025, 0.5, r'$\kappa < 0$', fontsize=10)
    plt.text(0.95, 0.95, r'"Go"', fontsize=10, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')
    plt.text(0.05, 0.05, r'"Grow"', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    fig.supxlabel(r'Local density $\rho_{\mathcal{N}}$', fontsize=10)
    label_axes(fig, labels=('C', 'D'))
    plt.tight_layout()
    plt.show()