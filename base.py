import sys

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from sympy.utilities.iterables import multiset_permutations

pi2 = 2 * np.pi


def update_progress(progress):
    """
    Simple progress bar update.
    :param progress: float. Fraction of the work done, to update bar.
    :return:
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rProgress: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 1),
                                               status)
    sys.stdout.write(text)
    sys.stdout.flush()


def colorbar_index(ncolors, cmap, use_gridspec=False):
    """Return a discrete colormap with n colors from the continuous colormap cmap and add correct tick labels

    :param ncolors: number of colors of the colormap
    :param cmap: continuous colormap to create discrete colormap from
    :param use_gridspec: optional, use option for colorbar
    :return: colormap instance
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    colorbar = plt.colorbar(mappable, use_gridspec=use_gridspec)
    colorbar.set_ticks(np.linspace(-0.5, ncolors + 0.5, 2 * ncolors + 1)[1::2])
    colorbar.set_ticklabels(list(range(ncolors)))
    return colorbar


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

        Example
            x = resize(arange(100), (5,100))
            djet = cmap_discretize(cm.jet, 5)
           imshow(x, cmap=djet)
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]

    return plt.matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


def estimate_figsize(array, x=8., cbar=False, dy=1.):
    lx, ly = array.shape
    if cbar:
        y = min([x * ly / lx - 1, 15.])
    else:
        y = min([x * ly / lx, 15.])  #
    y *= dy
    figsize = (x, y)
    return figsize


def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


def blank_fct():
    pass


def calc_nematic_tensor(v):
    return np.einsum('...i,...j->...ij', v, v) - 0.5 * np.diag(np.ones(2))[None, ...]


class LGCA_base():
    """
    Base class for a lattice-gas. Not meant to be used alone!
    """
    interactions = [
        'This is only a helper class, it cannot simulate! Use one the following classes: \n LGCA_1D, LGCA_SQUARE, LGCA_HEX']

    def __init__(self):
        return

    def get_interactions(self):
        print(self.interactions)

    def random_reset(self, density):
        """

        :param density:
        :return:
        """
        self.nodes = npr.random(self.nodes.shape) < density
        self.update_dynamic_fields()

    def update_dynamic_fields(self):
        """Update "fields" that store important variables to compute other dynamic steps

        :return:
        """
        self.cell_density = self.nodes.sum(-1)

    def timestep(self):
        """
        Update the state of the LGCA from time k to k+1.
        :return:
        """
        self.interaction(self)
        self.apply_boundaries()
        self.propagation()
        self.apply_boundaries()
        self.update_dynamic_fields()

    def calc_permutations(self):
        self.permutations = [np.array(list(multiset_permutations([1] * n + [0] * (self.K - n))), dtype=np.int8)
                             for n in range(self.K + 1)]
        self.j = [np.dot(self.c, self.permutations[n][:, :self.velocitychannels].T) for n in range(self.K + 1)]
        self.cij = np.einsum('ij,kj->jik', self.c, self.c) - 0.5 * np.diag(np.ones(2))[None, ...]
        self.si = [np.einsum('ij,jkl', self.permutations[n][:, :self.velocitychannels], self.cij) for n in
                   range(self.K + 1)]
