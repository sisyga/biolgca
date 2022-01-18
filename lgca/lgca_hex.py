# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

from lgca.base import *
from lgca.lgca_square import LGCA_Square, IBLGCA_Square, NoVE_LGCA_Square


class LGCA_Hex(LGCA_Square):
    """
    2d lattice-gas cellular automaton on a hexagonal lattice.
    """
    velocitychannels = 6
    cix = np.cos(np.arange(velocitychannels) * pi2 / velocitychannels)
    ciy = np.sin(np.arange(velocitychannels) * pi2 / velocitychannels)
    c = np.array([cix, ciy])
    r_poly = 0.5 / np.cos(np.pi / velocitychannels)
    dy = np.sin(2 * np.pi / velocitychannels)
    orientation = 0.

    def init_coords(self):
        if self.ly % 2 != 0:
            print('Warning: uneven number of rows; only use for plotting - boundary conditions do not work!')
        self.x = np.arange(self.lx) + self.r_int
        self.y = np.arange(self.ly) + self.r_int
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.coord_pairs = list(zip(self.xx.flat, self.yy.flat))

        # set all coords, including ghost cells
        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords.astype(float)
        self.ycoords = self.ycoords.astype(float)

        # shift coordinate of every other column of x by 0.5 to the right
        self.xcoords[:, 1::2] += 0.5
        self.ycoords *= self.dy
        self.xcoords = self.xcoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.ycoords = self.ycoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.nonborder = (self.xx, self.yy)

    def propagation(self):
        #print(self.nodes[self.nonborder])
        newcellnodes = np.zeros(self.nodes.shape, dtype=self.nodes.dtype)
        newcellnodes[..., 6:] = self.nodes[..., 6:]

        # prop in 0-direction
        newcellnodes[1:, :, 0] = self.nodes[:-1, :, 0]

        # prop in 1-direction
        newcellnodes[:, 1::2, 1] = self.nodes[:, :-1:2, 1]
        newcellnodes[1:, 2::2, 1] = self.nodes[:-1, 1:-1:2, 1]

        # prop in 2-direction
        newcellnodes[:-1, 1::2, 2] = self.nodes[1:, :-1:2, 2]
        newcellnodes[:, 2::2, 2] = self.nodes[:, 1:-1:2, 2]

        # prop in 3-direction
        newcellnodes[:-1, :, 3] = self.nodes[1:, :, 3]

        # prop in 4-direction
        newcellnodes[:, :-1:2, 4] = self.nodes[:, 1::2, 4]
        newcellnodes[:-1, 1:-1:2, 4] = self.nodes[1:, 2::2, 4]

        # prop in 5-direction
        newcellnodes[1:, :-1:2, 5] = self.nodes[:-1, 1::2, 5]
        newcellnodes[:, 1:-1:2, 5] = self.nodes[:, 2::2, 5]

        #(newcellnodes[self.nonborder])
        self.nodes = newcellnodes
        return self.nodes

    def apply_rbcx(self):
        # left boundary
        self.nodes[self.r_int, :, 0] += self.nodes[self.r_int - 1, :, 3]
        self.nodes[self.r_int, 2:-1:2, 1] += self.nodes[self.r_int - 1, 1:-2:2, 4]
        self.nodes[self.r_int, 2:-1:2, 5] += self.nodes[self.r_int - 1, 3::2, 2]

        # right boundary
        self.nodes[-self.r_int - 1, :, 3] += self.nodes[-self.r_int, :, 0]
        self.nodes[-self.r_int - 1, 1:-1:2, 4] += self.nodes[-self.r_int, 2::2, 1]
        self.nodes[-self.r_int - 1, 1:-1:2, 2] += self.nodes[-self.r_int, :-2:2, 5]

        self.apply_abcx()

    def apply_rbcy(self):
        lx, ly, _ = self.nodes.shape

        # lower boundary
        self.nodes[(1 - (self.r_int % 2)):, self.r_int, 1] += self.nodes[:lx - (1 - (self.r_int % 2)), self.r_int - 1,
                                                              4]
        self.nodes[:lx - (self.r_int % 2), self.r_int, 2] += self.nodes[(self.r_int % 2):, self.r_int - 1, 5]

        # upper boundary
        self.nodes[:lx - ((ly - 1 - self.r_int) % 2), -self.r_int - 1, 4] += self.nodes[((ly - 1 - self.r_int) % 2):,
                                                                             -self.r_int, 1]
        self.nodes[(1 - ((ly - 1 - self.r_int) % 2)):, -self.r_int - 1, 5] += self.nodes[
                                                                              :lx - (1 - ((ly - 1 - self.r_int) % 2)),
                                                                              -self.r_int, 2]
        self.apply_abcy()

    def gradient(self, qty):
        gx = np.zeros_like(qty, dtype=float)
        gy = np.zeros_like(qty, dtype=float)

        # x-component
        gx[:-1, ...] += self.cix[0] * qty[1:, ...]

        gx[1:, ...] += self.cix[3] * qty[:-1, ...]

        gx[:, :-1:2, ...] += self.cix[1] * qty[:, 1::2, ...]
        gx[:-1, 1:-1:2, ...] += self.cix[1] * qty[1:, 2::2, ...]

        gx[1:, :-1:2, ...] += self.cix[2] * qty[:-1, 1::2, ...]
        gx[:, 1:-1:2, ...] += self.cix[2] * qty[:, 2::2, ...]

        gx[:, 1::2, ...] += self.cix[4] * qty[:, :-1:2, ...]
        gx[1:, 2::2, ...] += self.cix[4] * qty[:-1, 1:-1:2, ...]

        gx[:-1, 1::2, ...] += self.cix[5] * qty[1:, :-1:2, ...]
        gx[:, 2::2, ...] += self.cix[5] * qty[:, 1:-1:2, ...]

        # y-component
        gy[:, :-1:2, ...] += self.ciy[1] * qty[:, 1::2, ...]
        gy[:-1, 1:-1:2, ...] += self.ciy[1] * qty[1:, 2::2, ...]

        gy[1:, :-1:2, ...] += self.ciy[2] * qty[:-1, 1::2, ...]
        gy[:, 1:-1:2, ...] += self.ciy[2] * qty[:, 2::2, ...]

        gy[:, 1::2, ...] += self.ciy[4] * qty[:, :-1:2, ...]
        gy[1:, 2::2, ...] += self.ciy[4] * qty[:-1, 1:-1:2, ...]

        gy[:-1, 1::2, ...] += self.ciy[5] * qty[1:, :-1:2, ...]
        gy[:, 2::2, ...] += self.ciy[5] * qty[:, 1:-1:2, ...]

        g = np.moveaxis(np.array([gx, gy]), 0, -1)
        return g

    def channel_weight(self, qty):
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, :, 0] = qty[1:, ...]
        weights[1:, :, 3] = qty[:-1, ...]

        weights[:, :-1:2, 1] = qty[:, 1::2, ...]
        weights[:-1, 1:-1:2, 1] = qty[1:, 2::2, ...]

        weights[1:, :-1:2, 2] = qty[:-1, 1::2, ...]
        weights[:, 1:-1:2, 2] = qty[:, 2::2, ...]

        weights[:, 1::2, 4] = qty[:, :-1:2, ...]
        weights[1:, 2::2, 4] = qty[:-1, 1:-1:2, ...]

        weights[:-1, 1::2, 5] = qty[1:, :-1:2, ...]
        weights[:, 2::2, 5] = qty[:, 1:-1:2, ...]

        return weights

    def nb_sum(self, qty):
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        sum[:, 1::2, ...] += qty[:, :-1:2, ...]
        sum[1:, 2::2, ...] += qty[:-1, 1:-1:2, ...]
        sum[:-1, 1::2, ...] += qty[1:, :-1:2, ...]
        sum[:, 2::2, ...] += qty[:, 1:-1:2, ...]
        sum[:, :-1:2, ...] += qty[:, 1::2, ...]
        sum[:-1, 1:-1:2, ...] += qty[1:, 2::2, ...]
        sum[1:, :-1:2, ...] += qty[:-1, 1::2, ...]
        sum[:, 1:-1:2, ...] += qty[:, 2::2, ...]
        return sum

    def setup_figure(self, figindex=None, figsize=(8, 8), tight_layout=True):
        fig, ax = super(LGCA_Hex, self).setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        plt.gca()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        return fig, ax

class IBLGCA_Hex(IBLGCA_Square, LGCA_Hex):
    """
    Identity-based LGCA simulator class.
    """
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath', 'birthdeath_discrete',
                    'only_propagation']


class NoVE_LGCA_Hex (NoVE_LGCA_Square, LGCA_Hex):

    def nb_sum(self, qty, addCenter=False):
        """
        Calculate sum of values in neighboring lattice sites of each lattice site.
        :param qty: ndarray in which neighboring values have to be added
                    first dimension indexes lattice sites
        :param addCenter: toggle adding central value
        :return: sum as ndarray
        """
        sum = np.zeros(qty.shape)
        # shift to left padding 0 and add to shift to the right padding 0
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        sum[:, 1::2, ...] += qty[:, :-1:2, ...]
        sum[1:, 2::2, ...] += qty[:-1, 1:-1:2, ...]
        sum[:-1, 1::2, ...] += qty[1:, :-1:2, ...]
        sum[:, 2::2, ...] += qty[:, 1:-1:2, ...]
        sum[:, :-1:2, ...] += qty[:, 1::2, ...]
        sum[:-1, 1:-1:2, ...] += qty[1:, 2::2, ...]
        sum[1:, :-1:2, ...] += qty[:-1, 1::2, ...]
        sum[:, 1:-1:2, ...] += qty[:, 2::2, ...]
        # add central value
        if addCenter:
            sum += qty
        return sum
