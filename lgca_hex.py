from __future__ import division

import numpy.random as npr

from lgca_square import LGCA_SQUARE
from tools import *


class LGCA_HEX(LGCA_SQUARE):
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

    def __init__(self, nodes=None, lx=50, ly=50, restchannels=1, density=0.1, bc='periodic', r_int=1, **kwargs):
        """
        Initialize class instance.
        :param nodes:
        :param l:
        :param restchannels:
        :param density:
        :param bc:
        :param r_int:
        :param kwargs:
        """
        self.dens_t, self.nodes_t, self.n_t = np.empty(3)  # placeholders to record dynamics
        assert r_int > 0
        self.r_int = r_int  # interaction range; must be at least 1 to handle propagation.
        if nodes is None:
            self.lx = lx
            self.ly = ly
            self.restchannels = restchannels
            self.K = self.velocitychannels + self.restchannels
            self.nodes = np.zeros((lx + 2 * self.r_int, ly + 2 * self.r_int, self.K), dtype=np.bool)
            self.nodes = npr.random(self.nodes.shape) < density
        if nodes is not None:
            assert len(nodes.shape) == 3
            self.lx, self.ly, self.K = nodes.shape
            self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K),
                                  dtype=np.bool)
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(np.bool)
            self.restchannels = self.K - self.velocitychannels

        self.x = np.arange(self.lx) + self.r_int
        self.y = np.arange(self.ly) + self.r_int
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.coord_pairs = zip(self.xx.flat, self.yy.flat)

        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords.astype(float)
        self.ycoords = self.ycoords.astype(float)

        self.xcoords[:, 1::2] += 0.5
        self.ycoords *= self.dy
        self.xcoords = self.xcoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.ycoords = self.ycoords[self.r_int:-self.r_int, self.r_int:-self.r_int]

        self.set_bc(bc)
        self.interaction = self.random_walk
        self.birth_death = blank_fct

        if 'interaction' in kwargs:
            self.set_interaction(**kwargs)

        self.cell_density = self.nodes.sum(-1)

    def propagation(self):
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

        self.nodes = newcellnodes
        return self.nodes

    def apply_pbc(self):
        self.nodes[:self.r_int, ...] = self.nodes[-2 * self.r_int:-self.r_int, ...]  # left boundary
        self.nodes[-self.r_int:, ...] = self.nodes[self.r_int:2 * self.r_int, ...]  # right boundary
        self.nodes[:, :self.r_int, :] = self.nodes[:, -2 * self.r_int:-self.r_int, :]  # upper boundary
        self.nodes[:, -self.r_int:, :] = self.nodes[:, self.r_int:2 * self.r_int, :]  # lower boundary

    def apply_rbc(self):
        lx, ly, _ = self.nodes.shape
        # left boundary
        self.nodes[self.r_int, :, 0] += self.nodes[self.r_int - 1, :, 3]
        self.nodes[self.r_int, 2:-1:2, 1] += self.nodes[self.r_int - 1, 1:-2:2, 4]
        self.nodes[self.r_int, 2:-1:2, 5] += self.nodes[self.r_int - 1, 3::2, 2]

        # lower boundary
        self.nodes[(1 - (self.r_int % 2)):, self.r_int, 1] += self.nodes[:lx - (1 - (self.r_int % 2)), self.r_int - 1,
                                                              4]
        self.nodes[:lx - (self.r_int % 2), self.r_int, 2] += self.nodes[(self.r_int % 2):, self.r_int - 1, 5]

        # right boundary
        self.nodes[-self.r_int - 1, :, 3] += self.nodes[-self.r_int, :, 0]
        self.nodes[-self.r_int - 1, 1:-1:2, 4] += self.nodes[-self.r_int, 2::2, 1]
        self.nodes[-self.r_int - 1, 1:-1:2, 2] += self.nodes[-self.r_int, :-2:2, 5]

        # upper boundary
        self.nodes[:lx - ((ly - 1 - self.r_int) % 2), -self.r_int - 1, 4] += self.nodes[((ly - 1 - self.r_int) % 2):,
                                                                             -self.r_int, 1]
        self.nodes[(1 - ((ly - 1 - self.r_int) % 2)):, -self.r_int - 1, 5] += self.nodes[
                                                                              :lx - (1 - ((ly - 1 - self.r_int) % 2)),
                                                                              -self.r_int, 2]
        self.apply_abc()

    def apply_abc(self):
        self.nodes[:self.r_int, ...] = 0
        self.nodes[-self.r_int:, ...] = 0
        self.nodes[:, :self.r_int, :] = 0
        self.nodes[:, -self.r_int:, :] = 0

    def gradient(self, qty):
        gx = np.zeros(qty.shape)
        gy = np.zeros(qty.shape)

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
        # gy[:-1, ...] += self.ciy[0] * qty[1:, ...]  # should be 0
        # gy[1:, ...] += self.ciy[3] * qty[:-1, ...]  # should be 0

        gy[:, :-1:2, ...] += self.ciy[1] * qty[:, 1::2, ...]
        gy[:-1, 1:-1:2, ...] += self.ciy[1] * qty[1:, 2::2, ...]

        gy[1:, :-1:2, ...] += self.ciy[2] * qty[:-1, 1::2, ...]
        gy[:, 1:-1:2, ...] += self.ciy[2] * qty[:, 2::2, ...]

        gy[:, 1::2, ...] += self.ciy[4] * qty[:, :-1:2, ...]
        gy[1:, 2::2, ...] += self.ciy[4] * qty[:-1, 1:-1:2, ...]

        gy[:-1, 1::2, ...] += self.ciy[5] * qty[1:, :-1:2, ...]
        gy[:, 2::2, ...] += self.ciy[5] * qty[:, 1:-1:2, ...]

        return gx, gy

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


if __name__ == '__main__':
    lx = 20
    ly = lx
    restchannels = 0
    nodes = np.zeros((lx, ly, 6 + restchannels))
    nodes[2, 1, 0] = 1
    # nodes[...] = 1
    # nodes[:lx//2, :, :6] = 1
    # nodes[:, ly//2:, 6:] = 1
    # nodes[0, :, :4] = 1
    lgca = LGCA_HEX(restchannels=restchannels, lx=lx, ly=ly, density=0.1, bc='pbc')  # , nodes=nodes)
    # lgca.plot_vectorfield(lgca.xcoords, lgca.ycoords, gx, gy, figindex=fig.number)
    lgca.set_interaction(interaction='nematic')
    # lgca.timeevo(timesteps=300, record=True)
    # ani = lgca.animate_flow(interval=500)
    # ani = lgca.animate_flux(interval=50)
    # ani = lgca.animate_density(interval=100)
    # ani = lgca.animate_density(density_t=refr, interval=50, vmax=lgca.restchannels)
    # ani2 = lgca.animate_density(density_t=exc, interval=50, vmax=lgca.velocitychannels)
    # lgca.plot_flux(edgecolor='k')
    # ani = lgca.live_animate_density(interval=100)
    # ani = lgca.live_animate_flux()
    # ani = lgca.live_animate_flow()
    ani = lgca.live_animate_config(interval=200, grid=False)
    # ani = lgca.animate_config(interval=10, grid=False)
    # lgca.plot_config(grid=False)
    # lgca.plot_density(edgecolor='k')
    plt.show()
