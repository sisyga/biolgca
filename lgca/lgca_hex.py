try:
    from .base import *
    from .lgca_square import LGCA_Square, IBLGCA_Square, LGCA_NoVE_2D

except ModuleNotFoundError:
    from base import *
    from lgca_square import LGCA_Square, IBLGCA_Square

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

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

        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords.astype(float)
        self.ycoords = self.ycoords.astype(float)

        self.xcoords[:, 1::2] += 0.5
        self.ycoords *= self.dy
        self.xcoords = self.xcoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.ycoords = self.ycoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.nonborder = (self.xx, self.yy)

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


class IBLGCA_Hex(IBLGCA_Square, LGCA_Hex):
    """
    Identity-based LGCA simulator class.
    """

    def init_nodes(self, density=0.1, nodes=None):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)

        else:
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.maxlabel = self.nodes.max()




class LGCA_NoVe_HEX (LGCA_NoVE_2D, LGCA_Hex):
    def init_coords(self):
        if self.ly % 2 != 0:
            print('Warning: uneven number of rows; only use for plotting - boundary conditions do not work!')
        self.x = np.arange(self.lx) + self.r_int
        self.y = np.arange(self.ly) + self.r_int
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.coord_pairs = list(zip(self.xx.flat, self.yy.flat))

        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords.astype(float)
        self.ycoords = self.ycoords.astype(float)

        self.xcoords[:, 1::2] += 0.5
        self.ycoords *= self.dy
        self.xcoords = self.xcoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.ycoords = self.ycoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.nonborder = (self.xx, self.yy)

    def calc_polar_alignment_parameter(self):

        sumx = 0
        sumy = 0

        abb = self.calc_flux(self.nodes)[self.nonborder]
        x = len(abb)
        y = len(abb[0])
        z = len(abb[0][0])

        for a in range(0, x):
            for b in range(0, y):
                for c in range(0, z):
                    if c == 0:
                        sumx = sumx + abb[a][b][c]
                    if c == 1:
                        sumy = sumy + abb[a][b][c]

        cells = self.nodes[self.nonborder].sum()

        sumy = sumy / cells
        sumx = sumx / cells

        magnitude = np.sqrt(sumx**2 + sumy**2)

        return magnitude





    def nbofnodes(self):
        return self.nodes[self.nonborder].sum()

    def vectorsum(self):
        return self.calc_flux(self.nodes)[self.nonborder].sum()



    def calc_mean_alignment(self):

        no_neighbors = self.nb_sum(self.cell_density[
                                                   self.nonborder])  # neighborhood is defined s.t. border particles don't have them

        f = self.calc_flux(self.nodes[self.nonborder])

        d = self.cell_density[self.nonborder]

        fsum = self.nb_sum(f)

        x = len(fsum)
        y = len(fsum[0])
        z = len(fsum[0][0])

        no_nbdiv = np.where(no_neighbors > 0, no_neighbors, 1)

        for a in range(0, x):
            for b in range(0, y):
                for c in range(0, z):
                    fsum[a][b][c] = (fsum[a][b][c]) / no_nbdiv[a][b]

        dot_p = []

        for a in range(0, x):
            for b in range(0, y):
                dot_p.append(np.dot(fsum[a][b], f[a][b]))

        tot = 0
        for i in range(0, len(dot_p)):
            tot = tot + dot_p[i]

        return tot / d.sum()


    def calc_entropy(self):
        """
        Calculate entropy of the lattice.
        :return: entropy according to information theory as scalar
        """
        # calculate relative frequencies
        rel_freq = self.nodes[self.nonborder].sum(-1)/self.nodes[self.nonborder].sum()
        # empty lattice sites are not considered in the sum
        a = np.where(rel_freq > 0, np.log(rel_freq), 0)
        return -np.multiply(rel_freq, a).sum()


    def calc_normalized_entropy(self):
        """
        Calculate entropy of the lattice normalized to maximal possible entropy.
        :return: normalized entropy as scalar
        """
        n_ofnodes = self.dims[0] * self.dims[1]
        smax = np.log(n_ofnodes)

        self.smax = smax

        return 1 - self.calc_entropy()/smax




if __name__ == '__main__':
    lx = 200
    ly = lx
    restchannels = 6
    nodes = np.zeros((lx, ly, 6 + restchannels))
    nodes[lx // 2, ly // 2, :] = 1
    # nodes[...] = 1
    # nodes[:lx // 2, :, -2:] = 1
    # nodes[..., -1] = 1
    # nodes[:, ly//2:, 6:] = 1
    # nodes[0, :, :4] = 1
    # lgca = LGCA_Hex(restchannels=restchannels, dims=(lx, ly), density=0.5 / (6 + restchannels), bc='pbc',
    #                 interaction='wetting', beta=20., gamma=10)
    # lgca.ecm = np.zeros_like(lgca.cell_density, dtype=bool)
    lgca = IBLGCA_Hex(nodes=nodes, interaction='birthdeath', std=0.05, r_b=0.1)
    # lgca.set_interaction('contact_guidance', beta=2)
    # cProfile.run('lgca.timeevo(timesteps=1000)')
    # lgca.timeevo(timesteps=50, record=True)
    # ani = lgca.animate_flow(interval=500)
    # ani = lgca.animate_flux(interval=50)
    # ani = lgca.animate_density(interval=50)
    # ani = lgca.animate_density(density_t=refr, interval=50, vmax=lgca.restchannels)
    # ani2 = lgca.animate_density(density_t=exc, interval=50, vmax=lgca.velocitychannels)
    # ani = lgca.animate_config(interval=10, grid=False)

    # ani = lgca.live_animate_density(interval=100, vmax=lgca.restchannels, channels=range(6, lgca.K))
    # ani2 = lgca.live_animate_density(interval=100, vmax=lgca.velocitychannels, channels=range(6))
    # ani = lgca.live_animate_flow()
    # ani = lgca.live_animate_density()
    #ani = lgca.live_animate_config()
    # plt.streamplot(lgca.xcoords[:, 0], lgca.ycoords[-1], lgca.g[1:-1, 1:-1, 0].T, lgca.g[1:-1, 1:-1, 1].T, density=.5,
    #               arrowstyle='->', color='orange', linewidth=2.)
    # ani = lgca.live_animate_density()
    # lgca.plot_config()
    # lgca.plot_density(edgecolor='k')
    lgca.timeevo(10, record=True)
    print(len(lgca.props_t))
    # lgca.plot_prop_spatial(propname='r_b', cbarlabel='$r_b$', vmin=0, vmax=1)
    # plt.show()
