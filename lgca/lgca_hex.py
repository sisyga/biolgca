try:
    from base import *
    from lgca_square import LGCA_Square, IBLGCA_Square

except ModuleNotFoundError:
    from .base import *
    from .lgca_square import LGCA_Square, IBLGCA_Square


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


if __name__ == '__main__':
    lx = 50
    ly = lx
    restchannels = 0
    nodes = np.zeros((lx, ly, 6 + restchannels))
    nodes[0] = 1
    # lgca = LGCA_Hex(restchannels=restchannels, dims=(lx, ly), density=0.5 / (6 + restchannels), bc='pbc',
    #                 interaction='wetting', beta=20., gamma=10)
    lgca = IBLGCA_Hex(nodes=nodes, interaction='go_and_grow', bc='refl', r_b=0.1)
    # lgca.set_interaction('contact_guidance', beta=2)
    # cProfile.run('lgca.timeevo(timesteps=1000)')
    lgca.timeevo(timesteps=200, record=True)
    # ani = lgca.animate_flow(interval=500)
    # ani = lgca.animate_flux(interval=50)
    # ani = lgca.animate_density(interval=50)
    lgca.plot_prop_spatial()
    plt.show()

    # l = 50
    # l_spheroid = 2
    # dims = (l, l)
    # tmax = 100
    # restc = 3
    # rho_0 = 3
    # nodes = np.zeros((l, l, restc + 6), dtype=bool)
    # nodes[..., :l_spheroid, -rho_0:] = 1
    # lgca = get_lgca(geometry='hex', interaction='wetting', beta=10., gamma=5., bc='rbc', density=0, restchannels=restc,
    #                 nodes=nodes, rho_0=rho_0)
    # lgca.r_b = .05
    # lgca.spheroid = np.zeros_like(lgca.cell_density, dtype=bool)
    # lgca.spheroid[lgca.r_int:-lgca.r_int, :lgca.r_int + l_spheroid] = 1
    # lgca.ecm = 1 - lgca.spheroid.astype(float)
    # lgca.ecm *= 0.
    # lgca.timestep()
    # lgca.timeevo(12, record=True)
    # lgca.plot_flow(cbar=False)
    # ani = lgca.live_animate_flow(cbar=False)
    # plt.show()
