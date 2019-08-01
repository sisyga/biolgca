import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch

try:
    from base import *
except ModuleNotFoundError:
    from .base import *


class LGCA_Square(LGCA_base):
    """
    2D version of a LGCA on the square lattice.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation',
                    'random_walk', 'excitable_medium', 'nematic', 'persistant_motion', 'chemotaxis', 'contact_guidance']
    velocitychannels = 4
    cix = np.array([1, 0, -1, 0], dtype=float)
    ciy = np.array([0, 1, 0, -1], dtype=float)
    c = np.array([cix, ciy])
    r_poly = 0.5 / np.cos(np.pi / velocitychannels)
    dy = np.sin(2 * np.pi / velocitychannels)
    orientation = np.pi / velocitychannels

    def set_dims(self, dims=None, nodes=None, restchannels=0):
        if nodes is not None:
            self.lx, self.ly, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels
            return

        elif dims is None:
            dims = (50, 50)

        try:
            self.lx, self.ly = dims
        except TypeError:
            self.lx, self.ly = dims, dims

        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density=0.1, nodes=None):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.bool)
        if nodes is None:
            self.random_reset(density)

        else:
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(np.bool)

    def init_coords(self):
        self.x = np.arange(self.lx) + self.r_int
        self.y = np.arange(self.ly) + self.r_int
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.nonborder = (self.xx, self.yy)

        self.coord_pairs = list(zip(self.xx.flat, self.yy.flat))
        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords[self.nonborder].astype(float)
        self.ycoords = self.ycoords[self.nonborder].astype(float)

    def propagation(self):
        """

        :return:
        """
        newnodes = np.zeros(self.nodes.shape, dtype=self.nodes.dtype)
        # resting particles stay
        newnodes[..., 4:] = self.nodes[..., 4:]

        # prop. to the right
        newnodes[1:, :, 0] = self.nodes[:-1, :, 0]

        # prop. to the left
        newnodes[:-1, :, 2] = self.nodes[1:, :, 2]

        # prop. upwards
        newnodes[:, 1:, 1] = self.nodes[:, :-1, 1]

        # prop. downwards
        newnodes[:, :-1, 3] = self.nodes[:, 1:, 3]

        self.nodes = newnodes

    def apply_pbcx(self):
        self.nodes[:self.r_int, ...] = self.nodes[-2 * self.r_int:-self.r_int, ...]  # left boundary
        self.nodes[-self.r_int:, ...] = self.nodes[self.r_int:2 * self.r_int, ...]  # right boundary

    def apply_pbcy(self):
        self.nodes[:, :self.r_int, :] = self.nodes[:, -2 * self.r_int:-self.r_int, :]  # upper boundary
        self.nodes[:, -self.r_int:, :] = self.nodes[:, self.r_int:2 * self.r_int, :]  # lower boundary

    def apply_pbc(self):
        self.apply_pbcx()
        self.apply_pbcy()

    def apply_rbcx(self):
        self.nodes[self.r_int, :, 0] += self.nodes[self.r_int - 1, :, 2]
        self.nodes[-self.r_int - 1, :, 2] += self.nodes[-self.r_int, :, 0]
        self.apply_abcx()

    def apply_rbcy(self):
        self.nodes[:, self.r_int, 1] += self.nodes[:, self.r_int - 1, 3]
        self.nodes[:, -self.r_int - 1, 3] += self.nodes[:, -self.r_int, 1]
        self.apply_abcy()

    def apply_rbc(self):
        self.apply_rbcx()
        self.apply_rbcy()

    def apply_abcx(self):
        self.nodes[:self.r_int, ...] = 0
        self.nodes[-self.r_int:, ...] = 0

    def apply_abcy(self):
        self.nodes[:, :self.r_int, :] = 0
        self.nodes[:, -self.r_int:, :] = 0

    def apply_abc(self):
        self.apply_abcx()
        self.apply_abcy()

    def apply_inflowbc(self):
        """
        Boundary condition for a inflow from x=0, y=:, with reflecting boundary conditions along the y-axis and periodic
        boundaries along the x-axis. Nodes at (x=0, y) are set to a homogeneous state with a constant average density
        given by the attribute 0 <= self.inflow <= 1.
        If there is no such attribute, the nodes are filled with the maximum density.
        :return:
        """
        if hasattr(self, 'inflow'):
            self.nodes[:, self.r_int, ...] = npr.random(self.nodes[0].shape) < self.inflow

        else:
            self.nodes[:, self.r_int, ...] = 1

        self.apply_rbc()
        # self.apply_pbcy()

    def nb_sum(self, qty):
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        sum[:, :-1, ...] += qty[:, 1:, ...]
        sum[:, 1:, ...] += qty[:, :-1, ...]
        return sum

    def gradient(self, qty):
        return np.moveaxis(np.asarray(np.gradient(qty, 2)), 0, -1)

    def channel_weight(self, qty):
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, :, 0] = qty[1:, ...]
        weights[1:, :, 2] = qty[:-1, ...]
        weights[:, :-1, 1] = qty[:, 1:, ...]
        weights[:, 1:, 3] = qty[:, :-1, ...]

        return weights

    def calc_vorticity(self, nodes):
        flux = self.calc_flux(nodes)
        dens = nodes.sum(-1)
        flux = np.divide(flux, dens[..., None], where=dens[..., None] > 0, out=np.zeros_like(flux))
        fx, fy = flux[..., 0], flux[..., 1]
        dfx = self.gradient(fx)
        dfy = self.gradient(fy)
        dfxdy = dfx[..., 1]
        dfydx = dfy[..., 0]
        vorticity = dfydx - dfxdy
        return vorticity

    def timeevo(self, timesteps=100, record=False, recordN=False, recorddens=True, showprogress=True):
        self.update_dynamic_fields()
        if record:
            self.nodes_t = np.zeros((timesteps + 1, self.lx, self.ly, self.K), dtype=self.nodes.dtype)
            self.nodes_t[0, ...] = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, ...]
        if recordN:
            self.n_t = np.zeros(timesteps + 1, dtype=np.int)
            self.n_t[0] = self.nodes.sum()
        if recorddens:
            self.dens_t = np.zeros((timesteps + 1, self.lx, self.ly))
            self.dens_t[0, ...] = self.cell_density[self.r_int:-self.r_int, self.r_int:-self.r_int]
        for t in range(1, timesteps + 1):
            self.timestep()
            if record:
                self.nodes_t[t, ...] = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :]
            if recordN:
                self.n_t[t] = self.cell_density.sum()
            if recorddens:
                self.dens_t[t, ...] = self.cell_density[self.r_int:-self.r_int, self.r_int:-self.r_int]
            if showprogress:
                update_progress(1.0 * t / timesteps)

    def setup_figure(self, figindex=None, figsize=None, tight_layout=True):
        dy = self.r_poly * np.cos(self.orientation)
        fig = plt.figure(num=figindex, figsize=figsize, tight_layout=tight_layout)
        ax = fig.add_subplot(111)
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + dy
        ymin = self.ycoords.min() - dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        ax.set_aspect('equal')
        plt.xlabel('$x \\; (\\varepsilon)$')
        plt.ylabel('$y \\; (\\varepsilon)$')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_autoscale_on(False)
        return fig, ax

    def plot_config(self, nodes=None, figindex=None, figsize=None, tight_layout=True, grid=False, ec='none'):
        r_circle = self.r_poly * 0.25
        # bbox_props = dict(boxstyle="Circle,pad=0.3", fc="white", ec="k", lw=1.5)
        bbox_props = None
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        density = nodes.sum(-1)
        if figsize is None:
            figsize = estimate_figsize(density, cbar=False, dy=self.dy)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)

        xx, yy = self.xcoords, self.ycoords
        x1, y1 = ax.transData.transform((0, 1.5 * r_circle))
        x2, y2 = ax.transData.transform((1.5 * r_circle, 0))
        dpx = np.mean([abs(x2 - x1), abs(y2 - y1)])
        fontsize = dpx * 72. / fig.dpi
        lw_circle = fontsize / 5
        lw_arrow = 0.5 * lw_circle

        colors = ('none', 'k')
        arrows = []
        for i in range(self.velocitychannels):
            cx = self.c[0, i] * 0.5
            cy = self.c[1, i] * 0.5
            arrows += [FancyArrowPatch((x, y), (x + cx, y + cy), mutation_scale=.3, fc=colors[occ], ec=ec, lw=lw_arrow)
                       for x, y, occ in zip(xx.ravel(), yy.ravel(), nodes[..., i].ravel())]

        arrows = PatchCollection(arrows, match_original=True)
        ax.add_collection(arrows)

        if self.restchannels > 0:
            circles = [Circle(xy=(x, y), radius=r_circle, fc='white', ec='k', lw=lw_circle * bool(n), visible=bool(n))
                       for x, y, n in
                       zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].sum(-1).ravel())]
            texts = [ax.text(x, y - 0.5 * r_circle, str(n), ha='center', va='baseline', fontsize=fontsize,
                             fontname='sans-serif', fontweight='bold', bbox=bbox_props, visible=bool(n))
                     for x, y, n in zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].sum(-1).ravel())]
            circles = PatchCollection(circles, match_original=True)
            ax.add_collection(circles)

        else:
            circles = []
            texts = []

        if grid:
            hexagons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly, lw=lw_arrow,
                                       orientation=self.orientation, facecolor='None', edgecolor='k')
                        for x, y in zip(self.xcoords.ravel(), self.ycoords.ravel())]
            ax.add_collection(PatchCollection(hexagons, match_original=True))

        else:
            ymin = -0.5 * self.c[1, 1]
            ymax = self.ycoords.max() + 0.5 * self.c[1, 1]
            plt.ylim(ymin, ymax)

        return fig, arrows, circles, texts

    def animate_config(self, nodes_t=None, figindex=None, figsize=None, interval=100, tight_layout=True, grid=False):
        if nodes_t is None:
            nodes_t = self.nodes_t

        fig, arrows, circles, texts = self.plot_config(nodes=nodes_t[0], figindex=figindex, figsize=figsize,
                                                       tight_layout=tight_layout, grid=grid)
        title = plt.title('Time $k =$0')
        arrow_color = np.zeros(nodes_t[..., :self.velocitychannels].shape + (4,))
        arrow_color = arrow_color.reshape(nodes_t.shape[0], -1, 4)
        arrow_color[..., -1] = np.moveaxis(nodes_t[..., :self.velocitychannels], -1, 1).reshape(nodes_t.shape[0], -1)

        if self.restchannels:
            circle_color = np.zeros(nodes_t[..., 0].shape + (4,))
            circle_color = circle_color.reshape(nodes_t.shape[0], -1, 4)
            circle_color[..., -1] = np.any(nodes_t[..., self.velocitychannels:], axis=-1).reshape(nodes_t.shape[0], -1)
            circle_fcolor = np.ones(circle_color.shape)
            circle_fcolor[..., -1] = circle_color[..., -1]
            resting_t = nodes_t[..., self.velocitychannels:].sum(-1).reshape(nodes_t.shape[0], -1)

            def update(n):
                title.set_text('Time $k =${}'.format(n))
                arrows.set(color=arrow_color[n])
                circles.set(edgecolor=circle_color[n], facecolor=circle_fcolor[n])
                for text, i in zip(texts, resting_t[n]):
                    text.set_text(str(i))
                    text.set(alpha=bool(i))
                return arrows, circles, texts, title

            ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
            return ani

        else:
            def update(n):
                title.set_text('Time $k =${}'.format(n))
                arrows.set(color=arrow_color[n])
                return arrows, title

            ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
            return ani

    def live_animate_config(self, figindex=None, figsize=None, interval=100, tight_layout=True, grid=False):
        fig, arrows, circles, texts = self.plot_config(figindex=figindex, figsize=figsize,
                                                       tight_layout=tight_layout, grid=grid)
        title = plt.title('Time $k =$0')
        nodes = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int]
        arrow_color = np.zeros(nodes[..., :self.velocitychannels].ravel().shape + (4,))
        if self.restchannels:
            circle_color = np.zeros(nodes[..., 0].ravel().shape + (4,))
            circle_fcolor = np.ones(circle_color.shape)

            def update(n):
                self.timestep()
                nodes = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int]
                arrow_color[:, -1] = np.moveaxis(nodes[..., :self.velocitychannels], -1, 0).ravel()
                circle_color[:, -1] = np.any(nodes[..., self.velocitychannels:], axis=-1).ravel()
                circle_fcolor[:, -1] = circle_color[:, -1]
                resting_t = nodes[..., self.velocitychannels:].sum(-1).ravel()
                title.set_text('Time $k =${}'.format(n))
                arrows.set(color=arrow_color)
                circles.set(edgecolor=circle_color, facecolor=circle_fcolor)
                for text, i in zip(texts, resting_t):
                    text.set_text(str(i))
                    text.set(alpha=bool(i))
                return arrows, circles, texts, title

            ani = animation.FuncAnimation(fig, update, interval=interval)
            return ani

        else:
            def update(n):
                self.timestep()
                nodes = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int]
                arrow_color[:, -1] = np.moveaxis(nodes[..., :self.velocitychannels], -1, 0).ravel()
                title.set_text('Time $k =${}'.format(n))
                arrows.set(color=arrow_color)
                return arrows, title

            ani = animation.FuncAnimation(fig, update, interval=interval)
            return ani

    def live_animate_density(self, figindex=None, figsize=None, cmap='viridis', interval=100, vmax=None,
                             channels=slice(None), tight_layout=True, edgecolor='None'):

        fig, pc, cmap = self.plot_density(figindex=figindex, figsize=figsize, cmap=cmap, vmax=vmax,
                                          tight_layout=tight_layout, edgecolor=edgecolor)
        title = plt.title('Time $k =$0')

        def update(n):
            self.timestep()
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=cmap.to_rgba(
                self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, channels].sum(-1).ravel()))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval)
        return ani

    def plot_flow(self, nodes=None, figindex=None, figsize=None, tight_layout=True, cmap='viridis', vmax=None):

        if nodes is None:
            nodes = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :]

        if vmax is None:
            K = self.K

        else:
            K = vmax

        nodes = nodes.astype(float)
        density = nodes.sum(-1)
        xx, yy = self.xcoords, self.ycoords
        jx, jy = np.moveaxis(self.calc_flux(nodes), -1, 0)

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        ax.set_aspect('equal')
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_under(alpha=0.0)
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
        cmap.set_array(density)
        plot = plt.quiver(xx, yy, jx, jy, facecolor=cmap.to_rgba(density.flatten()), pivot='mid', angles='xy',
                          scale_units='xy', scale=1)
        cbar = fig.colorbar(cmap, extend='min', use_gridspec=True)
        cbar.set_label('Particle number $n$')
        cbar.set_ticks(np.linspace(0., K + 1, 2 * K + 3, endpoint=True)[1::2])
        cbar.set_ticklabels(1 + np.arange(K))
        return fig, plot, cmap

    def animate_flow(self, nodes_t=None, figindex=None, figsize=None, interval=100, tight_layout=True, cmap='viridis'):
        if nodes_t is None:
            nodes_t = self.nodes_t

        nodes = nodes_t.astype(float)
        density = nodes.sum(-1)
        jx, jy = np.moveaxis(self.calc_flux(nodes.astype(float)), -1, 0)

        fig, plot, cmap = self.plot_flow(nodes[0], figindex=figindex, figsize=figsize, tight_layout=tight_layout,
                                         cmap=cmap)
        title = plt.title('Time $k =$0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            plot.set_UVC(jx[n], jy[n], cmap.to_rgba(density[n]))
            return plot, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
        return ani

    def live_animate_flow(self, figindex=None, figsize=None, interval=100, tight_layout=True, cmap='viridis',
                          vmax=None):
        fig, plot, cmap = self.plot_flow(figindex=figindex, figsize=figsize, tight_layout=tight_layout, cmap=cmap,
                                         vmax=None)
        title = plt.title('Time $k =$0')

        def update(n):
            self.timestep()
            jx, jy = np.moveaxis(self.calc_flux(self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int]), -1, 0)
            title.set_text('Time $k =${}'.format(n))
            plot.set_UVC(jx, jy)
            plot.set(
                facecolor=cmap.to_rgba(self.cell_density[self.r_int:-self.r_int, self.r_int:-self.r_int].flatten()))
            return plot, title

        ani = animation.FuncAnimation(fig, update, interval=interval)
        return ani

    def plot_density(self, density=None, figindex=None, figsize=None, tight_layout=True, cmap='viridis', vmax=None,
                     edgecolor='None', cbar=True):
        if density is None:
            density = self.cell_density[self.nonborder]

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True, dy=self.dy)

        if vmax is None:
            K = self.K

        else:
            K = vmax

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_under(alpha=0.0)
        if K > 1:
            cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
        else:
            cmap = plt.cm.ScalarMappable(cmap=cmap)
        cmap.set_array(density)
        hexagons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(density.ravel()))]
        pc = PatchCollection(hexagons, match_original=True)
        ax.add_collection(pc)
        if cbar:
            cbar = fig.colorbar(cmap, extend='min', use_gridspec=True)
            cbar.set_label('Particle number $n$')
            cbar.set_ticks(np.linspace(0., K + 1, 2 * K + 3, endpoint=True)[1::2])
            cbar.set_ticklabels(1 + np.arange(K))
        return fig, pc, cmap

    def plot_vectorfield(self, x, y, vfx, vfy, figindex=None, figsize=None, tight_layout=True, cmap='viridis'):
        l = np.sqrt(vfx ** 2 + vfy ** 2)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        ax.set_aspect('equal')
        plot = plt.quiver(x, y, vfx, vfy, l, cmap=cmap, pivot='mid', angles='xy', scale_units='xy', scale=1,
                          width=0.007, norm=colors.Normalize(vmin=0, vmax=1))
        return fig, plot

    def plot_flux(self, nodes=None, figindex=None, figsize=None, tight_layout=True, edgecolor='None', cbar=True):
        if nodes is None:
            nodes = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :]

        nodes = nodes.astype(np.int8)
        density = nodes.sum(-1).astype(float) / self.K

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        cmap = plt.cm.get_cmap('hsv')
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=0, vmax=360))

        jx, jy = np.moveaxis(self.calc_flux(nodes), -1, 0)
        angle = np.zeros(density.shape, dtype=complex)
        angle.real = jx
        angle.imag = jy
        angle = np.angle(angle, deg=True) % 360.
        cmap.set_array(angle)
        angle = cmap.to_rgba(angle)
        angle[..., -1] = np.sqrt(density)
        angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.
        hexagons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c,
                                   edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), angle.reshape(-1, 4))]
        pc = PatchCollection(hexagons, match_original=True)
        ax.add_collection(pc)
        if cbar:
            cbar = fig.colorbar(cmap, use_gridspec=True)
            cbar.set_label('Direction of movement $(\degree)$')
            cbar.set_ticks(np.arange(self.velocitychannels) * 360 / self.velocitychannels)
        return fig, pc, cmap


    def animate_density(self, density_t=None, figindex=None, figsize=None, cmap='viridis', interval=500, vmax=None,
                        tight_layout=True, edgecolor='None'):
        if density_t is None:
            density_t = self.dens_t

        fig, pc, cmap = self.plot_density(density_t[0], figindex=figindex, figsize=figsize, cmap=cmap, vmax=vmax,
                                          tight_layout=tight_layout, edgecolor=edgecolor)
        title = plt.title('Time $k =$0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=cmap.to_rgba(density_t[n, ...].ravel()))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=density_t.shape[0])
        return ani

    def animate_flux(self, nodes_t=None, figindex=None, figsize=None, interval=200, tight_layout=True,
                     edgecolor='None', cbar=True):
        if nodes_t is None:
            nodes_t = self.nodes_t

        nodes = nodes_t.astype(float)
        density = nodes.sum(-1) / self.K
        jx, jy = np.moveaxis(self.calc_flux(nodes), -1, 0)

        angle = np.zeros(density.shape, dtype=complex)
        angle.real = jx
        angle.imag = jy
        angle = np.angle(angle, deg=True) % 360.
        fig, pc, cmap = self.plot_flux(nodes=nodes[0], figindex=figindex, figsize=figsize, tight_layout=tight_layout,
                                       edgecolor=edgecolor, cbar=cbar)
        angle = cmap.to_rgba(angle[None, ...])[0]
        angle[..., -1] = np.sqrt(density)
        angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.
        title = plt.title('Time $k =$ 0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=angle[n, ...].reshape(-1, 4))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
        return ani

    def live_animate_flux(self, figindex=None, figsize=None, cmap='viridis', interval=100, tight_layout=True,
                          edgecolor='None'):

        fig, pc, cmap = self.plot_flux(figindex=figindex, figsize=figsize, tight_layout=tight_layout,
                                       edgecolor=edgecolor)
        title = plt.title('Time $k =$0')

        def update(n):
            self.timestep()
            jx, jy = np.moveaxis(self.calc_flux(self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :]), -1, 0)
            density = self.cell_density[self.r_int:-self.r_int, self.r_int:-self.r_int] / self.K

            angle = np.empty(density.shape, dtype=complex)
            angle.real = jx
            angle.imag = jy
            angle = np.angle(angle, deg=True) % 360.
            angle = cmap.to_rgba(angle)
            angle[..., -1] = np.sqrt(density)
            angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=angle.reshape(-1, 4))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval)
        return ani


if __name__ == '__main__':
    lx = 50
    ly = lx
    restchannels = 2
    nodes = np.zeros((lx, ly, 4 + restchannels))
    nodes[..., (0, 1)] = 1
    lgca = LGCA_Square(restchannels=restchannels, lx=lx, ly=ly, density=0.4, bc='refl')  # , nodes=nodes)
    lgca.set_interaction(interaction='aggregation', beta=10)
    # ani = lgca.animate_flow(interval=50)
    # ani = lgca.animate_flux(interval=100)
    # ani = lgca.animate_density(interval=100)
    # ani = lgca.live_animate_flux()
    ani = lgca.live_animate_density()
    # lgca.plot_flux()
    # lgca.plot_density()
    # lgca.plot_config(grid=True)
    plt.show()
