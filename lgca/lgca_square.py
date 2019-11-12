import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            self.dims = self.lx, self.ly
            return

        elif dims is None:
            dims = (50, 50)

        try:
            self.lx, self.ly = dims
        except TypeError:
            self.lx, self.ly = dims, dims

        self.dims = self.lx, self.ly
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

    def calc_vorticity(self, nodes=None):
        if nodes is None:
            nodes = self.nodes
        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

        flux = self.calc_flux(nodes)
        # dens = nodes.sum(-1)
        # flux = np.divide(flux, dens[..., None], where=dens[..., None] > 0, out=np.zeros_like(flux))
        fx, fy = flux[..., 0], flux[..., 1]
        dfx = self.gradient(fx)
        dfy = self.gradient(fy)
        dfxdy = dfx[..., 1]
        dfydx = dfy[..., 0]
        vorticity = dfydx - dfxdy
        return vorticity

    def calc_velocity_correlation(self, nodes=None):
        if nodes is None:
            nodes = self.nodes
        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

        flux = self.calc_flux(nodes)
        flux_norm = np.linalg.norm(flux, axis=-1)
        nb_flux = self.nb_sum(flux)
        nb_flux_norm = np.linalg.norm(nb_flux, axis=-1)
        corr = np.einsum('...i, ...i', flux, nb_flux)
        corr = np.divide(corr, flux_norm, where=flux_norm > 1e-6, out=np.zeros_like(corr))
        corr = np.divide(corr, nb_flux_norm, where=nb_flux_norm > 1e-6, out=np.zeros_like(corr))
        return corr

    def setup_figure(self, figindex=None, figsize=(8, 8), tight_layout=True):
        dy = self.r_poly * np.cos(self.orientation)
        if figindex is None:
            fig = plt.gcf()
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        else:
            fig = plt.figure(num=figindex)
            fig.set_size_inches(figsize)
            fig.set_tight_layout(tight_layout)

        ax = plt.gca()
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

    def plot_config(self, nodes=None, figsize=None, grid=False, ec='none', **kwargs):
        r_circle = self.r_poly * 0.25
        # bbox_props = dict(boxstyle="Circle,pad=0.3", fc="white", ec="k", lw=1.5)
        bbox_props = None
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        occupied = nodes.astype('bool')
        density = occupied.sum(-1)
        if figsize is None:
            figsize = estimate_figsize(density, cbar=False, dy=self.dy)

        fig, ax = self.setup_figure(figsize=figsize, **kwargs)

        xx, yy = self.xcoords, self.ycoords
        x1, y1 = ax.transData.transform((0, 1.5 * r_circle))
        x2, y2 = ax.transData.transform((1.5 * r_circle, 0))
        dpx = np.mean([abs(x2 - x1), abs(y2 - y1)])
        fontsize = dpx * 72. / fig.dpi
        lw_circle = fontsize / 5
        lw_arrow = 0.5 * lw_circle

        colors = 'none', 'k'
        arrows = []
        for i in range(self.velocitychannels):
            cx = self.c[0, i] * 0.5
            cy = self.c[1, i] * 0.5
            arrows += [FancyArrowPatch((x, y), (x + cx, y + cy), mutation_scale=.3, fc=colors[occ], ec=ec, lw=lw_arrow)
                       for x, y, occ in zip(xx.ravel(), yy.ravel(), occupied[..., i].ravel())]

        arrows = PatchCollection(arrows, match_original=True)
        ax.add_collection(arrows)

        if self.restchannels > 0:
            circles = [Circle(xy=(x, y), radius=r_circle, fc='white', ec='k', lw=lw_circle * bool(n), visible=bool(n))
                       for x, y, n in
                       zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].sum(-1).ravel())]
            texts = [ax.text(x, y - 0.5 * r_circle, str(n), ha='center', va='baseline', fontsize=fontsize,
                             fontname='sans-serif', fontweight='bold', bbox=bbox_props, visible=bool(n))
                     for x, y, n in zip(xx.ravel(), yy.ravel(), occupied[..., self.velocitychannels:].sum(-1).ravel())]
            circles = PatchCollection(circles, match_original=True)
            ax.add_collection(circles)

        else:
            circles = []
            texts = []

        if grid:
            polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly, lw=lw_arrow,
                                       orientation=self.orientation, facecolor='None', edgecolor='k')
                        for x, y in zip(self.xcoords.ravel(), self.ycoords.ravel())]
            ax.add_collection(PatchCollection(polygons, match_original=True))

        else:
            ymin = -0.5 * self.c[1, 1]
            ymax = self.ycoords.max() + 0.5 * self.c[1, 1]
            plt.ylim(ymin, ymax)

        return fig, arrows, circles, texts

    def animate_config(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t

        fig, arrows, circles, texts = self.plot_config(nodes=nodes_t[0], **kwargs)
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

    def live_animate_config(self, interval=100, **kwargs):
        fig, arrows, circles, texts = self.plot_config(**kwargs)
        title = plt.title('Time $k =$0')
        nodes = self.nodes[self.nonborder]
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

    def live_animate_density(self, interval=100, **kwargs):

        fig, pc, cmap = self.plot_density(**kwargs)
        title = plt.title('Time $k =$0')

        def update(n):
            self.timestep()
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=cmap.to_rgba(self.cell_density[self.nonborder].ravel()))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval)
        return ani

    def plot_flow(self, nodes=None, figsize=None, cmap='viridis', vmax=None, cbar=False, **kwargs):

        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if vmax is None:
            K = self.K

        else:
            K = vmax

        nodes = nodes.astype(float)
        density = nodes.sum(-1)
        xx, yy = self.xcoords, self.ycoords
        jx, jy = np.moveaxis(self.calc_flux(nodes), -1, 0)
        # jx = np.ma.masked_where(density==0, jx)  # using masked arrays would also have been possible

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True)

        fig, ax = self.setup_figure(**kwargs)
        ax.set_aspect('equal')
        plot = plt.quiver(xx, yy, jx, jy, density.ravel(), pivot='mid', angles='xy', scale_units='xy', scale=1./self.r_poly)

        if cbar:
            plot.set_cmap(cmap)
            cmap = plot.get_cmap()
            plot.set_clim([1, K])
            cmap.set_under(alpha=0.0)
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
            mappable.set_array(np.arange(K))
            cbar = fig.colorbar(mappable, extend='min', use_gridspec=True)
            cbar.set_label('Particle number $n$')
            cbar.set_ticks(np.linspace(0., K + 1, 2 * K + 3, endpoint=True)[1::2])
            cbar.set_ticklabels(1 + np.arange(K))

        else:
            plot.set_cmap('Greys')
            plot.set_clim([0, 1])
            cmap = plot.get_cmap()
            cmap.set_under(alpha=0.0)

            # cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(1), cmap.N))
            # cmap.set_array(np.arange(1))

        # plot = plt.quiver(xx, yy, jx, jy, # color=cmap.to_rgba(density.ravel()),
        #                   pivot='mid', angles='xy', scale_units='xy', scale=1./self.r_poly)
        return fig, plot

    def animate_flow(self, nodes_t=None, interval=100, cbar=False, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t

        nodes = nodes_t.astype(float)
        density = nodes.sum(-1)
        jx, jy = np.moveaxis(self.calc_flux(nodes.astype(float)), -1, 0)

        fig, plot = self.plot_flow(nodes[0], cbar=cbar, **kwargs)
        title = plt.title('Time $k =$0')


        def update(n):
            title.set_text('Time $k =${}'.format(n))
            plot.set_UVC(jx[n], jy[n], density[n])
            return plot, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
        return ani

    def live_animate_flow(self, interval=100, **kwargs):
        fig, plot = self.plot_flow(**kwargs)
        title = plt.title('Time $k =$0')

        def update(n):
            self.timestep()
            jx, jy = np.moveaxis(self.calc_flux(self.nodes[self.nonborder]), -1, 0)
            density = self.cell_density[self.nonborder]
            title.set_text('Time $k =${}'.format(n))
            plot.set_UVC(jx, jy, density)
            return plot, title

        ani = animation.FuncAnimation(fig, update, interval=interval)
        return ani

    def plot_scalarfield(self, field, cmap='cividis', cbar=True, edgecolor='none', mask=None,
                         cbarlabel='Scalar field', vmin=None, vmax=None, **kwargs):
        fig, ax = self.setup_figure(**kwargs)
        try:
            assert field.shape == self.dims

        except AssertionError:
            field = field[self.nonborder]

        if mask is None:
            mask = np.ones_like(field, dtype=bool)
        cmap = plt.cm.get_cmap(cmap)
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly, alpha=v,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c, v in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(field.ravel()), mask.ravel())]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(cmap, cax=cax, use_gridspec=True)
            cbar.set_label(cbarlabel)
            plt.sca(ax)

        return fig, pc, cmap

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
        polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(density.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(cmap, extend='min', use_gridspec=True, cax=cax)
            cbar.set_label('Particle number $n$')
            cbar.set_ticks(np.linspace(0., K + 1, 2 * K + 3, endpoint=True)[1::2])
            cbar.set_ticklabels(1 + np.arange(K))
            plt.sca(ax)

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
            nodes = self.nodes[self.nonborder]

        if nodes.dtype != 'bool':
            nodes = nodes.astype('bool')

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
        angle[..., -1] = np.sign(density)  # np.sqrt(density)
        angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.5
        polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c,
                                   edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), angle.reshape(-1, 4))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(cmap, use_gridspec=True, cax=cax)
            cbar.set_label('Direction of movement $(\degree)$')
            cbar.set_ticks(np.arange(self.velocitychannels) * 360 / self.velocitychannels)
            plt.sca(ax)

        return fig, pc, cmap

    def animate_density(self, density_t=None, interval=100, **kwargs):
        if density_t is None:
            density_t = self.dens_t

        fig, pc, cmap = self.plot_density(density_t[0], **kwargs)
        title = plt.title('Time $k =$0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=cmap.to_rgba(density_t[n, ...].ravel()))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=density_t.shape[0])
        return ani

    def animate_flux(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t

        nodes = nodes_t.astype(float)
        density = nodes.sum(-1) / self.K
        jx, jy = np.moveaxis(self.calc_flux(nodes), -1, 0)

        angle = np.zeros(density.shape, dtype=complex)
        angle.real = jx
        angle.imag = jy
        angle = np.angle(angle, deg=True) % 360.
        fig, pc, cmap = self.plot_flux(nodes=nodes[0], **kwargs)
        angle = cmap.to_rgba(angle[None, ...])[0]
        angle[..., -1] = np.sign(density)  # np.sqrt(density)
        angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.5
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
            jx, jy = np.moveaxis(self.calc_flux(self.nodes[self.nonborder]), -1, 0)
            density = self.cell_density[self.nonborder] / self.K

            angle = np.empty(density.shape, dtype=complex)
            angle.real = jx
            angle.imag = jy
            angle = np.angle(angle, deg=True) % 360.
            angle = cmap.to_rgba(angle)
            angle[..., -1] = np.sign(density)  # np.sqrt(density)
            angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.5
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=angle.reshape(-1, 4))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval)
        return ani


class IBLGCA_Square(IBLGCA_base, LGCA_Square):
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

    def plot_prop_spatial(self, nodes=None, props=None, propname=None, **kwargs):
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if props is None:
            props = self.props

        if propname is None:
            propname = list(props)[0]

        lx, ly, _ = nodes.shape
        mask = np.any(nodes, axis=-1)
        meanprop = self.calc_prop_mean(propname=propname, props=props, nodes=nodes)
        fig, pc, cmap = self.plot_scalarfield(meanprop, mask=mask, **kwargs)
        return fig, pc, cmap


if __name__ == '__main__':
    lx = 100
    ly = lx
    restchannels = 4
    nodes = np.zeros((lx, ly, 4 + restchannels))
    nodes[lx // 2, ly // 2, :] = 1
    lgca = IBLGCA_Square(restchannels=restchannels, lx=lx, ly=ly, bc='refl', nodes=nodes,
                         interaction='go_and_grow', r_b=0.1, std=0.1)
    lgca.timeevo(100, record=True)
    lgca.plot_prop_spatial(propname='r_b', cbarlabel='$r_b$')
    # print(lgca.cell_density[lgca.nonborder])
    # lgca.set_interaction(interaction='go_and_grow')
    # ani = lgca.animate_flow(interval=50)
    # ani = lgca.animate_flux(interval=100)
    # ani = lgca.animate_density(interval=100)
    # ani = lgca.live_animate_flux()
    # ani = lgca.live_animate_flux()
    # lgca.plot_flux()
    # lgca.plot_density()
    # lgca.plot_config(grid=True)
    plt.show()
