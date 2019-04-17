import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch

from interactions import *


class LGCA_Square(LGCA_base):
    """
    2D version of a LGCA on the square lattice.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation',
                    'random_walk', 'excitable_medium', 'nematic', 'persistant_motion']
    velocitychannels = 4
    cix = np.array([1, 0, -1, 0], dtype=float)
    ciy = np.array([0, 1, 0, -1], dtype=float)
    c = np.array([cix, ciy])
    r_poly = 0.5 / np.cos(np.pi / velocitychannels)
    dy = np.sin(2 * np.pi / velocitychannels)
    orientation = np.pi / velocitychannels

    def __init__(self, nodes=None, lx=50, ly=50, restchannels=1, density=0.1, bc='periodic', **kwargs):
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
        self.r_int = 1  # interaction range; must be at least 1 to handle propagation.
        if nodes is None:
            self.lx = lx
            self.ly = ly
            self.restchannels = restchannels
            self.K = self.velocitychannels + self.restchannels

        if nodes is not None:
            self.lx, self.ly, self.K = nodes.shape
            self.restchannels = self.K - self.velocitychannels

        self.set_interaction(**kwargs)
        self.init_nodes(density, nodes=nodes)
        self.set_bc(bc)
        self.init_coords()
        self.cell_density = self.nodes.sum(-1)

    def init_nodes(self, density, nodes=None):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.bool)
        if nodes is None:
            self.random_reset(density)

        else:
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(np.bool)

    def init_coords(self):
        self.x = np.arange(self.lx) + self.r_int
        self.y = np.arange(self.ly) + self.r_int
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.coord_pairs = list(zip(self.xx.flat, self.yy.flat))
        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords[self.r_int:-self.r_int, self.r_int:-self.r_int].astype(float)
        self.ycoords = self.ycoords[self.r_int:-self.r_int, self.r_int:-self.r_int].astype(float)

    def set_bc(self, bc):
        if bc in ['absorbing', 'absorb', 'abs']:
            self.apply_boundaries = self.apply_abc
        elif bc in ['reflecting', 'reflect', 'refl']:
            self.apply_boundaries = self.apply_rbc
        elif bc in ['periodic', 'pbc']:
            self.apply_boundaries = self.apply_pbc
        else:
            print(bc, 'not defined, using periodic boundaries')
            self.apply_boundaries = self.apply_pbc

        self.apply_boundaries()

    def set_interaction(self, **kwargs):
        if 'interaction' in kwargs:
            interaction = kwargs['interaction']
            if interaction == 'go_or_grow':
                self.interaction = go_or_grow_interaction
                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.01
                    print('death rate set to r_d = ', self.r_d)
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)
                if 'kappa' in kwargs:
                    self.kappa = kwargs['kappa']
                else:
                    self.kappa = 5.
                    print('switch rate set to kappa = ', self.kappa)
                if 'theta' in kwargs:
                    self.theta = kwargs['theta']
                else:
                    self.theta = 0.75
                    print('switch threshold set to theta = ', self.theta)
                if self.restchannels < 2:
                    print('WARNING: not enough rest channels - system will die out!!!')

            elif interaction == 'go_and_grow':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

            elif interaction == 'alignment':
                self.interaction = alignment
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'persistent_motion':
                self.interaction = persistent_walk
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'chemotaxis':
                self.interaction = chemotaxis
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 5.
                    print('sensitivity set to beta = ', self.beta)

                if 'gradient' in kwargs:
                    self.g = kwargs['gradient']
                else:
                    x_source = npr.normal(self.xcoords.mean(), 1)
                    y_source = npr.normal(self.ycoords.mean(), 1)
                    rx = self.xcoords - x_source
                    ry = self.ycoords - y_source
                    r = np.sqrt(rx ** 2 + ry ** 2)
                    self.concentration = np.exp(-r / self.ycoords.var())
                    self.g = self.gradient(np.pad(self.concentration, 1, 'constant'))

            elif interaction == 'contact_guidance':
                self.interaction = contact_guidance
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

                if 'director' in kwargs:
                    self.g = kwargs['director']
                else:
                    self.g = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, 2))
                    self.g[..., 0] = 1
                    self.guiding_tensor = calc_nematic_tensor(self.g)

            elif interaction == 'nematic':
                self.interaction = nematic
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'aggregation':
                self.interaction = aggregation
                self.calc_permutations()

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('sensitivity set to beta = ', self.beta)

            elif interaction == 'wetting':
                self.interaction = wetting
                self.calc_permutations()
                self.r_int = 2

                if 'beta' in kwargs:
                    self.beta = kwargs['beta']
                else:
                    self.beta = 2.
                    print('adhesion sensitivity set to beta = ', self.beta)

                if 'gamma' in kwargs:
                    self.gamma = kwargs['gamma']
                else:
                    self.gamma = 2.
                    print('alignment sensitivity set to gamma = ', self.gamma)

                if 'alpha' in kwargs:
                    self.alpha = kwargs['alpha']
                else:
                    self.alpha = 2.
                    print('substrate sensitivity set to alpha = ', self.alpha)

            elif interaction == 'random_walk':
                self.interaction = random_walk

            elif interaction == 'birth':
                self.interaction = birth
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

            elif interaction == 'birth_death':
                self.interaction = birth_death
                if 'r_b' in kwargs:
                    self.r_b = kwargs['r_b']
                else:
                    self.r_b = 0.2
                    print('birth rate set to r_b = ', self.r_b)

                if 'r_d' in kwargs:
                    self.r_d = kwargs['r_d']
                else:
                    self.r_d = 0.05
                    print('death rate set to r_d = ', self.r_d)

            elif interaction == 'excitable_medium':
                self.interaction = excitable_medium
                if 'beta' in kwargs:
                    self.beta = kwargs['beta']

                else:
                    self.beta = .05
                    print('alignment sensitivity set to beta = ', self.beta)

                if 'alpha' in kwargs:
                    self.alpha = kwargs['alpha']
                else:
                    self.alpha = 1.
                    print('aggregation sensitivity set to alpha = ', self.alpha)

                if 'N' in kwargs:
                    self.N = kwargs['N']
                else:
                    self.N = 50
                    print('repetition of fast reaction set to N = ', self.N)

            else:
                print('interaction', kwargs['interaction'], 'is not defined! Random walk used instead.')
                print('Implemented interactions:', self.interactions)
                self.interaction = random_walk

        else:
            print('Random walk interaction is used.')
            self.interaction = random_walk

    def propagation(self):
        """

        :param nodes:
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

    def apply_pbc(self):
        self.nodes[:self.r_int, ...] = self.nodes[-2 * self.r_int:-self.r_int, ...]  # left boundary
        self.nodes[-self.r_int:, ...] = self.nodes[self.r_int:2 * self.r_int, ...]  # right boundary
        self.nodes[:, :self.r_int, :] = self.nodes[:, -2 * self.r_int:-self.r_int, :]  # upper boundary
        self.nodes[:, -self.r_int:, :] = self.nodes[:, self.r_int:2 * self.r_int, :]  # lower boundary

    def apply_rbc(self):
        self.nodes[self.r_int, :, 0] += self.nodes[self.r_int - 1, :, 2]
        self.nodes[-self.r_int - 1, :, 2] += self.nodes[-self.r_int, :, 0]
        self.nodes[:, self.r_int, 1] += self.nodes[:, self.r_int - 1, 3]
        self.nodes[:, -self.r_int - 1, 3] += self.nodes[:, -self.r_int, 1]
        self.apply_abc()

    def apply_abc(self):
        self.nodes[:self.r_int, ...] = 0
        self.nodes[-self.r_int:, ...] = 0
        self.nodes[:, :self.r_int, :] = 0
        self.nodes[:, -self.r_int:, :] = 0

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

    def calc_flux(self, nodes):
        return np.einsum('ij,...j', self.c, nodes[..., :self.velocitychannels])

    def calc_vorticity(self, nodes):
        flux = self.calc_flux(nodes)
        dens = nodes.sum(-1)
        flux = np.divide(flux, dens[..., None], where=dens > 0, out=np.zeros_like(flux))
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
        plt.xlabel('$x \\; [\\varepsilon]$')
        plt.ylabel('$y \\; [\\varepsilon]$')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_autoscale_on(False)
        return fig, ax

    def plot_config(self, nodes=None, figindex=None, figsize=None, tight_layout=True, grid=False):
        r_circle = self.r_poly * 0.25
        # bbox_props = dict(boxstyle="Circle,pad=0.3", fc="white", ec="k", lw=1.5)
        bbox_props = None
        if nodes is None:
            nodes = self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int]

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
        arrows = []
        for i in range(self.velocitychannels):
            cx = self.c[0, i] * 0.5
            cy = self.c[1, i] * 0.5
            arrows += [FancyArrowPatch((x, y), (x + cx, y + cy), alpha=occ, mutation_scale=.3, fc='k', ec='none')
                       for x, y, occ in zip(xx.ravel(), yy.ravel(), nodes[..., i].ravel())]

        arrows = PatchCollection(arrows, match_original=True)
        ax.add_collection(arrows)

        if self.restchannels > 0:
            circles = [Circle(xy=(x, y), radius=r_circle, fc='none', ec='k', lw=lw_circle, alpha=occ)
                       for x, y, occ in
                       zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].any(axis=-1).ravel())]
            texts = [ax.text(x, y - 0.5 * r_circle, str(n), ha='center', va='baseline', fontsize=fontsize,
                             fontname='sans-serif', fontweight='bold', bbox=bbox_props, alpha=bool(n))
                     for x, y, n in zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].sum(-1).ravel())]
            circles = PatchCollection(circles, match_original=True)
            ax.add_collection(circles)

        else:
            circles = []
            texts = []

        if grid:
            hexagons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
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
                             channels=slice(None),
                             tight_layout=True, edgecolor='None'):

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
                     edgecolor='None'):
        if density is None:
            density = self.cell_density[self.r_int:-self.r_int, self.r_int:-self.r_int]

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True, dy=self.dy)

        if vmax is None:
            K = self.K

        else:
            K = vmax

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_under(alpha=0.0)
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
        cmap.set_array(density)
        hexagons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(density.ravel()))]
        pc = PatchCollection(hexagons, match_original=True)
        ax.add_collection(pc)
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

    def plot_flux(self, nodes=None, figindex=None, figsize=None, tight_layout=True, edgecolor='None'):
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
        cbar = fig.colorbar(cmap, use_gridspec=True)
        cbar.set_label('Particle flux $\\arg \\left( \\vec{J} \\right)$ $(\degree)$')
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
                     edgecolor='None'):
        if nodes_t is None:
            nodes_t = self.nodes_t

        nodes = nodes_t.astype(float)
        density = nodes.sum(-1) / self.K
        jx, jy = self.calc_flux(nodes)

        angle = np.zeros(density.shape, dtype=complex)
        angle.real = jx
        angle.imag = jy
        angle = np.angle(angle, deg=True) % 360.

        fig, pc, cmap = self.plot_flux(nodes=nodes[0], figindex=figindex, figsize=figsize, tight_layout=tight_layout,
                                       edgecolor=edgecolor)
        angle = cmap.to_rgba(angle)
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

        cmap = plt.cm.get_cmap('hsv')
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
    restchannels = 0
    nodes = np.zeros((lx, ly, 4 + restchannels))
    nodes[..., (0, 1)] = 1
    lgca = LGCA_Square(restchannels=restchannels, lx=lx, ly=ly, density=0.1)  # , nodes=nodes)

    lgca.set_interaction(interaction='alignment', beta=3)
    # ani = lgca.animate_flow(interval=50)
    # ani = lgca.animate_flux(interval=100)
    # ani = lgca.animate_density(interval=100)
    ani = lgca.live_animate_flux()
    # lgca.plot_flux()
    # lgca.plot_density()
    # lgca.plot_config(grid=True)
    plt.show()
