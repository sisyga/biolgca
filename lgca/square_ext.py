# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Extended 2D square LGCA models.

Includes identity-based and no-volume-exclusion simulators.
"""


from __future__ import annotations

import numpy as np
from lgca.ib_base import IBLGCA_base
from lgca.list_utils import get_arr_of_empty_lists
from lgca.nove_base import NoVE_LGCA_base
from lgca.nove_ib_base import NoVE_IBLGCA_base
from lgca.lgca_square import LGCA_Square
try:  # optional plotting dependencies
    import matplotlib.animation as animation
    import matplotlib.colors as colors
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:  # pragma: no cover - handled at runtime
    from lgca.base import _MissingPlotLib  # reuse stub

    animation = colors = mticker = FuncFormatter = PatchCollection = Normalize = (
        RegularPolygon
    ) = Circle = FancyArrowPatch = cm = make_axes_locatable = _MissingPlotLib(
        "matplotlib"
    )



class IBLGCA_Square(IBLGCA_base, LGCA_Square):
    """
    Identity-based LGCA simulator class.
    """
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath', 'birthdeath_discrete',
                    'only_propagation', 'go_and_grow_mutations']

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)

        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.apply_boundaries()

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

    def plot_density(self, density=None, channels=slice(None), **kwargs):
        # needs to be overridden because of the sum of channels if no density is provided
        if density is None:
            nodes = self.nodes[self.nonborder].astype('bool')
            density = nodes[..., channels].sum(-1)

        fig, pc, cmap = LGCA_Square.plot_density(self, density=density, channels=channels, **kwargs)
        return fig, pc, cmap

    def animate_config(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t.astype(bool)

        return super().animate_config(nodes_t=nodes_t, interval=interval, **kwargs)

    def plot_config(self, nodes=None, **kwargs):
        if nodes is None:
            nodes = self.occupied[self.nonborder]

        return super().plot_config(nodes=nodes, **kwargs)

    def live_animate_config(self, interval=100, **kwargs):
        fig, arrows, circles, texts = self.plot_config(**kwargs)
        title = plt.title('Time $k =$0')
        nodes = self.occupied[self.nonborder]
        if self.restchannels:
            def update(n):
                self.timestep()
                nodes = self.occupied[self.nonborder]
                arrow_color = np.moveaxis(nodes[..., :self.velocitychannels], -1, 0).ravel().astype(float)
                circle_color = np.any(nodes[..., self.velocitychannels:], axis=-1).ravel().astype(float)
                resting_t = nodes[..., self.velocitychannels:].sum(-1).ravel()
                title.set_text('Time $k =${}'.format(n))
                arrows.set(alpha=arrow_color)
                circles.set(alpha=circle_color)
                for text, i in zip(texts, resting_t):
                    text.set_text(str(i))
                    text.set(alpha=bool(i))
                return arrows, circles, texts, title

            ani = animation.FuncAnimation(fig, update, interval=interval)
            return ani

        else:
            def update(n):
                self.timestep()
                nodes = self.occupied[self.nonborder]
                arrow_color = np.moveaxis(nodes[..., :self.velocitychannels], -1, 0).ravel().astype(float)
                title.set_text('Time $k =${}'.format(n))
                arrows.set(alpha=arrow_color)
                return arrows, title

            ani = animation.FuncAnimation(fig, update, interval=interval)
            return ani


class NoVE_LGCA_Square(LGCA_Square, NoVE_LGCA_base):
    """
    2D square version of an LGCA without volume exclusion.
    """
    interactions = ['dd_alignment', 'di_alignment', 'go_or_grow', 'go_or_rest']

    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None):
        """
        Set the dimensions of the instance according to given values. Sets self.l, self.K, self.dims and self.restchannels
        :param dims: desired lattice size (int or array-like)
        :param nodes: existing lattice to use (ndarray)
        :param restchannels: desired number of resting channels, will be capped to 1 if >1 because of no volume exclusion
        :param capacity: reference value for density calculation. If number of cells = capacity, density = 1.0
        """
        # set instance dimensions according to passed lattice
        if nodes is not None:
            try:
                self.lx, self.ly, self.K = nodes.shape
            except ValueError as e:
                raise ValueError("Node shape does not match the 2D geometry! Shape must be (x,y,channels)") from e
            # set number of rest channels to <= 1 because >1 cells are allowed per channel
                # for now, raise Exception if format of nodes does no fit
                # (To Do: just sum the cells in surplus rest channels in init_nodes and print a warning)
            if self.K - self.velocitychannels > 1:
                raise RuntimeError('Only one resting channel allowed, but {} resting channels specified!'.format(self.K - self.velocitychannels))
            elif self.K < self.velocitychannels:
                raise RuntimeError('Not enough channels specified for the chosen geometry! Required: {}, provided: {}'.format(
                    self.velocitychannels, self.K))
            else:
                self.restchannels = self.K - self.velocitychannels
        # set instance dimensions according to required dimensions
        elif dims is not None:
            if isinstance(dims, tuple):
                if len(dims) == 2:
                    self.lx, self.ly = dims
                elif len(dims) > 2:
                    self.lx, self.ly = dims[0], dims[1]
                    print("Dimensions provided with too many values! " + str(dims))
                else:
                    self.lx, self.ly = dims[0], dims[0]
                    print("Dimensions provided as tuple " + str(dims) + ", but only one value for 2D lattice!")
            elif isinstance(dims, int):
                self.lx, self.ly = dims, dims
            else:
                self.lx, self.ly = (50, 50)
                print("Dimensions provided in wrong format, must be tuple of 2 elements or integer. Dimensions set to default 50x50.")
        # set default for dimension
        else:
            self.lx, self.ly = (50, 50)
            print("Dimensions set to default 50x50.")
        self.dims = self.lx, self.ly

        # set number of rest channels to <= 1 because >1 cells are allowed per channel
        if nodes is None and restchannels is not None:
            if restchannels > 1:
                self.restchannels = 1
            elif 0 <= restchannels <= 1:
                self.restchannels = restchannels
        elif nodes is None:
            self.restchannels = 0
        self.K = self.velocitychannels + self.restchannels

        # set capacity according to keyword or specified resting channels
        if capacity is not None:
            self.capacity = capacity
        elif restchannels is not None and restchannels > 1:
            self.capacity = self.velocitychannels + restchannels
        else:
            self.capacity = self.K

    def init_nodes(self, density=4, nodes=None):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(np.uint)
            self.apply_boundaries()

    def plot_density(self, density=None, figindex=None, figsize=None, tight_layout=True, cmap='viridis', vmax=None,
                     edgecolor='None', cbar=True, cbarlabel='Particle number $n$', channels=slice(None)):

        if density is None:
            nodes = self.nodes[self.nonborder]
            density = nodes[..., channels].sum(-1)

        if figsize is None:
            figsize = estimate_figsize(density, cbar=cbar, dy=self.dy)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)


        cmap = get_cmap(density, ax=ax, cmap=cmap, cbarlabel=cbarlabel, cbar=cbar)


        polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(density.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)

        return fig, pc, cmap

    def animate_flux(self, nodes_t=None, figindex=None, figsize=None, interval=200, tight_layout=True,
                     edgecolor='None', cbar=True):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call mylgca.timeevo with keyword record=True")

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
        angle[..., -1] = np.sign(density)

        angle[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.
        title = plt.title('Time $k =$ 0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=angle[n, ...].reshape(-1, 4))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
        return ani

    def animate_density(self, density_t=None, figindex=None, figsize=None, cmap='viridis', interval=200, vmax=None,
                        tight_layout=True, edgecolor='None'):
        if density_t is None:
            if hasattr(self, 'dens_t'):
                density_t = self.dens_t
            else:
                raise RuntimeError("Node-wise state of the lattice required for density plotting but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword recorddens=True")

        if vmax is not None:
            vmax_val = vmax
        else:
            vmax_val = int(density_t.max())

        fig, pc, cmap = self.plot_density(density_t[0], figindex=figindex, figsize=figsize, cmap=cmap, vmax=vmax_val,
                                          tight_layout=tight_layout, edgecolor=edgecolor)
        title = plt.title('Time $k =$0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=cmap.to_rgba(density_t[n, ...].ravel()))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=density_t.shape[0])
        return ani

    def live_animate_density(self, interval=100, channels=slice(None), **kwargs):
        # colourbar update is an issue
        warnings.warn("Live density animation not available for LGCA without volume exclusion yet.")

    def plot_config(self, nodes=None, figsize=None, grid=False, ec='none', rel_arrowlen=0.6, cmap='viridis', cbar=True,
                    cbarlabel='Particle number $n$', vmax=None, **kwargs):
        r_circle = self.r_poly * 0.25
        # bbox_props = dict(boxstyle="Circle,pad=0.3", fc="white", ec="k", lw=1.5)
        bbox_props = None
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        density = nodes.sum(-1)
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

        # colors = 'none', 'k'
        vmax = nodes.max() if vmax is None else vmax
        cmap = get_cmap(density, ax=ax, vmax=vmax, cmap=cmap, cbar=cbar, cbarlabel=cbarlabel)
        arrows = []
        for i in range(self.velocitychannels):
            cx = self.c[0, i] * 0.5
            cy = self.c[1, i] * 0.5
            arrows += [FancyArrowPatch((x + cx * (1 - rel_arrowlen), y + cy * (1 - rel_arrowlen)), (x + cx, y + cy),
                                       mutation_scale=.3, fc=c, ec=ec, lw=lw_arrow)
                       for x, y, c in zip(xx.ravel(), yy.ravel(), cmap.to_rgba(nodes[..., i].ravel()))]

        arrows = PatchCollection(arrows, match_original=True)
        ax.add_collection(arrows)

        if self.restchannels > 0:
            circles = [Circle(xy=(x, y), radius=r_circle, fc=c, ec='k', lw=0, fill=True) for x, y, c in
                       zip(xx.ravel(), yy.ravel(), cmap.to_rgba(nodes[..., self.velocitychannels:].sum(-1).ravel()))]
            circles = PatchCollection(circles, match_original=True)
            ax.add_collection(circles)

        else:
            circles = []

        if grid:
            polygons = [
                RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly, lw=lw_arrow,
                               orientation=self.orientation, facecolor='None', edgecolor='k')
                for x, y in zip(self.xcoords.ravel(), self.ycoords.ravel())]
            ax.add_collection(PatchCollection(polygons, match_original=True))

        else:
            ymin = -0.5 * self.c[1, 1]
            ymax = self.ycoords.max() + 0.5 * self.c[1, 1]
            plt.ylim(ymin, ymax)

        return fig, arrows, circles, cmap

    def animate_config(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError(
                    "Channel-wise state of the lattice required for plotting the configuration but not " +
                    "recorded in past LGCA run, call lgca.timeevo with keyword record=True")

        tmax = nodes_t.shape[0]
        fig, arrows, circles, cmap = self.plot_config(nodes=nodes_t[0], vmax=nodes_t.max(), **kwargs)
        title = plt.title('Time $k =$0')
        arrow_color = cmap.to_rgba(np.moveaxis(nodes_t[..., :self.velocitychannels], -1, 1)[None, ...]).reshape(tmax, -1, 4)

        if self.restchannels:
            circle_color = cmap.to_rgba(nodes_t[..., self.velocitychannels:].sum(-1)[None, ...]).reshape(tmax, -1, 4)

            def update(n):
                title.set_text('Time $k =${}'.format(n))
                arrows.set(color=arrow_color[n])
                circles.set(facecolor=circle_color[n])
                return arrows, circles, title

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
        warnings.warn("Live config animation not available for LGCA without volume exclusion yet.")


class NoVE_IBLGCA_Square(NoVE_IBLGCA_base, NoVE_LGCA_Square):
    """Identity-based lgca without volume exclusion on the square lattice.
    """
    def init_nodes(self, density=0.1, nodes=None):
        self.nodes = get_arr_of_empty_lists((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K))
        if nodes is None:
            self.random_reset(density)

        elif nodes.dtype == object:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes

        else:
            self._warn_nodes_shape(nodes)
            occ = nodes.astype(int)
            self.nodes[self.nonborder] = self.convert_int_to_ib(occ)

        self.calc_max_label()

    def propagation(self):
        """

        :return:
        """
        newnodes = get_arr_of_empty_lists(self.nodes.shape)
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

    def _apply_rbcx(self):
        self.nodes[self.r_int, :, 0] = self.nodes[self.r_int, :, 0] + self.nodes[self.r_int - 1, :, 2]
        self.nodes[-self.r_int - 1, :, 2] = self.nodes[-self.r_int - 1, :, 2] + self.nodes[-self.r_int, :, 0]
        self._apply_abcx()

    def _apply_rbcy(self):
        self.nodes[:, self.r_int, 1] = self.nodes[:, self.r_int, 1] + self.nodes[:, self.r_int - 1, 3]
        self.nodes[:, -self.r_int - 1, 3] = self.nodes[:, -self.r_int - 1, 3] + self.nodes[:, -self.r_int, 1]
        self._apply_abcy()

    def _apply_abcx(self):
        self.nodes[:self.r_int, ...] = get_arr_of_empty_lists(self.nodes[:self.r_int, ...].shape)
        self.nodes[-self.r_int:, ...] = get_arr_of_empty_lists(self.nodes[-self.r_int:, ...].shape)

    def _apply_abcy(self):
        self.nodes[:, :self.r_int, :] = get_arr_of_empty_lists(self.nodes[:, :self.r_int, :].shape)
        self.nodes[:, -self.r_int:, :] = get_arr_of_empty_lists(self.nodes[:, -self.r_int:, :].shape)

    def plot_density(self, density=None, channels=slice(None), **kwargs):
        if density is None:
            nodes = self.nodes[self.nonborder]
            density = self.length_checker(nodes[..., channels].sum(-1))

        return super().plot_density(density=density, **kwargs)

    def plot_flux(self, nodes=None, **kwargs):
        if nodes is None:
            if hasattr(self, 'nodes'):
                nodes = self.length_checker(self.nodes[self.nonborder])
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call mylgca.timeevo with keyword record=True")

        return super().plot_flux(nodes=nodes, **kwargs)

    def plot_config(self, nodes=None, **kwargs):
        if nodes is None:
            if hasattr(self, 'nodes'):
                nodes = self.length_checker(self.nodes[self.nonborder])
            else:
                raise RuntimeError("Channel-wise state of the lattice required for config calculation but not recorded " +
                                   "in past LGCA run, call mylgca.timeevo with keyword record=True")

        return super().plot_config(nodes=nodes, **kwargs)

    def animate_config(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes = self.length_checker(self.nodes_t)
            else:
                raise RuntimeError("Channel-wise state of the lattice required for config calculation but not recorded " +
                                   "in past LGCA run, call mylgca.timeevo with keyword record=True")

        return super().animate_config(nodes_t=nodes, **kwargs)

    def animate_flux(self, nodes_t=None, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes = self.length_checker(self.nodes_t)
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call mylgca.timeevo with keyword record=True")

        return super().animate_flux(nodes_t=nodes, **kwargs)

    def plot_prop_spatial(self, nodes=None, props=None, propname=None, **kwargs):
        if nodes is None:
            nodes = self.nodes[self.nonborder]
        if props is None:
            props = self.props
        if propname is None:
            propname = next(iter(props))

        if self.mean_prop_t == {}:
            self.calc_prop_mean_spatiotemp()

        mean_prop = self.mean_prop_t[propname][-1]
        if 'cbarlabel' not in kwargs:
            kwargs.update({'cbarlabel': str(propname)})

        return super().plot_scalarfield(mean_prop, **kwargs)


