# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from copy import copy

from lgca.base import *

class LGCA_Square(LGCA_base):
    """
    2D version of a LGCA on the square lattice.
    """
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation',
                    'random_walk', 'excitable_medium', 'nematic', 'persistant_motion', 'chemotaxis', 'contact_guidance',
                    'only_propagation']
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

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=bool)
        if 'hom' in kwargs:
            hom = kwargs['hom']
        else:
            hom = None
        if nodes is None and hom:
            self.homogeneous_random_reset(density)
        elif nodes is None:
            self.random_reset(density)
        else:
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(bool)
            self.apply_boundaries()

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
        Boundary condition for an inflow from x=0, y=:, with reflecting boundary conditions along the y-axis and periodic
        boundaries along the x-axis. Nodes at (x=0, y) are set to a homogeneous state with a constant average density
        given by the attribute 0 <= self.inflow <= 1.
        If there is no such attribute, the nodes are filled with the maximum density.
        :return:
        """
        self.apply_rbcx()

        if hasattr(self, 'inflow'):
            self.nodes[self.r_int, ...] = npr.random(self.nodes[0].shape) < self.inflow
        else:
            self.nodes[self.r_int, ...] = 1

        self.apply_pbcy()

    def nb_sum(self, qty):
        sum = np.zeros(qty.shape)
        sum[:-1, ...] += qty[1:, ...]
        sum[1:, ...] += qty[:-1, ...]
        sum[:, :-1, ...] += qty[:, 1:, ...]
        sum[:, 1:, ...] += qty[:, :-1, ...]
        return sum

    def gradient(self, qty):
        return np.moveaxis(np.asarray(np.gradient(qty, 0.5)), 0, -1)

    def channel_weight(self, qty):
        """
        Return an array of shape qty.shape + velocitychannels. 'qty' should be scalar field with the same dimension
        as the lgca lattice, so that the returned array assigns a weight to each velocity channel that is equal to the
        value of 'qty' in the neighboring node corresponding to the velocity channel.
        :param qty:
        :return:
        """
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, :, 0] = qty[1:, ...]
        weights[1:, :, 2] = qty[:-1, ...]
        weights[:, :-1, 1] = qty[:, 1:, ...]
        weights[:, 1:, 3] = qty[:, :-1, ...]

        return weights

    def calc_vorticity(self, nodes=None):
        """
        Calculate the vorticity of the flow field corresponding to the lgca state 'nodes'. The vorticity is used to
        characterize rotations in a field. For more, see https://en.wikipedia.org/wiki/Vorticity
        :param nodes: lgca state. by default it is the current state of the lgca class instance
        :return:
        """
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
        """
        Calculate the correlation between the node fluxes and the mean node flux in the neighborhood. Used to quantify
        correlated movement.
        :param nodes:
        :return:
        """
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
        if self.dy >= 1:
            minstep = self.dy
            integer = True
        else:
            minstep = 1
            integer = False
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[minstep, 2*self.dy, 5*self.dy, 10*self.dy], integer=integer))
        ax.yaxis.set_major_formatter(lambda x, pos: (int(x/self.dy)))
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_autoscale_on(False)
        return fig, ax

    def plot_config(self, nodes=None, figsize=None, grid=False, ec='none', rel_arrowlen=0.6, **kwargs):
        r_circle = self.r_poly * 0.25
        # bbox_props = dict(boxstyle="Circle,pad=0.3", fc="white", ec="k", lw=1.5)
        bbox_props = None
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if figsize is None:
            figsize = estimate_figsize(nodes[..., -1], cbar=False, dy=self.dy)

        fig, ax = self.setup_figure(figsize=figsize, **kwargs)

        xx, yy = self.xcoords, self.ycoords
        x1, y1 = ax.transData.transform((0, 1.5 * r_circle))
        x2, y2 = ax.transData.transform((1.5 * r_circle, 0))
        dpx = np.mean([abs(x2 - x1), abs(y2 - y1)])
        fontsize = dpx * 72. / fig.dpi
        lw_circle = fontsize / 5
        lw_arrow = 0.5 * lw_circle

        arrows = []
        for i in range(self.velocitychannels):
            cx = self.c[0, i] * 0.5
            cy = self.c[1, i] * 0.5
            arrows += [FancyArrowPatch((x + cx*(1-rel_arrowlen), y + cy*(1-rel_arrowlen)), (x + cx, y + cy),
                                       mutation_scale=.3, fc='k', ec=ec, lw=lw_arrow, alpha=occ)
                       for x, y, occ in zip(xx.ravel(), yy.ravel(), nodes[..., i].astype(float).ravel())]

        arrows = PatchCollection(arrows, match_original=True)
        ax.add_collection(arrows)

        if self.restchannels > 0:
            circles = [Circle(xy=(x, y), radius=r_circle, fc='white', ec='k', lw=lw_circle, fill=True, alpha=occ)
                       for x, y, occ in
                       zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].sum(-1).ravel().astype(bool).astype(float))]
            texts = [ax.text(x, y - 0.5 * r_circle, str(n), ha='center', va='baseline', fontsize=fontsize,
                             fontname='sans-serif', fontweight='bold', bbox=bbox_props, alpha=float(bool(n)))
                     for x, y, n in zip(xx.ravel(), yy.ravel(), nodes[..., self.velocitychannels:].sum(-1).ravel())]
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
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError("Channel-wise state of the lattice required for plotting the configuration but not "+
                                   "recorded in past LGCA run, call lgca.timeevo with keyword record=True")

        fig, arrows, circles, texts = self.plot_config(nodes=nodes_t[0], **kwargs)
        title = plt.title('Time $k =$0')
        arrow_color = np.zeros(nodes_t[..., :self.velocitychannels].shape)
        arrow_color = arrow_color.reshape(nodes_t.shape[0], -1)
        arrow_color = np.moveaxis(nodes_t[..., :self.velocitychannels], -1, 1).reshape(nodes_t.shape[0], -1)

        if self.restchannels:
            circle_color = np.zeros(nodes_t[..., 0].shape)
            circle_color = circle_color.reshape(nodes_t.shape[0], -1)
            circle_color = np.any(nodes_t[..., self.velocitychannels:], axis=-1).reshape(nodes_t.shape[0], -1).astype(float)
            # circle_fcolor = np.ones(circle_color.shape)
            # circle_fcolor[..., -1] = circle_color[..., -1]
            resting_t = nodes_t[..., self.velocitychannels:].sum(-1).reshape(nodes_t.shape[0], -1)

            def update(n):
                title.set_text('Time $k =${}'.format(n))
                arrows.set(alpha=arrow_color[n])
                circles.set(alpha=circle_color[n])
                for text, i in zip(texts, resting_t[n]):
                    text.set_text(str(i))
                    text.set(alpha=bool(i), visible=bool(i))
                return arrows, circles, texts, title

            ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
            return ani

        else:
            def update(n):
                title.set_text('Time $k =${}'.format(n))
                arrows.set(alpha=arrow_color[n])
                return arrows, title

            ani = animation.FuncAnimation(fig, update, interval=interval, frames=nodes_t.shape[0])
            return ani

    def live_animate_config(self, interval=100, **kwargs):
        fig, arrows, circles, texts = self.plot_config(**kwargs)
        title = plt.title('Time $k =$0')
        nodes = self.nodes[self.nonborder]
        arrow_color = np.zeros(nodes[..., :self.velocitychannels].ravel().shape)
        if self.restchannels:
            circle_color = np.zeros(nodes[..., 0].ravel().shape)

            def update(n):
                self.timestep()
                nodes = self.nodes[self.nonborder]
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
                nodes = self.nodes[self.nonborder]
                arrow_color= np.moveaxis(nodes[..., :self.velocitychannels], -1, 0).ravel().astype(float)
                title.set_text('Time $k =${}'.format(n))
                arrows.set(alpha=arrow_color)
                return arrows, title

            ani = animation.FuncAnimation(fig, update, interval=interval)
            return ani

    def live_animate_density(self, interval=100, channels=slice(None), **kwargs):

        fig, pc, cmap = self.plot_density(channels=channels, **kwargs)
        title = plt.title('Time $k =$0')

        def update(n):
            self.timestep()
            title.set_text('Time $k =${}'.format(n))
            nodes = self.nodes[self.nonborder].astype('bool')  # makes it work for IBLGCA and doesn't hurt here
            dens = nodes[..., channels].sum(-1)
            pc.set(facecolor=cmap.to_rgba(dens.ravel()))
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

        fig, ax = self.setup_figure(figsize=figsize, **kwargs)
        ax.set_aspect('equal')
        plot = plt.quiver(xx, yy, jx, jy, density.ravel(), pivot='mid', angles='xy', scale_units='xy',
                          scale=1./self.r_poly, minlength=0.)

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cmap = copy(cm.get_cmap(cmap))
            cmap.set_under(alpha=0.0)
            plot.set_cmap(cmap)
            # cmap = plot.get_cmap()
            plot.set_clim([1, K])
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
            mappable.set_array(np.arange(K))
            cbar = fig.colorbar(mappable, extend='min', use_gridspec=True, cax=cax)
            cbar.set_label('Particle number $n$')
            cbar.set_ticks(np.linspace(0., K + 1, 2 * K + 3, endpoint=True)[3::2])
            cax.yaxis.set_major_formatter(lambda x, pos: (int(x - 0.5)))
            # cbar.set_ticklabels(1 + np.arange(K)) # np.arange(K+1)
            plt.sca(ax)
        else:
            cmap = copy(cm.get_cmap('Greys'))
            cmap.set_under(alpha=0.0)
            plot.set_cmap(cmap)
            # cmap = plot.get_cmap()
            plot.set_clim([0, 1])

            # cmap = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(1), cmap.N))
            # cmap.set_array(np.arange(1))

        # plot = plt.quiver(xx, yy, jx, jy, # color=cmap.to_rgba(density.ravel()),
        #                   pivot='mid', angles='xy', scale_units='xy', scale=1./self.r_poly)
        return fig, plot

    def animate_flow(self, nodes_t=None, interval=100, cbar=False, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flow calculation but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword record=True")

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
            if hasattr(field, 'mask'):
                mask = field.mask

            else: mask = np.zeros_like(field, dtype=bool)


        cmap = plt.cm.get_cmap(cmap)
        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly, alpha=v,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c, v in
                    zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(field.ravel()),
                        1 - mask.ravel().astype(float))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(cmap, cax=cax, use_gridspec=True)
            cbar.set_label(cbarlabel)
            plt.sca(ax)

        return fig, pc, cmap

    def plot_density(self, density=None, channels=slice(None), figindex=None, figsize=None, tight_layout=True,
                     cmap='viridis', vmax=None, edgecolor='None', cbar=True, cbarlabel='Particle number $n$'):
        if density is None:
            nodes = self.nodes[self.nonborder]
            density = nodes[..., channels].sum(-1)

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True, dy=self.dy)

        if vmax is None:
            K = self.K

        else:
            K = vmax

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        cmap = copy(cm.get_cmap(cmap))  # do not modify a globally registered colormap in matplotlib > 3.3.2
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
            cbar.set_label(cbarlabel)
            cbar.set_ticks(np.linspace(0., K + 1, 2 * K + 3, endpoint=True)[3::2])
            cax.yaxis.set_major_formatter(lambda x, pos: (int(x - 0.5)))
            #cbar.set_ticklabels(1 + np.arange(K)) # np.arange(K+1)
            plt.sca(ax)

        return fig, pc, cmap

    def plot_vectorfield(self, x, y, vfx, vfy, figindex=None, figsize=None, tight_layout=True, cmap='viridis'):
        l = np.sqrt(vfx ** 2 + vfy ** 2)

        if figsize is None:
            figsize = estimate_figsize(x, cbar=True)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        ax.set_aspect('equal')
        plot = plt.quiver(x, y, vfx, vfy, l, cmap=cmap, pivot='mid', angles='xy', scale_units='xy',
                          scale=1./self.r_poly, norm=colors.Normalize(vmin=0, vmax=1), minlength=0.)
        return fig, plot

    def plot_flux(self, nodes=None, figindex=None, figsize=None, tight_layout=True, edgecolor='None', cbar=True):
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        # if nodes.dtype != 'bool':
        #     nodes = nodes.astype('bool')

        nodes = nodes.astype(np.int8)
        density = nodes.sum(-1).astype(float) / self.K

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        cmap = plt.cm.get_cmap('gist_rainbow')
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
            cbar.set_label('Direction of flux')
            cbar.set_ticks(np.arange(self.velocitychannels) * 360 / self.velocitychannels)
            cbar.set_ticklabels([r'${} \degree$'.format(int(i)) for i in
                                 np.arange(self.velocitychannels) * 360 / self.velocitychannels])
            plt.sca(ax)

        return fig, pc, cmap

    def animate_density(self, density_t=None, interval=100, channels=slice(None), repeat=True, **kwargs):

        if density_t is None:
            if hasattr(self, 'dens_t'):
                if channels == slice(None):
                    density_t = self.dens_t
                else:
                    if hasattr(self, 'nodes_t'):
                        nodes_t = self.nodes_t[..., channels]
                        density_t = nodes_t.sum(-1)
                    else:
                        raise RuntimeError(
                            "Channel-wise state of the lattice required for density plotting for required channels only " +
                            "but not recorded in past LGCA run, call lgca.timeevo with keyword record=True")
            else:
                raise RuntimeError("Node-wise state of the lattice required for density plotting but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword recorddens=True")

        fig, pc, cmap = self.plot_density(density_t[0], **kwargs)
        title = plt.title('Time $k =$0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            pc.set(facecolor=cmap.to_rgba(density_t[n, ...].ravel()))
            return pc, title

        ani = animation.FuncAnimation(fig, update, interval=interval, frames=density_t.shape[0], repeat=repeat)
        return ani

    def animate_flux(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError("Channel-wise state of the lattice required for flux calculation but not recorded " +
                                   "in past LGCA run, call lgca.timeevo with keyword record=True")

        nodes = nodes_t.astype(float)
        density = nodes.sum(-1) / self.K
        jx, jy = np.moveaxis(self.calc_flux(nodes), -1, 0)

        angle = np.zeros(density.shape, dtype=complex)
        angle.real = jx
        angle.imag = jy
        angle = np.angle(angle, deg=True) % 360.
        fig, pc, cmap = self.plot_flux(nodes=nodes[0], **kwargs)
        angle = cmap.to_rgba(angle[None, ...])[0]
        angle[..., -1] = np.sign(density)
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
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath', 'birthdeath_discrete',
                    'only_propagation', 'go_and_grow_mutations']

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            self.random_reset(density)

        else:
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

    def init_nodes(self, density=4, nodes=None, hom=None):
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.K), dtype=np.uint)
        if nodes is None:
            if hom:
                self.homogeneous_random_reset(density)
            else:
                self.random_reset(density)
        else:
            self.nodes[self.r_int:-self.r_int, self.r_int:-self.r_int, :] = nodes.astype(np.uint)
            self.apply_boundaries()

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
        # add shift up padding 0 and shift down padding 0
        sum[:, :-1, ...] += qty[:, 1:, ...]
        sum[:, 1:, ...] += qty[:, :-1, ...]
        # add central value
        if addCenter:
            sum += qty
        return sum

    def plot_density(self, density=None, figindex=None, figsize=None, tight_layout=True, cmap='viridis', vmax=None,
                     edgecolor='None', cbar=True, cbarlabel='Particle number $n$', channels=slice(None)):

        if density is None:
            nodes = self.nodes[self.nonborder]
            density = nodes[..., channels].sum(-1)

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True, dy=self.dy)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)


        cmap = get_cmap(density, cmap=cmap, cbarlabel=cbarlabel)


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
        cmap = get_cmap(density, vmax=vmax, cmap=cmap, cbar=cbar, cbarlabel=cbarlabel)
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
            self.nodes[self.nonborder] = nodes.astype(np.uint)

        else:
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

    def apply_rbcx(self):
        self.nodes[self.r_int, :, 0] = self.nodes[self.r_int, :, 0] + self.nodes[self.r_int - 1, :, 2]
        self.nodes[-self.r_int - 1, :, 2] = self.nodes[-self.r_int - 1, :, 2] + self.nodes[-self.r_int, :, 0]
        self.apply_abcx()

    def apply_rbcy(self):
        self.nodes[:, self.r_int, 1] = self.nodes[:, self.r_int, 1] + self.nodes[:, self.r_int - 1, 3]
        self.nodes[:, -self.r_int - 1, 3] = self.nodes[:, -self.r_int - 1, 3] + self.nodes[:, -self.r_int, 1]
        self.apply_abcy()

    def apply_abcx(self):
        self.nodes[:self.r_int, ...] = get_arr_of_empty_lists(self.nodes[:self.r_int, ...].shape)
        self.nodes[-self.r_int:, ...] = get_arr_of_empty_lists(self.nodes[-self.r_int:, ...].shape)

    def apply_abcy(self):
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

