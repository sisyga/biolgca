# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Plotting utilities for square lattice LGCA.

Provides helper functions for visualizing square LGCA simulations.
"""


import numpy as np
import warnings

try:
    import matplotlib.animation as animation
    import matplotlib.colors as colors
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:  # pragma: no cover - handled at runtime
    from lgca.base import _MissingPlotLib

    animation = colors = mticker = FuncFormatter = PatchCollection = Normalize = (
        RegularPolygon
    ) = Circle = FancyArrowPatch = make_axes_locatable = _MissingPlotLib(
        "matplotlib"
    )

from copy import copy

from .plot_data import (
    _reject_sparse_implicit_history,
    select_density,
    select_density_history,
    select_scalar_field,
    validate_species,
)
from .plots import estimate_figsize, get_cmap


class SquarePlotMixin:
    """Plotting helpers for square lattice LGCAs."""
    def setup_figure(self, figindex=None, figsize=(8, 8), tight_layout=True):
        """
        Create a :py:mod:`matplotlib` figure and manage basic layout.

        Used by the class' plotting functions.

        Parameters
        ----------
        figindex : int or str, optional
            An identifier for the figure (passed to :py:func:`matplotlib.pyplot.figure`). If it is a string, the
            figure label and the window title is set to this value.
        figsize : tuple of int or tuple of float with 2 elements, default=(8,8)
            Desired figure size in inches ``(x, y)``.
        tight_layout : bool, default=True
            If :py:meth:`matplotlib.figure.Figure.tight_layout` is called for padding between and around subplots.

        Returns
        -------
        fig : :py:class:`matplotlib.figure.Figure`
            New customized figure.
        ax : :py:class:`matplotlib.axes.Axes`
            Drawing axis associated with `fig`.

        See Also
        --------
        plot_density : Plot particle density over time.
        plot_flux : Plot flux over time.

        """
        # calculate y axis scaling for polygons (squares or hexagons)
        dy = self.r_poly * np.cos(self.orientation)

        # create or retrieve figure, set size and layout
        if figindex is None:
            fig = plt.gcf()
            fig.set_size_inches(figsize)
            fig.set_layout_engine("tight" if tight_layout else None)

        else:
            fig = plt.figure(num=figindex)
            fig.set_size_inches(figsize)
            fig.set_layout_engine("tight" if tight_layout else None)

        # retrieve drawing axis and scale
        ax = plt.gca()
        xmax = self.xcoords.max() + 0.5
        xmin = self.xcoords.min() - 0.5
        ymax = self.ycoords.max() + dy
        ymin = self.ycoords.min() - dy
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        ax.set_aspect('equal')

        # label axes, set tick positions and adjust their appearance
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        if self.dy >= 1:
            minstep = self.dy
            integer = True
        else:
            minstep = 1
            integer = False
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[minstep, 2 * self.dy, 5 * self.dy, 10 * self.dy], integer=integer)
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x / self.dy)))
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
            cmap = copy(plt.get_cmap(cmap))
            cmap.set_under(alpha=0.0)
            plot.set_cmap(cmap)
            # cmap = plot.get_cmap()
            plot.set_clim([1, K])
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=colors.BoundaryNorm(1 + np.arange(K + 1), cmap.N))
            mappable.set_array(np.arange(K))
            cbar = fig.colorbar(mappable, extend='min', use_gridspec=True, cax=cax)
            cbar.set_label('Particle number $n$')
            cbar.set_ticks(np.linspace(0.0, K + 1, 2 * K + 3, endpoint=True)[3::2])
            cax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x - 0.5)))
            # cbar.set_ticklabels(1 + np.arange(K)) # np.arange(K+1)
            plt.sca(ax)
        else:
            cmap = copy(plt.get_cmap('Greys'))
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
        field = select_scalar_field(self, field)

        if mask is None:
            if hasattr(field, 'mask'):
                mask = field.mask

            else: mask = np.zeros_like(field, dtype=bool)


        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        if self.geometry == 'square':
            masked_field = np.ma.array(field.T, mask=np.asarray(mask).T)
            image = ax.imshow(
                masked_field,
                origin='lower',
                interpolation='nearest',
                extent=(self.xcoords.min() - 0.5, self.xcoords.max() + 0.5,
                        self.ycoords.min() - 0.5, self.ycoords.max() + 0.5),
                cmap=cmap,
                norm=norm,
            )
            if cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                colorbar = fig.colorbar(image, cax=cax, use_gridspec=True)
                colorbar.set_label(cbarlabel)
                plt.sca(ax)
            return fig, image, image

        cmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
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

    def plot_density(self, density=None, channels=slice(None), species=None, figindex=None, figsize=None, tight_layout=True,
                     cmap='viridis', vmax=None, edgecolor='None', cbar=True, cbarlabel='Particle number $n$'):
        """
        Plot particle density in the lattice. A color bar on the right side shows the color coding of density values.
        Empty nodes are white.

        Parameters
        ----------
        cbar : bool, default=True
            Whether to draw a colorbar for the plot on an extra axis to the right.
        cbarlabel : str, default='Particle number $n$'
            Label of the colorbar.
        channels : slice
            Indices of the velocity/resting channels that should be considered for the density calculation if `density`
            is None.
        cmap : str or :py:class:`matplotlib.colors.Colormap`, default='viridis'
            Color map for the density values. Used to construct a discretized version of the colormap.
        colorbarwidth : float
            Width of the additional axis for the color bar, passed to
            :py:meth:`mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        density : :py:class:`numpy.ndarray`, optional
            Particle density values for a lattice to plot. If set to None and a simulation has been performed
            before, the result of the simulation is plotted. Dimensions: ``self.dims``.
        edgecolor : {:py:mod:`matplotlib` color, 'None', 'auto'}, default 'None'
            Color of the polygon edges for the lattice nodes.
        figindex : int or str, optional
            An identifier for the figure (passed to :py:func:`matplotlib.pyplot.figure`). If it is a string, the
            figure label and the window title is set to this value.
        figsize : tuple of int or tuple of float with 2 elements, default=(8,8)
            Desired figure size in inches ``(x, y)``.
        tight_layout : bool, default=True
            If :py:meth:`matplotlib.figure.Figure.tight_layout` is called for padding between and around subplots.
        vmax : int, optional
            Maximum density value for the color scaling. The minimum value is zero. All density values higher than
            `vmax` are drawn in the color at the end of the color bar. If None, `vmax` is set to the number of channels
            ``self.K``.
        **kwargs
            Arguments to be passed on to :py:meth:`setup_figure`.

        Returns
        -------
        :py:class:`matplotlib.image.AxesImage`
            Density plot over time.

        See Also
        --------
        setup_figure : Manage basic layout.

        """
        density = select_density(self, density=density, channels=channels, species=species)

        # specify image size
        if figsize is None:
            figsize = estimate_figsize(density, cbar=True, dy=self.dy)

        # set limit for coloring
        if vmax is None:
            K = self.K
        else:
            K = vmax

        # set up figure
        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        # set up density translation to color
        color_map = copy(plt.get_cmap(cmap))  # do not modify a globally registered colormap in matplotlib > 3.3.2
        color_map.set_under(alpha=0.0)
        if K > 1:
            norm = colors.BoundaryNorm(1 + np.arange(K + 1), color_map.N)
        else:
            norm = colors.Normalize(vmin=0, vmax=1)
        if self.geometry == 'square':
            image = ax.imshow(
                density.T,
                origin='lower',
                interpolation='nearest',
                extent=(self.xcoords.min() - 0.5, self.xcoords.max() + 0.5,
                        self.ycoords.min() - 0.5, self.ycoords.max() + 0.5),
                cmap=color_map,
                norm=norm,
            )
            if cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                colorbar = fig.colorbar(image, extend='min', use_gridspec=True, cax=cax)
                colorbar.set_label(cbarlabel)
                colorbar.set_ticks(np.linspace(0.0, K + 1, 2 * K + 3, endpoint=True)[3::2])
                cax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x - 0.5)))
                plt.sca(ax)
            return fig, image, image

        cmap = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cmap.set_array(density)
        # draw polygons
        polygons = [RegularPolygon(xy=(x, y), numVertices=self.velocitychannels, radius=self.r_poly,
                                   orientation=self.orientation, facecolor=c, edgecolor=edgecolor)
                    for x, y, c in zip(self.xcoords.ravel(), self.ycoords.ravel(), cmap.to_rgba(density.ravel()))]
        pc = PatchCollection(polygons, match_original=True)
        ax.add_collection(pc)
        # draw colorbar
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(cmap, extend='min', use_gridspec=True, cax=cax)
            cbar.set_label(cbarlabel)
            cbar.set_ticks(np.linspace(0.0, K + 1, 2 * K + 3, endpoint=True)[3::2])
            cax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x - 0.5)))
            #cbar.set_ticklabels(1 + np.arange(K)) # np.arange(K+1)
            plt.sca(ax)

        return fig, pc, cmap

    def _validate_plot_species(self, species):
        return validate_species(self, species)

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

        nodes = nodes.astype(np.int8)
        density = nodes.sum(-1).astype(float) / self.K

        if figsize is None:
            figsize = estimate_figsize(density, cbar=True)

        fig, ax = self.setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        cmap = plt.get_cmap('gist_rainbow')
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

    def animate_density(self, density_t=None, interval=100, channels=slice(None), species=None, repeat=True, **kwargs):

        if density_t is None:
            _reject_sparse_implicit_history(self, "density_t")
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

        density_t = select_density_history(self, density_t, species=species)

        fig, pc, cmap = self.plot_density(density_t[0], **kwargs)
        title = plt.title('Time $k =$0')

        def update(n):
            title.set_text('Time $k =${}'.format(n))
            if hasattr(pc, 'set_data'):
                pc.set_data(density_t[n, ...].T)
            else:
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
