# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Extended 3D cubic LGCA models.

Adds identity-based and no-volume-exclusion variants for cubic geometries.
"""


from __future__ import annotations

import numpy as np
import warnings
from lgca.ib_base import IBLGCA_base
from lgca.list_utils import get_arr_of_empty_lists
from lgca.nove_base import NoVE_LGCA_base
from lgca.nove_ib_base import NoVE_IBLGCA_base
from .lgca_cubic import LGCA_Cubic
try:  # optional plotting dependency
    from mayavi import mlab
except ImportError:  # pragma: no cover - handled at runtime
    from lgca.base import _MissingPlotLib
    mlab = _MissingPlotLib("mayavi")


class IBLGCA_Cubic(IBLGCA_base, LGCA_Cubic):
    """
    Identity-based LGCA simulator class for a 3D cubic lattice.
    """

    interactions = [
        "go_or_grow",
        "go_and_grow",
        "random_walk",
        "birth",
        "birthdeath",
        "birthdeath_discrete",
        "only_propagation",
        "go_and_grow_mutations",
    ]

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize the lattice for IBLGCA.
        """
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.K,
            ),
            dtype=np.uint,
        )
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = nodes.astype(np.uint)
            self.apply_boundaries()

    def plot_prop_spatial(self, nodes=None, props=None, propname=None, **kwargs):
        """
        Plot spatial distribution of a property.
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        if props is None:
            props = self.props

        if propname is None:
            propname = list(props)[0]

        mask = np.any(nodes, axis=-1)
        meanprop = self.calc_prop_mean(propname=propname, props=props, nodes=nodes)
        fig, pc, cmap = self.plot_scalarfield(
            meanprop, mask=mask, cbarlabel=propname, **kwargs
        )
        return fig, pc, cmap

    def plot_density(self, density=None, **kwargs):
        """
        Plot particle density in the lattice.
        """
        if density is None:
            self.update_dynamic_fields()
            density = self.cell_density[self.nonborder]

        fig, contour = LGCA_Cubic.plot_density(self, density=density, **kwargs)
        return fig, contour

    def plot_density_cubes(self, density=None, **kwargs):
        """
        Plot particle density in the lattice.
        """
        if density is None:
            self.update_dynamic_fields()
            density = self.cell_density[self.nonborder]

        fig, points = LGCA_Cubic.plot_density_cubes(self, density=density, **kwargs)
        return fig, points

    def animate_config(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            nodes_t = self.nodes_t.astype(bool)

        return super().animate_config(nodes_t=nodes_t, interval=interval, **kwargs)

    def plot_config(self, nodes=None, **kwargs):
        if nodes is None:
            nodes = self.occupied[self.nonborder]

        return super().plot_config(nodes=nodes, **kwargs)

    def animate_flux(
        self,
        nodes_t=None,
        scale_factor=1.0,
        opacity=0.6,
        interval=100,
        cbar=False,
        **kwargs,
    ):
        """
        Animate the flux vectors over time in the 3D lattice using Mayavi.

        Parameters
        ----------
        nodes_t : np.ndarray, optional
            Time series of node states. Shape: `(time, lx, ly, lz, K)`.
        scale_factor : float, default=1.0
            Scaling factor for the arrows.

        opacity : float, default=0.6
            Opacity of the flux vectors.
        interval : int, default=100
            Delay between frames in milliseconds.
        **kwargs
            Additional arguments passed to `mlab.quiver3d`.

        Returns
        -------
        None
        """
        if nodes_t is None:
            if hasattr(self, "nodes_t"):
                nodes_t = self.nodes_t.astype(bool)
            else:
                raise RuntimeError(
                    "Channel-wise state of the lattice required for flux calculation but not recorded. "
                    "Call lgca.timeevo with keyword record=True"
                )

        return super().animate_flux(
            nodes_t=nodes_t,
            scale_factor=scale_factor,
            opacity=opacity,
            interval=interval,
            cbar=cbar,
            **kwargs,
        )

    def live_animate_density(self, interval=100, channels=slice(None), **kwargs):
        # colourbar update is an issue
        warnings.warn(
            "Live density animation not available for this LGCA configuration yet."
        )

    def live_animate_configuration(self, interval=100, channels=slice(None), **kwargs):
        # colourbar update is an issue
        warnings.warn(
            "Live density animation not available for this LGCA configuration yet."
        )

    def live_animate_flux(self, interval=100, channels=slice(None), **kwargs):
        # colourbar update is an issue
        warnings.warn(
            "Live density animation not available for this LGCA configuration yet."
        )


class NoVE_LGCA_Cubic(LGCA_Cubic, NoVE_LGCA_base):
    """
    3D cubic version of an LGCA without volume exclusion.
    """

    interactions = ["dd_alignment", "di_alignment", "go_or_grow", "go_or_rest"]

    def set_dims(self, dims=None, nodes=None, restchannels=None, capacity=None):
        """
        Set the dimensions of the instance according to given values.
        """
        if nodes is not None:
            try:
                self.lx, self.ly, self.lz, self.K = nodes.shape
            except ValueError as e:
                raise ValueError(
                    "Node shape does not match the 3D geometry! Shape must be (x,y,z,channels)"
                ) from e

            if self.K < self.velocitychannels:
                raise RuntimeError(
                    f"Not enough channels specified! Required: {self.velocitychannels}, provided: {self.K}"
                )
            self.restchannels = self.K - self.velocitychannels
            if capacity is not None:
                self.capacity = capacity
            elif restchannels is not None:
                self.capacity = self.velocitychannels + restchannels
            else:
                self.capacity = self.K
            self.dims = (self.lx, self.ly, self.lz)
            return
        elif dims is not None:
            if isinstance(dims, tuple) and len(dims) == 3:
                self.lx, self.ly, self.lz = dims
            elif isinstance(dims, int):
                self.lx = self.ly = self.lz = dims
            else:
                raise TypeError(
                    "Keyword 'dims' must be a tuple of three integers or an int!"
                )
        else:
            self.lx = self.ly = self.lz = 50

        self.dims = (self.lx, self.ly, self.lz)
        self.restchannels = restchannels or 0
        self.K = self.velocitychannels + self.restchannels
        self.capacity = capacity if capacity is not None else self.K

    def init_nodes(self, density=4, nodes=None):
        """
        Initialize lattice nodes.
        """
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.K,
            ),
            dtype=np.uint,
        )
        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[
                self.r_int : -self.r_int,
                self.r_int : -self.r_int,
                self.r_int : -self.r_int,
                :,
            ] = nodes.astype(np.uint)
            self.apply_boundaries()

    def plot_flux(
        self, nodes=None, scale_factor=1.0, opacity=0.5, cbar=False, **kwargs
    ):
        """
        Plot the local flux vectors in the 3D lattice using Mayavi.

        Parameters
        ----------
        nodes : np.ndarray, optional
            State of the lattice. If None, uses `self.nodes`.
        scale_factor : float, default=1.0
            Scaling factor for the arrows.
        color : str or tuple, default='blue'
            Color of the flux vectors.
        opacity : float, default=0.6
            Opacity of the flux vectors.
        **kwargs
            Additional arguments passed to `mlab.quiver3d`.

        Returns
        -------
        quiver : mayavi.modules.vector_field.VectorField
            Mayavi quiver3d object.
        """
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        flux = self.calc_flux(nodes.astype(float))
        flux_norm = np.linalg.norm(flux, axis=-1)
        scatter_size = (
            (1 - np.sign(flux_norm)) * self.cell_density[self.nonborder] / self.K
        )

        x = self.xcoords
        y = self.ycoords
        z = self.zcoords
        u = flux[..., 0]
        v = flux[..., 1]
        w = flux[..., 2]
        fig = self.set_up_mayavi_fig()

        quiver = mlab.quiver3d(
            x,
            y,
            z,
            u,
            v,
            w,
            mode="arrow",
            color=(0, 0, 0),
            scale_factor=scale_factor,
            opacity=opacity,
            vmin=0,
            figure=fig,
            **kwargs,
        )
        scatter = mlab.points3d(
            x,
            y,
            z,
            scatter_size,
            scale_factor=scale_factor,
            color=(0, 0, 0),
            opacity=opacity,
            figure=fig,
        )
        fig = self.setup_mayavi_scene()
        mlab.title("Flux", size=0.4, color=(0, 0, 0))
        if cbar:
            mlab.colorbar(title="Flux Magnitude", orientation="vertical")
        return fig, quiver, scatter

    def plot_density(
        self,
        density=None,
        colormap="viridis",
        opacity=0.5,
        cbar=True,
        contours=5,
        vmax=None,
        **kwargs,
    ):
        """
        Plot a 3D surface based on the local density using a specified threshold using Mayavi.

        Parameters
        ----------
        threshold : float, default=0.5
            Density threshold for surface plotting.
        colormap : str, default='viridis'
            Colormap for the surface.
        opacity : float, default=0.5
            Opacity of the surface.
        **kwargs
            Additional arguments passed to `mlab.contour3d`.

        Returns
        -------
        contour : mayavi.modules.contour3d.Contour3D
            Mayavi contour3d object.
        """
        if density is None:
            self.update_dynamic_fields()
            density = self.cell_density[self.nonborder]

        x = self.xcoords
        y = self.ycoords
        z = self.zcoords
        fig = self.set_up_mayavi_fig()

        contour = mlab.contour3d(
            x,
            y,
            z,
            density,
            opacity=opacity,
            colormap=colormap,
            contours=contours,
            vmin=0,
            vmax=vmax,
            figure=fig,
            **kwargs,
        )
        fig = self.setup_mayavi_scene()
        mlab.title("Density Surface", size=0.4, color=(0, 0, 0))
        if cbar:
            colorbar = mlab.colorbar(
                title="Density", orientation="vertical", label_fmt="%.0f"
            )
            colorbar.label_text_property.color = (0, 0, 0)
            colorbar.title_text_property.color = (0, 0, 0)
            # reduce the size of the colorbar labels
            colorbar.scalar_bar.unconstrained_font_size = True
            colorbar.label_text_property.font_size = 14
            colorbar.label_text_property.bold = False
            colorbar.label_text_property.vertical_justification = "centered"

        return fig, contour

    def plot_density_cubes(
        self, density=None, colormap="viridis", opacity=0.5, cbar=True, **kwargs
    ):
        """
        Plot a 3D surface based on the local density using a specified threshold using Mayavi.

        Parameters
        ----------
        threshold : float, default=0.5
            Density threshold for surface plotting.
        colormap : str, default='viridis'
            Colormap for the surface.
        opacity : float, default=0.5
            Opacity of the surface.
        **kwargs
            Additional arguments passed to `mlab.contour3d`.

        Returns
        -------
        contour : mayavi.modules.contour3d.Contour3D
            Mayavi contour3d object.
        """
        if density is None:
            self.update_dynamic_fields()
            density = self.cell_density[self.nonborder]

        mask = density > 0
        x = self.xcoords[mask] + 0.5
        y = self.ycoords[mask] + 0.5
        z = self.zcoords[mask] + 0.5
        fig = self.set_up_mayavi_fig()

        points = mlab.points3d(
            x,
            y,
            z,
            density[mask],
            opacity=opacity,
            colormap=colormap,
            mode="cube",
            scale_mode="none",
            vmin=0,
            figure=fig,
            reset_zoom=False,
            scale_factor=1.0,
            **kwargs,
        )
        fig = self.setup_mayavi_scene()
        mlab.title("Density", size=0.4, color=(0, 0, 0))
        if cbar:
            colorbar = mlab.colorbar(
                title="Density", orientation="vertical", label_fmt="%.0f"
            )
            colorbar.label_text_property.color = (0, 0, 0)
            colorbar.title_text_property.color = (0, 0, 0)
            # reduce the size of the colorbar labels
            colorbar.scalar_bar.unconstrained_font_size = True
            colorbar.label_text_property.font_size = 14
            colorbar.label_text_property.bold = False
            colorbar.label_text_property.vertical_justification = "centered"

        return fig, points

    def plot_config(self, nodes=None, colormap="viridis", vmax=None, **kwargs):
        if nodes is None:
            nodes = self.nodes[self.nonborder]

        # create new array for x, y, z coordinates with shape (dims, K) from shape (dims)
        x = np.repeat(self.xcoords[..., None], self.velocitychannels, axis=-1)
        y = np.repeat(self.ycoords[..., None], self.velocitychannels, axis=-1)
        z = np.repeat(self.zcoords[..., None], self.velocitychannels, axis=-1)

        u = 0.5 * self.c[None, None, None, 0] * nodes[..., : self.velocitychannels]
        v = 0.5 * self.c[None, None, None, 1] * nodes[..., : self.velocitychannels]
        w = 0.5 * self.c[None, None, None, 2] * nodes[..., : self.velocitychannels]

        fig = self.set_up_mayavi_fig()
        quiver = mlab.quiver3d(
            x,
            y,
            z,
            u,
            v,
            w,
            mode="arrow",
            color=(0, 0, 0),
            scale_factor=1,
            opacity=0.5,
            figure=fig,
            **kwargs,
        )

        if self.restchannels > 0:
            rest = nodes[..., self.velocitychannels :].sum(axis=-1)
            scatter = mlab.points3d(
                self.xcoords,
                self.ycoords,
                self.zcoords,
                rest,
                scale_factor=0.5,
                scale_mode="scalar",
                colormap=colormap,
                opacity=0.5,
                figure=fig,
                vmin=0,
                vmax=vmax,
                color=(0.7, 0.7, 0.7),
            )

        else:
            scatter = None
        fig = self.setup_mayavi_scene()
        mlab.title("Configuration", size=0.4, color=(0, 0, 0))

        return (
            fig,
            quiver,
            scatter,
        )

    def animate_config(self, nodes_t=None, interval=100, **kwargs):
        if nodes_t is None:
            if hasattr(self, "nodes_t"):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError(
                    "Channel-wise state of the lattice required for plotting the configuration but not "
                    + "recorded in past LGCA run, call lgca.timeevo with keyword record=True"
                )

        fig, quiver, scatter = self.plot_config(nodes=nodes_t[0], **kwargs)
        mlab.title("Time 0", size=0.4, color=(0, 0, 0))
        u = (
            0.5
            * self.c[None, None, None, None, 0]
            * nodes_t[..., : self.velocitychannels]
        )
        v = (
            0.5
            * self.c[None, None, None, None, 1]
            * nodes_t[..., : self.velocitychannels]
        )
        w = (
            0.5
            * self.c[None, None, None, None, 2]
            * nodes_t[..., : self.velocitychannels]
        )

        rest = nodes_t[..., self.velocitychannels :].sum(axis=-1)

        @mlab.animate(delay=interval)
        def anim():
            for i in range(nodes_t.shape[0]):

                quiver.mlab_source.set(u=u[i], v=v[i], w=w[i])
                if self.restchannels > 0:
                    scatter.mlab_source.set(scalars=rest[i])
                mlab.title(f"Time {i}", size=0.4)
                yield

        anim()
        mlab.show()

    def animate_density(
        self,
        density_t=None,
        colormap="viridis",
        opacity=0.5,
        cbar=True,
        interval=100,
        **kwargs,
    ):
        """
        Animate the density surface over time using Mayavi.

        Parameters
        ----------
        density_t : np.ndarray, optional
            Time series of density states. Shape: `(time, lx, ly, lz)`.
        threshold : float, default=0.5
            Density threshold for surface plotting.
        colormap : str, default='viridis'
            Colormap for the surface.
        opacity : float, default=0.5
            Opacity of the surface.
        interval : int, default=100
            Delay between frames in milliseconds.
        **kwargs
            Additional arguments passed to `mlab.contour3d`.

        Returns
        -------
        None
        """
        if density_t is None:
            if hasattr(self, "dens_t"):
                density_t = self.dens_t
            else:
                raise RuntimeError(
                    "Density time series not found. Ensure to record density during simulation."
                )
        vmax = np.max(density_t)
        fig, contour = self.plot_density(
            density=density_t[0],
            colormap=colormap,
            opacity=opacity,
            cbar=cbar,
            vmax=vmax,
            **kwargs,
        )
        mlab.title("Time 0", size=0.4, color=(0, 0, 0))

        @mlab.animate(delay=interval)
        def anim():
            for i in range(density_t.shape[0]):
                contour.mlab_source.set(scalars=density_t[i])
                mlab.title(f"Time {i}", size=0.4)
                yield

        anim()
        mlab.show()

    def animate_flux(
        self,
        nodes_t=None,
        scale_factor=1.0,
        opacity=0.6,
        interval=100,
        cbar=False,
        **kwargs,
    ):
        """
        Animate the flux vectors over time in the 3D lattice using Mayavi.

        Parameters
        ----------
        nodes_t : np.ndarray, optional
            Time series of node states. Shape: `(time, lx, ly, lz, K)`.
        scale_factor : float, default=1.0
            Scaling factor for the arrows.

        opacity : float, default=0.6
            Opacity of the flux vectors.
        interval : int, default=100
            Delay between frames in milliseconds.
        **kwargs
            Additional arguments passed to `mlab.quiver3d`.

        Returns
        -------
        None
        """
        if nodes_t is None:
            if hasattr(self, "nodes_t"):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError(
                    "Channel-wise state of the lattice required for flux calculation but not recorded. "
                    "Call lgca.timeevo with keyword record=True"
                )

        flux_t = self.calc_flux(nodes_t.astype(float))
        time_steps = flux_t.shape[0]
        scatter_sizes = (1 - np.sign(np.linalg.norm(flux_t, axis=-1))) * self.dens_t

        fig, quiver, scatter = self.plot_flux(
            nodes=nodes_t[0],
            scale_factor=scale_factor,
            opacity=opacity,
            cbar=cbar,
            **kwargs,
        )
        mlab.title("Flux at Time 0", size=0.4)

        @mlab.animate(delay=interval)
        def anim():
            for i in range(time_steps):
                quiver.mlab_source.set(
                    u=flux_t[i, ..., 0], v=flux_t[i, ..., 1], w=flux_t[i, ..., 2]
                )
                scatter.mlab_source.set(scalars=scatter_sizes[i])
                mlab.title(f"Flux at Time {i}", size=0.4)
                yield

        anim()
        mlab.show()

    def live_animate_flux(self, scale_factor=1.0, opacity=0.5, cbar=False, **kwargs):
        """
        Live plot the flux vectors in the 3D lattice using Mayavi.

        This method updates the plot in real-time as the simulation progresses.

        Parameters
        ----------
        scale_factor : float, default=1.0
            Scaling factor for the arrows.
        opacity : float, default=0.6
            Opacity of the flux vectors.
        **kwargs
            Additional arguments passed to `mlab.quiver3d`.

        Returns
        -------
        quiver : mayavi.modules.vector_field.VectorField
            Mayavi quiver3d object.
        """
        nodes = self.nodes[self.nonborder]

        fig, quiver, scatter = self.plot_flux(
            nodes=nodes, scale_factor=scale_factor, opacity=opacity, cbar=cbar, **kwargs
        )

        def update_plot():
            while True:
                yield

        update_gen = update_plot()

        @mlab.animate(delay=100)
        def anim():
            for i in range(1000000):  # Arbitrary large number for continuous animation
                # Perform a timestep
                self.timestep()

                # Update flux
                nodes = self.nodes[self.nonborder]
                flux = self.calc_flux(nodes.astype(float))

                quiver.mlab_source.set(
                    u=flux[..., 0], v=flux[..., 1], w=flux[..., 2], vmin=0
                )
                scatter.mlab_source.set(
                    scalars=(1 - np.sign(np.linalg.norm(flux, axis=-1)))
                    * self.cell_density[self.nonborder]
                    / self.K
                )
                mlab.title(f"Time {i}", size=0.4)
                yield

        anim()
        mlab.show()

    def live_animate_density(self, **kwargs):
        """
        Live plot a 3D density surface based on a specified threshold using Mayavi.

        Parameters
        ----------
        threshold : float, default=0.5
            Density threshold for surface plotting.
        colormap : str, default='viridis'
            Colormap for the surface.
        opacity : float, default=0.5
            Opacity of the surface.
        **kwargs
            Additional arguments passed to `mlab.contour3d`.

        Returns
        -------
        None
        """
        fig, contour = self.plot_density(**kwargs)
        # def update():
        #     while True:
        #         yield
        #
        #
        # update_gen = update()

        @mlab.animate(delay=100)
        def anim():
            for i in range(1000000):
                self.timestep()
                contour.mlab_source.set(
                    scalars=self.cell_density[self.nonborder], vmin=0
                )
                mlab.title(f"Time {i}", size=0.4)
                yield

        anim()
        mlab.show()


class NoVE_IBLGCA_Cubic(NoVE_IBLGCA_base, LGCA_Cubic):
    """
    Identity-based LGCA without volume exclusion for a 3D cubic lattice.
    """

    def init_nodes(self, density=0.1, nodes=None):
        """
        Initialize the lattice nodes for NoVE_IBLGCA.
        """
        self.nodes = get_arr_of_empty_lists(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.K,
            )
        )
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
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.
        """
        newnodes = get_arr_of_empty_lists(self.nodes.shape)
        newnodes[..., self.velocitychannels :] = self.nodes[
            ..., self.velocitychannels :
        ]

        newnodes[1:, :, :, 0] = self.nodes[:-1, :, :, 0]
        newnodes[:-1, :, :, 1] = self.nodes[1:, :, :, 1]
        newnodes[:, 1:, :, 2] = self.nodes[:, :-1, :, 2]
        newnodes[:, :-1, :, 3] = self.nodes[:, 1:, :, 3]
        newnodes[:, :, 1:, 4] = self.nodes[:, :, :-1, 4]
        newnodes[:, :, :-1, 5] = self.nodes[:, :, 1:, 5]

        self.nodes = newnodes

    def _apply_rbc_x(self):
        self.nodes[self.r_int, :, :, 0] = (
            self.nodes[self.r_int, :, :, 0] + self.nodes[self.r_int - 1, :, :, 1]
        )
        self.nodes[-self.r_int - 1, :, :, 1] = (
            self.nodes[-self.r_int - 1, :, :, 1] + self.nodes[-self.r_int, :, :, 0]
        )

    def _apply_rbc_y(self):
        self.nodes[:, self.r_int, :, 2] = (
            self.nodes[:, self.r_int, :, 2] + self.nodes[:, self.r_int - 1, :, 3]
        )
        self.nodes[:, -self.r_int - 1, :, 3] = (
            self.nodes[:, -self.r_int - 1, :, 3] + self.nodes[:, -self.r_int, :, 2]
        )

    def _apply_rbc_z(self):
        self.nodes[:, :, self.r_int, 4] = (
            self.nodes[:, :, self.r_int, 4] + self.nodes[:, :, self.r_int - 1, 5]
        )
        self.nodes[:, :, -self.r_int - 1, 5] = (
            self.nodes[:, :, -self.r_int - 1, 5] + self.nodes[:, :, -self.r_int, 4]
        )

    def _apply_abc_x(self):
        """
        Apply absorbing boundary conditions in x-direction.
        """
        self.nodes[: self.r_int, :, :, :] = get_arr_of_empty_lists(
            self.nodes[: self.r_int, :, :, :].shape
        )
        self.nodes[-self.r_int :, :, :, :] = get_arr_of_empty_lists(
            self.nodes[-self.r_int :, :, :, :].shape
        )

    def _apply_abc_y(self):
        """
        Apply absorbing boundary conditions in y-direction.
        """
        self.nodes[:, : self.r_int, :, :] = get_arr_of_empty_lists(
            self.nodes[:, : self.r_int, :, :].shape
        )
        self.nodes[:, -self.r_int :, :, :] = get_arr_of_empty_lists(
            self.nodes[:, -self.r_int :, :, :].shape
        )

    def _apply_abc_z(self):
        """
        Apply absorbing boundary conditions in z-direction.
        """
        self.nodes[:, :, : self.r_int, :] = get_arr_of_empty_lists(
            self.nodes[:, :, : self.r_int, :].shape
        )
        self.nodes[:, :, -self.r_int :, :] = get_arr_of_empty_lists(
            self.nodes[:, :, -self.r_int :, :].shape
        )

    def plot_density(self, density=None, **kwargs):
        """
        Plot the density of the NoVE_IBLGCA_Cubic model.
        """
        if density is None:
            density = self.length_checker(self.nodes[self.nonborder].sum(-1))
        return NoVE_LGCA_Cubic.plot_density(self, density=density, **kwargs)

    def plot_flux(self, nodes=None, **kwargs):
        """
        Plot the flux of the NoVE_IBLGCA_Cubic model.
        """
        if nodes is None:
            nodes = self.length_checker(self.nodes[self.nonborder])
        return super().plot_flux(nodes=nodes, **kwargs)

    def plot_config(self, nodes=None, **kwargs):
        """
        Plot the configuration of the NoVE_IBLGCA_Cubic model.
        """
        if nodes is None:
            nodes = self.length_checker(self.nodes[self.nonborder])
        return super().plot_config(nodes=nodes, **kwargs)

    def animate_config(self, nodes_t=None, **kwargs):
        """
        Animate the configuration of the NoVE_IBLGCA_Cubic model.
        """
        if nodes_t is None:
            nodes_t = self.length_checker(self.nodes_t)
        return super().animate_config(nodes_t=nodes_t, **kwargs)

    def animate_flux(self, nodes_t=None, **kwargs):
        """
        Animate the flux of the NoVE_IBLGCA_Cubic model.
        """
        if nodes_t is None:
            nodes_t = self.length_checker(self.nodes_t)
        return super().animate_flux(nodes_t=nodes_t, **kwargs)

    def animate_density(self, density_t=None, **kwargs):
        """
        Animate the density of the NoVE_IBLGCA_Cubic model.
        """
        if density_t is None:
            if hasattr(self, "dens_t"):
                density_t = self.dens_t
            else:
                raise RuntimeError(
                    "Density time series not found. Ensure to record density during simulation."
                )
        return NoVE_LGCA_Cubic.animate_density(self, density_t=density_t, **kwargs)

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
        if "cbarlabel" not in kwargs:
            kwargs.update({"cbarlabel": str(propname)})

        return super().plot_scalarfield(mean_prop, **kwargs)


if __name__ == "__main__":
    # Initialize LGCA on a 3D cubic lattice
    from lgca import get_lgca

    L = 50
    nodes = np.zeros((L, L, L, 7), dtype=int)
    nodes[L // 2, L // 2, L // 2, -1] = 10
    lgca = get_lgca(
        ib=False,
        ve=True,
        geometry="cubic",
        interaction="go_or_grow",
        dims=50,
        density=0.05,
        beta=1,
    )
    lgca.timeevo(timesteps=100, record=True)

    # Plot flux using Mayavi
    # lgca.plot_flux()
    #
    # Animate flux
    # lgca.animate_flux()
    # lgca.live_animate_flux()
    # lgca.animate_config()
    # mlab.show()
    lgca.animate_density()
    # lgca.live_animate_density()
    # lgca.animate_flux()
    # mlab.show()
    # lgca.plot_config()
    # lgca.animate_config()
    # mlab.show()
