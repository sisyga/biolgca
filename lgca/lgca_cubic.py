# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""3D cubic lattice LGCA implementations.
"""


from lgca.base import LGCA_base, np
from lgca.mayavi_style import (
    INK,
    MUTED,
    add_colorbar,
    decorate_domain,
    mlab,
    new_figure,
    play,
    style_arrows,
    style_surface,
)

from .plot_data import (
    resolve_animation_history,
    select_density,
    select_density_history,
    select_scalar_field,
)


class LGCA_Cubic(LGCA_base):
    """
    Classical LGCA with volume exclusion on a 3D cubic lattice.

    It holds all methods and attributes that are specific for a cubic geometry. See :py:class:`lgca.base.LGCA_base` for
    the documentation of inherited attributes.

    Attributes
    ----------
    lx, ly, lz : int
        Lattice dimensions.
    xcoords, ycoords, zcoords : :py:class:`numpy.ndarray`
        Logical coordinates of non-border nodes starting with 0. Dimensions: ``(lgca.lx, lgca.ly, lgca.lz)``.

    See Also
    --------
    lgca.base.LGCA_base : Base class with geometry-independent methods and attributes.
    """

    # Set class attributes
    geometry = 'cubic'
    interactions = [
        "go_and_grow",
        "go_or_grow",
        "alignment",
        "aggregation",
        "random_walk",
        "excitable_medium",
        "nematic",
        "persistent_motion",
        "chemotaxis",
        "only_propagation",
    ]
    velocitychannels = 6  # +x, -x, +y, -y, +z, -z

    # Build velocity channel vectors
    cix = np.array([1, -1, 0, 0, 0, 0], dtype=float)
    ciy = np.array([0, 0, 1, -1, 0, 0], dtype=float)
    ciz = np.array([0, 0, 0, 0, 1, -1], dtype=float)
    c = np.array([cix, ciy, ciz])

    def set_dims(self, dims=None, nodes=None, restchannels=0, capacity=None):
        """
        Set LGCA dimensions.

        Initializes :py:attr:`self.K`, :py:attr:`self.restchannels`, :py:attr:`self.dims`, :py:attr:`self.lx`, :py:attr:`self.ly`, and :py:attr:`self.lz`.

        Parameters
        ----------
        dims : tuple of int, default=(10, 10, 10)
            Lattice dimensions. Must match with specified geometry.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration.
        restchannels : int, default=0
            Number of resting channels.

        See Also
        --------
        init_nodes : Initialize LGCA lattice configuration.
        init_coords : Initialize LGCA coordinates.
        """
        if nodes is not None:
            self.lx, self.ly, self.lz, self.K = nodes.shape
            if self.K < self.velocitychannels:
                raise RuntimeError(
                    'Not enough channels specified for the chosen geometry! '
                    f'Required: {self.velocitychannels}, provided: {self.K}'
                )
            self.restchannels = self.K - self.velocitychannels
            self.dims = (self.lx, self.ly, self.lz)
            return

        if dims is None:
            dims = (10, 10, 10)

        # set dimensions to keyword value
        if isinstance(dims, tuple):
            if len(dims) != 3:
                raise ValueError(
                    "For 3D cubic lattice, 'dims' must be a tuple of three integers."
                )
            self.lx, self.ly, self.lz = dims

        elif isinstance(dims, int):
            self.lx = self.ly = self.lz = dims
        else:
            raise TypeError("Keyword 'dims' must be a tuple of three integers or int!")

        self.dims = (self.lx, self.ly, self.lz)
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels
        self.capacity = capacity if capacity is not None else self.K

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize LGCA lattice configuration. Create the lattice and then assign particles to channels in the nodes.

        Initializes :py:attr:`self.nodes`. If `nodes` is not provided, the lattice is initialized randomly so that
        each node contains on average ``density`` particles.

        Parameters
        ----------
        density : float, default=0.1
            If `nodes` is None, initialize lattice randomly with this average number of particles per node.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration. Dimensions: ``(self.lx, self.ly, self.lz, self.K)``.

        See Also
        --------
        base.LGCA_base.random_reset : Initialize lattice nodes with average density `density`.
        set_dims : Set LGCA dimensions.
        init_coords : Initialize LGCA coordinates.
        """
        self.nodes = np.zeros(
            (
                self.lx + 2 * self.r_int,
                self.ly + 2 * self.r_int,
                self.lz + 2 * self.r_int,
                self.K,
            ),
            dtype=bool,
        )

        if nodes is None:
            self.random_reset(density)
        else:
            self._warn_nodes_shape(nodes)
            self.nodes[self.nonborder] = self._ensure_bool_nodes(nodes)
            self.apply_boundaries()

    def init_coords(self):
        """
        Initialize LGCA coordinates for a 3D cubic lattice.
        """
        x = np.arange(self.lx) + self.r_int
        y = np.arange(self.ly) + self.r_int
        z = np.arange(self.lz) + self.r_int
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        self.nonborder = (xx, yy, zz)
        self.coord_pairs = list(zip(xx.flat, yy.flat, zz.flat))
        self.xcoords, self.ycoords, self.zcoords = np.meshgrid(
            np.arange(self.lx + 2 * self.r_int) - self.r_int,
            np.arange(self.ly + 2 * self.r_int) - self.r_int,
            np.arange(self.lz + 2 * self.r_int) - self.r_int,
            indexing="ij",
        )
        self.xcoords = self.xcoords[self.nonborder].astype(float)
        self.ycoords = self.ycoords[self.nonborder].astype(float)
        self.zcoords = self.zcoords[self.nonborder].astype(float)

    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.

        Updates :py:attr:`self.nodes` such that resting particles stay in their position and particles in velocity channels
        are relocated according to the direction of the channel they reside in. Boundary conditions are enforced later by
        :py:meth:`apply_boundaries`.

        See Also
        --------
        base.LGCA_base.nodes : State of the lattice showing the structure of the ``lgca.nodes`` array.
        """
        newnodes = np.zeros_like(self.nodes)
        newnodes[..., self.velocitychannels :] = self.nodes[
            ..., self.velocitychannels :
        ]

        # propagation in each direction
        newnodes[1:, ..., 0] = self.nodes[:-1, ..., 0]
        newnodes[:-1, ..., 1] = self.nodes[1:, ..., 1]
        newnodes[:, 1:, ..., 2] = self.nodes[:, :-1, ..., 2]
        newnodes[:, :-1, ..., 3] = self.nodes[:, 1:, ..., 3]
        newnodes[:, :, 1:, ..., 4] = self.nodes[:, :, :-1, ..., 4]
        newnodes[:, :, :-1, ..., 5] = self.nodes[:, :, 1:, ..., 5]

        self.nodes = newnodes

    def _apply_pbc_x(self):
        self.nodes[: self.r_int, ...] = self.nodes[-2 * self.r_int : -self.r_int, ...]
        self.nodes[-self.r_int :, ...] = self.nodes[self.r_int : 2 * self.r_int, ...]

    def _apply_pbc_y(self):
        self.nodes[:, : self.r_int, ...] = self.nodes[:, -2 * self.r_int : -self.r_int, ...]
        self.nodes[:, -self.r_int :, ...] = self.nodes[:, self.r_int : 2 * self.r_int, ...]

    def _apply_pbc_z(self):
        self.nodes[:, :, : self.r_int, ...] = self.nodes[:, :, -2 * self.r_int : -self.r_int, ...]
        self.nodes[:, :, -self.r_int :, ...] = self.nodes[:, :, self.r_int : 2 * self.r_int, ...]

    def apply_pbc(self):
        self._apply_pbc_x()
        self._apply_pbc_y()
        self._apply_pbc_z()

    def _apply_rbc_x(self):
        self.nodes[self.r_int, ..., 0] += self.nodes[self.r_int - 1, ..., 1]
        self.nodes[-self.r_int - 1, ..., 1] += self.nodes[-self.r_int, ..., 0]

    def _apply_rbc_y(self):
        self.nodes[:, self.r_int, ..., 2] += self.nodes[:, self.r_int - 1, ..., 3]
        self.nodes[:, -self.r_int - 1, ..., 3] += self.nodes[:, -self.r_int, ..., 2]

    def _apply_rbc_z(self):
        self.nodes[:, :, self.r_int, ..., 4] += self.nodes[:, :, self.r_int - 1, ..., 5]
        self.nodes[:, :, -self.r_int - 1, ..., 5] += self.nodes[:, :, -self.r_int, ..., 4]

    def apply_rbc(self):
        self._apply_rbc_x()
        self._apply_rbc_y()
        self._apply_rbc_z()
        self.apply_abc()

    def _apply_abc_x(self):
        self.nodes[: self.r_int, ...] = 0
        self.nodes[-self.r_int :, ...] = 0

    def _apply_abc_y(self):
        self.nodes[:, : self.r_int, ...] = 0
        self.nodes[:, -self.r_int :, ...] = 0

    def _apply_abc_z(self):
        self.nodes[:, :, : self.r_int, ...] = 0
        self.nodes[:, :, -self.r_int :, ...] = 0

    def apply_abc(self):
        # Apply absorbing boundary conditions
        self._apply_abc_x()
        self._apply_abc_y()
        self._apply_abc_z()

    def nb_sum(self, qty):
        """
        For each node, sum up the contents of `qty` for the 6 neighboring nodes, excluding the center.

        Parameters
        ----------
        qty : :py:class:`numpy.ndarray`
            Array holding some quantity of the LGCA, e.g. a flux. Of shape ``self.dims + x``, where ``x`` is the shape
            of the quantity for one node, e.g. ``(2,)`` if it is a vector with 2 elements. ``self.dims`` ensures that
            lattice positions can be indexed the same way as in ``self.nodes``.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Sum of the content of `qty` in each node's neighborhood, shape: ``qty.shape``. Lattice positions can be
            indexed the same way as in ``self.nodes``.
        """
        sum = np.zeros_like(qty)
        sum[:-1, :, :, ...] += qty[1:, :, :, ...]
        sum[1:, :, :, ...] += qty[:-1, :, :, ...]
        sum[:, :-1, :, ...] += qty[:, 1:, :, ...]
        sum[:, 1:, :, ...] += qty[:, :-1, :, ...]
        sum[:, :, :-1, ...] += qty[:, :, 1:, ...]
        sum[:, :, 1:, ...] += qty[:, :, :-1, ...]
        return sum

    def gradient(self, qty):
        # documented in parent class
        return np.stack(np.gradient(qty, axis=(0, 1, 2)), axis=-1)

    def channel_weight(self, qty):
        """
        Calculate weights for the velocity channels in interactions depending on a field `qty`.

        The weight for each velocity channel is given by the value of `qty` of the respective neighboring node.

        Parameters
        ----------
        qty : :py:class:`numpy.ndarray`
            Scalar field with the same shape as ``self.cell_density``.

        Returns
        -------
        :py:class:`numpy.ndarray` of `float`
            Weights for the velocity channels of shape ``self.dims + (self.velocitychannels,)``.
        """
        weights = np.zeros(qty.shape + (self.velocitychannels,))
        weights[:-1, :, :, 0] = qty[1:, :, :, ...]
        weights[1:, :, :, 1] = qty[:-1, :, :, ...]
        weights[:, :-1, :, 2] = qty[:, 1:, :, ...]
        weights[:, 1:, :, 3] = qty[:, :-1, :, ...]
        weights[:, :, :-1, 4] = qty[:, :, 1:, ...]
        weights[:, :, 1:, 5] = qty[:, :, :-1, ...]
        return weights

    # ------------------------------------------------------------------
    # Three-dimensional plotting with Mayavi
    #
    # One implementation serves every model family on cubic and Moore
    # lattices: `_channel_counts` converts occupancy, particle counts,
    # identity labels and label lists to per-channel particle numbers.

    def _is_nove(self):
        """Whether the model has no volume exclusion, i.e. unbounded node occupancy."""
        from lgca.nove_base import NoVE_LGCA_base
        from lgca.nove_ib_base import NoVE_IBLGCA_base

        return isinstance(self, (NoVE_LGCA_base, NoVE_IBLGCA_base))

    def _count_limit(self, counts, capacity):
        """Colour and glyph-size limit for particle counts.

        With volume exclusion this is the fixed ``capacity`` of the plotted
        channels. Without volume exclusion it is the largest count in the data.
        """
        if self._is_nove():
            return max(float(np.max(counts, initial=0)), 1.0)
        return float(capacity)

    def _density_capacity(self, channels=slice(None)):
        """Maximum number of particles in the selected channels of one node with volume exclusion."""
        return len(range(self.K)[channels]) * getattr(self, "n_species", 1)

    def _node_density(self, density=None, channels=slice(None), species=None):
        """Particle number per node of the given or the current lattice state."""
        if density is None and species is None:
            return self._channel_counts(self.nodes[self.nonborder])[..., channels].sum(-1)
        return select_density(self, density=density, channels=channels, species=species)

    def _node_centres(self):
        """Coordinates of the node centres; node ``i`` occupies the unit cell ``[i, i + 1]``."""
        return self.xcoords + 0.5, self.ycoords + 0.5, self.zcoords + 0.5

    @staticmethod
    def _smoothed(density, smooth):
        """Gaussian-smoothed copy of ``density`` for display; ``smooth`` is the width in lattice units."""
        density = np.asarray(density, dtype=float)
        if not smooth:
            return density
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(density, smooth, mode="nearest")

    def _max_node_flux(self):
        """Largest net flux of one node with volume exclusion: all channels pointing into one half-space."""
        c = self.c.T
        directions = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)]) - 1
        largest = max(np.linalg.norm(c[c @ d > 0].sum(0)) for d in directions)
        return largest * getattr(self, "n_species", 1)

    @staticmethod
    def _arrow_scale(flux):
        """Arrow scale factor that draws the largest flux 0.9 lattice units long."""
        largest = float(np.max(np.linalg.norm(flux, axis=-1), initial=0))
        return 0.9 / largest if largest > 0 else 1.0

    def _voxels(self, fig, values, visible, colormap, opacity, vmin, vmax, cube_size, **kwargs):
        """Draw one cube per visible node, coloured by ``values``; None if no node is visible."""
        if not np.any(visible):
            return None
        x, y, z = (coords[visible] for coords in self._node_centres())
        cubes = mlab.points3d(x, y, z, values[visible], mode="cube", scale_mode="none", scale_factor=cube_size,
                              colormap=colormap, vmin=vmin, vmax=vmax, figure=fig, **kwargs)
        style_surface(cubes.actor, opacity)
        return cubes

    def _isosurfaces(self, fig, values, contours, colormap, opacity, vmin, vmax, **kwargs):
        """Draw translucent isosurfaces of ``values`` at ``contours`` evenly spaced or explicit levels."""
        if np.ndim(contours) == 0:
            contours = np.linspace(vmin, vmax, int(contours) + 2)[1:-1]
        contour = mlab.contour3d(*self._node_centres(), values, colormap=colormap, vmin=vmin, vmax=vmax, figure=fig,
                                 **kwargs)
        # Mayavi clips levels to the current data range; pin the range to the colour scale so that
        # levels stay fixed when animation frames change the data.
        component = contour.contour
        component.auto_update_range = False
        component.trait_set(_data_min=float(vmin), _data_max=float(vmax), trait_change_notify=False)
        component.contours = [float(c) for c in contours]
        style_surface(contour.actor, opacity)
        return contour

    def _density_figure(self, density, contours, colormap, opacity, vmax, smooth, cbar, cbarlabel, size, view,
                        **kwargs):
        fig = new_figure(size)
        contour = self._isosurfaces(fig, self._smoothed(density, smooth), contours, colormap, opacity, 0.0, vmax,
                                    **kwargs)
        decorate_domain(fig, self.dims, view=view)
        if cbar:
            add_colorbar(contour, cbarlabel, 0, vmax, integer=True, discrete=False)
        return fig, contour

    @staticmethod
    def _sphere_sizes(counts, limit):
        """Relative sphere diameters with volumes proportional to the particle number; `limit` particles give 1."""
        return np.cbrt(np.asarray(counts, dtype=float) / limit)

    def _stationary_sizes(self, flux, density, limit):
        """Sphere sizes that mark occupied nodes without net flux."""
        return (np.linalg.norm(flux, axis=-1) == 0) * self._sphere_sizes(density, limit)

    def _flux_figure(self, flux, density, scale_factor, density_limit, color, colormap, opacity, cbar, size, view,
                     **kwargs):
        fig = new_figure(size)
        x, y, z = self._node_centres()
        magnitude = np.linalg.norm(flux, axis=-1)
        quiver = mlab.quiver3d(x, y, z, flux[..., 0], flux[..., 1], flux[..., 2], scalars=magnitude, mode="arrow",
                               scale_mode="vector", color=color, colormap=colormap, scale_factor=scale_factor, vmin=0,
                               vmax=max(float(magnitude.max(initial=0)), 1e-12), figure=fig, **kwargs)
        if color is None:
            quiver.glyph.color_mode = "color_by_scalar"
        quiver.glyph.glyph_source.glyph_position = "center"
        style_arrows(quiver, opacity)
        scatter = mlab.points3d(x, y, z, self._stationary_sizes(flux, density, density_limit), scale_factor=0.5,
                                scale_mode="scalar", color=MUTED, resolution=16, figure=fig)
        style_surface(scatter.actor, opacity)
        decorate_domain(fig, self.dims, view=view)
        if cbar and color is None:
            add_colorbar(quiver, "Flux magnitude", 0, float(magnitude.max(initial=0)) or 1.0)
        return fig, quiver, scatter

    def _config_vectors(self, velocity):
        """Arrows from the node centre towards the neighbour of every occupied velocity channel."""
        occupied = 0.45 * (velocity > 0)
        return [self.c[axis] * occupied for axis in range(3)]

    def _config_figure(self, velocity, rest, velocity_limit, rest_limit, color, colormap, cbar, size, view,
                       **kwargs):
        # Arrows have a fixed length; with more than one possible particle per channel their colour shows
        # the channel population.
        fig = new_figure(size)
        x, y, z = (np.repeat(coords[..., None], self.velocitychannels, axis=-1) for coords in self._node_centres())
        u, v, w = self._config_vectors(velocity)
        coloured = velocity_limit > 1
        quiver = mlab.quiver3d(x, y, z, u, v, w, scalars=velocity, mode="arrow", scale_mode="vector",
                               scale_factor=1.0, color=None if coloured else INK, colormap=colormap, vmin=1,
                               vmax=max(velocity_limit, 2), figure=fig, **kwargs)
        if coloured:
            quiver.glyph.color_mode = "color_by_scalar"
        style_arrows(quiver)
        scatter = None
        if self.restchannels > 0:
            scatter = mlab.points3d(*self._node_centres(), self._sphere_sizes(rest, rest_limit), scale_factor=0.6,
                                    scale_mode="scalar", color=color, resolution=16, figure=fig)
            style_surface(scatter.actor)
        decorate_domain(fig, self.dims, view=view)
        if cbar and coloured:
            add_colorbar(quiver, "Particles", 1, velocity_limit, integer=True)
        return fig, quiver, scatter

    def plot_density(self, density=None, channels=slice(None), species=None, contours=3, colormap="viridis",
                     opacity=0.35, vmax=None, smooth=0.0, cbar=True, cbarlabel="Particles", size=None,
                     view=None, **kwargs):
        """
        Plot nested isosurfaces of the particle density with Mayavi.

        Parameters
        ----------
        density : :py:class:`numpy.ndarray`, optional
            Particle number per node, dimensions ``self.dims`` (plus a species axis for multi-species models).
            Default: the current lattice state.
        channels : slice, default=slice(None)
            Channels that count towards the density if `density` is None.
        species : int, optional
            Plot only this species of a multi-species model.
        contours : int or sequence of float, default=3
            Number of evenly spaced isosurfaces between 0 and `vmax`, or explicit density levels.
        colormap : str, default='viridis'
            Colormap for the isosurfaces.
        opacity : float, default=0.35
            Opacity of the isosurfaces; inner surfaces stay visible through outer ones.
        vmax : float, optional
            Upper limit of the colour scale. Default: the node capacity with volume exclusion, the largest density
            otherwise.
        smooth : float, default=0
            Width in lattice units of a Gaussian filter applied to the density before contouring. Only affects the
            display; use it to show the shape of a population instead of single-node fluctuations.
        cbar : bool, default=True
            Whether to draw a colour bar.
        cbarlabel : str, default='Particles'
            Colour bar title.
        size : tuple of int, optional
            Figure size in pixels.
        view : dict, optional
            Camera settings passed to :func:`mayavi.mlab.view`, e.g. ``{'azimuth': 0, 'elevation': 90}``.
        **kwargs
            Passed to :func:`mayavi.mlab.contour3d`.

        Returns
        -------
        fig : mayavi.core.scene.Scene
            The figure.
        contour : mayavi.modules.iso_surface.IsoSurface
            The isosurfaces.

        See Also
        --------
        plot_density_cubes : Show the density of every node as a coloured cube.
        """
        density = self._node_density(density, channels, species)
        if vmax is None:
            vmax = self._count_limit(density, self._density_capacity(channels))
        return self._density_figure(density, contours, colormap, opacity, vmax, smooth, cbar, cbarlabel, size, view,
                                    **kwargs)

    def plot_density_cubes(self, density=None, channels=slice(None), species=None, colormap="viridis", opacity=1.0,
                           vmax=None, cube_size=0.9, cbar=True, cbarlabel="Particles", size=None, view=None,
                           **kwargs):
        """
        Plot the particle density as one coloured cube per occupied node with Mayavi.

        Parameters
        ----------
        density, channels, species, colormap, vmax, cbar, cbarlabel, size, view
            As in :py:meth:`plot_density`.
        opacity : float, default=1.0
            Opacity of the cubes.
        cube_size : float, default=0.9
            Edge length of the cubes in lattice units; values below 1 leave gaps between neighbouring nodes.
        **kwargs
            Passed to :func:`mayavi.mlab.points3d`.

        Returns
        -------
        fig : mayavi.core.scene.Scene
            The figure.
        cubes : mayavi.modules.glyph.Glyph or None
            The cubes, or None if the lattice is empty.
        """
        density = self._node_density(density, channels, species)
        if vmax is None:
            vmax = self._count_limit(density, self._density_capacity(channels))
        fig = new_figure(size)
        cubes = self._voxels(fig, density, density > 0, colormap, opacity, 0, vmax, cube_size, **kwargs)
        decorate_domain(fig, self.dims, view=view)
        if cbar and cubes is not None:
            add_colorbar(cubes, cbarlabel, 0, vmax, integer=True)
        return fig, cubes

    def plot_scalarfield(self, field, mask=None, colormap="viridis", opacity=None, cbar=True,
                         cbarlabel="Scalar field", vmin=None, vmax=None, contours=3, cube_size=0.9, size=None,
                         view=None, **kwargs):
        """
        Plot a scalar field on the lattice with Mayavi.

        Nodes that are masked are hidden. A field with hidden nodes, such as the mean cell property of occupied
        nodes, is drawn as one cube per visible node; a complete field is drawn as isosurfaces.

        Parameters
        ----------
        field : :py:class:`numpy.ndarray` or :py:class:`numpy.ma.MaskedArray`
            Values with dimensions ``self.dims`` (or including the border). Masked entries are hidden.
        mask : :py:class:`numpy.ndarray` of bool, optional
            Additional nodes to hide (True = hidden).
        colormap : str, default='viridis'
            Colormap.
        opacity : float, optional
            Opacity; default 1 for cubes and 0.35 for isosurfaces.
        cbar : bool, default=True
            Whether to draw a colour bar.
        cbarlabel : str, default='Scalar field'
            Colour bar title.
        vmin, vmax : float, optional
            Colour scale limits. Default: range of the visible values.
        contours : int or sequence of float, default=3
            Isosurface count or levels for a complete field.
        cube_size : float, default=0.9
            Cube edge length for a field with hidden nodes.
        size, view
            As in :py:meth:`plot_density`.
        **kwargs
            Passed to :func:`mayavi.mlab.points3d` or :func:`mayavi.mlab.contour3d`.

        Returns
        -------
        fig : mayavi.core.scene.Scene
            The figure.
        obj : mayavi.modules.glyph.Glyph or mayavi.modules.iso_surface.IsoSurface or None
            The cubes or isosurfaces; None if every node is hidden.
        colormap : str
            The colormap.
        """
        field = select_scalar_field(self, field)
        hidden = np.ma.getmaskarray(field)
        if mask is not None:
            hidden = hidden | np.asarray(mask, dtype=bool)
        values = np.ma.getdata(field).astype(float)
        shown = values[~hidden]
        if vmin is None:
            vmin = float(shown.min()) if shown.size else 0.0
        if vmax is None:
            vmax = float(shown.max()) if shown.size else 1.0
        if vmax <= vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        fig = new_figure(size)
        if hidden.any():
            obj = self._voxels(fig, values, ~hidden, colormap, 1.0 if opacity is None else opacity, vmin, vmax,
                               cube_size, **kwargs)
        else:
            obj = self._isosurfaces(fig, values, contours, colormap, 0.35 if opacity is None else opacity, vmin,
                                    vmax, **kwargs)
        decorate_domain(fig, self.dims, view=view)
        if cbar and obj is not None:
            add_colorbar(obj, cbarlabel, vmin, vmax)
        return fig, obj, colormap

    def plot_flux(self, nodes=None, scale_factor=None, color=INK, colormap="viridis", opacity=1.0, cbar=False,
                  size=None, view=None, **kwargs):
        """
        Plot the net particle flux of every node as arrows with Mayavi.

        Grey spheres mark occupied nodes without net flux; their size shows the particle number.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`, optional
            Lattice configuration. Default: the current state.
        scale_factor : float, optional
            Arrow length per unit flux. Default: the largest flux is drawn 0.9 lattice units long.
        color : tuple of float or None, default=dark grey
            Arrow colour. None colours the arrows by flux magnitude with `colormap`.
        colormap : str, default='viridis'
            Colormap for ``color=None``.
        opacity : float, default=1.0
            Opacity of arrows and spheres.
        cbar : bool, default=False
            Whether to draw a colour bar of the flux magnitude (only with ``color=None``).
        size, view
            As in :py:meth:`plot_density`.
        **kwargs
            Passed to :func:`mayavi.mlab.quiver3d`.

        Returns
        -------
        fig : mayavi.core.scene.Scene
            The figure.
        quiver : mayavi.modules.vectors.Vectors
            The arrows.
        scatter : mayavi.modules.glyph.Glyph
            The spheres of occupied nodes without net flux.
        """
        counts = self._channel_counts(self.nodes[self.nonborder] if nodes is None else nodes).astype(float)
        flux = self.calc_flux(counts)
        density = counts.sum(-1)
        if scale_factor is None:
            scale_factor = self._arrow_scale(flux)
        limit = self._count_limit(density, self._density_capacity())
        return self._flux_figure(flux, density, scale_factor, limit, color, colormap, opacity, cbar, size, view,
                                 **kwargs)

    def plot_config(self, nodes=None, vmax=None, color=MUTED, colormap="viridis", cbar=True, size=None, view=None,
                    **kwargs):
        """
        Plot the channel configuration of every node with Mayavi.

        Each occupied velocity channel is an arrow from the node centre towards the corresponding neighbour. If a
        channel can hold more than one particle (without volume exclusion or with several species), the arrow colour
        shows its particle number. Spheres at the node centres show the resting particles; their volume is
        proportional to the particle number.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`, optional
            Lattice configuration. Default: the current state.
        vmax : float, optional
            Upper limit of the arrow colour scale. Default: the number of species with volume exclusion, the largest
            channel population otherwise.
        color : tuple of float, default=grey
            Colour of the rest-particle spheres.
        colormap : str, default='viridis'
            Colormap for the channel populations.
        cbar : bool, default=True
            Whether to draw a colour bar when the arrows are coloured.
        size, view
            As in :py:meth:`plot_density`.
        **kwargs
            Passed to :func:`mayavi.mlab.quiver3d`.

        Returns
        -------
        fig : mayavi.core.scene.Scene
            The figure.
        quiver : mayavi.modules.vectors.Vectors
            The channel arrows.
        scatter : mayavi.modules.glyph.Glyph or None
            The rest-particle spheres, or None without rest channels.
        """
        counts = self._channel_counts(self.nodes[self.nonborder] if nodes is None else nodes).astype(float)
        velocity = counts[..., : self.velocitychannels]
        rest = counts[..., self.velocitychannels:].sum(-1)
        n_species = getattr(self, "n_species", 1)
        velocity_limit = vmax or self._count_limit(velocity, n_species)
        rest_limit = self._count_limit(rest, self.restchannels * n_species)
        return self._config_figure(velocity, rest, velocity_limit, rest_limit, color, colormap, cbar, size, view,
                                   **kwargs)

    def animate_density(self, density_t=None, contours=3, colormap="viridis", opacity=0.35, cbar=True, interval=100,
                        steps=None, channels=slice(None), species=None, vmax=None, smooth=0.0, show=True, **kwargs):
        """
        Animate the density isosurfaces of a recorded simulation with Mayavi.

        Parameters
        ----------
        density_t : :py:class:`numpy.ndarray`, optional
            Density history with dimensions ``(time,) + self.dims``. Default: the recorded density.
        interval : int, default=100
            Delay between frames in milliseconds.
        steps : sequence of int, optional
            Time step of every frame for the time label. Default: the recorded sample times.
        vmax : float, optional
            Upper colour limit, fixed for the whole animation. Default: the node capacity with volume exclusion, the
            largest recorded density otherwise.
        show : bool, default=True
            Whether to enter the GUI event loop; see :func:`lgca.mayavi_style.play`.
        contours, colormap, opacity, cbar, channels, species, smooth
            As in :py:meth:`plot_density`.
        **kwargs
            As in :py:meth:`plot_density`.

        Returns
        -------
        mayavi.tools.animator.Animator
            The animation controller.
        """
        density_t, steps = resolve_animation_history(self, "density_t", density_t, steps, channels)
        density_t = select_density_history(self, density_t, species)
        if vmax is None:
            vmax = self._count_limit(density_t, self._density_capacity(channels))
        fig, contour = self._density_figure(density_t[0], contours, colormap, opacity, vmax, smooth, cbar,
                                            kwargs.pop("cbarlabel", "Particles"), kwargs.pop("size", None),
                                            kwargs.pop("view", None), **kwargs)

        def update(frame):
            contour.mlab_source.set(scalars=self._smoothed(density_t[frame], smooth))

        return play(fig, update, len(density_t), lambda frame: f"t = {steps[frame]}", interval, show)

    def animate_flux(self, nodes_t=None, scale_factor=None, color=INK, colormap="viridis", opacity=1.0, cbar=False,
                     interval=100, steps=None, show=True, **kwargs):
        """
        Animate the net particle flux of a recorded simulation with Mayavi.

        The arrow scale is fixed for the whole animation so that arrow lengths are comparable between frames.

        Parameters
        ----------
        nodes_t : :py:class:`numpy.ndarray`, optional
            Recorded lattice configurations with dimensions ``(time,) + self.dims + (self.K,)``. Default: the
            recorded node history.
        interval, steps, show
            As in :py:meth:`animate_density`.
        scale_factor, color, colormap, opacity, cbar
            As in :py:meth:`plot_flux`.
        **kwargs
            As in :py:meth:`plot_flux`.

        Returns
        -------
        mayavi.tools.animator.Animator
            The animation controller.
        """
        nodes_t, steps = resolve_animation_history(self, "nodes_t", nodes_t, steps)
        counts_t = self._channel_counts(nodes_t, history=True).astype(float)
        flux_t = self.calc_flux(counts_t)
        density_t = counts_t.sum(-1)
        if scale_factor is None:
            scale_factor = self._arrow_scale(flux_t)
        limit = self._count_limit(density_t, self._density_capacity())
        fig, quiver, scatter = self._flux_figure(flux_t[0], density_t[0], scale_factor, limit, color, colormap,
                                                 opacity, cbar, kwargs.pop("size", None), kwargs.pop("view", None),
                                                 **kwargs)

        def update(frame):
            flux = flux_t[frame]
            quiver.mlab_source.set(u=flux[..., 0], v=flux[..., 1], w=flux[..., 2],
                                   scalars=np.linalg.norm(flux, axis=-1))
            scatter.mlab_source.set(scalars=self._stationary_sizes(flux, density_t[frame], limit))

        return play(fig, update, len(flux_t), lambda frame: f"t = {steps[frame]}", interval, show)

    def animate_config(self, nodes_t=None, interval=100, steps=None, vmax=None, color=MUTED, colormap="viridis",
                       cbar=True, show=True, **kwargs):
        """
        Animate the channel configuration of a recorded simulation with Mayavi.

        Parameters
        ----------
        nodes_t : :py:class:`numpy.ndarray`, optional
            Recorded lattice configurations. Default: the recorded node history.
        interval, steps, show
            As in :py:meth:`animate_density`.
        vmax, color, colormap, cbar
            As in :py:meth:`plot_config`; the default `vmax` covers the whole history.
        **kwargs
            As in :py:meth:`plot_config`.

        Returns
        -------
        mayavi.tools.animator.Animator
            The animation controller.
        """
        nodes_t, steps = resolve_animation_history(self, "nodes_t", nodes_t, steps)
        counts_t = self._channel_counts(nodes_t, history=True).astype(float)
        velocity_t = counts_t[..., : self.velocitychannels]
        rest_t = counts_t[..., self.velocitychannels:].sum(-1)
        n_species = getattr(self, "n_species", 1)
        velocity_limit = vmax or self._count_limit(velocity_t, n_species)
        rest_limit = self._count_limit(rest_t, self.restchannels * n_species)
        fig, quiver, scatter = self._config_figure(velocity_t[0], rest_t[0], velocity_limit, rest_limit, color,
                                                   colormap, cbar, kwargs.pop("size", None), kwargs.pop("view", None),
                                                   **kwargs)

        def update(frame):
            u, v, w = self._config_vectors(velocity_t[frame])
            quiver.mlab_source.set(u=u, v=v, w=w, scalars=velocity_t[frame])
            if scatter is not None:
                scatter.mlab_source.set(scalars=self._sphere_sizes(rest_t[frame], rest_limit))

        return play(fig, update, len(counts_t), lambda frame: f"t = {steps[frame]}", interval, show)

    def live_animate_density(self, interval=100, channels=slice(None), species=None, contours=3, colormap="viridis",
                             opacity=0.35, vmax=None, smooth=0.0, cbar=True, show=True, **kwargs):
        """
        Simulate and show the density isosurfaces after every time step until the window is closed.

        Parameters
        ----------
        vmax : float, optional
            Upper colour limit. Default: the node capacity with volume exclusion, twice the initial maximum
            density otherwise.
        interval, show
            As in :py:meth:`animate_density`.
        channels, species, contours, colormap, opacity, smooth, cbar
            As in :py:meth:`plot_density`.
        **kwargs
            As in :py:meth:`plot_density`.

        Returns
        -------
        mayavi.tools.animator.Animator
            The animation controller.
        """
        density = self._node_density(None, channels, species)
        if vmax is None:
            vmax = self._count_limit(2 * density, self._density_capacity(channels))
        fig, contour = self._density_figure(density, contours, colormap, opacity, vmax, smooth, cbar,
                                            kwargs.pop("cbarlabel", "Particles"), kwargs.pop("size", None),
                                            kwargs.pop("view", None), **kwargs)

        def update(frame):
            if frame:
                self.timestep()
            contour.mlab_source.set(scalars=self._smoothed(self._node_density(None, channels, species), smooth))

        return play(fig, update, None, lambda frame: f"t = {frame}", interval, show)

    def live_animate_flux(self, interval=100, scale_factor=None, color=INK, colormap="viridis", opacity=1.0,
                          cbar=False, show=True, **kwargs):
        """
        Simulate and show the net particle flux after every time step until the window is closed.

        Parameters
        ----------
        scale_factor : float, optional
            Arrow length per unit flux. Default: the largest possible node flux with volume exclusion, or the largest
            initial flux without, is drawn 0.9 lattice units long.
        interval, show
            As in :py:meth:`animate_density`.
        color, colormap, opacity, cbar
            As in :py:meth:`plot_flux`.
        **kwargs
            As in :py:meth:`plot_flux`.

        Returns
        -------
        mayavi.tools.animator.Animator
            The animation controller.
        """
        def state():
            counts = self._channel_counts(self.nodes[self.nonborder]).astype(float)
            return self.calc_flux(counts), counts.sum(-1)

        flux, density = state()
        if scale_factor is None:
            scale_factor = self._arrow_scale(flux) if self._is_nove() else 0.9 / self._max_node_flux()
        limit = self._count_limit(2 * density, self._density_capacity())
        fig, quiver, scatter = self._flux_figure(flux, density, scale_factor, limit, color, colormap, opacity, cbar,
                                                 kwargs.pop("size", None), kwargs.pop("view", None), **kwargs)

        def update(frame):
            if frame:
                self.timestep()
            flux, density = state()
            quiver.mlab_source.set(u=flux[..., 0], v=flux[..., 1], w=flux[..., 2],
                                   scalars=np.linalg.norm(flux, axis=-1))
            scatter.mlab_source.set(scalars=self._stationary_sizes(flux, density, limit))

        return play(fig, update, None, lambda frame: f"t = {frame}", interval, show)


def __getattr__(name):
    """Lazily expose cubic-lattice extension classes."""
    if name == "IBLGCA_Cubic":
        from .cubic_ext import IBLGCA_Cubic
        return IBLGCA_Cubic
    if name == "NoVE_LGCA_Cubic":
        from .cubic_ext import NoVE_LGCA_Cubic
        return NoVE_LGCA_Cubic
    if name == "NoVE_IBLGCA_Cubic":
        from .cubic_ext import NoVE_IBLGCA_Cubic
        return NoVE_IBLGCA_Cubic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
