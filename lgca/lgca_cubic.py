import numpy as np
from lgca.base import *
from mayavi import mlab

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
    interactions = ['go_and_grow', 'go_or_grow', 'alignment', 'aggregation',
                    'random_walk', 'excitable_medium', 'nematic', 'persistent_motion',
                    'chemotaxis', 'contact_guidance', 'only_propagation']
    velocitychannels = 6  # +x, -x, +y, -y, +z, -z

    # Build velocity channel vectors
    cix = np.array([1, -1, 0, 0, 0, 0], dtype=float)
    ciy = np.array([0, 0, 1, -1, 0, 0], dtype=float)
    ciz = np.array([0, 0, 0, 0, 1, -1], dtype=float)
    c = np.array([cix, ciy, ciz])

    def set_dims(self, dims=None, nodes=None, restchannels=0):
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
            self.restchannels = self.K - self.velocitychannels
            self.dims = (self.lx, self.ly, self.lz)
            return

        if dims is None:
            dims = (10, 10, 10)

        # set dimensions to keyword value
        if isinstance(dims, tuple):
            if len(dims) != 3:
                raise ValueError("For 3D cubic lattice, 'dims' must be a tuple of three integers.")
            self.lx, self.ly, self.lz = dims

        elif isinstance(dims, int):
            self.lx = self.ly = self.lz = dims
        else:
            raise TypeError("Keyword 'dims' must be a tuple of three integers or int!")

        self.dims = (self.lx, self.ly, self.lz)
        self.restchannels = restchannels
        self.K = self.velocitychannels + self.restchannels

    def init_nodes(self, density=0.1, nodes=None, **kwargs):
        """
        Initialize LGCA lattice configuration. Create the lattice and then assign particles to channels in the nodes.

        Initializes :py:attr:`self.nodes`. If `nodes` is not provided, the lattice is initialized with particles
        randomly so that the average lattice density is `density`.

        Parameters
        ----------
        density : float, default=0.1
            If `nodes` is None, initialize lattice randomly with this particle density.
        nodes : :py:class:`numpy.ndarray`
            Custom initial lattice configuration. Dimensions: ``(self.lx, self.ly, self.lz, self.K)``.

        See Also
        --------
        base.LGCA_base.random_reset : Initialize lattice nodes with average density `density`.
        set_dims : Set LGCA dimensions.
        init_coords : Initialize LGCA coordinates.
        """
        self.nodes = np.zeros((self.lx + 2 * self.r_int, self.ly + 2 * self.r_int, self.lz + 2 * self.r_int,
                               self.K), dtype=bool)

        if nodes is None:
            self.random_reset(density)
        else:
            self.nodes[self.nonborder] = nodes.astype(bool)
            self.apply_boundaries()

    def init_coords(self):
        """
        Initialize LGCA coordinates for a 3D cubic lattice.
        """
        x = np.arange(self.lx) + self.r_int
        y = np.arange(self.ly) + self.r_int
        z = np.arange(self.lz) + self.r_int
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.nonborder = (xx, yy, zz)
        self.coord_triples = list(zip(xx.flat, yy.flat, zz.flat))
        self.xcoords, self.ycoords, self.zcoords = np.meshgrid(
            np.arange(self.lx + 2 * self.r_int) - self.r_int,
            np.arange(self.ly + 2 * self.r_int) - self.r_int,
            np.arange(self.lz + 2 * self.r_int) - self.r_int,
            indexing='ij'
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
        newnodes[..., self.velocitychannels:] = self.nodes[..., self.velocitychannels:]

        # propagation in each direction
        newnodes[1:, :, :, 0] = self.nodes[:-1, :, :, 0]
        newnodes[:-1, :, :, 1] = self.nodes[1:, :, :, 1]
        newnodes[:, 1:, :, 2] = self.nodes[:, :-1, :, 2]
        newnodes[:, :-1, :, 3] = self.nodes[:, 1:, :, 3]
        newnodes[:, :, 1:, 4] = self.nodes[:, :, :-1, 4]
        newnodes[:, :, :-1, 5] = self.nodes[:, :, 1:, 5]

        self.nodes = newnodes

    def _apply_pbc_x(self):
        self.nodes[:self.r_int, ...] = self.nodes[-2 * self.r_int:-self.r_int, ...]
        self.nodes[-self.r_int:, ...] = self.nodes[self.r_int:2 * self.r_int, ...]

    def _apply_pbc_y(self):
        self.nodes[:, :self.r_int, :] = self.nodes[:, -2 * self.r_int:-self.r_int, :]
        self.nodes[:, -self.r_int:, :] = self.nodes[:, self.r_int:2 * self.r_int, :]

    def _apply_pbc_z(self):
        self.nodes[:, :, :self.r_int] = self.nodes[:, :, -2 * self.r_int:-self.r_int]
        self.nodes[:, :, -self.r_int:] = self.nodes[:, :, self.r_int:2 * self.r_int]

    def apply_pbc(self):
        self._apply_pbc_x()
        self._apply_pbc_y()
        self._apply_pbc_z()

    def _apply_rbc_x(self):
        self.nodes[self.r_int, :, :, 0] += self.nodes[self.r_int - 1, :, :, 1]
        self.nodes[-self.r_int - 1, :, :, 1] += self.nodes[-self.r_int, :, :, 0]

    def _apply_rbc_y(self):
        self.nodes[:, self.r_int, :, 2] += self.nodes[:, self.r_int - 1, :, 3]
        self.nodes[:, -self.r_int - 1, :, 3] += self.nodes[:, -self.r_int, :, 2]

    def _apply_rbc_z(self):
        self.nodes[:, :, self.r_int, 4] += self.nodes[:, :, self.r_int - 1, 5]
        self.nodes[:, :, -self.r_int - 1, 5] += self.nodes[:, :, -self.r_int, 4]

    def apply_rbc(self):
        self._apply_rbc_x()
        self._apply_rbc_y()
        self._apply_rbc_z()
        self.apply_abc()

    def _apply_abc_x(self):
        self.nodes[:self.r_int, :, :, :] = 0
        self.nodes[-self.r_int:, :, :, :] = 0

    def _apply_abc_y(self):
        self.nodes[:, :self.r_int, :, :] = 0
        self.nodes[:, -self.r_int:, :, :] = 0

    def _apply_abc_z(self):
        self.nodes[:, :, :self.r_int, :] = 0
        self.nodes[:, :, -self.r_int:, :] = 0

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
        # Calculate the gradient of a quantity in the lattice
        gx, gy, gz = np.gradient(qty, 0.5)
        return np.stack((gx, gy, gz), axis=-1)

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

    def setup_mayavi_scene(self):
        # Use the current figure if available, otherwise create a new one.
        fig = mlab.gcf() if mlab.gcf() is not None else mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
        # Add an outline and axes to the current figure
        mlab.outline(color=(0, 0, 0), extent=[0, self.lx, 0, self.ly, 0, self.lz], opacity=0.6, line_width=1)
        axes = mlab.axes(nb_labels=5, xlabel='X', ylabel='Y', zlabel='Z', color=(0, 0, 0),
                         extent=[0, self.lx, 0, self.ly, 0, self.lz])
        axes.label_text_property.color = (0, 0, 0)
        axes.title_text_property.color = (0, 0, 0)
        return fig

    def plot_flux(self, nodes=None, scale_factor=1.0, opacity=0.5, cbar=False, **kwargs):
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
        scatter_size = (1 - np.sign(flux_norm)) * self.cell_density[self.nonborder] / self.K

        x = self.xcoords
        y = self.ycoords
        z = self.zcoords
        u = flux[..., 0]
        v = flux[..., 1]
        w = flux[..., 2]
        fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

        quiver = mlab.quiver3d(x, y, z, u, v, w,
                               mode='arrow', color=(0, 0, 0),
                               scale_factor=scale_factor,
                               opacity=opacity, vmax=np.sqrt(3), vmin=0, figure=fig,
                               **kwargs)
        scatter = mlab.points3d(x, y, z, scatter_size, scale_factor=scale_factor, color=(0, 0, 0),
                                opacity=opacity, figure=fig)
        fig = self.setup_mayavi_scene()
        mlab.title('Flux', size=0.4, color=(0, 0, 0))
        if cbar:
            mlab.colorbar(title='Flux Magnitude', orientation='vertical')
        return quiver, scatter

    def plot_density_surface(self, density=None, colormap='viridis', opacity=0.5, cbar=True, **kwargs):
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

        contour = mlab.contour3d(x, y, z, density, opacity=opacity, colormap=colormap, contours=self.K+1,
                                 vmin=0, vmax=self.K, **kwargs)
        fig = self.setup_mayavi_scene()
        mlab.title('Density Surface', size=0.4, color=(0, 0, 0))
        if cbar:
            colorbar = mlab.colorbar(title='Density', orientation='vertical', nb_labels=self.K + 1, label_fmt='%.0f',
                                     nb_colors=self.K + 1)
            colorbar.label_text_property.color = (0, 0, 0)
            colorbar.title_text_property.color = (0, 0, 0)
            # reduce the size of the colorbar labels
            colorbar.scalar_bar.unconstrained_font_size = True
            colorbar.label_text_property.font_size = 14
            colorbar.label_text_property.bold = False
            colorbar.label_text_property.vertical_justification = 'centered'

            # Assume self.K defines the number of discrete levels
            num_levels = self.K + 1  # Including zero or background

            # Create colorbar
            colorbar = mlab.colorbar(title='Density', orientation='vertical',
                                     nb_labels=num_levels, label_fmt='%.0f', nb_colors=num_levels)

            # Adjust text properties
            colorbar.label_text_property.color = (0, 0, 0)
            colorbar.title_text_property.color = (0, 0, 0)
            colorbar.label_text_property.font_size = 14
            colorbar.label_text_property.bold = False
        return contour

    def animate_density_surface(self, density_t=None, threshold=0.5, colormap='viridis', opacity=0.5, cbar=True,
                                       interval=100, **kwargs):
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
            if hasattr(self, 'dens_t'):
                density_t = self.dens_t
            else:
                raise RuntimeError("Density time series not found. Ensure to record density during simulation.")

        contour = self.plot_density_surface(density=density_t[0], colormap=colormap, opacity=opacity, cbar=cbar, **kwargs)
        mlab.title('Time 0', size=0.4)
        @mlab.animate(delay=interval)
        def anim():
            for i in range(density_t.shape[0]):
                contour.mlab_source.set(scalars=density_t[i], vmin=0, vmax=self.K)
                mlab.title(f'Time {i}', size=0.4)
                yield

        anim()
        mlab.show()

    def animate_flux(self, nodes_t=None, scale_factor=1.0, opacity=0.6, interval=100, cbar=False, **kwargs):
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
            if hasattr(self, 'nodes_t'):
                nodes_t = self.nodes_t
            else:
                raise RuntimeError(
                    "Channel-wise state of the lattice required for flux calculation but not recorded. "
                    "Call lgca.timeevo with keyword record=True")

        flux_t = self.calc_flux(nodes_t.astype(float))
        time_steps = flux_t.shape[0]
        scatter_sizes = (1 - np.sign(np.linalg.norm(flux_t, axis=-1))) * self.dens_t / self.K

        quiver, scatter = self.plot_flux(nodes=nodes_t[0], scale_factor=scale_factor, opacity=opacity, cbar=cbar, **kwargs)
        mlab.title('Flux at Time 0', size=0.4)
        @mlab.animate(delay=interval)
        def anim():
            for i in range(time_steps):
                quiver.mlab_source.set(u=flux_t[i, ..., 0],
                                       v=flux_t[i, ..., 1],
                                       w=flux_t[i, ..., 2])
                scatter.mlab_source.set(scalars=scatter_sizes[i])
                mlab.title(f'Flux at Time {i}', size=0.4)
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

        quiver, scatter = self.plot_flux(nodes=nodes, scale_factor=scale_factor, opacity=opacity, cbar=cbar, **kwargs)
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

                quiver.mlab_source.set(u=flux[..., 0],
                                       v=flux[..., 1],
                                       w=flux[..., 2], vmin=0, vmax=np.sqrt(3))
                scatter.mlab_source.set(scalars=(1 - np.sign(np.linalg.norm(flux, axis=-1))) * self.cell_density[self.nonborder] / self.K)
                mlab.title(f'Time {i}', size=0.4)
                yield

        anim()
        mlab.show()

    def live_animate_density_surface(self, **kwargs):
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
        fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
        contour = self.plot_density_surface(**kwargs)
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
                contour.mlab_source.set(scalars=self.cell_density[self.nonborder], vmin=0, vmax=self.K)
                mlab.title(f'Time {i}', size=0.4)
                yield

        anim()
        mlab.show()

if __name__ == "__main__":
    # Initialize LGCA on a 3D cubic lattice
    from __init__ import get_lgca
    L = 50
    nodes = np.zeros((L, L, L, 12), dtype=bool)
    nodes[L//2, L//2, L//2, :] = True
    lgca = get_lgca(geometry='cubic', interaction='birth',  nodes=nodes)
    # lgca.timeevo(timesteps=100, record=True)


    # Plot flux using Mayavi
    # lgca.plot_flux()
    #
    # # Plot density surface with threshold using Mayavi
    # lgca.plot_density_surface_mayavi(threshold=5, colormap='viridis')
    #
    # # Animate density surface
    # lgca.animate_density_surface_mayavi(density_t=lgca.dens_t, threshold=5, colormap='viridis')
    #
    # Animate flux
    # lgca.animate_flux()
    # lgca.live_animate_flux()
    # lgca.plot_density_surface()
    # lgca.animate_density_surface()
    lgca.live_animate_density_surface()
