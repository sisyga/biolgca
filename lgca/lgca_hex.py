# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2022 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
# The full license notice is found in the file lgca/__init__.py.

"""
Classes for two-dimensional LGCA on a hexagonal arm-share lattice. They specify
geometry-dependent LGCA behavior and inherit properties and structure from the
respective abstract base classes.
Objects of these classes can be used to simulate.

Supported LGCA types:

- classical LGCA (:py:class:`LGCA_Hex`)
- identity-based LGCA (:py:class:`IBLGCA_Hex`)
- LGCA without volume exclusion (:py:class:`NoVE_LGCA_Hex`)
"""

from lgca.base import *
from lgca.lgca_square import LGCA_Square, IBLGCA_Square, NoVE_LGCA_Square, NoVE_IBLGCA_Square

pi2 = 2 * np.pi

class LGCA_Hex(LGCA_Square):
    """
    Classical LGCA with volume exclusion on a 2D hexagonal lattice.

    It holds all methods and attributes that are specific for a hexagonal geometry.

    Attributes
    ----------
    coord_pairs : list of tuple
        Indices of non-border nodes in the :py:attr:`lgca.nodes` array, linearized, each tuple is (x-index, y-index).
    dy : float
        Scaling factor for the y axis. 0.87 for hexagonal geometry.
    lx, ly : int
        Lattice dimensions in x and y direction.
    orientation : float
        Attribute for drawing polygons that represent the nodes. Orientation of the polygon in rad.
        This is passed to :py:func:`matplotlib.patches.RegularPolygon()`. 0 for hexagonal geometry.
    r_poly : float
        Attribute for drawing polygons that represent the nodes. Distance between polygon center and vertices.
    xcoords, ycoords : np.ndarray
        Logical coordinates of non-border nodes starting with 0. Dimensions: ``(lgca.lx, lgca.ly)``.

    Warnings
    --------
    Boundary conditions only work for hexagonal LGCA with an even number of rows. LGCA with an uneven number of rows 
    should only be used for plotting.

    See Also
    --------
    base.LGCA_base : Base class with geometry-independent methods and attributes.

    """
    # set class attributes
    # interactions are inherited from 2D square LGCA
    velocitychannels = 6
    cix = np.cos(np.arange(velocitychannels) * pi2 / velocitychannels)
    ciy = np.sin(np.arange(velocitychannels) * pi2 / velocitychannels)
    c = np.array([cix, ciy])
    # attributes to draw polygons representing the nodes
    r_poly = 0.5 / np.cos(np.pi / velocitychannels)
    dy = np.sin(2 * np.pi / velocitychannels)
    orientation = 0.

    def init_coords(self):
        """
        Initialize LGCA coordinates.

        These are used to index the lattice nodes logically and programmatically (see below). 
        Initializes :py:attr:`self.nonborder`, :py:attr:`self.xcoords`, :py:attr:`self.ycoords` and 
        :py:attr:`self.coord_pairs`.
        
        See Also
        --------
        set_dims : Set LGCA dimensions.
        init_nodes : Initialize LGCA lattice configuration.
        set_r_int : Change the interaction radius.

        Notes
        --------
        :py:attr:`self.xcoords` and :py:attr:`self.ycoords` hold the logical coordinates of non-border nodes in x- and 
        y-direction starting with 0. Non-border nodes belong to the lattice in the mathematical definition of the LGCA, 
        while border nodes (=shadow nodes) are only included in order to implement boundary conditions. The coordinate 
        of every other row is shifted to the right by 0.5 in order to create a zig-zag boundary. 
        Note that since the lattice is two-dimensional, so are the coordinates.

        >>> lgca = get_lgca(geometry='hex', dims=2)
        >>> lgca.xcoords
        array([[0.5, 0. ],
               [1.5, 1. ]])
        >>> lgca.ycoords
        array([[0., 1.],
               [0., 1.]])

        A column in the printout is a row in the LGCA lattice. 
        :py:attr:`self.nonborder` holds the programmatical coordinates of non-border nodes, i.e. the indices of the 
        :py:attr:`self.nodes` array where non-border nodes are stored. This is why it is a tuple: Because it 
        is used to index a numpy array. All non-border lattice nodes can be called as ``self.nodes[self.nonborder]``.

        >>> lgca = get_lgca(geometry='hex', dims=2)  # default: periodic boundary conditions
        >>> lgca.r_int
        1
        >>> lgca.nodes.sum(-1)  # show contents of the lattice
        array([[1, 0, 1, 0],
               [0, 0, 0, 0],
               [1, 0, 1, 0],
               [0, 0, 0, 0]])
        >>> lgca.nodes[lgca.nonborder].sum(-1)
        array([[0, 0],
               [0, 1]])

        Summing along the last axis means summing over all channels of a node since we are interested in the geometry. 
        The first and the last row and column in the output of ``lgca.nodes.sum(-1)`` are the contents of the border 
        (=shadow) nodes, which reflects the interaction radius of 1. The innermost four elements are the contents of 
        the non-border nodes. Accordingly we find their indices to be:

        >>> lgca.nonborder
        (array([[1, 1],
                [2, 2]]),
         array([[1, 2],
                [1, 2]]))

        The first element of the tuple is the index in x-direction, the second element the index in y-direction. 
        Changing the interaction radius updates the shape of :py:attr:`self.nodes` by including more border (=shadow) 
        nodes. This also changes the coordinates. With an interaction radius of 2, there is 2 border nodes on each side 
        enveloping the non-border nodes whose contents remain the same. Therefore the first non-border node has the 
        index 2 in each direction.

        >>> lgca.set_r_int(2)  # change the interaction radius
        >>> lgca.r_int
        2
        >>> lgca.nodes.sum(-1)  # show contents of the lattice
        array([[0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1]])
        >>> lgca.nonborder
        (array([[2, 2],
                [3, 3]]),
         array([[2, 3],
                [2, 3]]))

        :py:attr:`self.coord_pairs` is a list of programmatical (x,y) coordinate tuples for iterating through nodes one
        by one.

        >>> lgca.set_r_int(1)
        >>> lgca.coord_pairs
        [(1, 1), (1, 2), (2, 1), (2, 2)]

        """
        if self.ly % 2 != 0:
            print('Warning: uneven number of rows; only use for plotting - boundary conditions do not work!')
        x = np.arange(self.lx) + self.r_int
        y = np.arange(self.ly) + self.r_int
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.coord_pairs = list(zip(xx.flat, yy.flat))

        # set all coords, including ghost cells
        self.xcoords, self.ycoords = np.meshgrid(np.arange(self.lx + 2 * self.r_int) - self.r_int,
                                                 np.arange(self.ly + 2 * self.r_int) - self.r_int, indexing='ij')
        self.xcoords = self.xcoords.astype(float)
        self.ycoords = self.ycoords.astype(float)

        # shift coordinate of every other row (x const.) by 0.5 to the right
        self.xcoords[:, 1::2] += 0.5
        self.ycoords *= self.dy
        self.xcoords = self.xcoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.ycoords = self.ycoords[self.r_int:-self.r_int, self.r_int:-self.r_int]
        self.nonborder = (xx, yy)

    def propagation(self):
        """
        Perform the transport step of the LGCA: Move particles through the lattice according to their velocity.

        Updates :py:attr:`self.nodes` such that resting particles (the contents of ``self.nodes[:, 6:]``) stay in their 
        position and particles in velocity channels (the contents of ``self.nodes[:, :6]``) are relocated according to 
        the direction of the channel they reside in. Boundary conditions are enforced later by 
        :py:meth:`apply_boundaries`.

        See Also
        --------
        base.LGCA_base.nodes : State of the lattice showing the structure of the ``lgca.nodes`` array.

        Notes
        --------
        >>> # set up the node configuration
        >>> nodes = np.zeros((4,4,7)).astype(bool)
        >>> nodes[1,1,:] = True
        >>> lgca = get_lgca(geometry='hex', nodes=nodes)
        >>> lgca.cell_density[lgca.nonborder]
        array([[0, 0, 0, 0],
               [0, 7, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
        >>> lgca.nodes[lgca.nonborder]
               # leftmost column of the lattice
        array([[[False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]],
               # column 1
               [[False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True], # node (1,1): all channels are filled
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]],
               # column 2
               [[False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]],
               # rightmost column
               [[False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]]])

        Before propagation, seven particles occupy node (1,1). It lies one node away from the bottom left of the 
        lattice. One particle resides in each velocity channel and one in the resting channel.

        >>> lgca.propagation()
        >>> lgca.update_dynamic_fields()  # to update lgca.cell_density
        >>> lgca.cell_density[lgca.nonborder]
        array([[1, 1, 1, 0], # left column of the lattice
               [1, 1, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]]) # right column of the lattice
        >>> lgca.nodes[lgca.nonborder]
               # leftmost column of the lattice
        array([[[False, False, False, False,  True, False, False], # node (0,0): particle moving diagonally downwards left
                [False, False, False,  True, False, False, False], # node (0,1): particle moving to the left
                [False, False,  True, False, False, False, False], # node (0,2): particle moving diagonally upwards left
                [False, False, False, False, False, False, False]],
               # column 1
               [[False, False, False, False, False,  True, False], # node (1,0): particle moving diagonally downwards right
                [False, False, False, False, False, False,  True], # node (1,1): resting particle
                [False,  True, False, False, False, False, False], # node (1,2): particle moving diagonally upwards right
                [False, False, False, False, False, False, False]],
               # column 2
               [[False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False], # node (2,1): particle moving to the right
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]],
               # rightmost column
               [[False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]]])
        >>> # perform lgca.plot_density() to visualise for clarity

        To interpret the cell density output, note that every other row of nodes is shifted to the right. Therefore 
        node (1,2) is positioned diagonally upwards to the right from node (1,1). There is no node straight upwards 
        from node (1,1).
        The particle with velocity to the right has moved to the right velocity channel in node (2,1) to the right of 
        node (1,1) and the particles in the other velocity channels have also moved according to their direction (see 
        output annotation). The resting particle stayed in its channel in node (1,1).

        """
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

    def _apply_rbcx(self):
        # documented in parent class
        # left boundary
        self.nodes[self.r_int, :, 0] += self.nodes[self.r_int - 1, :, 3]
        self.nodes[self.r_int, 2:-1:2, 1] += self.nodes[self.r_int - 1, 1:-2:2, 4]
        self.nodes[self.r_int, 2:-1:2, 5] += self.nodes[self.r_int - 1, 3::2, 2]

        # right boundary
        self.nodes[-self.r_int - 1, :, 3] += self.nodes[-self.r_int, :, 0]
        self.nodes[-self.r_int - 1, 1:-1:2, 4] += self.nodes[-self.r_int, 2::2, 1]
        self.nodes[-self.r_int - 1, 1:-1:2, 2] += self.nodes[-self.r_int, :-2:2, 5]

        self._apply_abcx()

    def _apply_rbcy(self):
        # documented in parent class
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
        self._apply_abcy()

    def gradient(self, qty):
        # documented in parent class
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
        """
        Calculate weights for the velocity channels in interactions depending on a field `qty`.

        The weight for the right/diagonal up right/diagonal up left/left/diagonal down left/diagonal down right 
        velocity channel is given by the value of `qty` of the respective neighboring node.

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
        """
        For each node, sum up the contents of `qty` for the 6 nodes in the von Neumann neughborhood, excluding the 
        center.

        `qty` is assumed to contain the value of a calculated quantity for each node in the lattice. `nb_sum` calculates
        the "neighborhood sum" of this quantity for each node, excluding the value for the node's own position.

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

        Examples
        --------
        >>> lgca = get_lgca(geometry='hex', density=0.3, dims=4) # periodic boundary conditions
        >>> lgca.cell_density[lgca.nonborder]
        array([[1, 2, 2, 1],
               [0, 2, 0, 0],
               [2, 2, 0, 1],
               [1, 2, 2, 3]])
        >>> lgca.nb_sum(lgca.cell_density).astype(int)[lgca.nonborder]
        array([[ 6, 10,  7,  9],
               [ 8,  7,  7,  5],
               [ 9,  6, 10,  5],
               [11,  9, 10,  7]])
        >>> # perform lgca.plot_density() to visualise for clarity

        ``lgca.cell_density`` is used as the argument `qty`. The value at each position in the resulting array is the 
        sum of the values at the neighboring positions in the source array. Note that the reduction to the non-border 
        nodes can only be done after the sum calculation in order to preserve boundary conditions. To interpret the 
        cell density output, note that every other row of nodes is shifted to the right. Therefore node (1,1) has the 
        neighborhood (0,0), (0,1), (0,2), (1,0), (1,2) and (2,1).

        """
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

    def setup_figure(self, figindex=None, figsize=(8, 8), tight_layout=True):
        # documented in parent class
        # create figure from parent method
        fig, ax = super(LGCA_Hex, self).setup_figure(figindex=figindex, figsize=figsize, tight_layout=tight_layout)
        # correct labels for y axis scaling
        plt.gca()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        return fig, ax


class IBLGCA_Hex(IBLGCA_Square, LGCA_Hex):
    """
    Identity-based LGCA simulator class.
    """
    interactions = ['go_or_grow', 'go_and_grow', 'random_walk', 'birth', 'birthdeath', 'birthdeath_discrete',
                    'only_propagation']


class NoVE_LGCA_Hex(NoVE_LGCA_Square, LGCA_Hex):

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
        sum[:, 1::2, ...] += qty[:, :-1:2, ...]
        sum[1:, 2::2, ...] += qty[:-1, 1:-1:2, ...]
        sum[:-1, 1::2, ...] += qty[1:, :-1:2, ...]
        sum[:, 2::2, ...] += qty[:, 1:-1:2, ...]
        sum[:, :-1:2, ...] += qty[:, 1::2, ...]
        sum[:-1, 1:-1:2, ...] += qty[1:, 2::2, ...]
        sum[1:, :-1:2, ...] += qty[:-1, 1::2, ...]
        sum[:, 1:-1:2, ...] += qty[:, 2::2, ...]
        # add central value
        if addCenter:
            sum += qty
        return sum


class NoVE_IBLGCA_Hex(NoVE_IBLGCA_Square, LGCA_Hex,  ):

    def propagation(self):
        newcellnodes = get_arr_of_empty_lists(self.nodes.shape)
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
        self.nodes[self.r_int, :, 0] = self.nodes[self.r_int - 1, :, 3] + self.nodes[self.r_int, :, 0]
        self.nodes[self.r_int, 2:-1:2, 1] = self.nodes[self.r_int - 1, 1:-2:2, 4] + self.nodes[self.r_int, 2:-1:2, 1]
        self.nodes[self.r_int, 2:-1:2, 5] = self.nodes[self.r_int - 1, 3::2, 2] + self.nodes[self.r_int, 2:-1:2, 5]

        # right boundary
        self.nodes[-self.r_int - 1, :, 3] = self.nodes[-self.r_int, :, 0] + self.nodes[-self.r_int - 1, :, 3]
        self.nodes[-self.r_int - 1, 1:-1:2, 4] = self.nodes[-self.r_int, 2::2, 1] + self.nodes[-self.r_int - 1, 1:-1:2, 4]
        self.nodes[-self.r_int - 1, 1:-1:2, 2] = self.nodes[-self.r_int, :-2:2, 5] + self.nodes[-self.r_int - 1, 1:-1:2, 2]

        self.apply_abcx()

    def apply_rbcy(self):
        lx, ly, _ = self.nodes.shape

        # lower boundary
        self.nodes[(1 - (self.r_int % 2)):, self.r_int, 1] = self.nodes[:lx - (1 - (self.r_int % 2)), self.r_int - 1,
            4] + self.nodes[(1 - (self.r_int % 2)):, self.r_int, 1]
        self.nodes[:lx - (self.r_int % 2), self.r_int, 2] = self.nodes[(self.r_int % 2):, self.r_int - 1,
            5] + self.nodes[:lx - (self.r_int % 2), self.r_int, 2]

        # upper boundary
        self.nodes[:lx - ((ly - 1 - self.r_int) % 2), -self.r_int - 1, 4] = self.nodes[((ly - 1 - self.r_int) % 2):,
            -self.r_int, 1] + self.nodes[:lx - ((ly - 1 - self.r_int) % 2), -self.r_int - 1, 4]
        self.nodes[(1 - ((ly - 1 - self.r_int) % 2)):, -self.r_int - 1, 5] = self.nodes[:lx - (1 - ((ly - 1 - self.r_int) % 2)),
            -self.r_int, 2] + self.nodes[(1 - ((ly - 1 - self.r_int) % 2)):, -self.r_int - 1, 5]
        self.apply_abcy()

