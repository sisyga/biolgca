import numpy as np
import numpy.random as npr
import pytest
from abc import ABC
from lgca import get_lgca


def calc_init_particles(ve, density, restchannels, K=None, b=None):
    if ve:
        return min(int(density * K) + 1, K) if restchannels % 2 else int(density * K)
    else:
        return int(density * (restchannels + b)) + 1 if restchannels % 2 else int(density * (restchannels + b))

class T_LGCA_Common(ABC):
    """
    Test setup that ve/nove, ib/non-ib have in common: parameters and test functions.
    Abstract base class: cannot be used to test directly
    """
    # common parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # reproducible but non-uniform fixups
    rng = npr.default_rng(1)
    # absolute tolerance for random densities
    density_epsilon = 0.1
    # 1D
    xdim_1d = 5
    b_1d = 2
    # 2D square
    xdim_square = 3
    ydim_square = 4
    b_square = 4
    # 2D hex
    b_hex = 6
    xdim_hex = 4
    ydim_hex = 6


# prefix or postfix testing functions with "test_"/"_test" - those cannot be parametrised normally (use fixups)
# prefix classes that contain tests with "Test" - ignored otherwise
class Test_LGCA_General:
    """
    Tests that can be generalised to do ve/nove, ib/non-ib via parametrisation.
    """
    com = T_LGCA_Common

    # classical parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    density_ve = 0.5
    # 1D
    restchannels_ve_1d = 4
    K_ve_1d = com.b_1d + restchannels_ve_1d
    nodes_ve_1d = com.rng.integers(low=0, high=1, endpoint=True,
                               size=(com.xdim_1d, com.b_1d + restchannels_ve_1d))
    # 2D square
    restchannels_ve_square = 2
    K_ve_square = com.b_square + restchannels_ve_square
    nodes_ve_square = com.rng.integers(low=0, high=1, endpoint=True,
                                   size=(com.xdim_square, com.ydim_square, com.b_square + restchannels_ve_square))
    # 2D hex
    restchannels_ve_hex = restchannels_ve_square
    K_ve_hex = com.b_hex + restchannels_ve_hex
    nodes_ve_hex = com.rng.integers(low=0, high=1, endpoint=True,
                                size=(com.xdim_hex, com.ydim_hex, com.b_hex + restchannels_ve_hex))

    # ib parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # nove parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # nodes must be defined in the correct size! Maximum size of last dimension = b+1
    density_nove_1 = 0.5
    density_nove_2 = 1.5
    # 1D
    restchannels_nove_1d = 3
    capacity_nove_1d = 5
    nodes_nove_1d = com.rng.integers(low=0, high=int(1.5 * capacity_nove_1d),
                                 size=(com.xdim_1d, com.b_1d + 1), endpoint=True)
    # 2D square
    restchannels_nove_square = 2
    capacity_nove_square = 6
    nodes_nove_square = com.rng.integers(low=0, high=int(1.5 * capacity_nove_square),
                                     size=(com.xdim_square, com.ydim_square, com.b_square + 1), endpoint=True)
    # 2D hex
    restchannels_nove_hex = restchannels_nove_square
    capacity_nove_hex = 8
    nodes_nove_hex = com.rng.integers(low=0, high=int(1.5 * capacity_nove_hex),
                                  size=(com.xdim_hex, com.ydim_hex, com.b_hex + 1), endpoint=True)


    # parameter tuples for 'node' keyword check
    @pytest.mark.parametrize("geom,ve,ib,nodes,dims,restchannels", [
        # classical LGCA (ve, non-ib)
        # (geom,    ve,   ib,    nodes,             dims,                               restchannels)
        ('lin',     True, False, nodes_ve_1d,       (com.xdim_1d,),                     restchannels_ve_1d),
        ('square',  True, False, nodes_ve_square,   (com.xdim_square, com.ydim_square), restchannels_ve_square),
        ('hex',     True, False, nodes_ve_hex,      (com.xdim_hex, com.ydim_hex),       restchannels_ve_hex),
        # IBLGCA (ve, ib)
        # NoVE_LGCA (nove, non-ib)
        # (geom,    ve,    ib,    nodes,                dims,                               restchannels)
        ('lin',     False, False, nodes_nove_1d,        (com.xdim_1d,),                     1),
        ('square',  False, False, nodes_nove_square,    (com.xdim_square, com.ydim_square), 1),
        ('hex',     False, False, nodes_nove_hex,       (com.xdim_hex, com.ydim_hex),       1)
    ])
    def test_getlgca_lattice_setup_nodes(self, geom, ve, ib, nodes, dims, restchannels):
        # 'node' keyword check: provided lattice is adopted by LGCA
        # if nodes are provided, 'restchannels', 'dims', 'density' and 'hom' have to be ignored
        lgca = get_lgca(geometry=geom, ve=ve, ib=ib, nodes=nodes, dims=dims, density=0.01, restchannels=restchannels+1,
                        hom=True, interaction='only_propagation')
        assert np.array_equal(lgca.nodes[lgca.nonborder], nodes), "Nodes not adopted correctly"
        assert lgca.restchannels == restchannels, "Wrong number of rest channels defined if nodes are given"
        assert lgca.dims == nodes.shape[:-1], "Wrong dimensions defined if nodes are given"
        if ve:
            assert np.max(lgca.nodes.astype(int)) <= 1, "Volume exclusion principle is not respected"

    # parameter tuples for 'hom', 'density' keyword check
    # init_particles = number of particles expected in each node int(density*capacity)
    @pytest.mark.parametrize("geom,ve,ib,dims,restchannels,density,init_particles", [
        # classical LGCA (ve, non-ib)
        # (geom,    ve,   ib,    dims,                               restchannels,              density,    init_particles)
        ('lin',     True, False, (com.xdim_1d,),                     restchannels_ve_1d,        density_ve, calc_init_particles(True, density_ve, restchannels_ve_1d, K=K_ve_1d)),
        ('square',  True, False, (com.xdim_square, com.ydim_square), restchannels_ve_square,    density_ve, calc_init_particles(True, density_ve, restchannels_ve_square, K=K_ve_square)),
        ('hex',     True, False, (com.xdim_hex, com.ydim_hex),       restchannels_ve_hex,       density_ve, calc_init_particles(True, density_ve, restchannels_ve_hex, K=K_ve_hex)),
        # IBLGCA (ve, ib)
        # NoVE_LGCA (nove, non-ib)
         # written assuming that capacity for nove is set to velocitychannels + restchannels
        # (geom,    ve,    ib,    dims,                               restchannels,             density,        init_particles)
        ('lin',     False, False, (com.xdim_1d,),                     restchannels_nove_1d,     density_nove_1, calc_init_particles(False, density_nove_1, restchannels_nove_1d, b=com.b_1d)),
        ('square',  False, False, (com.xdim_square, com.ydim_square), restchannels_nove_square, density_nove_1, calc_init_particles(False, density_nove_1, restchannels_nove_square, b=com.b_square)),
        ('hex',     False, False, (com.xdim_hex, com.ydim_hex),       restchannels_nove_hex,    density_nove_1, calc_init_particles(False, density_nove_1, restchannels_nove_hex, b=com.b_hex)),
        # (geom,    ve,    ib,    dims,                               restchannels,             density,        init_particles)
        ('lin',     False, False, (com.xdim_1d,),                     restchannels_nove_1d,     density_nove_2, calc_init_particles(False, density_nove_2, restchannels_nove_1d, b=com.b_1d)),
        ('square',  False, False, (com.xdim_square, com.ydim_square), restchannels_nove_square, density_nove_2, calc_init_particles(False, density_nove_2, restchannels_nove_square, b=com.b_square)),
        ('hex',     False, False, (com.xdim_hex, com.ydim_hex),       restchannels_nove_hex,    density_nove_2, calc_init_particles(False, density_nove_2, restchannels_nove_hex, b=com.b_hex))
    ])
    def test_getlgca_lattice_setup_homogeneous(self, geom, ve, ib, dims, restchannels, density, init_particles):
        # 'hom', 'density' keyword check
        lgca = get_lgca(geometry=geom, ve=ve, ib=ib, dims=dims, density=density, hom=True, restchannels=restchannels,
                        interaction='only_propagation')
        assert lgca.dims == dims, "Wrong dimensions defined"
        # in the excluded case the lattice will be fully filled
        if not (ve and density == 1):
            # compare flattened node config to config moved one node forward to see if nodes all have the same config
            assert not np.array_equal(lgca.nodes[lgca.nonborder].flatten(),
                                      np.roll(lgca.nodes[lgca.nonborder].flatten(), lgca.nodes.shape[-1])), \
                "Node configuration is not random"
        assert np.all(lgca.nodes[lgca.nonborder].sum(-1) == init_particles), \
            "Density not reached or not reached homogeneously"
        if ve:
            assert np.max(lgca.nodes.astype(int)) <= 1, "Volume exclusion principle is not respected"

    # parameter tuples for 'density' keyword check, random reset
    @pytest.mark.parametrize("geom,ve,ib,dims_large,restchannels,density,capacity", [
        # classical LGCA (ve, non-ib)
        # (geom,    ve,   ib,    dims_large,    restchannels,              density,    capacity)
        ('lin',     True, False, (400,),        restchannels_ve_1d,        density_ve, K_ve_1d),
         ('square', True, False, (20, 20),      restchannels_ve_square,    density_ve, K_ve_square),
         ('hex',    True, False, (20, 20),      restchannels_ve_hex,       density_ve, K_ve_hex),
         # IBLGCA (ve, ib)
         # NoVE_LGCA (nove, non-ib)
          # written assuming that capacity for nove is set to velocitychannels + restchannels
         # (geom,   ve,    ib,    dims_large,   restchannels,             density,        capacity)
         ('lin',    False, False, (400,),       restchannels_nove_1d,     density_nove_1, com.b_1d+restchannels_nove_1d),
         ('square', False, False, (20, 20),     restchannels_nove_square, density_nove_1, com.b_square+restchannels_nove_square),
         ('hex',    False, False, (20, 20),     restchannels_nove_hex,    density_nove_1, com.b_hex+restchannels_nove_hex),
         # (geom,   ve,    ib,    dims_large,   restchannels,             density,        capacity)
         ('lin',    False, False, (400,),       restchannels_nove_1d,     density_nove_2, com.b_1d+restchannels_nove_1d),
         ('square', False, False, (20, 20),     restchannels_nove_square, density_nove_2, com.b_square+restchannels_nove_square),
         ('hex',    False, False, (20, 20),     restchannels_nove_hex,    density_nove_2, com.b_hex+restchannels_nove_hex)
    ])
    def test_getlgca_lattice_setup_random(self, geom, ve, ib, dims_large, restchannels, density, capacity):
        # 'density' keyword check, random reset
        lgca = get_lgca(geometry=geom, ve=ve, ib=ib, dims=dims_large, density=density, restchannels=restchannels,
                        interaction='only_propagation')
        assert lgca.dims == dims_large, "Wrong dimensions defined"
        # in the excluded case the lattice will be fully filled
        if not(ve and density==1):
            # compare flattened node config to config moved one node forward to see if nodes all have the same config
            assert not np.array_equal(lgca.nodes[lgca.nonborder].flatten(),
                                      np.roll(lgca.nodes[lgca.nonborder].flatten(), lgca.nodes.shape[-1])), \
                "Node configuration is not random"
            assert lgca.nodes[lgca.nonborder].sum(-1).min() != lgca.nodes[lgca.nonborder].sum(-1).max(), \
                "Number of particles is homogeneous when it should differ randomly"
        assert np.abs(lgca.nodes[lgca.nonborder].sum() / (capacity * lgca.cell_density[lgca.nonborder].size) - density) \
               < self.com.density_epsilon, "Wrong density reached"
        if ve:
            assert np.max(lgca.nodes.astype(int)) <= 1, "Volume exclusion principle is not respected"


