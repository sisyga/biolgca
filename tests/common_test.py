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
    ve = None
    ib = None

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

    # classical parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1D
    restchannels_ve_1d = 4
    K_ve_1d = b_1d + restchannels_ve_1d
    nodes_ve_1d = rng.integers(low=0, high=1, endpoint=True,
                               size=(xdim_1d, b_1d + restchannels_ve_1d))
    # 2D square
    restchannels_ve_square = 2
    K_ve_square = b_square + restchannels_ve_square
    nodes_ve_square = rng.integers(low=0, high=1, endpoint=True,
                                   size=(xdim_square, ydim_square, b_square + restchannels_ve_square))
    # 2D hex
    restchannels_ve_hex = restchannels_ve_square
    K_ve_hex = b_hex + restchannels_ve_hex
    nodes_ve_hex = rng.integers(low=0, high=1, endpoint=True,
                                size=(xdim_hex, ydim_hex, b_hex + restchannels_ve_hex))

    # ib parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # reuses some classical parameters
    # 1D
    no_particles_1d = rng.integers(low=1, high=(xdim_1d * (b_1d + restchannels_ve_1d)), endpoint=True,
                                       size=1)
    nodes_ib_1d = np.append(np.arange(no_particles_1d) + 1, np.zeros(xdim_1d * (b_1d + restchannels_ve_1d)
                                                                     - no_particles_1d))
    rng.shuffle(nodes_ib_1d)
    nodes_ib_1d = nodes_ib_1d.reshape((xdim_1d, b_1d + restchannels_ve_1d))
    # 2D square
    no_particles_square = rng.integers(low=1, high=(xdim_square * ydim_square *
                                                        (b_square + restchannels_ve_square)), endpoint=True, size=1)
    nodes_ib_square = np.append(np.arange(no_particles_square) + 1, np.zeros(xdim_square * ydim_square *
                                                                             (b_square + restchannels_ve_square) - no_particles_square))
    rng.shuffle(nodes_ib_square)
    nodes_ib_square = nodes_ib_square.reshape((xdim_square, ydim_square, b_square + restchannels_ve_square))
    # 2D hex
    no_particles_hex = rng.integers(low=1, high=(xdim_hex * ydim_hex *
                                                     (b_hex + restchannels_ve_hex)), endpoint=True, size=1)
    nodes_ib_hex = np.append(np.arange(no_particles_hex) + 1, np.zeros(xdim_hex * ydim_hex *
                                                                       (b_hex + restchannels_ve_hex) - no_particles_hex))
    rng.shuffle(nodes_ib_hex)
    nodes_ib_hex = nodes_ib_hex.reshape((xdim_hex, ydim_hex, b_hex + restchannels_ve_hex))

    # nove parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # nodes must be defined in the correct size! Maximum size of last dimension = b+1
    # 1D
    restchannels_nove_1d = 3
    capacity_nove_1d = 5
    nodes_nove_1d = rng.integers(low=0, high=int(1.5 * capacity_nove_1d),
                                     size=(xdim_1d, b_1d + 1), endpoint=True)
    # 2D square
    restchannels_nove_square = 2
    capacity_nove_square = 6
    nodes_nove_square = rng.integers(low=0, high=int(1.5 * capacity_nove_square),
                                         size=(xdim_square, ydim_square, b_square + 1), endpoint=True)
    # 2D hex
    restchannels_nove_hex = restchannels_nove_square
    capacity_nove_hex = 8
    nodes_nove_hex = rng.integers(low=0, high=int(1.5 * capacity_nove_hex),
                                      size=(xdim_hex, ydim_hex, b_hex + 1), endpoint=True)

    def t_propagation_template(self, geom, nodes, expected_output, bc='pbc'):
        # check that propagation and rest channels work: all particles should move into one node within one timestep
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=nodes, bc=bc, interaction='only_propagation')
        lgca.timeevo(timesteps=1, recorddens=False, showprogress=False)
        print(lgca.__class__.__name__)

        assert lgca.nodes[lgca.nonborder].sum() == expected_output.sum(), "Particles appear or disappear"
        assert np.array_equal(lgca.nodes[lgca.nonborder],
                              expected_output), "Node configuration after propagation not correct"
        if self.ib:
            assert np.array_equal(lgca.occupied[lgca.nonborder].sum(-1), lgca.cell_density[lgca.nonborder]), \
                "Cell density field not updated correctly from node configuration"
            assert np.array_equal(lgca.occupied, lgca.nodes.astype(bool)), \
                "Occupation field not updated correctly from node configuration"
        else:
            assert np.array_equal(lgca.nodes[lgca.nonborder].sum(-1), lgca.cell_density[lgca.nonborder]), \
               "Cell density field not updated correctly from node configuration"
        return lgca  # for further tests


# prefix or postfix testing functions with "test_"/"_test" - those cannot be parametrised normally (use fixups)
# prefix classes that contain tests with "Test" - ignored otherwise
class Test_LGCA_General:
    """
    Tests that can be generalised to do ve/nove, ib/non-ib via parametrisation.
    """
    com = T_LGCA_Common

    # classical parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    density_ve = 0.5

    # ib parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # reuses some classical parameters

    # nove parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # nodes must be defined in the correct size! Maximum size of last dimension = b+1
    density_nove_1 = 0.5
    density_nove_2 = 1.5

    # parameter tuples for 'node' keyword check
    @pytest.mark.parametrize("geom,ve,ib,nodes,dims,restchannels", [
        # classical LGCA (ve, non-ib)
        # (geom,    ve,   ib,    nodes,                 dims,                               restchannels)
        ('lin',     True, False, com.nodes_ve_1d,       (com.xdim_1d,),                     com.restchannels_ve_1d),
        ('square',  True, False, com.nodes_ve_square,   (com.xdim_square, com.ydim_square), com.restchannels_ve_square),
        ('hex',     True, False, com.nodes_ve_hex,      (com.xdim_hex, com.ydim_hex),       com.restchannels_ve_hex),
        # IBLGCA (ve, ib)
        # (geom,    ve,   ib,   nodes,                  dims,                               restchannels)
        ('lin',     True, True, com.nodes_ib_1d,        (com.xdim_1d,),                     com.restchannels_ve_1d),
        ('square',  True, True, com.nodes_ib_square,    (com.xdim_square, com.ydim_square), com.restchannels_ve_square),
        ('hex',     True, True, com.nodes_ib_hex,       (com.xdim_hex, com.ydim_hex),       com.restchannels_ve_hex),
        # NoVE_LGCA (nove, non-ib)
        # (geom,    ve,    ib,    nodes,                    dims,                               restchannels)
        ('lin',     False, False, com.nodes_nove_1d,        (com.xdim_1d,),                     1),
        ('square',  False, False, com.nodes_nove_square,    (com.xdim_square, com.ydim_square), 1),
        ('hex',     False, False, com.nodes_nove_hex,       (com.xdim_hex, com.ydim_hex),       1)
    ])
    def test_getlgca_lattice_setup_nodes(self, geom, ve, ib, nodes, dims, restchannels):
        # 'node' keyword check: provided lattice is adopted by LGCA
        # if nodes are provided, 'restchannels', 'dims', 'density' and 'hom' have to be ignored
        lgca = get_lgca(geometry=geom, ve=ve, ib=ib, nodes=nodes, dims=dims, density=0.01, restchannels=restchannels+1,
                        hom=True, interaction='only_propagation')
        assert np.array_equal(lgca.nodes[lgca.nonborder], nodes), "Nodes not adopted correctly"
        assert lgca.restchannels == restchannels, "Wrong number of rest channels defined if nodes are given"
        assert lgca.dims == nodes.shape[:-1], "Wrong dimensions defined if nodes are given"
        if ve and not ib:
            assert np.max(lgca.nodes.astype(int)) <= 1, "Volume exclusion principle is not respected"
        if ib:
            assert np.array_equal(lgca.occupied, lgca.nodes.astype(bool)), \
                "Occupation field not updated correctly from node configuration"
            if lgca.nodes[lgca.nonborder].min() == 0:
                assert np.unique(lgca.nodes[lgca.nonborder], return_counts=True)[1][1:].max() <= 1, \
                        "Uniqueness principle is broken"
            else:
                assert np.unique(lgca.nodes[lgca.nonborder], return_counts=True)[1].max() <= 1, \
                        "Uniqueness principle is broken"
            # check ID updates
            assert lgca.nodes.max() == lgca.maxlabel, "IDs not updated/set up correctly"
            # check property updates
            if lgca.props != {}:
                for propname in lgca.props.keys():
                    assert len(lgca.props[propname]) == lgca.maxlabel + 1, "Properties not set up for all particles"

    # parameter tuples for 'hom', 'density' keyword check
    # init_particles = number of particles expected in each node int(density*capacity)
    @pytest.mark.parametrize("geom,ve,dims,restchannels,density,init_particles", [
        # classical LGCA (ve, non-ib)
        # (geom,    ve,   dims,                               restchannels,                  density,    init_particles)
        ('lin',     True, (com.xdim_1d,),                     com.restchannels_ve_1d,        density_ve, calc_init_particles(True, density_ve, com.restchannels_ve_1d, K=com.K_ve_1d)),
        ('square',  True, (com.xdim_square, com.ydim_square), com.restchannels_ve_square,    density_ve, calc_init_particles(True, density_ve, com.restchannels_ve_square, K=com.K_ve_square)),
        ('hex',     True, (com.xdim_hex, com.ydim_hex),       com.restchannels_ve_hex,       density_ve, calc_init_particles(True, density_ve, com.restchannels_ve_hex, K=com.K_ve_hex)),
        # NoVE_LGCA (nove, non-ib)
        # written assuming that capacity for nove is set to velocitychannels + restchannels
        # (geom,    ve,    dims,                               restchannels,                 density,        init_particles)
        ('lin',     False, (com.xdim_1d,),                     com.restchannels_nove_1d,     density_nove_1, calc_init_particles(False, density_nove_1, com.restchannels_nove_1d, b=com.b_1d)),
        ('square',  False, (com.xdim_square, com.ydim_square), com.restchannels_nove_square, density_nove_1, calc_init_particles(False, density_nove_1, com.restchannels_nove_square, b=com.b_square)),
        ('hex',     False, (com.xdim_hex, com.ydim_hex),       com.restchannels_nove_hex,    density_nove_1, calc_init_particles(False, density_nove_1, com.restchannels_nove_hex, b=com.b_hex)),
        # (geom,    ve,    dims,                               restchannels,                 density,        init_particles)
        ('lin',     False, (com.xdim_1d,),                     com.restchannels_nove_1d,     density_nove_2, calc_init_particles(False, density_nove_2, com.restchannels_nove_1d, b=com.b_1d)),
        ('square',  False, (com.xdim_square, com.ydim_square), com.restchannels_nove_square, density_nove_2, calc_init_particles(False, density_nove_2, com.restchannels_nove_square, b=com.b_square)),
        ('hex',     False, (com.xdim_hex, com.ydim_hex),       com.restchannels_nove_hex,    density_nove_2, calc_init_particles(False, density_nove_2, com.restchannels_nove_hex, b=com.b_hex))
    ])
    def test_getlgca_lattice_setup_homogeneous(self, geom, ve, dims, restchannels, density, init_particles):
        # 'hom', 'density' keyword check
        lgca = get_lgca(geometry=geom, ve=ve, ib=False, dims=dims, density=density, hom=True, restchannels=restchannels,
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
        # (geom,    ve,   ib,    dims_large,    restchannels,                  density,    capacity)
        ('lin',     True, False, (400,),        com.restchannels_ve_1d,        density_ve, com.K_ve_1d),
        ('square',  True, False, (20, 20),      com.restchannels_ve_square,    density_ve, com.K_ve_square),
        ('hex',     True, False, (20, 20),      com.restchannels_ve_hex,       density_ve, com.K_ve_hex),
        # IBLGCA (ve, ib)
        # (geom,    ve,   ib,   dims_large,    restchannels,                density,    capacity)
        ('lin',     True, True, (400,),        com.restchannels_ve_1d,      density_ve, com.K_ve_1d),
        ('square',  True, True, (20, 20),      com.restchannels_ve_square,  density_ve, com.K_ve_square),
        ('hex',     True, True, (20, 20),      com.restchannels_ve_hex,     density_ve, com.K_ve_hex),
        # NoVE_LGCA (nove, non-ib)
        # written assuming that capacity for nove is set to velocitychannels + restchannels
        # (geom,   ve,    ib,    dims_large,   restchannels,                 density,        capacity)
        ('lin',    False, False, (400,),       com.restchannels_nove_1d,     density_nove_1, com.b_1d+com.restchannels_nove_1d),
        ('square', False, False, (20, 20),     com.restchannels_nove_square, density_nove_1, com.b_square+com.restchannels_nove_square),
        ('hex',    False, False, (20, 20),     com.restchannels_nove_hex,    density_nove_1, com.b_hex+com.restchannels_nove_hex),
        # (geom,   ve,    ib,    dims_large,   restchannels,                 density,        capacity)
        ('lin',    False, False, (400,),       com.restchannels_nove_1d,     density_nove_2, com.b_1d+com.restchannels_nove_1d),
        ('square', False, False, (20, 20),     com.restchannels_nove_square, density_nove_2, com.b_square+com.restchannels_nove_square),
        ('hex',    False, False, (20, 20),     com.restchannels_nove_hex,    density_nove_2, com.b_hex+com.restchannels_nove_hex)
    ])
    def test_getlgca_lattice_setup_random(self, geom, ve, ib, dims_large, restchannels, density, capacity):
        # 'density' keyword check, random reset
        lgca = get_lgca(geometry=geom, ve=ve, ib=ib, dims=dims_large, density=density, restchannels=restchannels,
                        interaction='only_propagation')
        assert lgca.dims == dims_large, "Wrong dimensions defined"
        if ib:
            assert np.array_equal(lgca.occupied, lgca.nodes.astype(bool)), \
                "Occupation field not updated correctly from node configuration"
            if lgca.nodes[lgca.nonborder].min() == 0:
                assert np.unique(lgca.nodes[lgca.nonborder], return_counts=True)[1][1:].max() <= 1, \
                "Uniqueness principle is broken"
            else:
                assert np.unique(lgca.nodes[lgca.nonborder], return_counts=True)[1].max() <= 1, \
                "Uniqueness principle is broken"
            # check ID updates
            assert lgca.nodes.max() == lgca.maxlabel, "IDs not updated/set up correctly"
            # check property updates
            if lgca.props != {}:
                for propname in lgca.props.keys():
                    assert len(lgca.props[propname]) == lgca.maxlabel + 1, "Properties not set up for all particles"
            lgca.nodes = lgca.occupied  # in order to do the following tests like for the other LGCA species
        # in the excluded case the lattice will be fully filled
        if not(ve and density == 1):
            # compare flattened node config to config moved one node forward to see if nodes all have the same config
            assert not np.array_equal(lgca.nodes[lgca.nonborder].flatten(),
                                      np.roll(lgca.nodes[lgca.nonborder].flatten(), lgca.nodes.shape[-1])), \
                "Node configuration is not random"
            assert lgca.nodes[lgca.nonborder].sum(-1).min() != lgca.nodes[lgca.nonborder].sum(-1).max(), \
                "Number of particles is homogeneous when it should differ randomly"
        assert np.abs(lgca.nodes[lgca.nonborder].sum() / (capacity * lgca.cell_density[lgca.nonborder].size) - density) \
               < self.com.density_epsilon, "Wrong density reached"
        if ve and not ib:
            assert np.max(lgca.nodes.astype(int)) <= 1, "Volume exclusion principle is not respected"
