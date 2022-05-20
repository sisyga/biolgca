import numpy as np
import pytest
import copy

from lgca import get_lgca
from tests.common_test import T_LGCA_Common
from tests.classical_test import Test_LGCA_classical as T_LGCA_classical  # rename to avoid duplicate execution of these tests

com = T_LGCA_Common


@pytest.fixture
def nodes_1d_rbound():
    # fixtures for 1D boundary conditions
    # right
    nodes = np.zeros((com.xdim_1d, com.b_1d + 1))
    nodes[-1, 0] = 2  # particles that cross boundary
    nodes[-1, 2] = 1  # resting reference particle
    nodes[2, 0] = 3  # moving reference particles
    return nodes


@pytest.fixture
def out_1d_rbound(nodes_1d_rbound):
    # fixtures for 1D boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_1d_rbound.shape)
    expected_output[-1, 2] = 1  # resting reference particle
    expected_output[3, 0] = 3  # moving reference particles
    return expected_output


@pytest.fixture
def nodes_1d_lbound():
    # fixtures for 1D boundary conditions
    # left
    nodes = np.zeros((com.xdim_1d, com.b_1d + 1))
    nodes[0, 1] = 2  # particles that cross boundary
    nodes[0, 2] = 1  # resting reference particle
    nodes[2, 1] = 3  # moving reference particles
    return nodes


@pytest.fixture
def out_1d_lbound(nodes_1d_lbound):
    # fixtures for 1D boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_1d_lbound.shape)
    expected_output[1, 1] = 3  # moving reference particles
    expected_output[0, 2] = 1  # resting reference particle
    return expected_output


@pytest.fixture
def nodes_sq_rbound():
    # fixtures for 2D square boundary conditions
    # right
    nodes = np.zeros((3, 4, com.b_square + 1))
    nodes[-1, 3, 0] = 1  # particles that cross boundary
    nodes[-1, 2, 0] = 2  # particles that cross boundary
    nodes[-1, 1, 0] = 3  # particles that cross boundary
    nodes[-1, 0, 0] = 4  # particles that cross boundary
    nodes[0, 1, 0] = 5  # moving reference particles
    nodes[-1, 1, 4] = 6  # resting reference particles
    return nodes


@pytest.fixture
def out_sq_rbound(nodes_sq_rbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_sq_rbound.shape)
    expected_output[-1, 1, 4] = 6  # resting reference particles
    expected_output[1, 1, 0] = 5  # moving reference particles
    return expected_output


@pytest.fixture
def nodes_sq_lbound():
    # fixtures for 2D square boundary conditions
    # left
    nodes = np.zeros((3, 4, com.b_square + 1))
    nodes[0, 3, 2] = 1  # particles that cross boundary
    nodes[0, 2, 2] = 2  # particles that cross boundary
    nodes[0, 1, 2] = 3  # particles that cross boundary
    nodes[0, 0, 2] = 4  # particles that cross boundary
    nodes[0, 1, 4] = 6  # resting reference particles
    nodes[2, 1, 2] = 5  # moving reference particles
    return nodes


@pytest.fixture
def out_sq_lbound(nodes_sq_lbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output left
    expected_output = np.zeros(nodes_sq_lbound.shape)
    expected_output[0, 1, 4] = 6  # resting reference particles
    expected_output[1, 1, 2] = 5  # moving reference particles
    return expected_output


@pytest.fixture
def nodes_sq_tbound():
    # fixtures for 2D square boundary conditions
    # top
    nodes = np.zeros((3, 4, com.b_square + 1))
    nodes[0, -1, 1] = 1  # particles that cross boundary
    nodes[1, -1, 1] = 2  # particles that cross boundary
    nodes[2, -1, 1] = 3  # particles that cross boundary
    nodes[1, -1, 4] = 5  # resting reference particles
    nodes[1, 1, 1] = 4  # moving reference particles
    return nodes


@pytest.fixture
def out_sq_tbound(nodes_sq_tbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output top
    expected_output = np.zeros(nodes_sq_tbound.shape)
    expected_output[1, -1, 4] = 5  # resting reference particles
    expected_output[1, 2, 1] = 4  # moving reference particles
    return expected_output


@pytest.fixture
def nodes_sq_bbound():
    # fixtures for 2D square boundary conditions
    # bottom
    nodes = np.zeros((3, 4, com.b_square + 1))
    nodes[0, 0, 3] = 1  # particles that cross boundary
    nodes[1, 0, 3] = 2  # particles that cross boundary
    nodes[2, 0, 3] = 3  # particles that cross boundary
    nodes[1, 0, 4] = 5  # resting reference particles
    nodes[1, 2, 3] = 4  # moving reference particles
    return nodes


@pytest.fixture
def out_sq_bbound(nodes_sq_bbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output bottom
    expected_output = np.zeros(nodes_sq_bbound.shape)
    expected_output[1, 0, 4] = 5  # resting reference particles
    expected_output[1, 1, 3] = 4  # moving reference particles
    return expected_output


@pytest.fixture
def nodes_hex_rbound():
    # fixtures for 2D hex boundary conditions
    # right
    nodes = np.zeros((4, 6, com.b_hex + 1))
    nodes[-1, :, 0] = np.arange(1, 7)  # particles that cross boundary
    nodes[1, 2, 0] = 7  # moving reference particles
    nodes[-1, 2, 6] = 8  # resting reference particles
    nodes[-1, 1, 6] = 9  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_rbound(nodes_hex_rbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_hex_rbound.shape)
    expected_output[2, 2, 0] = 7  # moving reference particles
    expected_output[-1, 2, 6] = 8  # resting reference particles
    expected_output[-1, 1, 6] = 9  # resting reference particles
    return expected_output


@pytest.fixture
def nodes_hex_trbound():
    # fixtures for 2D hex boundary conditions
    # top right
    nodes = np.zeros((4, 6, com.b_hex + 1))
    nodes[-1, :, 1] = np.arange(1, 7)  # particles on the boundary
    nodes[0, -1, 1] = 7  # particles that cross boundary
    nodes[1, -1, 1] = 8  # particles that cross boundary
    nodes[0, 3, 1] = 9  # moving reference particles
    nodes[-1, 4, 6] = 12  # resting reference particles
    nodes[-1, -1, 6] = 11  # resting reference particles
    nodes[-2, -1, 6] = 10  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_trbound(nodes_hex_trbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output top right
    expected_output = np.zeros(nodes_hex_trbound.shape)
    expected_output[-1, -2, 1] = 4  # particles that do not cross boundary
    expected_output[-1, 2, 1] = 2  # particles that do not cross boundary
    expected_output[0, 4, 1] = 9  # moving reference particles
    expected_output[-1, 4, 6] = 12  # resting reference particles
    expected_output[-1, -1, 6] = 11  # resting reference particles
    expected_output[-2, -1, 6] = 10  # resting reference particles
    return expected_output


@pytest.fixture
def nodes_hex_tlbound():
    # fixtures for 2D hex boundary conditions
    # top left
    nodes = np.zeros((4, 6, com.b_hex + 1))
    nodes[0, :, 2] = np.arange(1, 7)  # particles at boundary
    nodes[1, -1, 2] = 7  # particle that crosses boundary
    nodes[-1, -1, 2] = 8  # particle that crosses boundary
    nodes[-1, 2, 2] = 9  # moving reference particles
    nodes[0, -1, 6] = 11  # resting reference particles
    nodes[1, -1, 6] = 10  # resting reference particles
    nodes[0, -2, 6] = 12  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_tlbound(nodes_hex_tlbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output top left
    expected_output = np.zeros(nodes_hex_tlbound.shape)
    expected_output[0, 1::2, 2] = np.array([1, 3, 5])  # particles that do not cross boundary
    expected_output[-1, 3, 2] = 9  # moving reference particles
    expected_output[0, -1, 6] = 11  # resting reference particles
    expected_output[1, -1, 6] = 10  # resting reference particles
    expected_output[0, -2, 6] = 12  # resting reference particles
    return expected_output


@pytest.fixture
def nodes_hex_lbound():
    # fixtures for 2D hex boundary conditions
    # left
    nodes = np.zeros((4, 6, com.b_hex + 1))
    nodes[0, :, 3] = np.arange(1, 7)  # particles that cross boundary
    nodes[2, 2, 3] = 7  # moving reference particles
    nodes[0, 2, 6] = 8  # resting reference particles
    nodes[0, 1, 6] = 9  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_lbound(nodes_hex_lbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output left
    expected_output = np.zeros(nodes_hex_lbound.shape)
    expected_output[1, 2, 3] = 7  # moving reference particles
    expected_output[0, 2, 6] = 8  # resting reference particles
    expected_output[0, 1, 6] = 9  # resting reference particles
    return expected_output


@pytest.fixture
def nodes_hex_blbound():
    # fixtures for 2D hex boundary conditions
    # bottom left
    nodes = np.zeros((4, 6, com.b_hex + 1))
    nodes[0, :, 4] = np.arange(1, 7)  # particles on boundary
    nodes[-2:, 0, 4] = np.array([7, 8])  # particles that cross boundary
    nodes[-1, 2, 4] = 9  # moving reference particles
    nodes[0, 0, 6] = 11  # resting reference particles
    nodes[1, 0, 6] = 12  # resting reference particles
    nodes[0, 1, 6] = 10  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_blbound(nodes_hex_blbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output bottom left
    expected_output = np.zeros(nodes_hex_blbound.shape)
    expected_output[0, 1, 4] = 3  # particles that do not cross boundary
    expected_output[0, 3, 4] = 5  # particles that do not cross boundary
    expected_output[-1, 1, 4] = 9  # moving reference particles
    expected_output[0, 0, 6] = 11  # resting reference particles
    expected_output[1, 0, 6] = 12  # resting reference particles
    expected_output[0, 1, 6] = 10  # resting reference particles
    return expected_output


@pytest.fixture
def nodes_hex_brbound():
    # fixtures for 2D hex boundary conditions
    # bottom right
    nodes = np.zeros((4, 6, com.b_hex + 1))
    nodes[-1, :, 5] = np.arange(1, 7)  # particles on boundary
    nodes[0, 0, 5] = 7  # particle that crosses boundary
    nodes[1, 0, 5] = 8  # particle that crosses boundary
    nodes[0, 3, 5] = 9  # moving reference particle
    nodes[-2:, 0, 6] = np.array([10, 11])  # resting reference particles
    nodes[-1, 1, 6] = 12  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_brbound(nodes_hex_brbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output bottom right
    expected_output = np.zeros(nodes_hex_brbound.shape)
    expected_output[-1, 0::2, 5] = np.array([2, 4, 6])  # particles that do not cross boundary
    expected_output[0, 2, 5] = 9  # moving reference particles
    expected_output[-2:, 0, 6] = np.array([10, 11])  # resting reference particles
    expected_output[-1, 1, 6] = 12  # resting reference particles
    return expected_output


class Test_LGCA_NoVE(T_LGCA_Common):
    """
        Class for testing LGCA without volume exclusion (not identity-based).
        * propagation
        * recording options during simulation
        * boundary conditions (uses tests from classical LGCA)
    """

    com = T_LGCA_Common
    ve = False
    ib = False
    # reuse absorbing boundary condition tests from classical LGCA
    test_abc_1d = T_LGCA_classical.test_abc_1d
    test_abc_square = T_LGCA_classical.test_abc_square
    test_abc_hex = T_LGCA_classical.test_abc_hex

    @pytest.mark.parametrize("geom,dims", [
        ('lin', (com.xdim_1d,)),
        ('square', (com.xdim_square, com.ydim_square)),
        ('hex', (com.xdim_hex, com.ydim_hex))
    ])
    def test_recording(self, geom, dims):
        # timeevo and recording: check if all properties are available when requested
        # keywords: record=False -> nodes_t, recordN=False n_t, recorddens=True -> dens_t, recordpertype=False -> velcells_t, restcells_t
        # recordorderparams=False -> ent_t, normEnt_t, polAlParam_t, meanAlign_t
        lgca_1 = get_lgca(geometry=geom, ve=False, dims=dims, density=0.5, interaction='only_propagation')
        lgca_2 = copy.deepcopy(lgca_1)

        lgca_1.timeevo(timesteps=2, recorddens=False, showprogress=False)
        assert not hasattr(lgca_1, 'dens_t'), "Records density when it should not"
        assert not hasattr(lgca_1, 'nodes_t'), "Records node configuration when it should not"
        assert not hasattr(lgca_1, 'n_t'), "Records particle number when it should not"
        assert not hasattr(lgca_1, 'restcells_t') and not hasattr(lgca_1, 'velcells_t'), "Records density per channel type when it should not"
        assert not hasattr(lgca_1, 'ent_t') and not hasattr(lgca_1, 'normEnt_t') and not hasattr(lgca_1, 'polAlParam_t') \
               and not hasattr(lgca_1, 'meanAlign_t'), "Records order parameters when it should not"
        del lgca_1

        lgca_3 = copy.deepcopy(lgca_2)
        lgca_2.timeevo(timesteps=2, showprogress=False)
        assert hasattr(lgca_2, 'dens_t'), "Does not record density"
        del lgca_2

        lgca_4 = copy.deepcopy(lgca_3)
        lgca_3.timeevo(timesteps=2, record=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_3, 'nodes_t'), "Does not record node configuration"
        del lgca_3

        lgca_5 = copy.deepcopy(lgca_4)
        lgca_4.timeevo(timesteps=2, recordN=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_4, 'n_t'), "Does not record particle number"
        del lgca_4

        lgca_6 = copy.deepcopy(lgca_5)
        lgca_5.timeevo(timesteps=2, recordpertype=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_5, 'restcells_t') and hasattr(lgca_5, 'velcells_t'), "Does not record density per channel type"
        del lgca_5

        lgca_6.timeevo(timesteps=2, recordorderparams=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_6, 'ent_t') and hasattr(lgca_6, 'normEnt_t') and hasattr(lgca_6, 'polAlParam_t') \
               and hasattr(lgca_6, 'meanAlign_t'), "Does not record order parameters"

    def test_propagation(self):
        # 1D
        # input
        nodes = np.zeros((self.xdim_1d, self.b_1d + 1))
        nodes[0, 0] = 1  # particle moving to the right
        nodes[1, 2] = 2  # particles resting
        nodes[2, 1] = 3  # particles moving to the left
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_1d, self.b_1d + 1))
        expected_output[1, 0] = 1  # particle moving to the right
        expected_output[1, 2] = 2  # particles resting
        expected_output[1, 1] = 3  # particles moving to the left
        self.t_propagation_template('lin', nodes, expected_output)

        # 2D square
        # input
        nodes = np.zeros((self.xdim_square, self.ydim_square, self.b_square + 1))
        nodes[0, 1, 0] = 1  # particle moving to the right
        nodes[1, 0, 1] = 2  # particles moving up
        nodes[2, 1, 2] = 3  # particles moving to the left
        nodes[1, 2, 3] = 4  # particles moving down
        nodes[1, 1, 4] = 5  # particles resting
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_square, self.ydim_square, self.b_square + 1))
        expected_output[1, 1, 0] = 1  # particle moving to the right
        expected_output[1, 1, 1] = 2  # particles moving up
        expected_output[1, 1, 2] = 3  # particles moving to the left
        expected_output[1, 1, 3] = 4  # particles moving down
        expected_output[1, 1, 4] = 5  # particles resting
        self.t_propagation_template('square', nodes, expected_output)

        # 2D hex
        # input
        nodes = np.zeros((self.xdim_hex, self.ydim_hex, self.b_hex + 1))
        nodes[0, 1, 0] = 1  # particle moving right
        nodes[0, 0, 1] = 2  # particles moving to the upper right
        nodes[1, 0, 2] = 3  # particles moving to the upper left
        nodes[2, 1, 3] = 4  # particles moving left
        nodes[1, 2, 4] = 5  # particles moving to the lower left
        nodes[0, 2, 5] = 6  # particles moving to the lower right
        nodes[1, 1, 6] = 7  # particles resting
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_hex, self.ydim_hex, self.b_hex + 1))
        expected_output[1, 1, 0] = 1  # particle moving right
        expected_output[1, 1, 1] = 2  # particles moving to the upper right
        expected_output[1, 1, 2] = 3  # particles moving to the upper left
        expected_output[1, 1, 3] = 4  # particles moving left
        expected_output[1, 1, 4] = 5  # particles moving to the lower left
        expected_output[1, 1, 5] = 6  # particles moving to the lower right
        expected_output[1, 1, 6] = 7  # particles resting
        self.t_propagation_template('hex', nodes, expected_output)

    @pytest.mark.parametrize("geom,nodes,b", [
        ('lin', com.nodes_nove_1d, com.b_1d),
        ('square', com.nodes_nove_square, com.b_square),
        ('hex', com.nodes_nove_hex, com.b_hex)
    ])
    def test_getlgca_capacity(self, geom, nodes, b):
        # lattice setup test with capacity: capacity defines how density is calculated
        capacity = 10
        restchannels = 3
        density = 0.5
        lgca = get_lgca(geometry=geom, ve=False, nodes=nodes, interaction='only_propagation')
        assert lgca.capacity == b+1, "Capacity not automatically defined from nodes"
        lgca = get_lgca(geometry=geom, ve=False, nodes=nodes, capacity=capacity, interaction='only_propagation')
        assert lgca.capacity == capacity, "Capacity keyword cannot overwrite format of nodes"
        lgca = get_lgca(geometry=geom, ve=False, density=density, capacity=capacity, interaction='only_propagation')
        assert lgca.capacity == capacity, "Capacity keyword not respected in random reset"
        lgca = get_lgca(geometry=geom, ve=False, density=density, hom=True, capacity=capacity, interaction='only_propagation')
        assert lgca.capacity == capacity, "Capacity keyword not respected in homogeneous random reset"
        lgca = get_lgca(geometry=geom, ve=False, nodes=nodes, restchannels=restchannels, interaction='only_propagation')
        assert lgca.capacity == b+restchannels, "Capacity not correctly calculated from provided geometry and rest channels"
        lgca = get_lgca(geometry=geom, ve=False, density=density, restchannels=restchannels, interaction='only_propagation')
        assert lgca.capacity == b+restchannels, "Capacity not correctly calculated from provided geometry and rest channels"
        lgca = get_lgca(geometry=geom, ve=False, density=density, hom=True, restchannels=restchannels, interaction='only_propagation')
        assert lgca.capacity == b+restchannels, "Capacity not correctly calculated from provided geometry and rest channels"

    @pytest.mark.parametrize("geom,nodes", [
        ('lin', com.nodes_nove_1d),
        ('square', com.nodes_nove_square),
        ('hex', com.nodes_nove_hex)
    ])
    def test_characteristics(self, geom, nodes):
        # volume exclusion does not have to be checked here, uniqueness neither
        # let interactions run for 50 timesteps to check attribute reference problems
        print("Starting characteristics test")
        ref_lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib)
        vchannels_mapping = {'lin':com.b_1d, 'square':com.b_square, 'hex':com.b_hex}
        for interaction in ref_lgca.interactions:
            print(interaction)
            # prevent velocity channel-only rules from crashing - quick fix
            if interaction == 'dd_alignment' or interaction == 'di_alignment':
                nodes_vonly = nodes[..., :vchannels_mapping[geom]]
                print('mapping', vchannels_mapping[geom])
                print('shape', nodes_vonly.shape)
                # test all boundary conditions in case of abuse of border nodes
                self.t_characteristics(geom, nodes_vonly, interaction, 'pbc')
                self.t_characteristics(geom, nodes_vonly, interaction, 'rbc')
                self.t_characteristics(geom, nodes_vonly, interaction, 'abc')
            else:
                # test all boundary conditions in case of abuse of border nodes
                self.t_characteristics(geom, nodes, interaction, 'pbc')
                self.t_characteristics(geom, nodes, interaction, 'rbc')
                self.t_characteristics(geom, nodes, interaction, 'abc')

    def t_characteristics(self, geom, nodes, interaction, bc):
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=nodes, interaction=interaction, bc=bc)
        lgca.timeevo(timesteps=50, recorddens=False, record=False, showprogress=False)

    def test_pbc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        # check periodic boundary conditions in 1D
        # right boundary
        out_1d_rbound[0, 0] = 2  # particles that cross boundary
        self.t_propagation_template('lin', nodes_1d_rbound, out_1d_rbound, bc='pbc')

        # left boundary
        out_1d_lbound[-1, 1] = 2  # particles that cross boundary
        self.t_propagation_template('lin', nodes_1d_lbound, out_1d_lbound, bc='pbc')

    def test_rbc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        # check reflecting boundary conditions in 1D
        # right boundary
        out_1d_rbound[-1, 1] = 2  # particle that crosses boundary
        self.t_propagation_template('lin', nodes_1d_rbound, out_1d_rbound, bc='rbc')

        # left boundary
        out_1d_lbound[0, 0] = 2  # particle that crosses boundary
        self.t_propagation_template('lin', nodes_1d_lbound, out_1d_lbound, bc='rbc')

    def test_pbc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        out_sq_rbound[0, 3, 0] = 1  # particles that cross boundary
        out_sq_rbound[0, 2, 0] = 2  # particles that cross boundary
        out_sq_rbound[0, 1, 0] = 3  # particles that cross boundary
        out_sq_rbound[0, 0, 0] = 4  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_rbound, out_sq_rbound, bc='pbc')

        # left boundary
        out_sq_lbound[-1, 3, 2] = 1  # particles that cross boundary
        out_sq_lbound[-1, 2, 2] = 2  # particles that cross boundary
        out_sq_lbound[-1, 1, 2] = 3  # particles that cross boundary
        out_sq_lbound[-1, 0, 2] = 4  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_lbound, out_sq_lbound, bc='pbc')

        # top boundary
        out_sq_tbound[0, 0, 1] = 1  # particles that cross boundary
        out_sq_tbound[1, 0, 1] = 2  # particles that cross boundary
        out_sq_tbound[2, 0, 1] = 3  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_tbound, out_sq_tbound, bc='pbc')

        # bottom boundary
        out_sq_bbound[0, -1, 3] = 1  # particles that cross boundary
        out_sq_bbound[1, -1, 3] = 2  # particles that cross boundary
        out_sq_bbound[2, -1, 3] = 3  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_bbound, out_sq_bbound, bc='pbc')

    def test_rbc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        # check reflecting boundary conditions in 2D square
        # right boundary
        out_sq_rbound[-1, 3, 2] = 1  # particles that cross boundary
        out_sq_rbound[-1, 2, 2] = 2  # particles that cross boundary
        out_sq_rbound[-1, 1, 2] = 3  # particles that cross boundary
        out_sq_rbound[-1, 0, 2] = 4  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_rbound, out_sq_rbound, bc='rbc')

        # left boundary
        out_sq_lbound[0, 3, 0] = 1  # particles that cross boundary
        out_sq_lbound[0, 2, 0] = 2  # particles that cross boundary
        out_sq_lbound[0, 1, 0] = 3  # particles that cross boundary
        out_sq_lbound[0, 0, 0] = 4  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_lbound, out_sq_lbound, bc='rbc')

        # top boundary
        out_sq_tbound[0, -1, 3] = 1  # particles that cross boundary
        out_sq_tbound[1, -1, 3] = 2  # particles that cross boundary
        out_sq_tbound[2, -1, 3] = 3  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_tbound, out_sq_tbound, bc='rbc')

        # bottom boundary
        out_sq_bbound[0, 0, 1] = 1  # particles that cross boundary
        out_sq_bbound[1, 0, 1] = 2  # particles that cross boundary
        out_sq_bbound[2, 0, 1] = 3  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_bbound, out_sq_bbound, bc='rbc')

    def test_pbc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        out_hex_rbound[0, :, 0] = np.arange(1, 7)  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_rbound, out_hex_rbound, bc='pbc')

        # top right boundary
        out_hex_trbound[0, 1::2, 1] = np.array([1, 3, 5])  # particles that cross boundary
        out_hex_trbound[-1, 0, 1] = 6
        out_hex_trbound[0, 0, 1] = 7
        out_hex_trbound[1, 0, 1] = 8
        self.t_propagation_template('hex', nodes_hex_trbound, out_hex_trbound, bc='pbc')

        # top left boundary
        out_hex_tlbound[-1, 2::2, 2] = np.array([2, 4])  # particles that cross boundary
        out_hex_tlbound[-1, 0, 2] = 6
        out_hex_tlbound[0, 0, 2] = 7
        out_hex_tlbound[-2, 0, 2] = 8
        self.t_propagation_template('hex', nodes_hex_tlbound, out_hex_tlbound, bc='pbc')

        # left boundary
        out_hex_lbound[-1, :, 3] = np.arange(1, 7)  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_lbound, out_hex_lbound, bc='pbc')

        # bottom left boundary
        out_hex_blbound[-1, 0::2, 4] = np.array([2, 4, 6])  # particles that cross boundary
        out_hex_blbound[-2:, -1, 4] = np.array([7, 8])
        out_hex_blbound[0, -1, 4] = 1
        self.t_propagation_template('hex', nodes_hex_blbound, out_hex_blbound, bc='pbc')

        # bottom right boundary
        out_hex_brbound[0, 1::2, 5] = np.array([3, 5, 1])  # particles that cross boundary
        out_hex_brbound[1, -1, 5] = 7
        out_hex_brbound[2, -1, 5] = 8
        self.t_propagation_template('hex', nodes_hex_brbound, out_hex_brbound, bc='pbc')

    def test_rbc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        out_hex_rbound[-1, :, 3] = np.arange(1, 7)  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_rbound, out_hex_rbound, bc='rbc')

        # top right boundary
        out_hex_trbound[-1, 0::2, 4] = np.array([1, 3, 5])  # particles that cross boundary
        out_hex_trbound[0, -1, 4] = 7
        out_hex_trbound[1, -1, 4] = 8
        out_hex_trbound[-1, -1, 4] = 6
        self.t_propagation_template('hex', nodes_hex_trbound, out_hex_trbound, bc='rbc')

        # top left boundary
        out_hex_tlbound[0, 1::2, 5] = np.array([2, 4, 6])  # particles that cross boundary
        out_hex_tlbound[1, -1, 5] = 7
        out_hex_tlbound[-1, -1, 5] = 8
        self.t_propagation_template('hex', nodes_hex_tlbound, out_hex_tlbound, bc='rbc')

        # left boundary
        out_hex_lbound[0, :, 0] = np.arange(1, 7)  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_lbound, out_hex_lbound, bc='rbc')

        # bottom left boundary
        out_hex_blbound[0, 1::2, 1] = np.array([2, 4, 6])  # particles that cross boundary
        out_hex_blbound[0, 0, 1] = 1
        out_hex_blbound[-2:, 0, 1] = np.array([7, 8])
        self.t_propagation_template('hex', nodes_hex_blbound, out_hex_blbound, bc='rbc')

        # bottom right boundary
        out_hex_brbound[-1, 0::2, 2] = np.array([1, 3, 5])  # particles that cross boundary
        out_hex_brbound[0, 0, 2] = 7
        out_hex_brbound[1, 0, 2] = 8
        self.t_propagation_template('hex', nodes_hex_brbound, out_hex_brbound, bc='rbc')
