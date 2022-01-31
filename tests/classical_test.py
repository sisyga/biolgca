import numpy as np
import pytest
import copy
import warnings

from lgca import get_lgca
from tests.common_test import T_LGCA_Common

com = T_LGCA_Common

@pytest.fixture
def nodes_1d_rbound():
    # fixtures for 1D boundary conditions
    # right
    nodes = np.zeros((com.xdim_1d, com.b_1d + 1))
    nodes[-1, 0] = 1  # particle that crosses boundary
    nodes[-1, 2] = 1  # resting reference particle
    nodes[2, 0] = 1  # moving reference particle
    return nodes


@pytest.fixture
def out_1d_rbound(nodes_1d_rbound):
    # fixtures for 1D boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_1d_rbound.shape)
    expected_output[-1, 2] = 1  # resting reference particle
    expected_output[3, 0] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_1d_lbound():
    # fixtures for 1D boundary conditions
    # left
    nodes = np.zeros((com.xdim_1d, com.b_1d + 1))
    nodes[0, 1] = 1  # particle that crosses boundary
    nodes[0, 2] = 1  # resting reference particle
    nodes[2, 1] = 1  # moving reference particle
    return nodes


@pytest.fixture
def out_1d_lbound(nodes_1d_lbound):
    # fixtures for 1D boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_1d_lbound.shape)
    expected_output[1, 1] = 1  # moving reference particle
    expected_output[0, 2] = 1  # resting reference particle
    return expected_output


@pytest.fixture
def nodes_sq_rbound():
    # fixtures for 2D square boundary conditions
    # right
    nodes = np.zeros((com.xdim_square, com.ydim_square, com.b_square + 1))
    nodes[-1, :, 0] = 1  # particles that cross boundary
    nodes[-1, 1, 4] = 1  # resting reference particle
    nodes[0, 1, 0] = 1  # moving reference particle
    return nodes


@pytest.fixture
def out_sq_rbound(nodes_sq_rbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_sq_rbound.shape)
    expected_output[-1, 1, 4] = 1  # resting reference particle
    expected_output[1, 1, 0] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_sq_lbound():
    # fixtures for 2D square boundary conditions
    # left
    nodes = np.zeros((com.xdim_square, com.ydim_square, com.b_square + 1))
    nodes[0, :, 2] = 1  # particles that cross boundary
    nodes[0, 1, 4] = 1  # resting reference particle
    nodes[2, 1, 2] = 1  # moving reference particle
    return nodes


@pytest.fixture
def out_sq_lbound(nodes_sq_lbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output left
    expected_output = np.zeros(nodes_sq_lbound.shape)
    expected_output[0, 1, 4] = 1  # resting reference particle
    expected_output[1, 1, 2] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_sq_tbound():
    # fixtures for 2D square boundary conditions
    # top
    nodes = np.zeros((com.xdim_square, com.ydim_square, com.b_square + 1))
    nodes[:, -1, 1] = 1  # particles that cross boundary
    nodes[1, -1, 4] = 1  # resting reference particle
    nodes[1, 1, 1] = 1  # moving reference particle
    return nodes


@pytest.fixture
def out_sq_tbound(nodes_sq_tbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output top
    expected_output = np.zeros(nodes_sq_tbound.shape)
    expected_output[1, -1, 4] = 1  # resting reference particle
    expected_output[1, 2, 1] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_sq_bbound():
    # fixtures for 2D square boundary conditions
    # bottom
    nodes = np.zeros((com.xdim_square, com.ydim_square, com.b_square + 1))
    nodes[:, 0, 3] = 1  # particles that cross boundary
    nodes[1, 0, 4] = 1  # resting reference particle
    nodes[1, 2, 3] = 1  # moving reference particle
    return nodes


@pytest.fixture
def out_sq_bbound(nodes_sq_bbound):
    # fixtures for 2D square boundary conditions
    # common part of expected output bottom
    expected_output = np.zeros(nodes_sq_bbound.shape)
    expected_output[1, 0, 4] = 1  # resting reference particle
    expected_output[1, 1, 3] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_hex_rbound():
    # fixtures for 2D hex boundary conditions
    # right
    nodes = np.zeros((com.xdim_hex, com.ydim_hex, com.b_hex + 1))
    nodes[-1, :, 0] = 1  # particles that cross boundary
    nodes[1, 2, 0] = 1  # moving reference particle
    nodes[-1, 1:3, 6] = 1  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_rbound(nodes_hex_rbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output right
    expected_output = np.zeros(nodes_hex_rbound.shape)
    expected_output[-1, 1:3, 6] = 1  # resting reference particles
    expected_output[2, 2, 0] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_hex_trbound():
    # fixtures for 2D hex boundary conditions
    # top right
    nodes = np.zeros((com.xdim_hex, com.ydim_hex, com.b_hex + 1))
    nodes[-1, :, 1] = 1  # particles that cross boundary
    nodes[0:2, -1, 1] = 1  # particles that cross boundary
    nodes[0, 3, 1] = 1  # moving reference particle
    nodes[-1, 4:, 6] = 1  # resting reference particles
    nodes[-2, -1, 6] = 1  # resting reference particle
    return nodes


@pytest.fixture
def out_hex_trbound(nodes_hex_trbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output top right
    expected_output = np.zeros(nodes_hex_trbound.shape)
    expected_output[-1, 2::2, 1] = 1  # particles that do not cross boundary
    expected_output[0, 4, 1] = 1  # moving reference particle
    expected_output[-1, 4:, 6] = 1  # resting reference particles
    expected_output[-2, -1, 6] = 1  # resting reference particle
    return expected_output


@pytest.fixture
def nodes_hex_tlbound():
    # fixtures for 2D hex boundary conditions
    # top left
    nodes = np.zeros((com.xdim_hex, com.ydim_hex, com.b_hex + 1))
    nodes[0, :, 2] = 1  # particles that cross boundary
    nodes[1, -1, 2] = 1  # particle that crosses boundary
    nodes[-1, -1, 2] = 1  # particle that crosses boundary
    nodes[-1, 2, 2] = 1  # moving reference particle
    nodes[0:2, -1, 6] = 1  # resting reference particles
    nodes[0, -2, 6] = 1  # resting reference particle
    return nodes


@pytest.fixture
def out_hex_tlbound(nodes_hex_tlbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output top left
    expected_output = np.zeros(nodes_hex_tlbound.shape)
    expected_output[0, 1::2, 2] = 1  # particles that do not cross boundary
    expected_output[0:2, -1, 6] = 1  # resting reference particles
    expected_output[0, -2, 6] = 1  # resting reference particle
    expected_output[-1, 3, 2] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_hex_lbound():
    # fixtures for 2D hex boundary conditions
    # left
    nodes = np.zeros((com.xdim_hex, com.ydim_hex, com.b_hex + 1))
    nodes[0, :, 3] = 1  # particles that cross boundary
    nodes[2, 2, 3] = 1  # moving reference particle
    nodes[0, 1:3, 6] = 1  # resting reference particles
    return nodes


@pytest.fixture
def out_hex_lbound(nodes_hex_lbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output left
    expected_output = np.zeros(nodes_hex_lbound.shape)
    expected_output[0, 1:3, 6] = 1  # resting reference particles
    expected_output[1, 2, 3] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_hex_blbound():
    # fixtures for 2D hex boundary conditions
    # bottom left
    nodes = np.zeros((com.xdim_hex, com.ydim_hex, com.b_hex + 1))
    nodes[0, :, 4] = 1  # particles that cross boundary
    nodes[-2:, 0, 4] = 1  # particles that cross boundary
    nodes[-1, 2, 4] = 1  # moving reference particle
    nodes[:2, 0, 6] = 1  # resting reference particles
    nodes[0, 1, 6] = 1  # resting reference particle
    return nodes


@pytest.fixture
def out_hex_blbound(nodes_hex_blbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output bottom left
    expected_output = np.zeros(nodes_hex_blbound.shape)
    expected_output[0, 1::2, 4] = 1  # particle that does not cross boundary
    expected_output[0, -1, 4] = 0  # remove last particle
    expected_output[:2, 0, 6] = 1  # resting reference particles
    expected_output[0, 1, 6] = 1  # resting reference particle
    expected_output[-1, 1, 4] = 1  # moving reference particle
    return expected_output


@pytest.fixture
def nodes_hex_brbound():
    # fixtures for 2D hex boundary conditions
    # bottom right
    nodes = np.zeros((com.xdim_hex, com.ydim_hex, com.b_hex + 1))
    nodes[-1, :, 5] = 1  # particles that cross boundary
    nodes[-2, 0, 5] = 1  # particle that crosses boundary
    nodes[0, 0, 5] = 1  # particle that crosses boundary
    nodes[0, 3, 5] = 1  # moving reference particle
    nodes[-2:, 0, 6] = 1  # resting reference particles
    nodes[-1, 1, 6] = 1  # resting reference particle
    return nodes


@pytest.fixture
def out_hex_brbound(nodes_hex_brbound):
    # fixtures for 2D hex boundary conditions
    # common part of expected output bottom right
    expected_output = np.zeros(nodes_hex_brbound.shape)
    expected_output[-1, 0::2, 5] = 1  # particles that do not cross boundary
    expected_output[-2:, 0, 6] = 1  # resting reference particles
    expected_output[-1, 1, 6] = 1  # resting reference particle
    expected_output[0, 2, 5] = 1  # moving reference particle
    return expected_output


class Test_LGCA_classical(T_LGCA_Common):
    """
    Class for testing classical LGCA (volume exclusion, not identity-based).
    * propagation
    * recording options during simulation
    * boundary conditions
    The tests for boundary conditions are more extensive than necessary so that the boundary which is causing the problem
    can be isolated more easily.
    """
    com = T_LGCA_Common
    ve = True
    ib = False

    @pytest.mark.parametrize("geom,dims", [
        ('lin', (com.xdim_1d,)),
        ('square', (com.xdim_square, com.ydim_square)),
        ('hex', (com.xdim_hex, com.ydim_hex))
    ])
    def test_recording(self, geom, dims):
        # timeevo and recording: check if all properties are available when requested
        # keywords: record=False -> nodes_t, recordN=False n_t, recorddens=True -> dens_t, recordpertype=False -> velcells_t, restcells_t
        lgca_1 = get_lgca(geometry=geom, ve=True, dims=dims, density=0.5, interaction='only_propagation')
        lgca_2 = copy.deepcopy(lgca_1)

        lgca_1.timeevo(timesteps=2, recorddens=False, showprogress=False)
        assert not hasattr(lgca_1, 'dens_t'), "Records density when it should not"
        assert not hasattr(lgca_1, 'nodes_t'), "Records node configuration when it should not"
        assert not hasattr(lgca_1, 'n_t'), "Records particle number when it should not"
        assert not hasattr(lgca_1, 'restcells_t') and not hasattr(lgca_1,
                                                                  'velcells_t'), "Records density per channel type when it should not"
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

        lgca_5.timeevo(timesteps=2, recordpertype=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_5, 'restcells_t') and hasattr(lgca_5,
                                                          'velcells_t'), "Does not record density per channel type"

    def test_propagation(self):
        # 1D
        # input
        restchannels = 1
        nodes = np.zeros((self.xdim_1d, restchannels + self.b_1d))
        nodes[0, 0] = 1  # particle moving to the right
        nodes[1, 2] = 1  # particle resting
        nodes[2, 1] = 1  # particle moving to the left
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_1d, restchannels + self.b_1d))
        expected_output[1, :] = 1
        self.t_propagation_template('lin', nodes, expected_output)

        # 2D square
        # input
        restchannels = 2
        nodes = np.zeros((self.xdim_square, self.ydim_square, restchannels + self.b_square))
        nodes[0, 1, 0] = 1  # particle moving to the right
        nodes[1, 0, 1] = 1  # particle moving up
        nodes[2, 1, 2] = 1  # particle moving to the left
        nodes[1, 2, 3] = 1  # particle moving down
        nodes[1, 1, 4] = 1  # particle resting
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_square, self.ydim_square, restchannels + self.b_square))
        expected_output[1, 1, 0:5] = 1
        self.t_propagation_template('square', nodes, expected_output)

        # 2D hex
        # input
        restchannels = 2
        nodes = np.zeros((self.xdim_hex, self.ydim_hex, restchannels + self.b_hex))
        nodes[0, 1, 0] = 1  # particle moving right
        nodes[0, 0, 1] = 1  # particle moving to the upper right
        nodes[1, 0, 2] = 1  # particle moving to the upper left
        nodes[2, 1, 3] = 1  # particle moving left
        nodes[1, 2, 4] = 1  # particle moving to the lower left
        nodes[0, 2, 5] = 1  # particle moving to the lower right
        nodes[1, 1, 6] = 1  # particle resting
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_hex, self.ydim_hex, restchannels + self.b_hex))
        expected_output[1, 1, 0:7] = 1
        self.t_propagation_template('hex', nodes, expected_output)

    def test_pbc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        # check periodic boundary conditions in 1D
        # right boundary
        out_1d_rbound[0, 0] = 1  # particle that crosses boundary
        self.t_propagation_template('lin', nodes_1d_rbound, out_1d_rbound, bc='pbc')

        # left boundary
        out_1d_lbound[-1, 1] = 1  # particle that crosses boundary
        self.t_propagation_template('lin', nodes_1d_lbound, out_1d_lbound, bc='pbc')

    def test_rbc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        # check reflecting boundary conditions in 1D
        # right boundary
        out_1d_rbound[-1, 1] = 1  # particle that crosses boundary
        self.t_propagation_template('lin', nodes_1d_rbound, out_1d_rbound, bc='rbc')

        # left boundary
        out_1d_lbound[0, 0] = 1  # particle that crosses boundary
        self.t_propagation_template('lin', nodes_1d_lbound, out_1d_lbound, bc='rbc')

    def test_abc_1d(self, nodes_1d_rbound, out_1d_rbound, nodes_1d_lbound, out_1d_lbound):
        # check absorbing boundary conditions in 1D
        # right boundary
        self.t_propagation_template('lin', nodes_1d_rbound, out_1d_rbound, bc='abc')

        # left boundary
        self.t_propagation_template('lin', nodes_1d_lbound, out_1d_lbound, bc='abc')

    def test_pbc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        out_sq_rbound[0, :, 0] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_rbound, out_sq_rbound, bc='pbc')

        # left boundary
        out_sq_lbound[-1, :, 2] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_lbound, out_sq_lbound, bc='pbc')

        # top boundary
        out_sq_tbound[:, 0, 1] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_tbound, out_sq_tbound, bc='pbc')

        # bottom boundary
        out_sq_bbound[:, -1, 3] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_bbound, out_sq_bbound, bc='pbc')

    def test_rbc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        # check reflecting boundary conditions in 2D square
        # right boundary
        out_sq_rbound[-1, :, 2] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_rbound, out_sq_rbound, bc='rbc')

        # left boundary
        out_sq_lbound[0, :, 0] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_lbound, out_sq_lbound, bc='rbc')

        # top boundary
        out_sq_tbound[:, -1, 3] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_tbound, out_sq_tbound, bc='rbc')

        # bottom boundary
        out_sq_bbound[:, 0, 1] = 1  # particles that cross boundary
        self.t_propagation_template('square', nodes_sq_bbound, out_sq_bbound, bc='rbc')

    def test_abc_square(self, nodes_sq_rbound, out_sq_rbound, nodes_sq_lbound, out_sq_lbound,
                        nodes_sq_tbound, out_sq_tbound, nodes_sq_bbound, out_sq_bbound):
        # check absorbing boundary conditions in 2D square
        # right boundary
        self.t_propagation_template('square', nodes_sq_rbound, out_sq_rbound, bc='abc')

        # left boundary
        self.t_propagation_template('square', nodes_sq_lbound, out_sq_lbound, bc='abc')

        # top boundary
        self.t_propagation_template('square', nodes_sq_tbound, out_sq_tbound, bc='abc')

        # bottom boundary
        self.t_propagation_template('square', nodes_sq_bbound, out_sq_bbound, bc='abc')

    def test_pbc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        out_hex_rbound[0, :, 0] = 1  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_rbound, out_hex_rbound, bc='pbc')

        # top right boundary
        out_hex_trbound[0, 1::2, 1] = 1  # particles that cross boundary
        out_hex_trbound[0:2, 0, 1] = 1
        out_hex_trbound[-1, 0, 1] = 1
        self.t_propagation_template('hex', nodes_hex_trbound, out_hex_trbound, bc='pbc')

        # top left boundary
        out_hex_tlbound[0, 0, 2] = 1  # particles that cross boundary
        out_hex_tlbound[-2, 0, 2] = 1
        out_hex_tlbound[-1, 0::2, 2] = 1
        self.t_propagation_template('hex', nodes_hex_tlbound, out_hex_tlbound, bc='pbc')

        # left boundary
        out_hex_lbound[-1, :, 3] = 1  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_lbound, out_hex_lbound, bc='pbc')

        # bottom left boundary
        out_hex_blbound[-1, 0::2, 4] = 1  # particles that cross boundary
        out_hex_blbound[-2:, -1, 4] = 1
        out_hex_blbound[0, -1, 4] = 1
        self.t_propagation_template('hex', nodes_hex_blbound, out_hex_blbound, bc='pbc')

        # bottom right boundary
        out_hex_brbound[0, 1::2, 5] = 1  # particles that cross boundary
        out_hex_brbound[1, -1, 5] = 1
        out_hex_brbound[-1, -1, 5] = 1
        self.t_propagation_template('hex', nodes_hex_brbound, out_hex_brbound, bc='pbc')

    def test_rbc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        out_hex_rbound[-1, :, 3] = 1  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_rbound, out_hex_rbound, bc='rbc')

        # top right boundary
        out_hex_trbound[-1, 0::2, 4] = 1  # particles that cross boundary
        out_hex_trbound[0:2, -1, 4] = 1
        out_hex_trbound[-1, -1, 4] = 1
        self.t_propagation_template('hex', nodes_hex_trbound, out_hex_trbound, bc='rbc')

        # top left boundary
        out_hex_tlbound[0, 1::2, 5] = 1  # particles that cross boundary
        out_hex_tlbound[1, -1, 5] = 1
        out_hex_tlbound[-1, -1, 5] = 1
        self.t_propagation_template('hex', nodes_hex_tlbound, out_hex_tlbound, bc='rbc')

        # left boundary
        out_hex_lbound[0, :, 0] = 1  # particles that cross boundary
        self.t_propagation_template('hex', nodes_hex_lbound, out_hex_lbound, bc='rbc')

        # bottom left boundary
        out_hex_blbound[0, 1::2, 1] = 1  # particles that cross boundary
        out_hex_blbound[0, 0, 1] = 1
        out_hex_blbound[-2:, 0, 1] = 1
        self.t_propagation_template('hex', nodes_hex_blbound, out_hex_blbound, bc='rbc')

        # bottom right boundary
        out_hex_brbound[-1, 0::2, 2] = 1  # particles that cross boundary
        out_hex_brbound[0, 0, 2] = 1
        out_hex_brbound[-2, 0, 2] = 1
        self.t_propagation_template('hex', nodes_hex_brbound, out_hex_brbound, bc='rbc')

    def test_abc_hex(self, nodes_hex_rbound, out_hex_rbound, nodes_hex_trbound, out_hex_trbound,
                     nodes_hex_tlbound, out_hex_tlbound, nodes_hex_lbound, out_hex_lbound,
                     nodes_hex_blbound, out_hex_blbound, nodes_hex_brbound, out_hex_brbound):
        # check periodic boundary conditions in 2D square
        # right boundary
        self.t_propagation_template('hex', nodes_hex_rbound, out_hex_rbound, bc='abc')

        # top right boundary
        self.t_propagation_template('hex', nodes_hex_trbound, out_hex_trbound, bc='abc')

        # top left boundary
        self.t_propagation_template('hex', nodes_hex_tlbound, out_hex_tlbound, bc='abc')

        # left boundary
        self.t_propagation_template('hex', nodes_hex_lbound, out_hex_lbound, bc='abc')

        # bottom left boundary
        self.t_propagation_template('hex', nodes_hex_blbound, out_hex_blbound, bc='abc')

        # bottom right boundary
        self.t_propagation_template('hex', nodes_hex_brbound, out_hex_brbound, bc='abc')

    @pytest.mark.parametrize("geom,nodes", [
        ('lin', com.nodes_ve_1d),
        ('square', com.nodes_ve_square),
        ('hex', com.nodes_ve_hex)
    ])
    def test_characteristics(self, geom, nodes):
        # test compliance in all interactions to:
        # * volume exclusion principle
        ref_lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib)
        for interaction in ref_lgca.interactions:
            # test all boundary conditions in case of abuse of border nodes
            self.t_characteristics(geom, nodes, interaction, 'pbc')
            self.t_characteristics(geom, nodes, interaction, 'rbc')
            self.t_characteristics(geom, nodes, interaction, 'abc')

    def t_characteristics(self, geom, nodes, interaction, bc):
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=nodes, interaction=interaction, bc=bc)
        lgca.timeevo(timesteps=100, recorddens=False, record=True, showprogress=False)
        assert np.max(lgca.nodes_t.astype(int)) <= 1, "Volume exclusion principle is not respected"
        if lgca.nodes_t[-1].max() == 0 and lgca.nodes_t[0].max() != 0 and bc!='abc':
            warnings.warn("System died out in " + str(interaction))
