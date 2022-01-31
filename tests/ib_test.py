import numpy as np
import pytest
import copy
import warnings
import re


def matching_import(pattern, module, globals):
    """
    Imports only elements from a module that match the given regular expression
    (credit: https://stackoverflow.com/a/17226922)
    :param pattern: regular expression that describes the desired elements
    :param module: module to import the elements from (has to be imported first!)
    :param globals: dictionary containing the current scope's global variables: pass globals()
    """
    for key, value in module.__dict__.items():
        if re.findall(pattern, key):
            globals[key] = value


from lgca import get_lgca
# import other test modules for fixture and function reuse
# CAUTION: they need to be renamed and lose the "Test" prefix so that pytest does not executes these tests again
from tests.common_test import T_LGCA_Common
from tests.nove_test import Test_LGCA_NoVE as T_LGCA_NoVE  # rename to avoid duplicate execution of these tests
from tests.classical_test import Test_LGCA_classical as T_LGCA_classical
import tests.nove_test
matching_import("^nodes_", tests.nove_test, globals())
matching_import("^out_", tests.nove_test, globals())

com = T_LGCA_Common


class Test_LGCA_IB(T_LGCA_Common):
    """
        Class for testing identity-based LGCA (with volume exclusion).
        * propagation
        * recording options during simulation
        * boundary conditions (uses tests from classical LGCA)
    """

    com = T_LGCA_Common
    ve = True
    ib = True
    # reuse absorbing boundary condition tests from classical LGCA
    test_abc_1d = T_LGCA_classical.test_abc_1d
    test_abc_square = T_LGCA_classical.test_abc_square
    test_abc_hex = T_LGCA_classical.test_abc_hex
    # reuse fixtures from nove LGCA tests
    # # possible because the numbers of particles per channel are unique in the current fixtures
    test_pbc_1d = T_LGCA_NoVE.test_pbc_1d
    test_rbc_1d = T_LGCA_NoVE.test_rbc_1d
    test_pbc_square = T_LGCA_NoVE.test_pbc_square
    test_rbc_square = T_LGCA_NoVE.test_rbc_square
    test_pbc_hex = T_LGCA_NoVE.test_pbc_hex
    test_rbc_hex = T_LGCA_NoVE.test_rbc_hex

    @pytest.mark.parametrize("geom,dims", [
        ('lin', (com.xdim_1d,)),
        ('square', (com.xdim_square, com.ydim_square)),
        ('hex', (com.xdim_hex, com.ydim_hex))
    ])
    def test_recording(self, geom, dims):
        # timeevo and recording: check if all properties are available when requested
        # keywords: record=False -> nodes_t, recordN=False n_t, recorddens=True -> dens_t
        lgca_1 = get_lgca(geometry=geom, ve=True, ib=True, dims=dims, density=0.5, interaction='only_propagation')
        lgca_2 = copy.deepcopy(lgca_1)

        lgca_1.timeevo(timesteps=2, recorddens=False, showprogress=False)
        assert not hasattr(lgca_1, 'dens_t'), "Records density when it should not"
        assert not hasattr(lgca_1, 'nodes_t'), "Records node configuration when it should not"
        assert not hasattr(lgca_1, 'n_t'), "Records particle number when it should not"
        del lgca_1

        lgca_3 = copy.deepcopy(lgca_2)
        lgca_2.timeevo(timesteps=2, showprogress=False)
        assert hasattr(lgca_2, 'dens_t'), "Does not record density"
        del lgca_2

        lgca_4 = copy.deepcopy(lgca_3)
        lgca_3.timeevo(timesteps=2, record=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_3, 'nodes_t'), "Does not record node configuration"
        del lgca_3

        lgca_4.timeevo(timesteps=2, recordN=True, recorddens=False, showprogress=False)
        assert hasattr(lgca_4, 'n_t'), "Does not record particle number"

    def test_propagation(self):
        # 1D
        restchannels = 1
        # input
        nodes = np.zeros((self.xdim_1d, self.b_1d + restchannels))
        nodes[0, 0] = 1  # particle moving to the right
        nodes[1, 2] = 2  # particle resting
        nodes[2, 1] = 3  # particle moving to the left
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_1d, self.b_1d + restchannels))
        expected_output[1, 0] = 1  # particle moving to the right
        expected_output[1, 2] = 2  # particle resting
        expected_output[1, 1] = 3  # particle moving to the left
        self.t_propagation_template('lin', nodes, expected_output)

        # 2D square
        restchannels = 2
        # input
        nodes = np.zeros((self.xdim_square, self.ydim_square, self.b_square + restchannels))
        nodes[0, 1, 0] = 1  # particle moving to the right
        nodes[1, 0, 1] = 2  # particle moving up
        nodes[2, 1, 2] = 3  # particle moving to the left
        nodes[1, 2, 3] = 4  # particle moving down
        nodes[1, 1, 4] = 5  # particle resting
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_square, self.ydim_square, self.b_square + restchannels))
        expected_output[1, 1, 0] = 1  # particle moving to the right
        expected_output[1, 1, 1] = 2  # particle moving up
        expected_output[1, 1, 2] = 3  # particle moving to the left
        expected_output[1, 1, 3] = 4  # particle moving down
        expected_output[1, 1, 4] = 5  # particle resting
        self.t_propagation_template('square', nodes, expected_output)

        # 2D hex
        restchannels = 2
        # input
        nodes = np.zeros((self.xdim_hex, self.ydim_hex, self.b_hex + restchannels))
        nodes[0, 1, 0] = 1  # particle moving right
        nodes[0, 0, 1] = 2  # particle moving to the upper right
        nodes[1, 0, 2] = 3  # particle moving to the upper left
        nodes[2, 1, 3] = 4  # particle moving left
        nodes[1, 2, 4] = 5  # particle moving to the lower left
        nodes[0, 2, 5] = 6  # particle moving to the lower right
        nodes[1, 1, 6] = 7  # particle resting
        # output: all particles should move into one node within one timestep
        expected_output = np.zeros((self.xdim_hex, self.ydim_hex, self.b_hex + restchannels))
        expected_output[1, 1, 0] = 1  # particle moving right
        expected_output[1, 1, 1] = 2  # particle moving to the upper right
        expected_output[1, 1, 2] = 3  # particle moving to the upper left
        expected_output[1, 1, 3] = 4  # particle moving left
        expected_output[1, 1, 4] = 5  # particle moving to the lower left
        expected_output[1, 1, 5] = 6  # particle moving to the lower right
        expected_output[1, 1, 6] = 7  # particle resting
        self.t_propagation_template('hex', nodes, expected_output)

    @pytest.mark.parametrize("geom,nodes", [
        ('lin', com.nodes_ib_1d),
        ('square', com.nodes_ib_square),
        ('hex', com.nodes_ib_hex)
    ])
    def test_characteristics(self, geom, nodes):
        # test compliance in all interactions to:
        # * uniqueness of particle IDs for identification
        # * IDs are updated correctly (track of maximum ID)
        # * properties are updated for all particles
        ref_lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib)
        for interaction in ref_lgca.interactions:
            # test all boundary conditions in case of abuse of border nodes
            self.t_characteristics(geom, nodes, interaction, 'pbc')
            self.t_characteristics(geom, nodes, interaction, 'rbc')
            self.t_characteristics(geom, nodes, interaction, 'abc')

    def t_characteristics(self, geom, nodes, interaction, bc):
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=nodes, interaction=interaction, bc=bc)
        lgca.timeevo(timesteps=100, recorddens=False, record=True, showprogress=False)
        if lgca.nodes[lgca.nonborder].max() == 0 and lgca.nodes_t[0].max() != 0 and bc != 'abc':
            warnings.warn("System died out in " + str(interaction))
        for i in range(101):  # check throughout time evolution
            # uniqueness principle
            if lgca.nodes_t[i].min() == 0:  # 0 can occur more than once
                if not lgca.nodes_t[i].max() == 0:
                    # if only 0 occurs in this timestep, uniqueness is fulfilled, so only check the rest
                    assert np.unique(lgca.nodes_t[i], return_counts=True)[1][1:].max() <= 1, \
                        "Uniqueness principle is broken"
            else:  # if 0 does not occur, all IDs have to be unique
                assert np.unique(lgca.nodes_t[i], return_counts=True)[1].max() <= 1, \
                    "Uniqueness principle is broken"
            # check ID updates: max can be smaller if a particle died
            assert lgca.nodes.max() <= lgca.maxlabel, "IDs not updated correctly"
            # check property updates
            if lgca.props == {}:
                continue
            for propname in lgca.props.keys():
                assert len(lgca.props[propname]) == lgca.maxlabel + 1, "Properties not updated for all particles"
