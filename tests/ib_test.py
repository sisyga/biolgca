import numpy as np
import pytest
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
from lgca.lgca_3dmoore import LGCA_3dMoore
# import other test modules for fixture and function reuse
# CAUTION: they need to be renamed and lose the "Test" prefix so that pytest does not executes these tests again
from tests.common_test import T_LGCA_Common
from tests.nove_test import Test_LGCA_NoVE as T_LGCA_NoVE  # rename to avoid duplicate execution of these tests
from tests.classical_test import Test_LGCA_classical as T_LGCA_classical
import tests.nove_test
import tests.classical_test
matching_import("^nodes_", tests.nove_test, globals())
matching_import("^out_", tests.nove_test, globals())
matching_import("^nodes_cb_", tests.classical_test, globals())
matching_import("^out_cb_", tests.classical_test, globals())

com = T_LGCA_Common


class Test_LGCA_IB(T_LGCA_Common):
    """
        Class for testing identity-based LGCA (with volume exclusion).
        * propagation
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
    test_pbc_cubic = T_LGCA_classical.test_pbc_cubic
    test_rbc_cubic = T_LGCA_classical.test_rbc_cubic
    test_abc_cubic = T_LGCA_classical.test_abc_cubic
    test_pbc_moore = T_LGCA_classical.test_pbc_moore
    test_rbc_moore = T_LGCA_classical.test_rbc_moore
    test_abc_moore = T_LGCA_classical.test_abc_moore

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

        # 3D cubic
        restchannels = 2
        nodes = np.zeros((self.xdim_cubic, self.ydim_cubic, self.zdim_cubic, self.b_cubic + restchannels))
        nodes[0, 1, 1, 0] = 1
        nodes[2, 1, 1, 1] = 2
        nodes[1, 0, 1, 2] = 3
        nodes[1, 2, 1, 3] = 4
        nodes[1, 1, 0, 4] = 5
        nodes[1, 1, 2, 5] = 6
        nodes[1, 1, 1, 6] = 7
        expected_output = np.zeros((self.xdim_cubic, self.ydim_cubic, self.zdim_cubic, self.b_cubic + restchannels))
        expected_output[1, 1, 1, 0] = 1
        expected_output[1, 1, 1, 1] = 2
        expected_output[1, 1, 1, 2] = 3
        expected_output[1, 1, 1, 3] = 4
        expected_output[1, 1, 1, 4] = 5
        expected_output[1, 1, 1, 5] = 6
        expected_output[1, 1, 1, 6] = 7
        self.t_propagation_template('cubic', nodes, expected_output)

        # 3D moore
        restchannels = 2
        nodes = np.zeros((self.xdim_moore, self.ydim_moore, self.zdim_moore, self.b_moore + restchannels))
        nodes[0, 1, 1, 21] = 1  # +x
        nodes[2, 1, 1, 4] = 2   # -x
        nodes[1, 0, 1, 15] = 3  # +y
        nodes[1, 2, 1, 10] = 4  # -y
        nodes[1, 1, 0, 13] = 5  # +z
        nodes[1, 1, 2, 12] = 6  # -z
        nodes[1, 1, 1, 26] = 7  # rest
        expected_output = np.zeros((self.xdim_moore, self.ydim_moore, self.zdim_moore, self.b_moore + restchannels))
        expected_output[1, 1, 1, [21, 4, 15, 10, 13, 12, 26]] = [1, 2, 3, 4, 5, 6, 7]
        self.t_propagation_template('moore', nodes, expected_output)

    def test_propagation_moore_all_channels(self):
        restchannels = 2
        nodes = np.zeros(
            (self.xdim_moore, self.ydim_moore, self.zdim_moore, restchannels + self.b_moore),
            dtype=bool,
        )
        center = (1, 1, 1)
        for idx, (dx, dy, dz) in enumerate(LGCA_3dMoore.velocities):
            nodes[center[0] - dx, center[1] - dy, center[2] - dz, idx] = True
        nodes[center][self.b_moore :] = True
        expected = np.zeros_like(nodes)
        expected[center] = True
        self.t_propagation_template('moore', nodes, expected)

    @pytest.mark.parametrize("geom,nodes", [
        ('lin', com.nodes_ib_1d),
        ('square', com.nodes_ib_square),
        ('hex', com.nodes_ib_hex),
        ('cubic', com.nodes_ib_cubic),
        ('moore', com.nodes_ib_moore)
    ])
    def test_characteristics(self, geom, nodes):
        # in all interactions and at every boundary:
        # * every living cell keeps a label of its own, at every step
        # * labels never exceed maxlabel, and every label has its property rows
        # * motility interactions keep the same cells (labels) on closed lattices
        ref_lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib)
        for interaction in ref_lgca.interactions:
            for bc in ('pbc', 'rbc', 'abc'):
                self.t_characteristics(geom, nodes, interaction, bc)

    def t_characteristics(self, geom, nodes, interaction, bc):
        lgca = get_lgca(geometry=geom, ve=self.ve, ib=self.ib, nodes=nodes, interaction=interaction, bc=bc,
                        seed=1)
        lgca.timeevo(timesteps=20, recorddens=False, record=True, showprogress=False)
        initial_labels = np.sort(nodes[nodes > 0])
        for state in lgca.nodes_t:
            labels = state[state > 0]
            assert np.unique(labels).size == labels.size, "two cells share a label"
            assert labels.max(initial=0) <= lgca.maxlabel, "a label exceeds maxlabel"
            if interaction in ('only_propagation', 'random_walk') and bc != 'abc':
                np.testing.assert_array_equal(np.sort(labels), initial_labels,
                                              err_msg=f"{interaction} creates, loses or relabels cells")
        for propname, values in lgca.props.items():
            assert len(values) == lgca.maxlabel + 1, f"property {propname} misses rows of new cells"
