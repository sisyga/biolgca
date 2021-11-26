try:
    from classical_test import *
except ModuleNotFoundError:
    from .classical_test import *

class Test_LGCA_NoVE(Test_LGCA_classical):
    """
        Class for testing LGCA without volume exclusion (not identity-based).
        * propagation
        * recording options during simulation
        * boundary conditions (uses tests from classical LGCA)
    """

    com = T_LGCA_Common
    gen = Test_LGCA_General

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

    def t_propagation_template(self, geom, nodes, expected_output, bc='pbc'):
        # check that propagation and rest channels work: all particles should move into one node within one timestep
        lgca = get_lgca(geometry=geom, ve=False, nodes=nodes, bc=bc, interaction='only_propagation')
        lgca.timeevo(timesteps=1, recorddens=False, showprogress=False)

        assert lgca.nodes[lgca.nonborder].sum() == expected_output.sum(), "Particles appear or disappear"
        assert np.array_equal(lgca.nodes[lgca.nonborder],
                              expected_output), "Node configuration after propagation not correct"

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
        ('lin', gen.nodes_nove_1d, com.b_1d),
        ('square', gen.nodes_nove_square, com.b_square),
        ('hex', gen.nodes_nove_hex, com.b_hex)
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