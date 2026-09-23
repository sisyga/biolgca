"""Independent measurement oracles and observational purity checks."""

from copy import deepcopy

import numpy as np
import pytest

from lgca import get_lgca
from lgca.simulation import OrderParameterRecorder, SimulationRunner


@pytest.mark.parametrize("bc", ["periodic", "reflecting", "absorbing"])
@pytest.mark.parametrize("geometry,dims", [("lin", (4,)), ("square", (4, 4)), ("hex", (4, 4))])
def test_alignment_measurement_preserves_state_and_rng(bc, geometry, dims):
    model = get_lgca(geometry=geometry, dims=dims, ve=False, density=2,
                     bc=bc, seed=129, interaction="only_propagation")
    before = model.nodes.copy()
    density = model.cell_density.copy()
    rng = deepcopy(model.rng.bit_generator.state)

    SimulationRunner(model, timesteps=0, observers=[OrderParameterRecorder()], showprogress=False).run()

    assert np.isfinite(model.meanAlign_t).all()
    np.testing.assert_array_equal(model.nodes, before)
    np.testing.assert_array_equal(model.cell_density, density)
    assert model.rng.bit_generator.state == rng


@pytest.mark.parametrize("bc,expected", [("periodic", 1.0),
                                        ("reflecting", 0.75), ("absorbing", 0.75)])
def test_uniform_directors_have_hand_computed_wall_alignment(bc, expected):
    nodes = np.zeros((4, 4, 4), dtype=int)
    nodes[..., 0] = 2
    model = get_lgca(geometry="square", ve=False, nodes=nodes, bc=bc,
                     interaction="only_propagation")
    # Four neighbors per periodic site; an open 4x4 grid has 48 directed edges.
    assert model.calc_mean_alignment() == pytest.approx(expected)


@pytest.mark.parametrize("populations,expected", [([[2, 0], [0, 2]], 0),
                                                  ([[2, 0], [2, 0]], 1),
                                                  ([[2, 0], [0, 0]], 1),
                                                  ([[0, 0], [0, 0]], 0),
                                                  ([[3, 0], [0, 1]], .5)])
@pytest.mark.parametrize("reverse", [False, True])
def test_global_polarization_sums_species_before_norm(populations, expected, reverse):
    nodes = np.tile(np.array(populations)[None], (3, 1, 1))
    if reverse:
        nodes = nodes[:, ::-1]
    model = get_lgca(geometry="lin", ve=False, n_species=2, nodes=nodes,
                     interaction="only_propagation")
    aggregate = get_lgca(geometry="lin", ve=False, nodes=nodes.sum(axis=1),
                         interaction="only_propagation")
    assert model.calc_polar_alignment_parameter() == pytest.approx(expected)
    assert aggregate.calc_polar_alignment_parameter() == pytest.approx(expected)
