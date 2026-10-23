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


def _labelled_line(ve):
    # cells 1 and 2 at node 0, cell 3 at node 1, node 2 empty; one rest channel
    if ve:
        nodes = np.array([[1, 2, 0], [0, 0, 3], [0, 0, 0]])
    else:
        nodes = np.empty((3, 3), dtype=object)
        for index in np.ndindex(nodes.shape):
            nodes[index] = []
        nodes[0, 0], nodes[1, 1] = [1, 2], [3]
    model = get_lgca(geometry="lin", ib=True, ve=ve, nodes=nodes, interaction="only_propagation")
    model.props["r_b"] = [0., 0.2, 0.4, 0.9]
    return model


@pytest.mark.parametrize("ve", [True, False], ids=["ib", "nove_ib"])
def test_mean_property_of_a_node_averages_its_cells_and_masks_empty_nodes(ve):
    model = _labelled_line(ve)
    mean = model.calc_prop_mean(nodes=model.nodes[model.nonborder], props=model.props, propname="r_b")
    np.testing.assert_array_equal(np.ma.getmaskarray(mean), [False, False, True])
    np.testing.assert_allclose(np.ma.getdata(mean)[:2], [0.3, 0.9])


@pytest.mark.parametrize("ve", [True, False], ids=["ib", "nove_ib"])
def test_mean_property_defaults_to_the_current_cells(ve):
    model = _labelled_line(ve)
    np.testing.assert_allclose(np.ma.getdata(model.calc_prop_mean(propname="r_b"))[:2], [0.3, 0.9])
