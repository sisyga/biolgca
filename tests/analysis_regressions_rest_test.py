"""What a researcher reads off a model, against hand-made states: trait tables per node, density plots,
positions and distances, velocity correlations and nematic tensors."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lgca import get_lgca


@pytest.fixture
def nove_ib():
    """An identity-based model without volume exclusion with a few labelled cells at known nodes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgca = get_lgca(geometry="square", ib=True, ve=False, dims=(3, 3), density=1, seed=1,
                        interaction="random_walk")
    return lgca


@pytest.mark.parametrize("ve", [True, False], ids=["ve", "nove"])
def test_get_prop_reads_the_property_table_it_is_given(ve):
    # a student passing their own table, of a property the model does not have, gets its values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgca = get_lgca(geometry="square", ib=True, ve=ve, dims=(4, 4), density=1, seed=1,
                        interaction="random_walk")
    table = {"size": np.full(lgca.maxlabel + 1, 0.3)}
    values = np.asarray(lgca.get_prop(props=table))
    cells = values if not ve else values[lgca.nodes[lgca.nonborder] > 0]
    assert len(cells) == lgca.cell_density[lgca.nonborder].sum() > 0
    assert np.all(cells == 0.3)


def test_calc_prop_mean_defaults_to_the_first_property_of_the_model(nove_ib):
    labels = nove_ib.nodes[nove_ib.nonborder]
    nove_ib.props = {"label": np.arange(nove_ib.maxlabel + 1, dtype=float)}
    mean = nove_ib.calc_prop_mean()
    for index in np.ndindex(nove_ib.dims):
        cells = [label for channel in labels[index] for label in channel]
        if cells:
            assert mean[index] == pytest.approx(np.mean(cells))
        else:
            assert np.ma.is_masked(mean[index])


@pytest.mark.parametrize("geometry", ["square", "hex"])
def test_multispecies_density_plot_tells_a_full_node_from_a_half_full_one(geometry):
    # two species, K = 8 channels each: a full node holds 16 cells
    nodes = np.zeros((6, 6, 2, 8), dtype=bool)
    nodes[1, 1] = True  # 16 cells
    nodes[4, 4, 0] = True  # 8 cells
    lgca = get_lgca(geometry=geometry, n_species=2, nodes=nodes, interaction="only_propagation")
    _, _, mappable = lgca.plot_density()
    assert mappable.norm(16) != mappable.norm(8)
    assert mappable.norm(16) == mappable.norm(mappable.norm.vmax - 1)  # the top of the colour bar
    # each species alone holds up to K cells
    _, _, mappable = lgca.plot_density(species=0)
    assert mappable.norm.vmax == lgca.K + 1


def test_density_plot_marks_values_above_vmax():
    nodes = np.zeros((6, 6, 5), dtype=bool)
    nodes[1, 1] = True
    lgca = get_lgca(geometry="square", nodes=nodes, interaction="only_propagation")
    _, image, _ = lgca.plot_density(vmax=3)
    assert image.colorbar.extend == "both"
    plt.close("all")
    _, image, _ = lgca.plot_density()
    assert image.colorbar.extend == "min"


@pytest.mark.parametrize("y", [2, 3], ids=["even_row", "odd_row"])
def test_tumour_spheroid_positions_put_hex_neighbours_one_node_apart(y):
    # the example measures the spheroid, its unstirred layer and its rings with these positions
    from lgca.examples import tumour_spheroid

    dims = (6, 6)
    lgca = get_lgca(geometry="hex", dims=dims, density=0, restchannels=0, bc="periodic",
                    interaction="only_propagation")
    nodes = np.zeros(dims + (6,), dtype=bool)
    nodes[2, y] = True  # one cell in every velocity channel
    lgca.nodes[lgca.nonborder] = nodes
    lgca.timestep()
    x_pos, y_pos = tumour_spheroid._positions("hex", dims)
    targets = np.argwhere(lgca.cell_density[lgca.nonborder] > 0)
    assert len(targets) == 6
    distances = np.hypot(x_pos[tuple(targets.T)] - x_pos[2, y], y_pos[tuple(targets.T)] - y_pos[2, y])
    np.testing.assert_allclose(distances, 1.0)
    np.testing.assert_allclose(x_pos, lgca.xcoords)
    np.testing.assert_allclose(y_pos, lgca.ycoords)


def test_velocity_correlation_counts_cells_without_volume_exclusion():
    # three cells moving right and one moving left: the node moves right, like its right neighbour
    lgca = get_lgca(geometry="square", ve=False, dims=(5, 5), density=0, restchannels=0,
                    interaction="only_propagation", bc="reflecting")
    nodes = np.zeros((5, 5, 4), dtype=int)
    nodes[2, 2, 0], nodes[2, 2, 2], nodes[3, 2, 0] = 3, 1, 1
    lgca.nodes[lgca.nonborder] = nodes
    correlation = lgca.calc_velocity_correlation()[lgca.nonborder]
    assert correlation[2, 2] == pytest.approx(1.0)
    assert correlation[3, 2] == pytest.approx(1.0)


def test_nematic_tensors_of_the_states():
    # s = sum over occupied velocity channels of c c^T - I/2, the tensor of nematic rules
    lgca = get_lgca(geometry="hex", dims=(3, 3), density=0.2, restchannels=1, interaction="nematic", seed=1)
    for n in range(lgca.K + 1):
        states = lgca.get_permutations(n)[:, :lgca.velocitychannels]
        expected = np.einsum("si,ai,bi->sab", states, lgca.c, lgca.c) - 0.5 * states.sum(-1)[:, None, None] * np.eye(2)
        np.testing.assert_allclose(lgca.get_si_permutations(n), expected, atol=1e-12)
