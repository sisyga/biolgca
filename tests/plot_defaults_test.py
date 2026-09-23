"""Plotting methods must run with their default arguments and show particle counts."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="requires matplotlib for plotting tests")
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from lgca import get_lgca


FAMILIES = {
    "classical": dict(ve=True, ib=False, interaction="alignment", restchannels=1),
    "identity": dict(ve=True, ib=True, interaction="birth", restchannels=1),
    "nove": dict(ve=False, ib=False, interaction="dd_alignment", restchannels=0),
    "nove_identity": dict(ve=False, ib=True, interaction="go_or_grow", restchannels=1),
}
GEOMETRIES = {"lin": 12, "square": 6, "hex": (6, 6)}
DEFAULT_PLOTS = {
    "lin": ["plot_density", "plot_flux"],
    "square": ["plot_density", "plot_flux", "plot_flow", "plot_config"],
    "hex": ["plot_density", "plot_flux", "plot_flow", "plot_config"],
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _run(family, geometry, steps=4, seed=2):
    lgca = get_lgca(geometry=geometry, dims=GEOMETRIES[geometry], density=0.5, seed=seed,
                    **FAMILIES[family])
    lgca.timeevo(timesteps=steps, record=True, showprogress=False)
    return lgca


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("family", FAMILIES)
def test_plots_run_with_default_arguments(family, geometry):
    lgca = _run(family, geometry)
    for name in DEFAULT_PLOTS[geometry]:
        if name == "plot_config" and family in ("identity",):
            continue  # plot_config needs a boolean configuration for identity lattices
        getattr(lgca, name)()
        plt.close("all")


def test_identity_flow_colours_arrows_by_cell_count_not_labels():
    nodes = np.zeros((3, 3, 5), dtype=np.uint)
    nodes[1, 1, 0] = 300  # one cell with a large label moving right
    lgca = get_lgca(geometry="square", ib=True, nodes=nodes, interaction="only_propagation")

    _, quiver = lgca.plot_flow()

    assert quiver.get_array().max() == 1


@pytest.mark.parametrize("label", [1, 128, 256])
def test_identity_flux_plot_keeps_cells_with_any_label(label):
    nodes = np.zeros((3, 3, 5), dtype=np.uint)
    nodes[1, 1, 0] = label
    lgca = get_lgca(geometry="square", ib=True, nodes=nodes, interaction="only_propagation")

    _, collection, _ = lgca.plot_flux()

    alpha = collection.get_facecolor()[:, -1].reshape(3, 3)
    assert alpha[1, 1] == 1
    assert alpha.sum() == 1


def test_nove_identity_flux_counts_every_cell_in_a_channel():
    nodes = np.zeros((3, 3), dtype=int)
    nodes[1, 0] = 5
    lgca = get_lgca(geometry="lin", ib=True, ve=False, nodes=nodes, interaction="only_propagation")

    np.testing.assert_array_equal(lgca.calc_flux(lgca.nodes[lgca.nonborder]).ravel(), [0, 5, 0])


def test_nove_identity_spatial_property_plot_uses_latest_history():
    lgca = _run("nove_identity", "lin", steps=3)
    lgca.plot_prop_spatial(propname="kappa")
    lgca.timeevo(timesteps=8, record=True, showprogress=False)

    image = lgca.plot_prop_spatial(propname="kappa")

    assert image.get_array().shape == (9, 12)


def test_nove_identity_2d_property_plot_uses_given_state():
    lgca = _run("nove_identity", "square")
    empty = np.empty(lgca.nodes[lgca.nonborder].shape, dtype=object)
    for index in np.ndindex(empty.shape):
        empty[index] = []

    _, collection, _ = lgca.plot_prop_spatial(nodes=empty, propname="kappa")

    assert np.ma.getmaskarray(collection.get_array()).all()


def test_nove_identity_2d_property_histogram_without_seaborn():
    lgca = _run("nove_identity", "square")

    fig, (joint, top, right) = lgca.plot_prop_2dhist(propnames=("kappa", "theta"))

    assert joint.get_xlabel() == "kappa"
    assert joint.get_ylabel() == "theta"
    assert fig is joint.figure


def test_nove_identity_lists_living_families_with_crowded_nodes():
    lgca = get_lgca(geometry="square", dims=5, ib=True, ve=False, restchannels=1, density=3,
                    interaction="go_or_grow_glioblastoma", seed=1)
    lgca.timeevo(timesteps=3, showprogress=False)

    families = lgca.list_families_alive()

    assert families.size >= 1
    assert np.all(np.diff(families) > 0)
