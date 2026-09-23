"""Lattice plots choose their figure predictably, and recorded animations play and save."""

import gc

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lgca import get_lgca
from lgca.plots import LatticeAnimation, colorbar_index

FAMILIES = {"classical": {}, "nove": {"ve": False}, "ib": {"ib": True}, "nove_ib": {"ve": False, "ib": True}}


def _recorded(geometry, family, steps=3):
    dims = 10 if geometry == "lin" else (10, 8)
    restchannels = 0 if FAMILIES[family] == {"ve": False} else 1
    lgca = get_lgca(geometry=geometry, dims=dims, density=1, restchannels=restchannels, seed=1, **FAMILIES[family])
    lgca.timeevo(timesteps=steps, record=True, showprogress=False)
    return lgca


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("geometry", ["lin", "square", "hex"])
def test_consecutive_plots_open_separate_figures(geometry, family):
    lgca = _recorded(geometry, family)

    lgca.plot_density()
    lgca.plot_flux()

    assert len(plt.get_fignums()) == 2


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("geometry", ["lin", "square", "hex"])
def test_plots_draw_into_given_axes(geometry, family):
    lgca = _recorded(geometry, family)
    fig, (left, right) = plt.subplots(1, 2)

    lgca.plot_density(ax=left)
    lgca.plot_flux(ax=right)

    assert plt.get_fignums() == [fig.number]
    assert left.images or left.collections
    assert right.images or right.collections


def test_an_empty_current_figure_is_used_and_keeps_its_size():
    lgca = _recorded("square", "classical")
    fig = plt.figure(figsize=(3, 3))

    lgca.plot_density(figsize=None)

    assert plt.get_fignums() == [fig.number]
    assert fig.axes


def test_hexagonal_row_ticks_are_whole_rows_at_round_intervals():
    lgca = get_lgca(geometry="hex", dims=(60, 50), density=0.5, seed=1)

    _, ax = lgca.setup_figure()

    rows = [int(label.get_text()) for label in ax.get_yticklabels() if label.get_text()]
    assert rows and all(row % 10 == 0 for row in rows)
    np.testing.assert_allclose(ax.get_yticks() / lgca.dy, np.round(ax.get_yticks() / lgca.dy), atol=1e-9)


@pytest.mark.parametrize("ncolors, expected", [(4, [0, 1, 2, 3]), (13, [0, 2, 4, 6, 8, 10, 12])])
def test_discrete_colorbar_ticks_sit_on_bin_centres(ncolors, expected):
    fig, ax = plt.subplots()

    colorbar = colorbar_index(ncolors, "viridis", cax=ax)

    np.testing.assert_array_equal(colorbar.get_ticks(), expected)


@pytest.mark.parametrize("kind", ["density", "flux", "config", "flow"])
def test_recorded_animations_save_gifs_without_ffmpeg(kind, tmp_path):
    lgca = _recorded("hex", "classical", steps=1)
    path = tmp_path / f"{kind}.gif"

    getattr(lgca, f"animate_{kind}")(save_path=path, save_kwargs={"dpi": 20})

    assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["density", "flux", "config"])
def test_recorded_animations_of_every_family_are_lattice_animations(kind, family):
    lgca = _recorded("square", family, steps=1)
    anim = getattr(lgca, f"animate_{kind}")()

    assert isinstance(anim, LatticeAnimation)
    del anim
    gc.collect()  # delete the unrendered animation while its warning is filtered


def test_animation_displays_as_html_player_and_closes_its_static_figure():
    lgca = _recorded("square", "classical")
    anim = lgca.animate_density()

    html = anim._repr_html_()

    assert "<script" in html
    assert plt.get_fignums() == []


def test_colorbars_in_constrained_layout_stay_inside_their_panel():
    lgca = _recorded("hex", "classical")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")

    lgca.plot_density(ax=left)
    lgca.plot_density(ax=right)
    fig.canvas.draw()

    left_colorbar = left.child_axes[0].get_tightbbox()
    assert left_colorbar.x1 < right.get_tightbbox().x0


def _outside_figure(fig):
    """Labels and visible tick labels that extend beyond the figure."""
    fig.canvas.draw()
    bounds = fig.bbox
    texts = []
    for ax in fig.axes:
        texts += [ax.xaxis.label, ax.yaxis.label]
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            texts += [label for tick in axis.get_major_ticks() if low <= tick.get_loc() <= high
                      for label in (tick.label1, tick.label2) if label.get_visible()]
    return [text.get_text() for text in texts if text.get_text() and (
        text.get_window_extent().x0 < bounds.x0 - 1 or text.get_window_extent().x1 > bounds.x1 + 1
        or text.get_window_extent().y0 < bounds.y0 - 1 or text.get_window_extent().y1 > bounds.y1 + 1)]


@pytest.mark.parametrize("geometry, dims", [("square", (50, 50)), ("square", (80, 20)), ("hex", (80, 20)),
                                            ("lin", 50), ("lin", 200)])
@pytest.mark.parametrize("kind", ["density", "flux"])
def test_standalone_plots_keep_labels_and_colorbars_inside_the_figure(geometry, dims, kind):
    lgca = get_lgca(geometry=geometry, dims=dims, density=1, seed=1)
    lgca.timeevo(timesteps=20, record=True, showprogress=False)

    getattr(lgca, f"plot_{kind}")()

    assert _outside_figure(plt.gcf()) == []
