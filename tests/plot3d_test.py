"""Offscreen rendering tests for the Mayavi plots of cubic and Moore lattices.

Skipped unless the ``plot3d`` extra is installed.
"""

import warnings

import numpy as np
import pytest

mlab = pytest.importorskip("mayavi.mlab")

import lgca.lgca_cubic as cubic  # noqa: E402
from lgca import get_lgca  # noqa: E402

FAMILIES = {
    "classical": dict(geometry="cubic", ve=True, ib=False, interaction="random_walk", restchannels=1, density=0.3),
    "ib": dict(geometry="cubic", ve=True, ib=True, interaction="go_and_grow", restchannels=1, density=0.3),
    "nove": dict(geometry="cubic", ve=False, ib=False, interaction="go_or_grow", restchannels=1, density=1,
                 capacity=8),
    "nove_ib": dict(geometry="cubic", ve=False, ib=True, interaction="go_or_grow", restchannels=1, density=1,
                    capacity=8),
    "moore": dict(geometry="moore", ve=True, ib=False, interaction="random_walk", restchannels=1, density=0.3),
    "ms_moore": dict(geometry="moore", ve=True, ib=False, interaction="random_walk", restchannels=1, density=0.3,
                     n_species=2),
}


@pytest.fixture(autouse=True)
def offscreen():
    mlab.options.offscreen = True
    yield
    mlab.close(all=True)


def make_model(family, dims=(5, 5, 5), steps=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = get_lgca(dims=dims, seed=7, **FAMILIES[family])
    model.timeevo(timesteps=steps, record=True, showprogress=False)
    return model


def drawn_pixels(fig):
    image = mlab.screenshot(fig)
    return np.count_nonzero(np.any(image < 250, axis=-1))


def run_frames(monkeypatch, n_frames=None):
    """Replace the GUI animation loop by calling every frame once; return the frame labels."""
    labels = []

    def play(fig, update, frames, label, interval, show, **kwargs):
        for frame in range(frames if frames is not None else n_frames):
            update(frame)
            labels.append(label(frame))
        return "animator"

    monkeypatch.setattr(cubic, "play", play)
    return labels


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["plot_density", "plot_density_cubes", "plot_flux", "plot_config"])
def test_every_3d_plot_draws_the_lattice_for_every_model_family(family, method):
    fig = getattr(make_model(family), method)()[0]
    assert drawn_pixels(fig) > 1000


@pytest.mark.parametrize("family", ["ib", "nove_ib"])
def test_property_plot_shows_the_mean_property_of_occupied_nodes_only(family):
    model = make_model(family)
    nodes = model.nodes[model.nonborder]
    occupied = np.any(model._channel_counts(nodes), axis=-1)
    propname = next(iter(model.props))
    mean = np.ma.getdata(model.calc_prop_mean(propname=propname, props=model.props, nodes=nodes))

    _, cubes, _ = model.plot_prop_spatial(propname=propname)

    np.testing.assert_allclose(np.sort(cubes.mlab_source.scalars), np.sort(mean[occupied]))
    values = np.asarray(model.props[propname], dtype=float)
    assert values.min() <= cubes.mlab_source.scalars.min() <= cubes.mlab_source.scalars.max() <= values.max()


def test_glyphs_sit_at_node_centres_inside_the_domain_box():
    model = make_model("classical", dims=(4, 5, 6), steps=0)
    density = np.zeros(model.dims)
    density[1, 2, 3] = 2

    _, cubes = model.plot_density_cubes(density=density)

    np.testing.assert_array_equal(np.c_[cubes.mlab_source.x, cubes.mlab_source.y, cubes.mlab_source.z],
                                  [[1.5, 2.5, 3.5]])


def test_flux_arrows_are_centred_on_the_node_and_the_largest_is_drawn_09_lattice_units_long():
    model = make_model("classical", dims=(3, 3, 3), steps=0)
    nodes = np.zeros(model.dims + (model.K,), dtype=bool)
    nodes[1, 1, 1, [0, 2]] = True  # +x and +y: flux (1, 1, 0)

    _, quiver, _ = model.plot_flux(nodes=nodes)

    assert quiver.glyph.glyph_source.glyph_position == "center"
    assert quiver.glyph.glyph.scale_factor == pytest.approx(0.9 / np.sqrt(2))
    centre = (model.xcoords == 1) & (model.ycoords == 1) & (model.zcoords == 1)
    np.testing.assert_array_equal(quiver.mlab_source.u[centre], [1])
    np.testing.assert_array_equal(quiver.mlab_source.v[centre], [1])


def test_isosurface_levels_stay_fixed_when_frames_change_the_data_range():
    model = make_model("nove", steps=0)
    density = np.zeros(model.dims)
    density[2, 2, 2] = 8

    _, contour = model.plot_density(density=density, contours=[2.0, 6.0], vmax=8)
    density[2, 2, 2] = 3
    contour.mlab_source.set(scalars=density)

    assert list(contour.contour.contours) == [2.0, 6.0]


def test_integer_colour_bar_centres_one_bin_and_label_on_each_particle_number():
    model = make_model("classical", steps=0)

    _, cubes = model.plot_density_cubes()

    colorbar = cubes.module_manager.scalar_lut_manager
    assert tuple(colorbar.data_range) == (-0.5, model.K + 0.5)
    assert colorbar.number_of_colors == model.K + 1
    labels = colorbar.scalar_bar.custom_labels.to_array()
    np.testing.assert_array_equal(labels, np.arange(model.K + 1))


def test_nove_config_draws_full_length_arrows_coloured_by_channel_population():
    model = make_model("nove", dims=(3, 3, 3), steps=0)
    nodes = np.zeros(model.dims + (model.K,), dtype=int)
    nodes[0, 0, 0, 0] = 3  # three particles moving in +x
    nodes[2, 2, 2, 3] = 1  # one particle moving in -y

    _, quiver, _ = model.plot_config(nodes=nodes)

    source = quiver.mlab_source
    lengths = np.sqrt(source.u ** 2 + source.v ** 2 + source.w ** 2).ravel()
    np.testing.assert_allclose(np.sort(lengths[lengths > 0]), [0.45, 0.45])
    np.testing.assert_array_equal(np.sort(np.ravel(source.scalars)[lengths > 0]), [1, 3])
    assert quiver.glyph.color_mode == "color_by_scalar"


@pytest.mark.parametrize("kind", ["density", "flux", "config"])
def test_animations_show_every_recorded_frame(kind, monkeypatch):
    model = make_model("nove_ib", steps=4)
    labels = run_frames(monkeypatch)
    figures = []
    build = getattr(model, f"_{kind}_figure")
    monkeypatch.setattr(model, f"_{kind}_figure", lambda *args, **kwargs: figures.append(build(*args, **kwargs))
                        or figures[-1])

    assert getattr(model, "animate_" + kind)() == "animator"

    assert labels == [f"t = {step}" for step in range(5)]
    source = figures[0][1].mlab_source
    last = model._channel_counts(model.nodes_t[-1], history=False)
    if kind == "density":
        np.testing.assert_array_equal(source.scalars, model.dens_t[-1])
    elif kind == "flux":
        np.testing.assert_allclose(source.u, model.calc_flux(last.astype(float))[..., 0])
    else:
        np.testing.assert_array_equal(np.ravel(source.scalars), np.ravel(last[..., : model.velocitychannels]))


def test_config_animation_without_rest_channels_runs_without_numerical_warnings(monkeypatch):
    model = get_lgca(geometry="cubic", dims=(3, 3, 3), restchannels=0, density=0.5, seed=1,
                     interaction="random_walk")
    model.timeevo(timesteps=2, record=True, showprogress=False)
    run_frames(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.animate_config()


def test_identity_based_live_density_animation_steps_the_simulation(monkeypatch):
    model = make_model("ib", steps=0)
    run_frames(monkeypatch, n_frames=3)
    figures = []
    build = model._density_figure
    monkeypatch.setattr(model, "_density_figure", lambda *args, **kwargs: figures.append(build(*args, **kwargs))
                        or figures[-1])
    reference = make_model("ib", steps=2)

    model.live_animate_density()

    np.testing.assert_array_equal(figures[0][1].mlab_source.scalars, reference.cell_density[reference.nonborder])


def open_scenes():
    from mayavi.core.registry import registry

    return sum(len(engine.scenes) for engine in registry.engines.values())


@pytest.mark.parametrize("kind", ["density", "flux", "config"])
def test_movie_has_one_frame_per_recorded_step_and_leaves_no_window(kind, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    model = make_model("nove", steps=4)
    mlab.close(all=True)
    mlab.options.offscreen = False  # saving must not depend on the global offscreen option

    path = getattr(model, "animate_" + kind)(save_path=tmp_path / f"{kind}.gif", size=(301, 241))

    assert mlab.options.offscreen is False
    assert open_scenes() == 0
    with Image.open(path) as movie:
        assert movie.n_frames == len(model.nodes_t)
        assert movie.size == (300, 240)  # cropped to even dimensions for video codecs


def test_mp4_movie_uses_ffmpeg(tmp_path):
    from matplotlib import animation

    if not animation.writers.is_available("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    model = make_model("classical", steps=2)

    path = model.animate_density(save_path=tmp_path / "density.mp4", save_kwargs={"fps": 4})

    assert path.stat().st_size > 0


def test_movie_resolution_is_set_by_figure_size_not_dpi(tmp_path):
    model = make_model("classical", steps=1)

    with pytest.raises(ValueError, match="size="):
        model.animate_density(save_path=tmp_path / "density.gif", save_kwargs={"dpi": 200})


def test_plotting_observers_write_3d_snapshots_and_movies(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from lgca.plotting import AnimationObserver, PlotSnapshotObserver
    from lgca.simulation import Schedule, SimulationRunner

    model = make_model("ib", steps=0)
    snapshots = PlotSnapshotObserver(kind="density_cubes", schedule=Schedule(steps={0, 3}),
                                     output_dir=tmp_path / "snapshots")
    movie = AnimationObserver(kind="flux", save_path=tmp_path / "movies" / "flux.gif")

    SimulationRunner(model, timesteps=3, observers=[snapshots, movie], showprogress=False).run()

    assert [path.name for path in snapshots.paths] == ["density_cubes_00000.png", "density_cubes_00003.png"]
    for path in snapshots.paths:
        with Image.open(path) as image:
            assert np.count_nonzero(np.any(np.asarray(image.convert("RGB")) < 250, axis=-1)) > 1000
    with Image.open(movie.animation) as frames:
        assert frames.n_frames == 4
    assert open_scenes() == 0
