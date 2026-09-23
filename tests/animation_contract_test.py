"""Public animation entry points share data, channels and time semantics."""

from contextlib import nullcontext
from types import SimpleNamespace
import importlib

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest

from lgca import get_lgca
from lgca.plotting import AnimationObserver, animate


@pytest.mark.parametrize("ib,ve", [(False, True), (True, True), (False, False), (True, False)])
@pytest.mark.parametrize("kind", ["config", "flux", "flow", "density"])
@pytest.mark.parametrize("entry", ["direct", "facade"])
def test_direct_and_facade_renderer_times(ib, ve, kind, entry):
    times = [0, 1, 6]  # non-uniform sample times must appear in the frame titles
    model = get_lgca(geometry="square", dims=(2, 2), ib=ib, ve=ve, restchannels=1,
                     density=2, seed=113, interaction="only_propagation")
    model.nodes_t = np.stack([model.nodes[model.nonborder]] * 3)
    model.dens_t = np.stack([model.cell_density[model.nonborder]] * 3)
    model.nodes_steps = model.dens_steps = np.array(times)
    animation = (animate(model, kind=kind) if entry == "facade"
                 else getattr(model, "animate_" + kind)())
    for row, time in enumerate(times):
        animation._func(row)
        assert animation._fig.axes[0].get_title() == f"Time $k =${time}"
    animation._draw_was_started = True
    plt.close(animation._fig)


@pytest.mark.parametrize("kind", ["config", "flux", "flow", "density"])
@pytest.mark.parametrize("times", [None, [2, 5, 9]])
def test_explicit_arrays_use_only_explicit_times(kind, times):
    model = get_lgca(geometry="square", dims=(2, 2), density=2, seed=148,
                     interaction="only_propagation")
    model.nodes_steps = model.dens_steps = np.array([100, 200, 300])
    data = model.cell_density[model.nonborder] if kind == "density" else model.nodes[model.nonborder]
    data = np.stack([data] * 3)
    animation = animate(model, kind=kind, data=data, steps=times)
    animation._func(2)
    assert animation._fig.axes[0].get_title() == f"Time $k =${2 if times is None else 9}"
    animation._draw_was_started = True
    plt.close(animation._fig)


@pytest.mark.parametrize("kind", ["config", "density", "flux"])
@pytest.mark.parametrize("entry", ["direct", "facade"])
def test_mayavi_animations_label_frames_with_resolved_times(kind, entry, monkeypatch):
    # Runs without Mayavi: the figure builders and the event loop are replaced.
    played = {}

    def play(fig, update, n_frames, label, interval, show, **kwargs):
        played["labels"] = [label(frame) for frame in range(n_frames)]
        for frame in range(n_frames):
            update(frame)
        return "animator"

    module = importlib.import_module("lgca.lgca_cubic")
    monkeypatch.setattr(module, "play", play)
    model = get_lgca(geometry="cubic", dims=(2, 2, 2), restchannels=1,
                     density=2, interaction="only_propagation")
    model.nodes_t = np.stack([model.nodes[model.nonborder]] * 3)
    model.dens_t = np.stack([model.cell_density[model.nonborder]] * 3)
    model.nodes_steps = model.dens_steps = np.array([0, 3, 20])
    updates = []
    artist = SimpleNamespace(mlab_source=SimpleNamespace(set=lambda **kwargs: updates.append(kwargs)))
    output = (None, artist) if kind == "density" else (None, artist, artist)
    monkeypatch.setattr(model, f"_{kind}_figure", lambda *args, **kwargs: output)
    result = animate(model, kind=kind) if entry == "facade" else getattr(model, "animate_" + kind)()
    assert result == "animator"
    assert played["labels"] == ["t = 0", "t = 3", "t = 20"]
    assert updates


@pytest.mark.parametrize("save_path", [None, "movie.mp4"])
def test_3d_animation_observer_never_blocks_and_forwards_movie_options(save_path, tmp_path, monkeypatch):
    # Runs without Mayavi: the figure builder and the event loop are replaced.
    from lgca.simulation import Schedule, SimulationRunner

    calls = []

    def play(fig, update, n_frames, label, interval, show, **kwargs):
        calls.append(dict(show=show, labels=[label(frame) for frame in range(n_frames)], **kwargs))
        return "result"

    module = importlib.import_module("lgca.lgca_cubic")
    monkeypatch.setattr(module, "play", play)
    monkeypatch.setattr(module, "offscreen", lambda enabled=True: nullcontext())
    model = get_lgca(geometry="cubic", dims=(2, 2, 2), density=1, interaction="random_walk", seed=1)
    artist = SimpleNamespace(mlab_source=SimpleNamespace(set=lambda **kwargs: None))
    monkeypatch.setattr(model, "_density_figure", lambda *args, **kwargs: (None, artist))
    path = None if save_path is None else tmp_path / "out" / save_path
    observer = AnimationObserver(kind="density", schedule=Schedule(steps=[0, 2, 3]), save_path=path,
                                 save_kwargs={"fps": 4})

    SimulationRunner(model, timesteps=3, observers=[observer], showprogress=False).run()

    (call,) = calls
    assert call["show"] is False
    assert call["labels"] == ["t = 0", "t = 2", "t = 3"]
    assert call["save_path"] == path
    assert call["save_kwargs"] == ({"fps": 4} if path else None)
    assert observer.animation == "result"
    if path is not None:
        assert path.parent.is_dir()


def test_animation_observer_captures_selected_channels():
    from lgca.plotting import AnimationObserver
    from lgca.simulation import SimulationRunner, Schedule

    nodes = np.zeros((2, 2, 5), dtype=bool)
    nodes[..., 4] = True
    model = get_lgca(geometry="square", nodes=nodes, interaction="only_propagation")
    observer = AnimationObserver(channels=slice(0, 4), schedule=Schedule(steps=[0, 2]), cbar=False)
    SimulationRunner(model, timesteps=2, observers=[observer], showprogress=False).run()
    observer.animation._func(1)
    axis = observer.animation._fig.axes[0]
    np.testing.assert_array_equal(axis.images[0].get_array(), 0)
    assert axis.get_title() == "Time $k =$2"
    observer.animation._draw_was_started = True
    plt.close(observer.animation._fig)
