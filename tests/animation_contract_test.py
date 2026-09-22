"""Public animation entry points share data, channels and time semantics."""

from types import SimpleNamespace
import importlib

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest

from lgca import get_lgca
from lgca.plotting import animate


@pytest.mark.parametrize("geometry", ["square", "hex"])
@pytest.mark.parametrize("ib,ve", [(False, True), (True, True), (False, False), (True, False)])
@pytest.mark.parametrize("kind", ["config", "flux", "flow", "density"])
@pytest.mark.parametrize("times", [[0, 1, 2], [0, 3, 6], [0, 1, 6]])
@pytest.mark.parametrize("entry", ["direct", "facade"])
def test_direct_and_facade_renderer_times(geometry, ib, ve, kind, times, entry):
    model = get_lgca(geometry=geometry, dims=(2, 2), ib=ib, ve=ve, restchannels=1,
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
def test_mayavi_adapter_passes_resolved_times_without_matplotlib_internals(kind, entry, monkeypatch):
    titles = []

    def animation_decorator(**kwargs):
        def decorate(function):
            return lambda: list(function())
        return decorate

    fake = SimpleNamespace(animate=animation_decorator,
                           title=lambda title, **kwargs: titles.append(title), show=lambda: None)
    module = importlib.import_module("lgca.lgca_cubic")
    monkeypatch.setattr(module, "mlab", fake)
    model = get_lgca(geometry="cubic", dims=(2, 2, 2), restchannels=1,
                     density=2, interaction="only_propagation")
    model.nodes_t = np.stack([model.nodes[model.nonborder]] * 3)
    model.dens_t = np.stack([model.cell_density[model.nonborder]] * 3)
    model.nodes_steps = model.dens_steps = np.array([0, 3, 20])
    artist = SimpleNamespace(mlab_source=SimpleNamespace(set=lambda **kwargs: None))
    output = (None, artist) if kind == "density" else (None, artist, artist)
    monkeypatch.setattr(model, "plot_" + kind, lambda **kwargs: output)
    if entry == "facade":
        assert animate(model, kind=kind) is None
    else:
        assert getattr(model, "animate_" + kind)() is None
    prefix = "Flux at Time " if kind == "flux" else "Time "
    assert titles == [prefix + str(step) for step in (0, 0, 3, 20)]


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
