import pytest

matplotlib = pytest.importorskip("matplotlib", reason="requires matplotlib for plotting tests")
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lgca import get_lgca
from lgca.plotting import AnimationObserver, PlotSnapshotObserver
from lgca.simulation import Schedule, SimulationRunner


def make_square_lgca(seed=1):
    return get_lgca(
        geometry="square",
        dims=(4, 4),
        density=0.5,
        restchannels=1,
        interaction="only_propagation",
        seed=seed,
    )


def test_plot_snapshot_observer_records_results_and_paths(tmp_path):
    lgca = make_square_lgca()
    observer = PlotSnapshotObserver(kind="density", output_dir=tmp_path, close=True, cbar=False)

    SimulationRunner(lgca, timesteps=1, observers=[observer], showprogress=False).run()

    assert [step for step, _ in observer.results] == [0, 1]
    assert len(observer.paths) == 2
    assert all(path.exists() for path in observer.paths)
    assert all(path.suffix == ".png" for path in observer.paths)
    plt.close("all")


def test_plot_snapshot_observer_uses_schedule():
    lgca = make_square_lgca(seed=2)
    observer = PlotSnapshotObserver(
        kind="density",
        schedule=Schedule(every=2),
        close=True,
        cbar=False,
    )

    SimulationRunner(lgca, timesteps=4, observers=[observer], showprogress=False).run()

    assert [step for step, _ in observer.results] == [0, 2, 4]
    plt.close("all")


def test_animation_observer_collects_runner_frames_and_builds_animation():
    lgca = make_square_lgca(seed=3)
    observer = AnimationObserver(kind="density", interval=10, cbar=False, close=True)

    SimulationRunner(lgca, timesteps=2, observers=[observer], showprogress=False).run()

    assert len(observer.frames) == 3
    assert observer.frames[0].shape == lgca.dims
    assert isinstance(observer.animation, FuncAnimation)
    # This test only verifies construction; it intentionally does not render or save.
    observer.animation._draw_was_started = True
    plt.close("all")
