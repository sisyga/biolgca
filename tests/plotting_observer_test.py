import pytest

matplotlib = pytest.importorskip("matplotlib", reason="requires matplotlib for plotting tests")
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from lgca import get_lgca
from lgca.plotting import AnimationObserver, PlotSnapshotObserver, animate, plot
from lgca.simulation import DensityRecorder, Schedule, SimulationRunner


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
    observer = PlotSnapshotObserver(
        kind="density", output_dir=tmp_path, close=True, retain_results=True, cbar=False
    )

    SimulationRunner(lgca, timesteps=1, observers=[observer], showprogress=False).run()

    assert [step for step, _ in observer.results] == [0, 1]
    assert len(observer.paths) == 2
    assert all(path.exists() for path in observer.paths)
    assert all(path.suffix == ".png" for path in observer.paths)
    plt.close("all")


def test_plot_snapshot_observer_works_without_cm_get_cmap(monkeypatch):
    import matplotlib.cm as cm

    lgca = make_square_lgca()
    observer = PlotSnapshotObserver(kind="density", close=True, retain_results=True, cbar=False)
    monkeypatch.delattr(cm, "get_cmap", raising=False)

    SimulationRunner(lgca, timesteps=1, observers=[observer], showprogress=False).run()

    assert [step for step, _ in observer.results] == [0, 1]
    plt.close("all")


def test_plot_snapshot_observer_uses_schedule():
    lgca = make_square_lgca(seed=2)
    observer = PlotSnapshotObserver(
        kind="density",
        schedule=Schedule(every=2),
        close=True,
        retain_results=True,
        cbar=False,
    )

    SimulationRunner(lgca, timesteps=4, observers=[observer], showprogress=False).run()

    assert [step for step, _ in observer.results] == [0, 2, 4]
    plt.close("all")


def test_animation_observer_collects_runner_frames_and_builds_animation():
    lgca = make_square_lgca(seed=3)
    observer = AnimationObserver(kind="density", interval=10, cbar=False, close=True)

    SimulationRunner(lgca, timesteps=2, observers=[observer], showprogress=False).run()

    assert observer.frames == []
    assert observer.frame_steps == [0, 1, 2]
    assert isinstance(observer.animation, FuncAnimation)
    # This test only verifies construction; it intentionally does not render or save.
    observer.animation._draw_was_started = True
    plt.close("all")


def test_sparse_recorded_history_is_rejected_by_implicit_animation():
    lgca = make_square_lgca(seed=4)
    recorder = DensityRecorder(schedule=Schedule(every=2))
    SimulationRunner(lgca, timesteps=4, observers=[recorder], showprogress=False).run()

    with pytest.raises(ValueError, match="sparse recorded history"):
        animate(lgca, kind="density", cbar=False)


def test_plot_snapshot_observer_reuse_resets_outputs(tmp_path):
    observer = PlotSnapshotObserver(
        kind="density", output_dir=tmp_path, retain_results=True, close=True, cbar=False
    )
    SimulationRunner(
        make_square_lgca(seed=5), timesteps=1, observers=[observer], showprogress=False
    ).run()
    SimulationRunner(
        make_square_lgca(seed=6), timesteps=0, observers=[observer], showprogress=False
    ).run()

    assert [step for step, _ in observer.results] == [0]
    assert len(observer.paths) == 1


def test_plot_snapshot_result_retention_is_opt_in(tmp_path):
    observer = PlotSnapshotObserver(kind="density", output_dir=tmp_path, cbar=False)

    SimulationRunner(
        make_square_lgca(seed=7), timesteps=2, observers=[observer], showprogress=False
    ).run()

    assert observer.results == []
    assert len(observer.paths) == 3


def test_animation_observer_reuse_resets_and_releases_frames():
    observer = AnimationObserver(kind="density", interval=10, cbar=False, close=True)
    SimulationRunner(
        make_square_lgca(seed=8), timesteps=2, observers=[observer], showprogress=False
    ).run()
    observer.animation._draw_was_started = True
    SimulationRunner(
        make_square_lgca(seed=9), timesteps=0, observers=[observer], showprogress=False
    ).run()

    assert observer.frames == []
    assert observer.frame_steps == [0]
    assert isinstance(observer.animation, FuncAnimation)
    observer.animation._draw_was_started = True
    plt.close("all")


def test_multispecies_density_plot_defaults_to_aggregate_and_selects_species():
    lgca = get_lgca(
        geometry="square", dims=(4, 4), density=0.5, n_species=2,
        interaction="only_propagation", seed=10,
    )

    fig, collection, _ = lgca.plot_density(cbar=False)
    assert collection.get_paths()
    fig2, collection2, _ = plot(lgca, kind="density", species=1, cbar=False)
    assert collection2.get_paths()
    with pytest.raises(ValueError, match="species"):
        plot(lgca, kind="density", species=2, cbar=False)
    plt.close(fig)
    plt.close(fig2)


def test_multispecies_zero_step_animation_observer_uses_aggregate_density():
    lgca = get_lgca(
        geometry="square", dims=(4, 4), density=0.5, n_species=2,
        interaction="only_propagation", seed=11,
    )
    observer = AnimationObserver(kind="density", interval=10, cbar=False, close=True)

    SimulationRunner(lgca, timesteps=0, observers=[observer], showprogress=False).run()

    assert observer.frame_steps == [0]
    assert isinstance(observer.animation, FuncAnimation)
    observer.animation._draw_was_started = True
    plt.close("all")


def test_plotting_observers_reject_unsupported_3d_backend_early():
    class CubicStub:
        geometry = "cubic"

    runner = type("Runner", (), {"timesteps": 0})()
    with pytest.raises(NotImplementedError, match="3-D"):
        PlotSnapshotObserver().setup(CubicStub(), runner)
    with pytest.raises(NotImplementedError, match="3-D"):
        AnimationObserver().setup(CubicStub(), runner)
