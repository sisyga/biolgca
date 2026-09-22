import numpy as np
import pytest

from lgca import get_lgca
from lgca.simulation import (
    CallbackObserver,
    DensityRecorder,
    NodeRecorder,
    PerTypeRecorder,
    PopulationRecorder,
    ScalarTimeSeriesRecorder,
    Schedule,
    SimulationRunner,
)


def test_per_type_recorder_counts_sparse_identity_labels():
    lgca = get_lgca(geometry="lin", ib=True, restchannels=1,
                    nodes=np.array([[1, 7, 2], [0, 0, 0]], dtype=np.uint),
                    interaction="only_propagation")
    SimulationRunner(lgca, timesteps=2, observers=[PerTypeRecorder(), PopulationRecorder()],
                     showprogress=False).run()
    np.testing.assert_array_equal(lgca.velcells_t[0], [2, 0])
    np.testing.assert_array_equal(lgca.restcells_t[0], [1, 0])
    np.testing.assert_array_equal((lgca.velcells_t + lgca.restcells_t).sum(axis=-1), lgca.n_t)


@pytest.mark.parametrize("same_schedule", [False, True])
@pytest.mark.parametrize("dtype", [None, np.float32])
def test_duplicate_recorders_fail_before_state_or_output_changes(same_schedule, dtype):
    from lgca.simulation import ChannelDensityRecorder, OrderParameterRecorder, FamilyPopulationRecorder

    lgca = get_lgca(geometry="lin", dims=3, density=1, seed=119, interaction="random_walk")
    before = lgca.nodes.copy()
    for kind in (NodeRecorder, DensityRecorder, PopulationRecorder, ChannelDensityRecorder,
                 PerTypeRecorder, OrderParameterRecorder, FamilyPopulationRecorder):
        first = kind(Schedule(every=2))
        second = kind(Schedule(every=2) if same_schedule else Schedule(steps=[0, 3]))
        if kind is DensityRecorder:
            second.dtype = dtype
        with pytest.raises(ValueError, match=f"Multiple {kind.__name__}"):
            SimulationRunner(lgca, timesteps=4, observers=[first, second], showprogress=False).run()
        np.testing.assert_array_equal(lgca.nodes, before)
        assert not hasattr(lgca, "nodes_t")


def test_recording_estimate_and_budget_precede_allocation():
    from types import SimpleNamespace
    from lgca.simulation import estimate_recording_bytes

    fake = SimpleNamespace(dims=(128, 128, 128), nodes=np.empty((0, 0, 0, 6), dtype=bool))
    assert estimate_recording_bytes(fake, 1000, [DensityRecorder()]) == 1001 * (128**3 * 8 + 8)
    assert estimate_recording_bytes(fake, 1000, [DensityRecorder(Schedule(every=100))]) == 11 * (128**3 * 8 + 8)
    lgca = get_lgca(geometry="lin", dims=3, density=1, seed=127, interaction="random_walk")
    before = lgca.nodes.copy()
    with pytest.raises(ValueError, match="Recording requires"):
        SimulationRunner(lgca, timesteps=3, observers=[DensityRecorder()],
                         max_recording_bytes=1, showprogress=False).run()
    assert not hasattr(lgca, "dens_t")
    np.testing.assert_array_equal(lgca.nodes, before)
    SimulationRunner(lgca, timesteps=3, observers=[DensityRecorder()],
                     max_recording_bytes=None, showprogress=False).run()
    assert lgca.dens_t.shape == (4, 3)


class StepCollector:
    def __init__(self, schedule=None):
        self.schedule = schedule
        self.steps = []

    def on_step(self, lgca, step):
        self.steps.append(step)


def test_runner_records_classical_outputs():
    lgca = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.5,
        restchannels=1,
        interaction="only_propagation",
        seed=1,
    )

    runner = SimulationRunner(
        lgca,
        timesteps=2,
        observers=[
            NodeRecorder(),
            PopulationRecorder(),
            DensityRecorder(),
        ],
        showprogress=False,
    )
    runner.run()

    assert lgca.nodes_t.shape == (3, 4, 5, lgca.K)
    assert lgca.n_t.shape == (3,)
    assert lgca.dens_t.shape == (3, 4, 5)
    np.testing.assert_array_equal(lgca.nodes_t[-1], lgca.nodes[lgca.nonborder])
    np.testing.assert_array_equal(lgca.dens_t[-1], lgca.cell_density[lgca.nonborder])
    assert lgca.n_t[-1] == lgca.cell_density[lgca.nonborder].sum()


def test_timeevo_uses_runner_compatible_recording():
    lgca = get_lgca(
        geometry="square",
        dims=(4, 5),
        density=0.5,
        restchannels=1,
        interaction="only_propagation",
        seed=2,
    )

    lgca.timeevo(
        timesteps=2,
        record=True,
        recordN=True,
        recorddens=True,
        recordpertype=True,
        showprogress=False,
    )

    assert lgca.nodes_t.shape == (3, 4, 5, lgca.K)
    assert lgca.n_t.shape == (3,)
    assert lgca.dens_t.shape == (3, 4, 5)
    assert lgca.velcells_t.shape == (3, 4, 5)
    assert lgca.restcells_t.shape == (3, 4, 5)
    np.testing.assert_array_equal(lgca.nodes_t[-1], lgca.nodes[lgca.nonborder])
    np.testing.assert_array_equal(lgca.dens_t[-1], lgca.cell_density[lgca.nonborder])


def test_observer_schedule_runs_at_selected_steps():
    lgca = get_lgca(
        geometry="1d",
        dims=(6,),
        density=0.5,
        interaction="only_propagation",
        seed=3,
    )
    collector = StepCollector(schedule=Schedule(every=2))

    SimulationRunner(lgca, timesteps=5, observers=[collector], showprogress=False).run()

    assert collector.steps == [0, 2, 4]


def test_callback_observer_calls_function_with_lgca_and_selected_step():
    """Removing callback invocation or scheduling must fail this test."""

    lgca = get_lgca(
        geometry="1d",
        dims=(6,),
        density=0.5,
        interaction="only_propagation",
        seed=30,
    )
    calls = []

    def collect(current_lgca, step):
        calls.append((current_lgca, step))

    observer = CallbackObserver(collect, schedule=Schedule(every=2))
    SimulationRunner(
        lgca,
        timesteps=3,
        observers=[observer],
        showprogress=False,
    ).run()

    assert calls == [(lgca, 0), (lgca, 2)]


def test_callback_observer_rejects_non_callable_during_construction():
    """A bad callback must fail before a simulation starts."""

    with pytest.raises(TypeError, match="callback must be callable"):
        CallbackObserver(None)


def test_runner_uses_pluggable_step_function_with_same_lifecycle():
    """A custom dynamics callback must share setup, scheduling, and finalization."""

    lgca = get_lgca(
        geometry="1d",
        dims=(4,),
        density=0.0,
        interaction="only_propagation",
        seed=31,
    )
    collector = StepCollector()
    called_steps = []

    def step_function(current_lgca, step, runner):
        called_steps.append(step)
        current_lgca.nodes[current_lgca.nonborder][0, 0] = True
        current_lgca.update_dynamic_fields()

    runner = SimulationRunner(
        lgca,
        timesteps=2,
        observers=[collector],
        showprogress=False,
        step_function=step_function,
    )

    assert runner.run() is lgca
    assert called_steps == [1, 2]
    assert collector.steps == [0, 1, 2]
    assert runner.elapsed_seconds >= 0.0


@pytest.mark.parametrize("every", [True, 1.5, 0, -1])
def test_schedule_rejects_invalid_every_without_coercion(every):
    with pytest.raises(ValueError, match="every"):
        Schedule(every=every)


@pytest.mark.parametrize("steps", [[True], [-1], [1.5]])
def test_schedule_rejects_invalid_explicit_steps(steps):
    with pytest.raises(ValueError, match="steps"):
        Schedule(steps=steps)


def test_sparse_array_recorders_store_only_samples_and_explicit_steps():
    lgca = get_lgca(
        geometry="square",
        dims=(3, 4),
        density=0.5,
        restchannels=1,
        interaction="only_propagation",
        seed=4,
    )
    schedule = Schedule(every=2)

    SimulationRunner(
        lgca,
        timesteps=4,
        observers=[
            NodeRecorder(schedule),
            DensityRecorder(schedule),
            PopulationRecorder(schedule),
            PerTypeRecorder(schedule),
        ],
        showprogress=False,
    ).run()

    expected_steps = np.array([0, 2, 4])
    for name in ("nodes_steps", "dens_steps", "n_steps", "velcells_steps", "restcells_steps"):
        np.testing.assert_array_equal(getattr(lgca, name), expected_steps)
    assert lgca.nodes_t.shape[0] == 3
    assert lgca.dens_t.shape[0] == 3
    assert lgca.n_t.shape == (3,)
    assert lgca.velcells_t.shape[0] == 3
    assert lgca.restcells_t.shape[0] == 3
    assert np.all(lgca.n_t > 0)


def test_default_dense_recorders_publish_dense_step_metadata():
    lgca = get_lgca(
        geometry="1d", dims=(4,), density=0.5, interaction="only_propagation", seed=5
    )
    SimulationRunner(
        lgca,
        timesteps=2,
        observers=[NodeRecorder(), DensityRecorder(), PopulationRecorder()],
        showprogress=False,
    ).run()

    for name in ("nodes_steps", "dens_steps", "n_steps"):
        np.testing.assert_array_equal(getattr(lgca, name), [0, 1, 2])


def test_scalar_recorder_can_be_reused_without_retaining_previous_run(tmp_path):
    recorder = ScalarTimeSeriesRecorder(output_path=tmp_path / "population.csv")
    lgca = get_lgca(
        geometry="1d", dims=(4,), density=0.5, interaction="only_propagation", seed=6
    )

    SimulationRunner(lgca, timesteps=2, observers=[recorder], showprogress=False).run()
    SimulationRunner(lgca, timesteps=1, observers=[recorder], showprogress=False).run()

    assert [record["step"] for record in recorder.records] == [0, 1]
