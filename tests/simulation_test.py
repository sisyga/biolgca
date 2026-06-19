import numpy as np

from lgca import get_lgca
from lgca.simulation import (
    DensityRecorder,
    NodeRecorder,
    PopulationRecorder,
    Schedule,
    SimulationRunner,
)


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
