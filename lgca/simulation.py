"""Simulation runner and observers for LGCA time evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from .list_utils import _copy_arr_of_lists, get_arr_of_empty_lists


@dataclass(frozen=True)
class Schedule:
    """Select timesteps at which an observer should run."""

    every: int = 1
    steps: frozenset[int] | None = None

    def __init__(self, every: int = 1, steps: Iterable[int] | None = None):
        if every < 1:
            raise ValueError("every must be a positive integer.")
        object.__setattr__(self, "every", int(every))
        object.__setattr__(self, "steps", None if steps is None else frozenset(steps))

    def should_run(self, step: int) -> bool:
        if self.steps is not None:
            return step in self.steps
        return step % self.every == 0


class Observer:
    """Base class for objects that observe simulation state over time."""

    def __init__(self, schedule: Schedule | None = None):
        self.schedule = schedule or Schedule()

    def setup(self, lgca, runner: "SimulationRunner") -> None:
        pass

    def on_step(self, lgca, step: int) -> None:
        pass

    def finalize(self, lgca, runner: "SimulationRunner") -> None:
        pass


class SimulationRunner:
    """Run an LGCA simulation and notify observers separately from dynamics."""

    def __init__(self, lgca, timesteps: int = 100, observers=None, showprogress: bool = True):
        if timesteps < 0:
            raise ValueError("timesteps must be non-negative.")
        self.lgca = lgca
        self.timesteps = int(timesteps)
        self.observers = list(observers or [])
        self.showprogress = showprogress

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def run(self):
        lgca = self.lgca
        lgca.update_dynamic_fields()
        for observer in self.observers:
            setup = getattr(observer, "setup", None)
            if setup is not None:
                setup(lgca, self)

        self._notify_observers(0)
        for step in tqdm(range(1, self.timesteps + 1), disable=not self.showprogress):
            lgca.timestep()
            self._notify_observers(step)

        for observer in self.observers:
            finalize = getattr(observer, "finalize", None)
            if finalize is not None:
                finalize(lgca, self)
        return lgca

    def _notify_observers(self, step: int) -> None:
        for observer in self.observers:
            schedule = getattr(observer, "schedule", None)
            if schedule is None or schedule.should_run(step):
                observer.on_step(self.lgca, step)


class NodeRecorder(Observer):
    """Record full node configurations in ``lgca.nodes_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        shape = (runner.timesteps + 1,) + lgca.nodes[lgca.nonborder].shape
        if lgca.nodes.dtype == object:
            lgca.nodes_t = get_arr_of_empty_lists(shape)
        else:
            lgca.nodes_t = np.zeros(shape, dtype=lgca.nodes.dtype)

    def on_step(self, lgca, step: int) -> None:
        if lgca.nodes.dtype == object:
            lgca.nodes_t[step, ...] = _copy_arr_of_lists(lgca.nodes[lgca.nonborder])
        else:
            lgca.nodes_t[step, ...] = lgca.nodes[lgca.nonborder]


class PopulationRecorder(Observer):
    """Record total population size in ``lgca.n_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        lgca.n_t = np.zeros(runner.timesteps + 1, dtype=np.uint)

    def on_step(self, lgca, step: int) -> None:
        lgca.n_t[step] = lgca.cell_density[lgca.nonborder].sum()


class DensityRecorder(Observer):
    """Record node or species density in ``lgca.dens_t``."""

    def __init__(self, schedule: Schedule | None = None, dtype=None):
        super().__init__(schedule=schedule)
        self.dtype = dtype

    def setup(self, lgca, runner: SimulationRunner) -> None:
        value = self._density(lgca)
        if self.dtype is None:
            lgca.dens_t = np.zeros((runner.timesteps + 1,) + value.shape)
        else:
            lgca.dens_t = np.zeros((runner.timesteps + 1,) + value.shape, dtype=self.dtype)

    def on_step(self, lgca, step: int) -> None:
        lgca.dens_t[step, ...] = self._density(lgca)

    @staticmethod
    def _density(lgca):
        if hasattr(lgca, "species_density"):
            return lgca.species_density[lgca.nonborder]
        return lgca.cell_density[lgca.nonborder]


class ChannelDensityRecorder(Observer):
    """Record channel populations in ``lgca.channel_pop_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        value = lgca.channel_pop[lgca.nonborder]
        lgca.channel_pop_t = np.zeros((runner.timesteps + 1,) + value.shape, dtype=np.uint)

    def on_step(self, lgca, step: int) -> None:
        lgca.channel_pop_t[step, ...] = lgca.channel_pop[lgca.nonborder]


class PerTypeRecorder(Observer):
    """Record moving and resting particle counts."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        velocity, resting = self._counts(lgca)
        lgca.velcells_t = np.zeros((runner.timesteps + 1,) + velocity.shape)
        lgca.restcells_t = np.zeros((runner.timesteps + 1,) + resting.shape)

    def on_step(self, lgca, step: int) -> None:
        velocity, resting = self._counts(lgca)
        lgca.velcells_t[step, ...] = velocity
        lgca.restcells_t[step, ...] = resting

    @staticmethod
    def _counts(lgca):
        channel_pop = getattr(lgca, "channel_pop", lgca.nodes)
        nodes = channel_pop[lgca.nonborder]
        velocity = nodes[..., :lgca.velocitychannels].sum(-1)
        resting = nodes[..., lgca.velocitychannels:].sum(-1)
        return velocity, resting


class OrderParameterRecorder(Observer):
    """Record NoVE order parameters."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        length = runner.timesteps + 1
        lgca.ent_t = np.zeros(length, dtype=float)
        lgca.normEnt_t = np.zeros(length, dtype=float)
        lgca.polAlParam_t = np.zeros(length, dtype=float)
        lgca.meanAlign_t = np.zeros(length, dtype=float)

    def on_step(self, lgca, step: int) -> None:
        lgca.ent_t[step] = lgca.calc_entropy()
        lgca.normEnt_t[step] = lgca.calc_normalized_entropy()
        lgca.polAlParam_t[step] = lgca.calc_polar_alignment_parameter()
        lgca.meanAlign_t[step] = lgca.calc_mean_alignment()


class FamilyPopulationRecorder(Observer):
    """Record family populations in ``lgca.fam_pop_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        if "family" not in lgca.props:
            raise RuntimeError(
                "Interaction does not deal with families, family population can therefore not be recorded."
            )
        self.is_mutating = self._is_mutating_family_interaction(lgca)
        if self.is_mutating:
            lgca.fam_pop_t = []
        else:
            lgca.fam_pop_t = np.zeros((runner.timesteps + 1, lgca.maxfamily + 1))

    def on_step(self, lgca, step: int) -> None:
        if self.is_mutating:
            lgca.fam_pop_t.append(lgca.calc_family_pop_alive())
        else:
            try:
                lgca.fam_pop_t[step, ...] = lgca.calc_family_pop_alive()
            except ValueError as exc:
                raise ValueError(
                    "Number of families has increased, interaction must be included in the case "
                    "distinction for the recordfampop keyword in the IBLGCA base timeevo function!"
                ) from exc

    def finalize(self, lgca, runner: SimulationRunner) -> None:
        if self.is_mutating:
            lgca._straighten_family_populations()

    @staticmethod
    def _is_mutating_family_interaction(lgca) -> bool:
        mutating_interactions = []
        try:
            from lgca.ib_interactions import go_and_grow_mutations

            mutating_interactions.append(go_and_grow_mutations)
        except ImportError:
            pass
        try:
            from lgca.nove_ib_interactions import evo_steric, go_or_grow_glioblastoma

            mutating_interactions.extend([evo_steric, go_or_grow_glioblastoma])
        except ImportError:
            pass
        return lgca.interaction in mutating_interactions


def run_timeevo(lgca, timesteps: int, observers, showprogress: bool) -> None:
    """Compatibility helper for existing ``timeevo`` methods."""

    SimulationRunner(lgca, timesteps=timesteps, observers=observers, showprogress=showprogress).run()
