"""Simulation runner and observers for LGCA time evolution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm.auto import tqdm

from .list_utils import _copy_arr_of_lists, get_arr_of_empty_lists


__all__ = [
    "CSVSnapshotObserver",
    "ChannelDensityRecorder",
    "DensityRecorder",
    "FamilyPopulationRecorder",
    "NodeRecorder",
    "Observer",
    "OrderParameterRecorder",
    "PerTypeRecorder",
    "PopulationRecorder",
    "ScalarTimeSeriesRecorder",
    "Schedule",
    "SimulationRunner",
    "run_timeevo",
]


@dataclass(frozen=True)
class Schedule:
    """Select timesteps at which an observer should run."""

    every: int = 1
    steps: frozenset[int] | None = None

    def __init__(self, every: int = 1, steps: Iterable[int] | None = None):
        if isinstance(every, bool) or int(every) != every or every < 1:
            raise ValueError("every must be a positive integer.")
        if steps is not None:
            steps = tuple(steps)
            if any(
                isinstance(step, bool) or int(step) != step or step < 0
                for step in steps
            ):
                raise ValueError("steps must contain only non-negative integers.")
        object.__setattr__(self, "every", int(every))
        object.__setattr__(
            self, "steps", None if steps is None else frozenset(int(step) for step in steps)
        )

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


def _setup_sample_indices(observer, lgca, runner, step_attribute: str) -> int:
    """Publish sampled steps and map absolute simulation steps to compact rows."""
    steps = np.fromiter(
        (
            step
            for step in range(runner.timesteps + 1)
            if observer.schedule.should_run(step)
        ),
        dtype=int,
    )
    setattr(lgca, step_attribute, steps)
    observer._sample_indices = {int(step): index for index, step in enumerate(steps)}
    return len(steps)


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
        length = _setup_sample_indices(self, lgca, runner, "nodes_steps")
        shape = (length,) + lgca.nodes[lgca.nonborder].shape
        if lgca.nodes.dtype == object:
            lgca.nodes_t = get_arr_of_empty_lists(shape)
        else:
            lgca.nodes_t = np.zeros(shape, dtype=lgca.nodes.dtype)

    def on_step(self, lgca, step: int) -> None:
        index = self._sample_indices[step]
        if lgca.nodes.dtype == object:
            lgca.nodes_t[index, ...] = _copy_arr_of_lists(lgca.nodes[lgca.nonborder])
        else:
            lgca.nodes_t[index, ...] = lgca.nodes[lgca.nonborder]


class PopulationRecorder(Observer):
    """Record total population size in ``lgca.n_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        length = _setup_sample_indices(self, lgca, runner, "n_steps")
        lgca.n_t = np.zeros(length, dtype=np.uint)

    def on_step(self, lgca, step: int) -> None:
        lgca.n_t[self._sample_indices[step]] = lgca.cell_density[lgca.nonborder].sum()


class DensityRecorder(Observer):
    """Record node or species density in ``lgca.dens_t``."""

    def __init__(self, schedule: Schedule | None = None, dtype=None):
        super().__init__(schedule=schedule)
        self.dtype = dtype

    def setup(self, lgca, runner: SimulationRunner) -> None:
        value = self._density(lgca)
        length = _setup_sample_indices(self, lgca, runner, "dens_steps")
        if self.dtype is None:
            lgca.dens_t = np.zeros((length,) + value.shape)
        else:
            lgca.dens_t = np.zeros((length,) + value.shape, dtype=self.dtype)

    def on_step(self, lgca, step: int) -> None:
        lgca.dens_t[self._sample_indices[step], ...] = self._density(lgca)

    @staticmethod
    def _density(lgca):
        if hasattr(lgca, "species_density"):
            return lgca.species_density[lgca.nonborder]
        return lgca.cell_density[lgca.nonborder]


class ChannelDensityRecorder(Observer):
    """Record channel populations in ``lgca.channel_pop_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        value = lgca.channel_pop[lgca.nonborder]
        length = _setup_sample_indices(self, lgca, runner, "channel_pop_steps")
        lgca.channel_pop_t = np.zeros((length,) + value.shape, dtype=np.uint)

    def on_step(self, lgca, step: int) -> None:
        lgca.channel_pop_t[self._sample_indices[step], ...] = lgca.channel_pop[lgca.nonborder]


class PerTypeRecorder(Observer):
    """Record moving and resting particle counts."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        velocity, resting = self._counts(lgca)
        length = _setup_sample_indices(self, lgca, runner, "velcells_steps")
        lgca.restcells_steps = lgca.velcells_steps.copy()
        lgca.velcells_t = np.zeros((length,) + velocity.shape)
        lgca.restcells_t = np.zeros((length,) + resting.shape)

    def on_step(self, lgca, step: int) -> None:
        velocity, resting = self._counts(lgca)
        index = self._sample_indices[step]
        lgca.velcells_t[index, ...] = velocity
        lgca.restcells_t[index, ...] = resting

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
        length = _setup_sample_indices(self, lgca, runner, "order_parameter_steps")
        lgca.ent_t = np.zeros(length, dtype=float)
        lgca.normEnt_t = np.zeros(length, dtype=float)
        lgca.polAlParam_t = np.zeros(length, dtype=float)
        lgca.meanAlign_t = np.zeros(length, dtype=float)

    def on_step(self, lgca, step: int) -> None:
        index = self._sample_indices[step]
        lgca.ent_t[index] = lgca.calc_entropy()
        lgca.normEnt_t[index] = lgca.calc_normalized_entropy()
        lgca.polAlParam_t[index] = lgca.calc_polar_alignment_parameter()
        lgca.meanAlign_t[index] = lgca.calc_mean_alignment()


class FamilyPopulationRecorder(Observer):
    """Record family populations in ``lgca.fam_pop_t``."""

    def setup(self, lgca, runner: SimulationRunner) -> None:
        if "family" not in lgca.props:
            raise RuntimeError(
                "Interaction does not deal with families, family population can therefore not be recorded."
            )
        length = _setup_sample_indices(self, lgca, runner, "fam_pop_steps")
        self.is_mutating = self._is_mutating_family_interaction(lgca, runner)
        if self.is_mutating:
            lgca.fam_pop_t = []
        else:
            lgca.fam_pop_t = np.zeros((length, lgca.maxfamily + 1))

    def on_step(self, lgca, step: int) -> None:
        if self.is_mutating:
            lgca.fam_pop_t.append(lgca.calc_family_pop_alive())
        else:
            try:
                lgca.fam_pop_t[self._sample_indices[step], ...] = lgca.calc_family_pop_alive()
            except ValueError as exc:
                raise ValueError(
                    "Number of families has increased, interaction must be included in the case "
                    "distinction for the recordfampop keyword in the IBLGCA base timeevo function!"
                ) from exc

    def finalize(self, lgca, runner: SimulationRunner) -> None:
        if self.is_mutating:
            lgca._straighten_family_populations()

    @staticmethod
    def _is_mutating_family_interaction(lgca, runner) -> bool:
        compiled = getattr(runner, "compiled", None)
        pipeline = getattr(compiled, "pipeline", None)
        if pipeline is not None:
            return any(operator.info.mutates_families for operator in pipeline.operators)
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


class CSVSnapshotObserver(Observer):
    """Write lattice snapshots to one CSV file per observed timestep."""

    def __init__(
        self,
        kind: str = "density",
        schedule: Schedule | None = None,
        output_dir=None,
        filename: str = "{kind}_{step:05d}.csv",
    ):
        super().__init__(schedule=schedule)
        self.kind = kind
        self.output_dir = Path("." if output_dir is None else output_dir)
        self.filename = filename
        self.paths: list[Path] = []

    def setup(self, lgca, runner: SimulationRunner) -> None:
        self.paths = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, lgca, step: int) -> None:
        values = _snapshot_values(lgca, self.kind)
        path = self.output_dir / self.filename.format(kind=self.kind, step=step)
        _write_array_csv(path, values)
        self.paths.append(path)


class ScalarTimeSeriesRecorder(Observer):
    """Record scalar metrics over time and write them to CSV."""

    def __init__(
        self,
        metrics: Mapping[str, Any] | None = None,
        schedule: Schedule | None = None,
        output_path=None,
    ):
        super().__init__(schedule=schedule)
        self.metrics = dict(metrics or {"population": _total_population})
        self.output_path = Path("time_series.csv" if output_path is None else output_path)
        self.records: list[dict[str, Any]] = []

    def setup(self, lgca, runner: SimulationRunner) -> None:
        self.records = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step(self, lgca, step: int) -> None:
        row = {"step": step}
        for name, metric in self.metrics.items():
            row[name] = _scalar_value(metric(lgca))
        self.records.append(row)

    def finalize(self, lgca, runner: SimulationRunner) -> None:
        if not self.records:
            return
        with self.output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)


def _snapshot_values(lgca, kind: str):
    key = kind.replace("_", "").lower()
    if key == "density":
        if hasattr(lgca, "species_density"):
            return lgca.species_density[lgca.nonborder]
        return lgca.cell_density[lgca.nonborder]
    if key in {"nodes", "config", "configuration"}:
        return lgca.nodes[lgca.nonborder]
    if key in {"channelpop", "channelpopulation"}:
        return lgca.channel_pop[lgca.nonborder]
    raise ValueError("Unknown CSV snapshot kind {!r}.".format(kind))


def _write_array_csv(path: Path, values) -> None:
    array = np.asarray(values)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flat_index", "value"])
        for index, value in enumerate(array.ravel()):
            writer.writerow([index, _scalar_value(value)])


def _total_population(lgca):
    if hasattr(lgca, "species_density"):
        return lgca.species_density[lgca.nonborder].sum()
    return lgca.cell_density[lgca.nonborder].sum()


def _scalar_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value
