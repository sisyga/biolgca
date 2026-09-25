"""Simulation runner and observers for LGCA time evolution."""

from __future__ import annotations

import csv
import time
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm.auto import tqdm

from .list_utils import _copy_arr_of_lists, get_arr_of_empty_lists


DEFAULT_RECORDING_LIMIT_BYTES = 512 * 1024 ** 2


def estimate_recording_bytes(lgca, timesteps, observers):
    """Estimate fixed recorder buffers in bytes before allocation.

    Object-state payloads and dynamically growing families are additional to
    this estimate. Sparse schedules count only their selected sample times.
    """
    total = 0
    spatial = math.prod(lgca.dims)
    channels = math.prod(lgca.nodes.shape[len(lgca.dims):])
    species = getattr(lgca, "n_species", 1)
    for observer in observers:
        schedule = getattr(observer, "schedule", None) or Schedule()
        samples = (timesteps // schedule.every + 1 if schedule.steps is None
                   else sum(step <= timesteps for step in schedule.steps))
        if isinstance(observer, NodeRecorder):
            per_frame = spatial * channels * lgca.nodes.dtype.itemsize
        elif isinstance(observer, DensityRecorder):
            per_frame = spatial * species * observer.resolve_dtype(lgca).itemsize
        elif isinstance(observer, PopulationRecorder):
            per_frame = np.dtype(np.uint).itemsize
        elif isinstance(observer, ChannelDensityRecorder):
            per_frame = spatial * channels * np.dtype(np.uint).itemsize
        elif isinstance(observer, PerTypeRecorder):
            per_frame = spatial * species * 2 * np.dtype(float).itemsize
        elif isinstance(observer, OrderParameterRecorder):
            per_frame = 4 * np.dtype(float).itemsize
        elif isinstance(observer, FamilyPopulationRecorder):
            per_frame = (int(getattr(lgca, "maxfamily", 0)) + 1) * np.dtype(float).itemsize
        else:
            continue
        total += samples * (per_frame + np.dtype(int).itemsize)
    return total


__all__ = [
    "CallbackObserver",
    "CSVSnapshotObserver",
    "ChannelDensityRecorder",
    "DensityRecorder",
    "FamilyPopulationRecorder",
    "NodeRecorder",
    "Observer",
    "OrderParameterRecorder",
    "PerTypeRecorder",
    "PopulationRecorder",
    "RECORDED",
    "RunData",
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


class CallbackObserver(Observer):
    """Call a Python function with ``(lgca, step)`` at scheduled steps."""

    def __init__(self, callback, schedule: Schedule | None = None):
        if not callable(callback):
            raise TypeError("callback must be callable")
        super().__init__(schedule=schedule)
        self.callback = callback

    def on_step(self, lgca, step: int) -> None:
        self.callback(lgca, step)


def _setup_sample_indices(observer, lgca, runner, step_attribute: str) -> int:
    """Publish local run steps and map them to compact sample rows."""
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

    def __init__(
        self,
        lgca,
        timesteps: int = 100,
        observers=None,
        showprogress: bool = True,
        step_function=None,
        context=None,
        max_recording_bytes=DEFAULT_RECORDING_LIMIT_BYTES,
    ):
        if timesteps < 0:
            raise ValueError("timesteps must be non-negative.")
        self.lgca = lgca
        self.timesteps = int(timesteps)
        self.observers = list(observers or [])
        self.showprogress = showprogress
        self.step_function = step_function
        self.context = context if context is not None else getattr(lgca, "_compiled_model", None)
        self.max_recording_bytes = max_recording_bytes
        self.elapsed_seconds = 0.0

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def run(self):
        self.start_step = int(getattr(self.context, "_step", 0))
        self.end_step = self.start_step + self.timesteps
        recorder_types = (NodeRecorder, DensityRecorder, PopulationRecorder,
                          ChannelDensityRecorder, PerTypeRecorder, OrderParameterRecorder,
                          FamilyPopulationRecorder)
        seen = set()
        for observer in self.observers:
            kind = next((kind for kind in recorder_types if isinstance(observer, kind)), None)
            if kind is not None:
                if kind in seen:
                    raise ValueError(f"Multiple {kind.__name__} instances share LGCA output arrays; "
                                     "use one recorder per type and select samples afterward")
                seen.add(kind)
        self.estimated_recording_bytes = estimate_recording_bytes(self.lgca, self.timesteps, self.observers)
        if (self.max_recording_bytes is not None
                and self.estimated_recording_bytes > self.max_recording_bytes):
            raise ValueError(f"Recording requires at least {self.estimated_recording_bytes:,} bytes; "
                             f"limit is {self.max_recording_bytes:,}. Use sparse schedules, "
                             "a smaller dtype, CSV streaming, or explicitly increase max_recording_bytes.")
        start = time.perf_counter()
        lgca = self.lgca
        lgca.recording_start_step = self.start_step
        lgca.recording_end_step = self.end_step
        lgca.update_dynamic_fields()
        for observer in self.observers:
            setup = getattr(observer, "setup", None)
            if setup is not None:
                setup(lgca, self)

        self._notify_observers(0)
        for step in tqdm(range(1, self.timesteps + 1), disable=not self.showprogress):
            if self.step_function is None:
                lgca.timestep()
            else:
                self.step_function(lgca, step, self)
            self._notify_observers(step)

        for observer in self.observers:
            finalize = getattr(observer, "finalize", None)
            if finalize is not None:
                finalize(lgca, self)
        self.elapsed_seconds = time.perf_counter() - start
        return lgca

    def _notify_observers(self, step: int) -> None:
        for observer in self.observers:
            schedule = getattr(observer, "schedule", None)
            if schedule is None or schedule.should_run(step):
                observer.on_step(self.lgca, step)


class NodeRecorder(Observer):
    """Record full node configurations in ``lgca.nodes_t``.

    Identity-based models without volume exclusion (periodic, reflecting or
    absorbing boundaries) record compact cell tables in ``lgca.cells_t``, a
    :class:`~lgca.cells.CellHistory` with the label, node and channel of every
    cell at every recorded time; ``lgca.nodes_t`` builds the lists of labels
    from them when it is first read.
    """

    def setup(self, lgca, runner: SimulationRunner) -> None:
        length = _setup_sample_indices(self, lgca, runner, "nodes_steps")
        start = getattr(lgca, "_start_cell_history", None)
        self._cells = start is not None and start(length)
        if self._cells:
            return
        shape = (length,) + lgca.nodes[lgca.nonborder].shape
        if lgca.nodes.dtype == object:
            lgca.nodes_t = get_arr_of_empty_lists(shape)
        else:
            lgca.nodes_t = np.zeros(shape, dtype=lgca.nodes.dtype)

    def on_step(self, lgca, step: int) -> None:
        index = self._sample_indices[step]
        if self._cells:
            lgca._record_cells(index)
        elif lgca.nodes.dtype == object:
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
    """Record node or species density in ``lgca.dens_t``.

    By default densities are stored as signed integers: ``int16`` for models
    with volume exclusion, whose nodes hold at most ``K`` particles per species,
    and ``int32`` without volume exclusion, widened to ``int64`` if a node ever
    exceeds that range. Pass ``dtype=float`` to record floating-point values.
    """

    def __init__(self, schedule: Schedule | None = None, dtype=None):
        super().__init__(schedule=schedule)
        self.dtype = dtype

    def resolve_dtype(self, lgca) -> np.dtype:
        """Return the storage type used for ``lgca``."""
        if self.dtype is not None:
            return np.dtype(self.dtype)
        if _has_unbounded_counts(lgca):
            return np.dtype(np.int32)
        return np.dtype(np.int16 if lgca.K <= np.iinfo(np.int16).max else np.int32)

    def setup(self, lgca, runner: SimulationRunner) -> None:
        value = self._density(lgca)
        length = _setup_sample_indices(self, lgca, runner, "dens_steps")
        dtype = self.resolve_dtype(lgca)
        self._widen_on_overflow = self.dtype is None and _has_unbounded_counts(lgca)
        lgca.dens_t = np.zeros((length,) + value.shape, dtype=dtype)

    def on_step(self, lgca, step: int) -> None:
        value = self._density(lgca)
        if (self._widen_on_overflow and value.size
                and value.max() > np.iinfo(lgca.dens_t.dtype).max):
            lgca.dens_t = lgca.dens_t.astype(np.int64)
        lgca.dens_t[self._sample_indices[step], ...] = value

    @staticmethod
    def _density(lgca):
        if hasattr(lgca, "species_density"):
            return lgca.species_density[lgca.nonborder]
        return lgca.cell_density[lgca.nonborder]


def _has_unbounded_counts(lgca) -> bool:
    """Return whether nodes may hold arbitrarily many particles (no volume exclusion)."""
    from .nove_base import NoVE_LGCA_base

    return isinstance(lgca, NoVE_LGCA_base)


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
        channel_pop = getattr(lgca, "channel_pop", getattr(lgca, "occupied", lgca.nodes))
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
        """Whether new families can appear: an operator, or one stacked in it, founds families.

        A function passed as the interaction may do anything, so it counts as founding them.
        """
        compiled = getattr(runner, "context", None)
        pipeline = getattr(compiled, "pipeline", None)
        if pipeline is None:
            return True
        operators = list(pipeline.operators)
        while operators:
            operator = operators.pop()
            if operator.info.mutates_families:
                return True
            operators.extend(getattr(operator, "operators", ()))
        return False


TIMEEVO_HINT = "Record it with lgca.timeevo(..., record=True, recordN=True, recordpertype=True)"


def run_timeevo(lgca, timesteps: int, observers, showprogress: bool) -> None:
    """Compatibility helper for existing ``timeevo`` methods; the recordings become ``lgca.data``."""

    SimulationRunner(lgca, timesteps=timesteps, observers=observers, showprogress=showprogress).run()
    lgca._run_data = RunData.from_run(lgca, observers, hint=TIMEEVO_HINT)


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
        self._output_root = None

    def setup(self, lgca, runner: SimulationRunner) -> None:
        self.paths = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, lgca, step: int) -> None:
        values = _snapshot_values(lgca, self.kind)
        filename = self.filename.format(kind=self.kind, step=step)
        path = (self.output_dir / filename).resolve()
        if self._output_root is not None:
            windows = PureWindowsPath(filename)
            if (Path(filename).anchor or windows.anchor or ".." in Path(filename).parts
                    or ".." in windows.parts or not path.is_relative_to(self._output_root)):
                raise ValueError("Snapshot output path escapes the run directory")
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
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
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


# name in ``result.data`` -> (attribute of the model, attribute of its steps, recorder, meaning)
RECORDED = {
    "population": ("n_t", "n_steps", "PopulationRecorder", "number of cells"),
    "density": ("dens_t", "dens_steps", "DensityRecorder", "cells per node, or per node and species"),
    "nodes": ("nodes_t", "nodes_steps", "NodeRecorder", "channel states of all nodes"),
    "channel_population": ("channel_pop_t", "channel_pop_steps", "ChannelDensityRecorder", "cells per channel"),
    "moving": ("velcells_t", "velcells_steps", "PerTypeRecorder", "cells in velocity channels per node"),
    "resting": ("restcells_t", "restcells_steps", "PerTypeRecorder", "cells in rest channels per node"),
    "family_population": ("fam_pop_t", "fam_pop_steps", "FamilyPopulationRecorder", "cells per family"),
    "entropy": ("ent_t", "order_parameter_steps", "OrderParameterRecorder", "entropy"),
    "normalized_entropy": ("normEnt_t", "order_parameter_steps", "OrderParameterRecorder", "normalized entropy"),
    "polar_alignment": ("polAlParam_t", "order_parameter_steps", "OrderParameterRecorder", "polar alignment"),
    "mean_alignment": ("meanAlign_t", "order_parameter_steps", "OrderParameterRecorder", "mean alignment"),
}
_DATA_ALIASES = {"n": "population"}


class RunData(Mapping):
    """The data recorded in a run, by name, with the steps at which it was recorded.

    ``data["population"]`` (alias ``data["n"]``) is an array with one entry per
    recorded step, ``data.steps("population")`` the steps (counted from the
    start of the run). The names are those of :data:`RECORDED` for the built-in
    recorders, e.g. ``"density"`` for :class:`DensityRecorder`, and the metric
    names of a :class:`ScalarTimeSeriesRecorder`; ``list(data)`` shows what a run
    recorded. The arrays are the ones on the model (``lgca.n_t``, ...) at the
    end of the run.

    Examples
    --------
    >>> from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, AnalysisSpec, run_model
    >>> from lgca.simulation import PopulationRecorder, Schedule
    >>> spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=20), state=StateSpec(density=0.5),
    ...                  time=TimeSpec(steps=10, seed=1),
    ...                  analysis=AnalysisSpec(observers=[PopulationRecorder(schedule=Schedule(every=5))]))
    >>> result = run_model(spec, showprogress=False)
    >>> list(result.data)
    ['population']
    >>> result.data.steps("n").tolist()
    [0, 5, 10]
    """

    def __init__(self, values: Mapping[str, Any] | None = None, steps: Mapping[str, Any] | None = None,
                 hint: str = "Add a recorder to ModelSpec.analysis, e.g. AnalysisSpec(observers=[DensityRecorder()])"):
        self._values = dict(values or {})  # name -> array, or a function that returns it
        self._steps = dict(steps or {})
        self._hint = hint

    @classmethod
    def from_run(cls, lgca, observers=(), **options) -> RunData:
        """The data of the recorders among ``observers`` of a run of ``lgca``."""
        recorders = {type(observer).__name__ for observer in observers}
        values, steps = {}, {}
        for name, (attribute, steps_attribute, recorder, _) in RECORDED.items():
            if recorder in recorders and steps_attribute in vars(lgca):
                # read lazily: the lists of labels of some identity-based models are built on demand
                values[name] = _reader(lgca, attribute)
                steps[name] = np.asarray(getattr(lgca, steps_attribute))
        for observer in observers:
            if isinstance(observer, ScalarTimeSeriesRecorder) and observer.records:
                recorded = np.array([row["step"] for row in observer.records])
                for name in observer.metrics:
                    if name not in values:
                        values[name] = np.array([row[name] for row in observer.records])
                        steps[name] = recorded
        return cls(values, steps, **options)

    def _name(self, name):
        name = _DATA_ALIASES.get(name, name)
        if name not in self._values:
            recorded = ", ".join(self._values) or "nothing"
            raise KeyError(f"{name!r} was not recorded; this run recorded {recorded}. {self._hint}")
        return name

    def __getitem__(self, name):
        name = self._name(name)
        value = self._values[name]
        if callable(value):
            value = self._values[name] = value()
        return value

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __contains__(self, name):
        return _DATA_ALIASES.get(name, name) in self._values

    def steps(self, name) -> np.ndarray:
        """The steps at which ``name`` was recorded, counted from the start of the run."""
        return self._steps[self._name(name)]

    def __repr__(self):
        return f"RunData({', '.join(self._values) or 'nothing recorded'})"


def _reader(lgca, attribute):
    """The recorded array, or a function that builds it (the lists of labels from a cell history)."""
    history = vars(lgca).get("cells_t")
    if attribute == "nodes_t" and history is not None and vars(lgca).get("_nodes_t") is None:
        return history.to_nodes
    return getattr(lgca, attribute)
