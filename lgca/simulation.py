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
    # not lgca.nodes: it would build the label lists of identity-based models without volume exclusion
    populations = getattr(lgca, "_channel_populations", None)
    nodes = populations() if populations is not None else lgca.nodes
    channels = math.prod(nodes.shape[len(lgca.dims):])
    species = getattr(lgca, "n_species", 1)
    for observer in observers:
        schedule = getattr(observer, "schedule", None) or Schedule()
        samples = (timesteps // schedule.every + 1 if schedule.steps is None
                   else sum(step <= timesteps for step in schedule.steps))
        if isinstance(observer, NodeRecorder):
            per_frame = spatial * channels * nodes.dtype.itemsize
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
        elif isinstance(observer, FieldRecorder):
            per_frame = spatial * len(observer.fields) * np.dtype(float).itemsize
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
    "FieldRecorder",
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
    """Base class for objects that observe simulation state over time.

    :class:`SimulationRunner` calls ``setup`` before the run, ``on_step`` at the
    scheduled steps (step 0 is the initial state) and ``finalize`` at the end,
    also of a run that failed; before that, ``truncate(lgca, step)`` asks the
    observer to keep only what it recorded up to ``step``, the last step that
    completed.
    """

    def __init__(self, schedule: Schedule | None = None):
        self.schedule = schedule or Schedule()

    def setup(self, lgca, runner: "SimulationRunner") -> None:
        pass

    def on_step(self, lgca, step: int) -> None:
        pass

    def truncate(self, lgca, step: int) -> None:
        """Keep what was recorded up to ``step``: the run failed after it, so later rows are empty."""
        for name in _recorded_names(self):
            attribute, steps = RECORDED[name][:2]
            recorded = getattr(lgca, steps, None)
            if recorded is None or not hasattr(lgca, attribute):
                continue
            count = int(np.searchsorted(recorded, step, side="right"))
            setattr(lgca, steps, recorded[:count])
            if not (attribute == "nodes_t" and getattr(self, "_cells", False)):  # built from cells_t
                setattr(lgca, attribute, getattr(lgca, attribute)[:count])

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


def check_observers(observers) -> None:
    """Check that the recorders of a run do not write the same outputs.

    Recorders of one built-in type share arrays of the model, so each type may appear once, and
    each field may be recorded by one :class:`FieldRecorder`. Two recorders may give a name of
    ``result.data`` only if they record the same quantity, on any schedules: a
    :class:`PopulationRecorder` and a :class:`ScalarTimeSeriesRecorder` of the default
    population, or one metric function in two scalar recorders. :class:`SimulationRunner` checks
    this before the first step, ``biolgca validate`` and ``biolgca run`` before they write a file.

    Parameters
    ----------
    observers : iterable of Observer
        The observers of the run.

    Raises
    ------
    ValueError
        If two recorders would write the same output.
    """
    observers = list(observers)
    recorder_types = (NodeRecorder, DensityRecorder, PopulationRecorder,
                      ChannelDensityRecorder, PerTypeRecorder, OrderParameterRecorder,
                      FamilyPopulationRecorder)
    seen = set()
    for observer in observers:
        kind = next((kind for kind in recorder_types if isinstance(observer, kind)), None)
        if kind is not None:
            if kind in seen:
                raise ValueError(f"Multiple {kind.__name__} instances share LGCA output arrays; "
                                 "use one recorder per type and select samples afterward")
            seen.add(kind)
    recorded_fields = [name for observer in observers if isinstance(observer, FieldRecorder)
                       for name in observer.fields]
    twice = sorted({name for name in recorded_fields if recorded_fields.count(name) > 1})
    if twice:
        raise ValueError(f"the fields {twice} are recorded by several FieldRecorders; record each "
                         "field with one")
    output_names = {}
    for observer in observers:
        names = _recorded_names(observer)
        duplicates = {name for name in names if name in output_names
                      and not _same_output(name, observer, output_names[name])}
        if duplicates:
            raise ValueError(f"Several recorders produce the same output names {sorted(duplicates)}; "
                             "use distinct names or one recorder per output")
        output_names.update(dict.fromkeys(names, observer))


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
        check_observers(self.observers)
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

        self.completed_step = self.failed_step = None
        step = 0
        try:
            self._notify_observers(0)
            self.completed_step = 0
            for step in tqdm(range(1, self.timesteps + 1), disable=not self.showprogress):
                if self.step_function is None:
                    lgca.timestep()
                else:
                    self.step_function(lgca, step, self)
                self._notify_observers(step)
                self.completed_step = step
        except BaseException as exc:
            self._stop(exc, step, start)
            raise
        self._finalize()
        self.elapsed_seconds = time.perf_counter() - start
        return lgca

    def _stop(self, exc, step, start):
        """A step failed: keep the recordings of the completed steps and write the outputs of the observers."""
        self.failed_step = step
        done = -1 if self.completed_step is None else self.completed_step
        for observer in self.observers:
            truncate = getattr(observer, "truncate", None)
            if truncate is not None:
                try:
                    truncate(self.lgca, done)
                except Exception as error:  # noqa: BLE001 - the step's error is the one raised
                    exc.add_note(f"{type(observer).__name__} could not drop its rows after step {done}: {error}")
        try:
            self._finalize()
        except Exception as error:  # noqa: BLE001
            exc.add_note(f"the observers could not finish ({type(error).__name__}: {error})")
        self.elapsed_seconds = time.perf_counter() - start
        recorded = "nothing was recorded" if done < 0 else f"recordings and output files hold steps 0 to {done}"
        exc.add_note(f"the run stopped at step {step} of {self.timesteps}; {recorded}")

    def _finalize(self):
        for observer in self.observers:
            finalize = getattr(observer, "finalize", None)
            if finalize is not None:
                finalize(self.lgca, self)

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

    def truncate(self, lgca, step: int) -> None:
        super().truncate(lgca, step)
        if self._cells:
            lgca.cells_t._truncate(len(lgca.nodes_steps))
            lgca.nodes_t = None  # built again from the cells

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
        # not getattr(..., lgca.nodes): reading nodes builds the label lists of identity-based models
        channel_pop = getattr(lgca, "channel_pop", None)
        if channel_pop is None:
            channel_pop = getattr(lgca, "occupied", None)
        if channel_pop is None:
            channel_pop = lgca.nodes
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
        if not hasattr(lgca, "props"):
            raise RuntimeError("FamilyPopulationRecorder needs an identity-based model that tracks families "
                               "(StateSpec(identity_based=True) and a rule that founds families)")
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
            # a rule may also found families in its body (cells.found_families, divide(new_family=True))
            populations = lgca.calc_family_pop_alive()
            missing = len(populations) - lgca.fam_pop_t.shape[1]
            if missing > 0:
                lgca.fam_pop_t = np.pad(lgca.fam_pop_t, ((0, 0), (0, missing)))
            lgca.fam_pop_t[self._sample_indices[step], :len(populations)] = populations

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


class FieldRecorder(Observer):
    """Record fields of the model, e.g. one that a ``pde`` operator updates.

    The history of each field is ``result.data[name]``, of shape
    ``(samples,) + dims`` (interior nodes), with ``result.data.steps(name)``.

    Parameters
    ----------
    fields : str or sequence of str
        Names of fields of ``StateSpec.fields`` with one value per node.
    schedule : Schedule, optional
        When to record; default every step.

    Examples
    --------
    >>> from lgca.simulation import FieldRecorder, Schedule
    >>> recorder = FieldRecorder("oxygen", schedule=Schedule(every=10))
    >>> recorder.fields
    ('oxygen',)
    """

    def __init__(self, fields, schedule: Schedule | None = None):
        super().__init__(schedule=schedule)
        fields = (fields,) if isinstance(fields, str) else tuple(fields)
        if not fields or not all(isinstance(name, str) and name for name in fields):
            raise ValueError("FieldRecorder needs the names of one or more fields")
        if len(set(fields)) != len(fields):
            raise ValueError("FieldRecorder lists a field twice")
        self.fields = fields
        self.values: dict[str, np.ndarray] = {}
        self.steps = np.zeros(0, dtype=int)

    def setup(self, lgca, runner: SimulationRunner) -> None:
        self._check(lgca)
        dims = tuple(lgca.dims)
        self.steps = np.fromiter((step for step in range(runner.timesteps + 1)
                                  if self.schedule.should_run(step)), dtype=int)
        self._sample_indices = {int(step): index for index, step in enumerate(self.steps)}
        self.values = {name: np.zeros((len(self.steps),) + dims) for name in self.fields}

    def on_step(self, lgca, step: int) -> None:
        index = self._sample_indices[step]
        for name in self.fields:
            self.values[name][index] = self._interior(lgca, name)

    def truncate(self, lgca, step: int) -> None:
        count = int(np.searchsorted(self.steps, step, side="right"))
        self.steps = self.steps[:count]
        self.values = {name: values[:count] for name, values in self.values.items()}

    def _check(self, lgca) -> None:
        """Raise if the model cannot record these fields (also ``biolgca validate`` and ``run``, before they
        write a file)."""
        dims = tuple(lgca.dims)
        for name in self.fields:
            if name in RECORDED or name in _DATA_ALIASES:
                raise ValueError(f"the field {name!r} has the name of a recording in result.data; "
                                 "rename the field")
            if not hasattr(lgca, name):
                raise ValueError(f"FieldRecorder: the model has no field {name!r}; declare it in "
                                 "StateSpec.fields")
            if self._interior(lgca, name).shape != dims:
                raise ValueError(f"FieldRecorder records fields with one value per node; {name!r} has "
                                 f"shape {np.shape(getattr(lgca, name))}")

    @staticmethod
    def _interior(lgca, name):
        values = np.asarray(getattr(lgca, name))
        if values.shape[:len(lgca.dims)] != tuple(lgca.dims):
            values = values[lgca.nonborder]
        return values


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
    """Record scalar metrics over time, and write them to a CSV file if ``output_path`` is given.

    Parameters
    ----------
    metrics : mapping, optional
        Name -> function of the model that returns a number, at least one. Default: the
        population. The values are in ``result.data`` under these names.
    schedule : Schedule, optional
        The steps at which the metrics are recorded. Default: every step.
    output_path : str or Path, optional
        CSV file with a column ``step`` and a column per metric, written at the end of
        the run. Default: no file. ``biolgca run`` writes ``time_series.csv`` in its
        output directory.
    """

    def __init__(
        self,
        metrics: Mapping[str, Any] | None = None,
        schedule: Schedule | None = None,
        output_path=None,
    ):
        super().__init__(schedule=schedule)
        self.metrics = dict({"population": _total_population} if metrics is None else metrics)
        if not self.metrics:
            raise ValueError("ScalarTimeSeriesRecorder needs at least one metric; omit metrics to record "
                             "the population")
        for name, metric in self.metrics.items():
            if not isinstance(name, str) or not name or name == "step" or name in _DATA_ALIASES:
                raise ValueError(f"Metric names must be non-empty strings other than 'step' and "
                                 f"the data aliases {list(_DATA_ALIASES)}, got {name!r}")
            if not callable(metric):
                raise TypeError(f"The metric {name!r} must be callable")
        self.output_path = None if output_path is None else Path(output_path)
        self.records: list[dict[str, Any]] = []

    def setup(self, lgca, runner: SimulationRunner) -> None:
        self.records = []
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step(self, lgca, step: int) -> None:
        row = {"step": step}
        for name, metric in self.metrics.items():
            row[name] = _scalar_value(metric(lgca))
        self.records.append(row)

    def finalize(self, lgca, runner: SimulationRunner) -> None:
        if not self.records or self.output_path is None:
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


def _recorded_names(observer):
    if isinstance(observer, FieldRecorder):
        return set(observer.fields)
    if isinstance(observer, ScalarTimeSeriesRecorder):
        return set(observer.metrics)
    return {name for name, (_, _, recorder, _) in RECORDED.items()
            if isinstance(observer, globals()[recorder])}


def _same_output(name, first, second):
    """Whether two recorders record the same quantity under ``name``: PopulationRecorder and the
    default metric of a ScalarTimeSeriesRecorder, or one metric function in two of them. Their
    schedules may differ; ``result.data`` keeps one of them, with its own steps."""
    def source(observer):
        if isinstance(observer, PopulationRecorder):
            return _total_population if name == "population" else None
        if isinstance(observer, ScalarTimeSeriesRecorder):
            return observer.metrics.get(name)
        return None

    return source(first) is not None and source(first) is source(second)


class RunData(Mapping):
    """The data recorded in a run, by name, with the steps at which it was recorded.

    ``data["population"]`` (alias ``data["n"]``) is an array with one entry per
    recorded step, ``data.steps("population")`` the steps (counted from the
    start of the run). The names are those of :data:`RECORDED` for the built-in
    recorders, e.g. ``"density"`` for :class:`DensityRecorder`, and the metric
    names of a :class:`ScalarTimeSeriesRecorder` and the field names of a
    :class:`FieldRecorder`; ``list(data)`` shows what a run
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
        values, steps = {}, {}
        for name, (attribute, steps_attribute, recorder, _) in RECORDED.items():
            if (any(isinstance(observer, globals()[recorder]) for observer in observers)
                    and steps_attribute in vars(lgca)):
                # read lazily: the lists of labels of some identity-based models are built on demand
                values[name] = _reader(lgca, attribute)
                steps[name] = np.asarray(getattr(lgca, steps_attribute))
        for observer in observers:
            if isinstance(observer, FieldRecorder):
                for name in observer.fields:
                    values[name] = observer.values[name]
                    steps[name] = observer.steps
        for observer in observers:
            if isinstance(observer, ScalarTimeSeriesRecorder):
                recorded = np.array([row["step"] for row in observer.records], dtype=int)
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
