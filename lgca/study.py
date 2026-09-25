"""Study a model: vary its parameters, sweep over them and over seeds, and collect the runs in a table.

:func:`vary` returns a copy of a :class:`~lgca.model.ModelSpec` with some values changed, named by
paths such as ``"time.steps"`` or ``"dynamics.operators[0].parameters.kappa"``, or by short names
such as ``"kappa"`` when only one place of the model has that name. :func:`sweep` runs a model for
every combination of values and seeds and returns a :class:`pandas.DataFrame` with one row per run
(or, with ``long=True``, one row per run and recorded step)::

    from lgca.study import sweep

    table = sweep(spec, grid={"beta": [0, 1, 2, 4]}, seeds=range(10),
                  measure={"population": final_population}, n_jobs=4)
    table.groupby("beta").population.agg(["mean", "std"])

Paths
-----
A path is a sequence of names separated by dots, with ``[i]`` for the ``i``-th entry of a list.

- ``[name]`` instead of ``[i]`` selects the entry with that name, e.g. ``operators[birth_death]``
  or ``terms[chemotaxis]``, if exactly one entry has it.
- Names that are not fields of an operator or term are its parameters: ``operators[0].kappa`` is
  short for ``operators[0].parameters.kappa``.
- A single name without dots, e.g. ``"kappa"``, is searched for among the fields of the space,
  state and time and the parameters (given or default) of all operators and terms; it must occur
  once.
"""

from __future__ import annotations

import difflib
import itertools
import logging
import multiprocessing
import pickle
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["final_population", "resolve_path", "sweep", "vary"]

logger = logging.getLogger("lgca")

_TOKEN = re.compile(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[([^\[\]]+)\]")
def vary(spec, changes: Mapping[str, Any]):
    """A copy of ``spec`` with the values at the given paths replaced.

    Parameters
    ----------
    spec : ModelSpec
        The model to vary; it is not changed.
    changes : mapping
        Path (see the module) -> new value, e.g. ``{"time.steps": 200, "kappa": 4.0}``.

    Returns
    -------
    ModelSpec

    Examples
    --------
    >>> from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec
    >>> from lgca.pipeline import InteractionPipelineSpec
    >>> spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=50), state=StateSpec(density=0.2),
    ...                  time=TimeSpec(steps=20, seed=1),
    ...                  dynamics=InteractionPipelineSpec(operators=[
    ...                      {"name": "birth_death", "parameters": {"birth_rate": 0.1}}]))
    >>> variant = vary(spec, {"time.steps": 40, "birth_rate": 0.3, "death_rate": 0.05})
    >>> variant.time.steps, variant.dynamics.operators[0]["parameters"]
    (40, {'birth_rate': 0.3, 'death_rate': 0.05})
    """
    if not isinstance(changes, Mapping):
        raise TypeError(f"changes must map paths to values, e.g. {{'time.steps': 200}}, got {changes!r}")
    for path, value in changes.items():
        spec = _set(spec, _tokens(resolve_path(spec, path)), value, "", None)
    return spec


def resolve_path(spec, path: str) -> str:
    """The full path of ``path``: itself, or the one place of the model with that short name."""
    if not isinstance(path, str) or not path:
        raise TypeError(f"a path must be a non-empty string, e.g. 'time.steps', got {path!r}")
    _tokens(path)  # check the syntax
    if "." in path or "[" in path or path in {f.name for f in fields(spec)}:
        return path
    places = _places(spec, path)
    if len(places) == 1:
        return places[0]
    if not places:
        raise KeyError(f"no field or parameter of the model is named {path!r}; give a path such as "
                       f"'dynamics.operators[0].parameters.{path}'")
    raise KeyError(f"{path!r} occurs in several places: {', '.join(places)}; give one of these paths")


def final_population(result) -> int:
    """The number of cells at the end of the run, the default measure of :func:`sweep`."""
    lgca = result.lgca
    density = lgca.species_density if hasattr(lgca, "species_density") else lgca.cell_density
    return int(np.asarray(density[lgca.nonborder]).sum())


def sweep(spec, grid: Mapping[str, Sequence] | Sequence[Mapping[str, Any]] | None = None, *,
          seeds: Iterable[int] | None = None, measure: Mapping[str, Any] | None = None, n_jobs: int = 1,
          backend: str = "processes", long: bool = False, plugins: Sequence[str] = (),
          showprogress: bool = True):
    """Run a model for every combination of parameter values and seeds; one table row per run.

    Parameters
    ----------
    spec : ModelSpec
        The model; its values are changed with :func:`vary` for every run.
    grid : mapping or list of mappings, optional
        Path (see the module) -> list of values; every combination of the values runs. A list of
        mappings gives the combinations explicitly, e.g. ``[{"kappa": 2, "theta": 0.3}, ...]``.
        Default: the model as it is.
    seeds : iterable of int, optional
        The seeds; every combination runs once per seed. Default: the seed of ``spec`` (one drawn
        from the operating system if it has none, and recorded in the table).
    measure : mapping, optional
        Column name -> what to measure in each run: a function of the run's
        :class:`~lgca.model.ModelRunResult` that returns a value (a number, an array or a
        :class:`pandas.Series` indexed by step), or the name of a recording in
        :attr:`result.data <lgca.model.ModelRunResult.data>`, e.g. ``"population"``, which
        measures its whole time series (the recorder is added if the model has none). Default:
        ``{"population": final_population}``.
    n_jobs : int, default=1
        Number of runs at the same time. With more than one, runs go to worker processes (or
        threads, see ``backend``); the results do not depend on ``n_jobs``.
    backend : {"processes", "threads"}, default="processes"
        Worker processes run in parallel; they need functions that can be sent to them (defined
        with ``def`` in a module, not lambdas) and rules they can import (``plugins``). Threads
        take any function and the rules defined in a notebook, but run only partly in parallel.
    long : bool, default=False
        One row per run and recorded step instead of one row per run: time series (named
        recordings and :class:`pandas.Series`) are spread over a column ``step``, and numbers are
        repeated on every row of the run.
    plugins : sequence of str, default=()
        Modules to import in every worker before it runs, e.g. the module that registers your
        own rules (``"my_project.rules"``).
    showprogress : bool, default=True
        Show a progress bar over the runs.

    Every run has its own copy of the model's observers. Files that observers would write (CSV
    snapshots and time series) are discarded; measure what you need instead, e.g. the metrics of
    a ``ScalarTimeSeriesRecorder`` by their names.

    Returns
    -------
    pandas.DataFrame
        A column per varied value (named by its last name, e.g. ``beta``, or by the shortest end of
        its path that tells it from the others, e.g. ``chemotaxis.beta``),
        ``seed`` and a column per measure; with ``long=True`` also ``step``. ``table.attrs`` holds
        the BioLGCA version and the paths of the varied values.

    Examples
    --------
    >>> from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec
    >>> from lgca.pipeline import InteractionPipelineSpec
    >>> spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=50), state=StateSpec(density=0.2),
    ...                  time=TimeSpec(steps=20),
    ...                  dynamics=InteractionPipelineSpec(operators=[
    ...                      {"name": "birth_death", "parameters": {"birth_rate": 0.1}}]))
    >>> table = sweep(spec, grid={"birth_rate": [0.1, 0.3]}, seeds=range(3), showprogress=False)
    >>> table.columns.tolist(), len(table)
    (['birth_rate', 'seed', 'population'], 6)
    >>> table.groupby("birth_rate").population.mean().is_monotonic_increasing
    True
    """
    import pandas as pd

    from .model import _package_version

    for module in plugins:  # their rules may be named in the grid
        import_module(module)
    combinations, paths = _combinations(spec, grid)
    columns = _column_names(paths)
    measures = _measures(measure)
    spec = _with_recorders(spec, measures)
    seeds = [spec.time.seed] if seeds is None else [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("seeds is empty; give at least one seed, e.g. seeds=range(10)")
    if seeds == [None]:
        seeds = [_drawn_seed()]
    jobs = [(combination, seed) for combination in combinations for seed in seeds]
    results = _execute(spec, jobs, measures, n_jobs, backend, tuple(plugins), showprogress, columns)
    rows = []
    for (combination, seed), measured in zip(jobs, results):
        base = {columns[path]: value for path, value in combination.items()}
        base["seed"] = seed
        rows.extend(_rows(base, measured, long))
    table = pd.DataFrame(rows)
    table.attrs["biolgca_version"] = _package_version()
    table.attrs["paths"] = {columns[path]: path for path in paths}
    return table


def _tokens(path):
    tokens, position = [], 0
    for match in _TOKEN.finditer(path):
        if match.start() != position or (position == 0 and path.startswith(".")):
            break
        name, index = match.groups()
        if name is not None:
            tokens.append(("name", name))
        else:
            index = index.strip()
            tokens.append(("index", int(index) if re.fullmatch(r"-?\d+", index) else index))
        position = match.end()
    if position != len(path) or not tokens or tokens[0][0] != "name":
        raise ValueError(f"invalid path {path!r}; write names separated by dots and [i] for list entries, "
                         f"e.g. 'dynamics.operators[0].parameters.kappa'")
    return tokens


def _name_of(entry):
    if isinstance(entry, Mapping):
        return entry.get("name")
    return getattr(entry, "name", None)


def _has_parameters(node):
    if is_dataclass(node):
        return "parameters" in {f.name for f in fields(node)}
    return isinstance(node, Mapping) and "name" in node


def _set(node, tokens, value, where, parent):
    if not tokens:
        return value
    (kind, key), rest = tokens[0], tokens[1:]
    if kind == "index":
        if not isinstance(node, (list, tuple)):
            raise KeyError(f"{where or 'the model'} is not a list; [{key}] needs a list")
        index = _select(node, key, where)
        items = list(node)
        # operators and terms take their parameters by name
        entry = "entry" if parent in ("operators", "terms") else None
        items[index] = _set(items[index], rest, value, f"{where}[{key}]", entry)
        return tuple(items) if isinstance(node, tuple) else items
    here = f"{where}.{key}" if where else key
    if is_dataclass(node) and not isinstance(node, type):
        names = [f.name for f in fields(node)]
        if key in names:
            return replace(node, **{key: _set(getattr(node, key), rest, value, here, key)})
        if "parameters" in names and parent == "entry":
            parameters = dict(node.parameters or {})
            return replace(node, parameters=_set(parameters, tokens, value, f"{where}.parameters", "parameters"))
        raise KeyError(_unknown(key, names, where))
    if isinstance(node, Mapping):
        if parent == "entry" and "name" in node and key != "name":
            # an operator given as a dict: other names are its parameters, which may be left out
            entry = dict(node)
            inner = rest if key == "parameters" else tokens
            entry["parameters"] = _set(dict(node.get("parameters") or {}), inner, value,
                                       f"{where}.parameters", "parameters")
            return entry
        if rest and key not in node:
            raise KeyError(_unknown(key, list(node), where))
        entry = dict(node)
        entry[key] = _set(node.get(key), rest, value, here, key)
        return entry
    raise KeyError(f"{where or 'the model'} has no entries; cannot set {key!r} in {type(node).__name__}")


def _select(items, key, where):
    if isinstance(key, int):
        if not -len(items) <= key < len(items):
            raise KeyError(f"{where}[{key}] does not exist; {where} has {len(items)} entries")
        return key % len(items)
    matches = [index for index, entry in enumerate(items) if _name_of(entry) == key]
    if len(matches) == 1:
        return matches[0]
    names = [_name_of(entry) for entry in items]
    if not matches:
        raise KeyError(f"{where} has no entry named {key!r}; its entries are {names}")
    raise KeyError(f"{where} has several entries named {key!r} (at {matches}); select one by index")


def _unknown(key, names, where):
    hint = difflib.get_close_matches(key, [str(name) for name in names], n=1)
    return (f"{where or 'the model'} has no {key!r}" + (f" (did you mean {hint[0]!r}?)" if hint else "")
            + f"; it has {', '.join(map(str, names))}")


def _places(spec, name):
    """The full paths of all places of the model with the short name ``name``."""
    places = [f"{part}.{name}" for part in ("space", "state", "time")
              if name in {f.name for f in fields(getattr(spec, part))}]
    for index, operator in enumerate(spec.dynamics.operators):
        where = f"dynamics.operators[{index}]"
        if hasattr(operator, "terms"):  # a ReorientationSpec
            if name in (operator.parameters or {}) or name in ("sweeps", "channels", "species"):
                places.append(f"{where}.parameters.{name}")
            for position, term in enumerate(operator.terms):
                term_where = f"{where}.terms[{position}]"
                if name in {f.name for f in fields(term)} - {"name", "parameters"}:
                    places.append(f"{term_where}.{name}")
                elif name in (term.parameters or {}) or name in _term_parameters(term.name):
                    places.append(f"{term_where}.parameters.{name}")
            continue
        parameters = operator.get("parameters") or {} if isinstance(operator, Mapping) else \
            getattr(operator, "parameters", None) or {}
        if name in parameters or name in _operator_parameters(_name_of(operator)):
            places.append(f"{where}.parameters.{name}")
    return places


def _operator_parameters(name):
    from .plugins import describe_plugin

    try:
        return set(describe_plugin(name).parameter_specs)
    except (KeyError, ValueError, TypeError):
        return set()


def _term_parameters(name):
    from .pipeline import _REORIENTATION_TERMS, _TERM_ALIASES

    definition = _REORIENTATION_TERMS.get(_TERM_ALIASES.get(name, name))
    return set(definition.info.parameter_specs) if definition is not None else set()


def _combinations(spec, grid):
    if grid is None:
        return [{}], []
    if isinstance(grid, Mapping):
        paths = [resolve_path(spec, path) for path in grid]
        values = []
        for path, options in zip(paths, grid.values()):
            if isinstance(options, (str, bytes, Mapping)) or not isinstance(options, Iterable):
                raise TypeError(f"grid[{path!r}] must be a list of values, got {options!r}")
            values.append(list(options))
        return [dict(zip(paths, point)) for point in itertools.product(*values)], paths
    points = [{resolve_path(spec, path): value for path, value in point.items()} for point in grid]
    paths = list(dict.fromkeys(path for point in points for path in point))
    return points, paths


def _column_names(paths):
    """Short column names: the last name of each path, or the shortest suffix that tells them apart."""
    labels = {path: [key for key in _tokens(path) if key != ("name", "parameters")] for path in paths}
    names = {}
    for path, tokens in labels.items():
        for length in range(1, len(tokens) + 1):
            suffix = tokens[-length:]
            if suffix[0][0] == "index" and isinstance(suffix[0][1], int):
                continue  # "[1].beta" reads badly; include the list's name
            name = _render(suffix)
            clashes = [other for other, others in labels.items()
                       if other != path and _render(others[-length:]) == name]
            if not clashes and name not in ("seed", "step"):
                names[path] = name
                break
        else:
            names[path] = path
    return names


def _render(tokens):
    text = ""
    for kind, key in tokens:
        if kind == "index" and isinstance(key, int):
            text += f"[{key}]"
        else:
            text += ("." if text else "") + str(key)
    return text


def _measures(measure):
    if measure is None:
        return {"population": final_population}
    if not isinstance(measure, Mapping) or not measure:
        raise TypeError("measure must map column names to functions of the result or names of recordings, "
                        "e.g. {'population': 'population'}")
    for name, what in measure.items():
        if not (callable(what) or isinstance(what, str)):
            raise TypeError(f"measure[{name!r}] must be a function of the result or the name of a recording, "
                            f"got {what!r}")
    return dict(measure)


def _with_recorders(spec, measures):
    from . import simulation
    from .model import AnalysisSpec

    observers = list(spec.analysis.observers) if spec.analysis is not None else []
    present = {type(observer).__name__ for observer in observers}
    for what in measures.values():
        if isinstance(what, str):
            name = simulation._DATA_ALIASES.get(what, what)
            recorder = simulation.RECORDED[name][2] if name in simulation.RECORDED else None
            if recorder is not None and recorder not in present:
                observers.append(getattr(simulation, recorder)())
                present.add(recorder)
    return replace(spec, analysis=AnalysisSpec(observers=tuple(observers)))


def _drawn_seed():
    return int(np.random.SeedSequence().generate_state(1, np.uint32)[0])


def _execute(spec, jobs, measures, n_jobs, backend, plugins, showprogress, columns):
    from tqdm.auto import tqdm

    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer, got {n_jobs!r}")
    if backend not in ("processes", "threads"):
        raise ValueError(f"backend must be 'processes' or 'threads', got {backend!r}")
    logger.info("sweep: %d runs, %d at a time", len(jobs), min(n_jobs, len(jobs)))
    results = [None] * len(jobs)
    progress = tqdm(total=len(jobs), disable=not showprogress)
    try:
        if n_jobs == 1 or len(jobs) == 1:
            for index, job in enumerate(jobs):
                results[index] = _guarded(spec, job, measures, (), columns, None)
                progress.update()
            return results
        processes = backend == "processes"
        if processes:
            _check_picklable(spec, measures)
            pool = ProcessPoolExecutor(n_jobs, mp_context=_start_method())
        else:
            pool = ThreadPoolExecutor(n_jobs)
        with pool:
            futures = {pool.submit(_guarded, spec, job, measures, plugins if processes else (), columns,
                                   backend): index for index, job in enumerate(jobs)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                progress.update()
    finally:
        progress.close()
    return results


def _start_method():
    """Fresh workers that import what they need, the same on every system (no fork of a threaded process)."""
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def _check_picklable(spec, measures):
    for name, what in measures.items():
        try:
            pickle.dumps(what)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:  # lambdas, local functions
            raise TypeError(f"measure[{name!r}] cannot be sent to worker processes ({exc}); define it with "
                            f"def at the top level of a module, or use backend='threads' or n_jobs=1") from None
    try:
        pickle.dumps(spec)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise TypeError(f"the model cannot be sent to worker processes ({exc}); use backend='threads' or "
                        f"n_jobs=1") from None


def _guarded(spec, job, measures, plugins, columns, backend):
    """One run, with the run named in errors."""
    combination, seed = job
    try:
        for module in plugins:
            import_module(module)
        return _run_one(spec, combination, seed, measures)
    except Exception as exc:
        label = ", ".join([f"{columns[path]}={value!r}" for path, value in combination.items()] + [f"seed={seed}"])
        hint = ""
        text = str(exc)
        if backend == "processes" and ("unknown operator" in text or "unknown reorientation term" in text
                                       or "Can't get attribute" in text):
            hint = (" Worker processes only know the rules and functions they can import: put them in a "
                    "module and pass plugins=['my_module'], or use backend='threads'.")
        raise RuntimeError(f"the run with {label} failed: {type(exc).__name__}: {exc}.{hint}") from exc


def _run_one(spec, combination, seed, measures):
    from .model import AnalysisSpec, run_model
    from .simulation import CSVSnapshotObserver, ScalarTimeSeriesRecorder

    variant = vary(spec, combination)
    variant = replace(variant, time=replace(variant.time, seed=seed))
    # every run its own observers (they keep state), and files written by observers go nowhere
    observers = deepcopy(tuple(variant.analysis.observers)) if variant.analysis is not None else ()
    with tempfile.TemporaryDirectory(prefix="lgca-sweep-") as scratch:
        for index, observer in enumerate(observers):
            if isinstance(observer, ScalarTimeSeriesRecorder):
                observer.output_path = Path(scratch) / f"series_{index}.csv"
            elif isinstance(observer, CSVSnapshotObserver):
                observer.output_dir = Path(scratch) / f"snapshots_{index}"
        variant = replace(variant, analysis=AnalysisSpec(observers=observers))
        result = run_model(variant, showprogress=False)
    measured = {}
    for name, what in measures.items():
        if isinstance(what, str):
            measured[name] = _Series(result.data.steps(what), result.data[what])
        else:
            value = what(result)
            measured[name] = _as_series(value)
    return measured


class _Series:
    """A time series: values and the steps at which they were recorded."""

    def __init__(self, steps, values):
        self.steps = np.asarray(steps)
        self.values = values
        if len(self.steps) != len(values):
            raise ValueError(f"a time series has {len(values)} values for {len(self.steps)} steps")


def _as_series(value):
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a dependency
        return value
    if isinstance(value, pd.Series):
        return _Series(value.index.to_numpy(), value.to_numpy())
    return value


def _rows(base, measured, long):
    if not long:
        row = dict(base)
        for name, value in measured.items():
            row[name] = value.values if isinstance(value, _Series) else value
        return [row]
    series = {name: value for name, value in measured.items() if isinstance(value, _Series)}
    constants = {name: value for name, value in measured.items() if not isinstance(value, _Series)}
    for name, value in constants.items():
        if np.ndim(value) > 0 and not isinstance(value, str):
            raise ValueError(f"measure {name!r} gave an array; for long=True give time series as the name of "
                             f"a recording or as a pandas.Series indexed by step")
    if not series:
        return [{**base, **constants}]
    steps = sorted(set().union(*[value.steps.tolist() for value in series.values()]))
    lookup = {name: dict(zip(value.steps.tolist(), value.values)) for name, value in series.items()}
    return [{**base, "step": step, **constants, **{name: lookup[name].get(step, np.nan) for name in series}}
            for step in steps]
