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
import hashlib
import itertools
import logging
import multiprocessing
import pickle
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ._warnings import warn_user

__all__ = ["final_population", "resolve_path", "sweep", "vary"]

logger = logging.getLogger("lgca")

_TOKEN = re.compile(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[([^\[\]]+)\]")
def vary(spec, changes: Mapping[str, Any]):
    """A new spec: ``spec`` with the values at the given paths replaced.

    Parameters
    ----------
    spec : ModelSpec
        The model to vary; it is not changed.
    changes : mapping
        Path (see the module) -> new value, e.g. ``{"time.steps": 200, "kappa": 4.0}``.

    Returns
    -------
    ModelSpec
        The parts that are not changed are those of ``spec``, not copies: change the
        variant with ``vary`` again rather than in place, which would change ``spec`` too.
        A model built from either keeps its own copy (see :func:`~lgca.model.build_model`).

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
          showprogress: bool = True, resource_base: str | Path | None = None, trusted_paths: bool = False,
          keep_files: bool = False, errors: str = "raise"):
    """Run a model for every combination of parameter values and seeds; one table row per run.

    Parameters
    ----------
    spec : ModelSpec
        The model; its values are changed with :func:`vary` for every run.
    grid : mapping or list of mappings, optional
        Path (see the module) -> list of values; every combination of the values runs. A list of
        mappings gives the combinations explicitly, e.g. ``[{"kappa": 2, "theta": 0.3}, ...]``.
        Default: the model as it is. The seed is not varied here but with ``seeds``.
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
    resource_base : str or Path, optional
        The directory of a model file, against which the files it reads are found, e.g. the file of
        a ``from_npz`` initializer; they must lie inside it. Without it, files are found as
        :func:`numpy.load` finds them (relative to the working directory, or absolute). As in
        :func:`~lgca.model.build_model`.
    trusted_paths : bool, default=False
        Allow files outside ``resource_base``, as in :func:`~lgca.model.build_model`.
    keep_files : bool, default=False
        Keep the files that observers write: every run writes into a folder of its own, named
        after its values and seed (e.g. ``kappa=2_seed=1``), inside the observer's destination,
        e.g. ``snapshots/kappa=2_seed=1/density_00010.png``.
    errors : {"raise", "record"}, default="raise"
        What a run that fails does. ``"raise"`` stops the sweep with the run's error, naming its
        values and seed; runs that have not started are cancelled. ``"record"`` goes on: the
        failed run gets a row with its values, its seed and the error (type and message) in a
        column ``error``, without measures, and a warning counts the failed runs; the other
        rows have ``None`` there. Interrupting the sweep (Ctrl-C) always stops it.

    Returns
    -------
    pandas.DataFrame
        A column per varied value (named by its last name, e.g. ``beta``, or by the shortest end of
        its path that tells it from the others, e.g. ``chemotaxis.beta``),
        ``seed`` and a column per measure; with ``long=True`` also ``step``. ``table.attrs`` holds
        the BioLGCA version, the paths of the varied values and the provenance of the runs (see
        Notes).

    Notes
    -----
    Every run has its own copy of the model's observers, and runs its own copy of the operator objects
    in the model and the grid (see :func:`~lgca.model.build_model`); a model or a value of the grid that
    cannot be copied is reported before the first run. By default a sweep keeps no files: observers
    that only draw or write files (plot snapshots, movies, CSV snapshots) do not run, and the files of
    time series are discarded; measure what you need instead, e.g. the metrics of a
    ``ScalarTimeSeriesRecorder`` by their names.

    Files that the runs read, the arrays of a ``from_npz`` initializer, are read once before the
    first run: a file that changes during the sweep changes no run. ``table.attrs["provenance"]``
    records the versions of Python and the packages, the platform, the modules in ``plugins``, the
    SHA-256 hashes of the rules' source code and of the files read (see :mod:`lgca.provenance`).

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
    if errors not in ("raise", "record"):
        raise ValueError(f"errors must be 'raise' or 'record', got {errors!r}")
    reserved = set(columns.values()) | {"seed"} | ({"step"} if long else set()) | (
        {"error"} if errors == "record" else set())
    conflicts = reserved.intersection(measures)
    if conflicts:
        raise ValueError(f"measure names conflict with sweep columns: {sorted(conflicts)}; "
                         "give these measures different names")
    spec = _with_recorders(spec, measures)
    from .simulation import check_observers

    check_observers(spec.analysis.observers)  # once, before any run
    from .model import _owned_spec

    # every run copies the model and its operator objects: fail before any run if they cannot be copied
    spec = _owned_spec(spec)
    _check_grid_values(combinations, spec)
    if seeds is None:
        from .model import _validate_seed

        _validate_seed(spec.time.seed)
        seeds = [spec.time.seed if spec.time.seed is not None else _drawn_seed()]
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds is empty; give at least one seed, e.g. seeds=range(10)")
    bad = [seed for seed in seeds if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0]
    if bad:
        raise ValueError(f"seeds must be non-negative integers, got {bad[0]!r}")
    seeds = [int(seed) for seed in seeds]
    jobs = [(combination, seed) for combination in combinations for seed in seeds]
    resources = {"resource_base": None if resource_base is None else str(Path(resource_base).resolve()),
                 "trusted_paths": bool(trusted_paths)}
    if resource_base is None:  # files as np.load finds them, also by workers that start in another directory
        resources = {"resource_base": str(Path.cwd()), "trusted_paths": True}
    if keep_files:
        spec = _absolute_destinations(spec)  # workers may start in another directory
        jobs = [(combination, seed, folder) for (combination, seed), folder in zip(jobs, _run_folders(jobs, columns))]
    else:
        jobs = [(combination, seed, None) for combination, seed in jobs]
    preloaded, inputs = _read_inputs(spec, jobs, resources)
    results = _execute(spec, jobs, measures, n_jobs, backend, tuple(plugins), showprogress, columns, resources,
                       errors, preloaded)
    rows, rules = [], {}
    for (combination, seed, _), outcome in zip(jobs, results):
        base = {columns[path]: value for path, value in combination.items()}
        base["seed"] = seed
        if errors == "record":
            base["error"] = outcome.message if isinstance(outcome, _RunError) else None
        measured = {}
        if not isinstance(outcome, _RunError):
            measured, provenance = outcome
            for name, source in provenance.get("rules", {}).items():  # runs may vary the rules
                rules.setdefault(name, source)
        rows.extend(_rows(base, measured, long))
    failed = sum(isinstance(outcome, _RunError) for outcome in results)
    if failed:
        warn_user(f"{failed} of {len(results)} runs failed; their errors are in the column 'error'")
    from .provenance import environment

    table = pd.DataFrame(rows)
    table.attrs["biolgca_version"] = _package_version()
    table.attrs["paths"] = {columns[path]: path for path in paths}
    table.attrs["provenance"] = {**environment(), "plugins": list(plugins), "rules": rules, "inputs": inputs}
    return table


def _read_inputs(spec, jobs, resources):
    """Read the files of the runs once: per job the bytes of its files by resolved path, and the hash of
    every file as given. A file that cannot be read is left to the run, which reports it."""
    from .initializers import resolve_resource_path
    from .provenance import sha256

    read, hashes, preloaded = {}, {}, []
    for combination, _, _ in jobs:
        initializer = vary(spec, combination).state.initializer if combination else spec.state.initializer
        files = {}
        if isinstance(initializer, Mapping) and initializer.get("name") == "from_npz":
            given = (initializer.get("parameters") or {}).get("path")
            try:
                path = str(resolve_resource_path(given, resource_base=resources["resource_base"],
                                                 trusted_paths=resources["trusted_paths"]))
                if path not in read:
                    read[path] = Path(path).read_bytes()
                    hashes[str(given)] = sha256(read[path])
                files[path] = read[path]
            except (OSError, TypeError, ValueError):
                pass
        preloaded.append(files)
    return preloaded, hashes


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
        if key == "parameters" and parent == "entry" and rest:  # its parameters are fields, e.g. PDESpec
            return _set(node, rest, value, where, "entry")
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


def _get(node, tokens, where="", parent=None):
    """The value at a path, or the default of a parameter the model leaves out; mirrors :func:`_set`."""
    if not tokens:
        return node
    (kind, key), rest = tokens[0], tokens[1:]
    if kind == "index":
        if not isinstance(node, (list, tuple)):
            raise KeyError(f"{where or 'the model'} is not a list; [{key}] needs a list")
        entry = parent if parent in ("operators", "terms") else None
        return _get(node[_select(node, key, where)], rest, f"{where}[{key}]", entry)
    here = f"{where}.{key}" if where else key
    if is_dataclass(node) and not isinstance(node, type):
        names = [f.name for f in fields(node)]
        if key == "parameters" and "parameters" in names and rest and parent in ("operators", "terms"):
            return _parameter(node, node.parameters, rest, where, parent)
        if key in names:
            return _get(getattr(node, key), rest, here, key)
        if "parameters" in names and parent in ("operators", "terms"):
            return _parameter(node, node.parameters, tokens, where, parent)
        if key == "parameters" and parent in ("operators", "terms") and rest:  # e.g. PDESpec
            return _get(node, rest, where, parent)
        raise KeyError(_unknown(key, names, where))
    if isinstance(node, Mapping):
        if parent in ("operators", "terms") and "name" in node and key != "name":
            if key == "parameters" and not rest:  # all its parameters
                return node.get("parameters") or {}
            return _parameter(node, node.get("parameters"), rest if key == "parameters" else tokens, where, parent)
        if key not in node:
            raise KeyError(_unknown(key, list(node), where))
        return _get(node[key], rest, here, key)
    raise KeyError(f"{where or 'the model'} has no entries; cannot read {key!r} in {type(node).__name__}")


def _parameter(entry, parameters, tokens, where, parent):
    parameters = parameters or {}
    (_, key), rest = tokens[0], tokens[1:]
    if key in parameters:
        return _get(parameters[key], rest, f"{where}.parameters.{key}", key)
    from .pipeline import _REORIENTATION_TERMS, _TERM_ALIASES
    from .plugins import describe_plugin

    name = _name_of(entry)
    try:
        if parent == "terms":
            specs = _REORIENTATION_TERMS[_TERM_ALIASES.get(name, name)].info.parameter_specs
        else:
            specs = describe_plugin(name).parameter_specs
    except (KeyError, ValueError, TypeError):
        specs = {}
    if key not in specs or rest:
        raise KeyError(_unknown(key, list(parameters) + list(specs), f"{where}.parameters"))
    return specs[key].default


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
    from .operator_base import InteractionOperator
    from .pipeline import ReorientationSpec

    for index, operator in enumerate(spec.dynamics.operators):
        if isinstance(operator, InteractionOperator):  # an operator object: vary cannot change it
            continue
        where = f"dynamics.operators[{index}]"
        if isinstance(operator, ReorientationSpec):
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
    aliases = {}

    def grid_path(path):
        resolved = _grid_path(spec, path)
        return aliases.setdefault(_canonical_path(spec, resolved), resolved)

    if isinstance(grid, Mapping):
        paths = [grid_path(path) for path in grid]
        _unique_grid_paths(spec, list(grid), paths)
        values = []
        for path, options in zip(paths, grid.values()):
            if isinstance(options, (str, bytes, Mapping)) or not isinstance(options, Iterable):
                raise TypeError(f"grid[{path!r}] must be a list of values, got {options!r}")
            values.append(list(options))
            if not values[-1]:
                raise ValueError(f"grid[{path!r}] is empty; give at least one value")
        return [dict(zip(paths, point)) for point in itertools.product(*values)], paths
    points = []
    for point in grid:
        point_paths = [grid_path(path) for path in point]
        _unique_grid_paths(spec, list(point), point_paths)
        points.append(dict(zip(point_paths, point.values())))
    if not points:
        raise ValueError("grid is empty; give at least one combination, e.g. [{'kappa': 2}], or no grid to "
                         "run the model as it is")
    paths = list(dict.fromkeys(path for point in points for path in point))
    return points, paths


def _check_grid_values(combinations, spec):
    """Fail before any run if a value of the grid cannot be copied, e.g. an operator object, also one in a
    list of operators; every run copies its model."""
    from .model import _copy_error, _shared_objects
    from .pipeline import _copy_templates

    shared = _shared_objects(spec)
    checked = set()
    for combination in combinations:
        for path, value in combination.items():
            if isinstance(value, (str, bytes, int, float, np.generic, type(None))) or id(value) in checked:
                continue
            checked.add(id(value))
            try:
                _copy_templates(value, dict(shared))
            except Exception as exc:  # noqa: BLE001 - any failure of the copy: named before the first run
                raise _copy_error(value, f"grid[{path!r}] value", shared, exc) from None


def _unique_grid_paths(spec, keys, paths):
    overlap = _overlap(spec, paths)
    if overlap is not None:
        first, second = (keys[index] for index in overlap)
        raise ValueError(f"grid keys {first!r} and {second!r} set the same model parameter, or one sets a "
                         "part of the other; give each parameter once")


def _overlap(spec, paths):
    """The indices of two paths that set the same value, or a value and a part of it; ``None`` if none do."""
    canonical = [_canonical_tokens(spec, _tokens(path)) for path in paths]
    for (i, first), (j, second) in itertools.combinations(enumerate(canonical), 2):
        if first[:len(second)] == second[:len(first)]:  # one is the start of the other
            return i, j
    return None


def _grid_path(spec, path):
    """The full path of a varied value; the seed is not one (every run gets its seed from ``seeds``)."""
    resolved = resolve_path(spec, path)
    if _tokens(resolved) == [("name", "time"), ("name", "seed")]:
        raise ValueError(f"the seed cannot be varied in the grid ({path!r}); give the seeds with seeds=, "
                         f"e.g. seeds=[11, 22] (CLI: --seeds 11,22), and every combination runs once per seed")
    return resolved


def _canonical_path(spec, path):
    return _render(_canonical_tokens(spec, _tokens(resolve_path(spec, path))))


def _canonical_tokens(node, tokens, where="", parent=None):
    """Normalize named/negative indices and implicit operator parameters to their actual location."""
    if not tokens:
        return []
    (kind, key), rest = tokens[0], tokens[1:]
    if kind == "index":
        if not isinstance(node, (list, tuple)):
            raise KeyError(f"{where or 'the model'} is not a list; [{key}] needs a list")
        index = _select(node, key, where)
        entry = "entry" if parent in ("operators", "terms") else None
        return [(kind, index), *_canonical_tokens(node[index], rest, f"{where}[{key}]", entry)]
    here = f"{where}.{key}" if where else key
    if is_dataclass(node) and not isinstance(node, type):
        names = [field.name for field in fields(node)]
        if key in names:
            return [(kind, key), *_canonical_tokens(getattr(node, key), rest, here, key)]
        if parent == "entry" and "parameters" in names:
            inner = _canonical_tokens(node.parameters or {}, tokens, f"{where}.parameters", "parameters")
            return [("name", "parameters"), *inner]
        if parent == "entry" and key == "parameters" and rest:
            return _canonical_tokens(node, rest, where, parent)  # PDESpec parameters are its fields
        raise KeyError(_unknown(key, names, where))
    if isinstance(node, Mapping):
        if parent == "entry" and "name" in node and key != "name":
            inner = rest if key == "parameters" else tokens
            return [("name", "parameters"), *_canonical_tokens(node.get("parameters") or {}, inner,
                                                               f"{where}.parameters", "parameters")]
        if rest and key not in node:
            raise KeyError(_unknown(key, list(node), where))
        return [(kind, key), *_canonical_tokens(node.get(key), rest, here, key)]
    raise KeyError(f"{where or 'the model'} has no entries; cannot set {key!r} in {type(node).__name__}")


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
    present = set().union(*(simulation._recorded_names(observer) for observer in observers))
    for what in measures.values():
        if isinstance(what, str):
            name = simulation._DATA_ALIASES.get(what, what)
            recorder = simulation.RECORDED[name][2] if name in simulation.RECORDED else None
            if recorder is not None and name not in present:
                observer = getattr(simulation, recorder)()
                observers.append(observer)
                present.update(simulation._recorded_names(observer))
    return replace(spec, analysis=AnalysisSpec(observers=tuple(observers)))


def _drawn_seed():
    return int(np.random.SeedSequence().generate_state(1, np.uint32)[0])


def _execute(spec, jobs, measures, n_jobs, backend, plugins, showprogress, columns, resources, errors="raise",
             preloaded=None):
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
                results[index] = _guarded(spec, job, measures, (), columns, None, resources, errors,
                                          preloaded[index] if preloaded else None)
                progress.update()
            return results
        processes = backend == "processes"
        if processes:
            if multiprocessing.current_process().name != "MainProcess":
                # a worker imports the script that called sweep, and the script calls sweep again
                raise RuntimeError(_MAIN_GUARD)
            _check_picklable(spec, measures)
            pool = ProcessPoolExecutor(n_jobs, mp_context=_start_method())
        else:
            pool = ThreadPoolExecutor(n_jobs)
        try:
            with pool:
                futures = {pool.submit(_guarded, spec, job, measures, plugins if processes else (), columns,
                                       backend, resources, errors, preloaded[index] if preloaded else None): index
                           for index, job in enumerate(jobs)}
                try:
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                        progress.update()
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)  # the runs that have not started do not
                    raise
        except (BrokenProcessPool, EOFError) as exc:
            raise RuntimeError(f"the worker processes stopped ({type(exc).__name__}). {_MAIN_GUARD}") from exc
    finally:
        progress.close()
    return results


_MAIN_GUARD = ("Worker processes import the script that runs the sweep; in a script, put the code that "
               "calls sweep under  if __name__ == \"__main__\":  so that it runs only once. In a notebook, "
               "define measures and rules in a module, or use backend=\"threads\".")


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


@dataclass(frozen=True)
class _RunError:
    """The error of a run that failed, in a sweep that records errors."""

    message: str


def _guarded(spec, job, measures, plugins, columns, backend, resources, errors="raise", preloaded=None):
    """One run, with the run named in errors; with ``errors="record"`` a failure gives a :class:`_RunError`."""
    combination, seed, folder = job
    try:
        for module in plugins:
            import_module(module)
        from .initializers import _PRELOADED

        token = _PRELOADED.set(preloaded or None)  # the files as the sweep read them before the first run
        try:
            return _run_one(spec, combination, seed, measures, resources, folder)
        finally:
            _PRELOADED.reset(token)
    except Exception as exc:
        label = ", ".join([f"{columns[path]}={value!r}" for path, value in combination.items()] + [f"seed={seed}"])
        hint = ""
        text = str(exc)
        if backend == "processes" and ("unknown operator" in text or "unknown reorientation term" in text
                                       or "Can't get attribute" in text):
            hint = (" Worker processes only know the rules and functions they can import: put them in a "
                    "module and pass plugins=['my_module'], or use backend='threads'.")
        if errors == "record":
            return _RunError(f"{type(exc).__name__}: {exc}.{hint}")
        raise RuntimeError(f"the run with {label} failed: {type(exc).__name__}: {exc}.{hint}") from exc


def _run_one(spec, combination, seed, measures, resources, folder=None):
    from .model import AnalysisSpec, run_model

    # the run's model copies the operator objects of the model and of the grid, so runs in threads do not
    # share them (build_model)
    variant = vary(spec, combination)
    variant = replace(variant, time=replace(variant.time, seed=seed))
    # every run its own observers (they keep state), and their files in a folder of the run or nowhere
    observers = deepcopy(tuple(variant.analysis.observers)) if variant.analysis is not None else ()
    observers = tuple(observer for observer in observers if _redirect(observer, folder))
    variant = replace(variant, analysis=AnalysisSpec(observers=observers))
    result = run_model(variant, showprogress=False, **resources)
    measured = {}
    for name, what in measures.items():
        if isinstance(what, str):
            measured[name] = _Series(result.data.steps(what), result.data[what])
        else:
            value = what(result)
            measured[name] = _as_series(value)
    return measured, {"rules": result.metadata.get("provenance", {}).get("rules", {})}


def _redirect(observer, folder):
    """Send the files of an observer to the run's ``folder`` in its destination, or with no folder
    nowhere; False if the observer only draws or writes files that nobody would see."""
    from .plotting import AnimationObserver, PlotSnapshotObserver
    from .simulation import CSVSnapshotObserver, ScalarTimeSeriesRecorder

    if isinstance(observer, ScalarTimeSeriesRecorder):
        if folder is None or observer.output_path is None:
            observer.output_path = None  # the values stay in result.data
        else:
            observer.output_path = observer.output_path.parent / folder / observer.output_path.name
    elif isinstance(observer, (CSVSnapshotObserver, PlotSnapshotObserver)):
        if folder is None or observer.output_dir is None:
            return False
        observer.output_dir = observer.output_dir / folder
    elif isinstance(observer, AnimationObserver):
        if folder is None or observer.save_path is None:
            return False
        observer.save_path = observer.save_path.parent / folder / observer.save_path.name
    return True


def _absolute_destinations(spec):
    """The model with the destinations of its observers' files as absolute paths."""
    from .model import AnalysisSpec

    if spec.analysis is None:
        return spec
    observers = deepcopy(tuple(spec.analysis.observers))
    for observer in observers:
        for attribute in ("output_path", "output_dir", "save_path"):
            path = getattr(observer, attribute, None)
            if isinstance(path, Path):
                setattr(observer, attribute, path.absolute())
    return replace(spec, analysis=AnalysisSpec(observers=observers))


def _run_folders(jobs, columns):
    """A folder name per run from its values and seed, e.g. ``kappa=2_seed=1``; unique."""
    names = []
    for combination, seed in jobs:
        parts = [f"{columns[path]}={value}" for path, value in combination.items()] + [f"seed={seed}"]
        name = re.sub(r"[^A-Za-z0-9=._+-]+", "-", "_".join(parts))
        if len(name) > 120:
            name = name[:96] + "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        names.append(name)
    counts = Counter(name.casefold() for name in names)  # folders on Windows and macOS ignore case
    return [f"{name}_run{index}" if counts[name.casefold()] > 1 else name for index, name in enumerate(names)]


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
    if not steps:  # nothing recorded: one row with an empty step keeps the run and its numbers
        return [{**base, "step": np.nan, **constants, **{name: np.nan for name in series}}]
    lookup = {name: dict(zip(value.steps.tolist(), value.values)) for name, value in series.items()}
    return [{**base, "step": step, **constants, **{name: lookup[name].get(step, np.nan) for name in series}}
            for step in steps]
