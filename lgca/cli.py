"""Command-line interface for portable BioLGCA model workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from pathlib import Path, PureWindowsPath

import numpy as np

from ._warnings import warn_user
from .examples import describe_example, example_names, save_example_spec
from .model import build_model, load_model_spec, save_model_spec
from .provenance import sha256, vcs_commit
from .simulation import (
    RECORDED,
    CSVSnapshotObserver,
    FieldRecorder,
    ScalarTimeSeriesRecorder,
    check_observers,
)


def main(argv=None) -> int:
    """Run the ``biolgca`` command and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biolgca")
    commands = parser.add_subparsers(dest="command", required=True)

    examples = commands.add_parser("examples", help="list or export curated models")
    example_commands = examples.add_subparsers(dest="examples_command", required=True)
    list_command = example_commands.add_parser("list", help="list curated examples")
    list_command.set_defaults(handler=_examples_list)
    export = example_commands.add_parser("export", help="export a curated model")
    export.add_argument("name")
    export.add_argument("path", type=Path)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=_examples_export)

    plugins_help = ("import these trusted modules first, e.g. the module that registers your own rules "
                    "(comma-separated or repeated)")

    validate = commands.add_parser("validate", help="validate and compile a model")
    validate.add_argument("model", type=Path)
    validate.add_argument("--trusted-paths", action="store_true")
    validate.add_argument("--plugins", action="append", default=[], metavar="MODULE", help=plugins_help)
    validate.set_defaults(handler=_validate)

    run = commands.add_parser("run", help="run a model into an explicit directory")
    run.add_argument("model", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--trusted-paths", action="store_true")
    run.add_argument("--show-progress", action="store_true")
    run.add_argument("--plugins", action="append", default=[], metavar="MODULE", help=plugins_help)
    run.set_defaults(handler=_run)

    sweep = commands.add_parser("sweep", help="run a model over parameter values and seeds into a table")
    sweep.add_argument("model", type=Path)
    sweep.add_argument("--output", type=Path, required=True, help="directory for table.csv and sweep.json")
    sweep.add_argument("--vary", action="append", default=[], metavar="PATH=VALUES",
                       help="values of a parameter, e.g. kappa=-4,0,4 or dynamics.operators[0].parameters.r_b=0.1,0.2;"
                            " every combination runs (repeat for several parameters)")
    sweep.add_argument("--seeds", default=None, help="seeds, e.g. 0:10 (0 to 9) or 1,2,3; default: the model's")
    sweep.add_argument("--measure", action="append", default=[], metavar="NAME",
                       help="a recording to measure as a time series, e.g. population (repeatable); "
                            "default: the population at the end")
    sweep.add_argument("--long", action="store_true", help="one row per run and recorded step")
    sweep.add_argument("--keep-files", action="store_true",
                       help="keep the files of the model's observers, in a folder per run (e.g. kappa=2_seed=1)")
    sweep.add_argument("--n-jobs", type=int, default=1, help="number of runs at the same time")
    sweep.add_argument("--errors", choices=("raise", "record"), default="raise",
                       help="a run that fails stops the sweep (raise), or gets a row with its error (record)")
    sweep.add_argument("--overwrite", action="store_true")
    sweep.add_argument("--trusted-paths", action="store_true")
    sweep.add_argument("--show-progress", action="store_true")
    sweep.add_argument("--plugins", action="append", default=[], metavar="MODULE", help=plugins_help)
    sweep.set_defaults(handler=_sweep)
    return parser


def _import_plugins(args) -> list[str]:
    modules = [name.strip() for entry in args.plugins for name in entry.split(",") if name.strip()]
    for module in modules:
        import_module(module)
    return modules


def _examples_list(args) -> int:
    for name in example_names():
        info = describe_example(name)
        print(f"{name}\t{info.title}")
    return 0


def _examples_export(args) -> int:
    target = args.path
    if target.exists() and not args.overwrite:
        raise ValueError(
            f"Output file already exists: {target}. Use --overwrite to replace it."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    save_example_spec(args.name, target)
    print(target)
    return 0


def _validate(args) -> int:
    _import_plugins(args)
    model_path = _existing_model_path(args.model)
    spec, _ = _load(model_path)
    _validate_output_declarations(spec, trusted_paths=args.trusted_paths)
    build_model(
        spec,
        resource_base=model_path.parent,
        trusted_paths=args.trusted_paths,
    )
    print(f"Valid BioLGCA model: {model_path}")
    return 0


def _run(args) -> int:
    _import_plugins(args)
    model_path = _existing_model_path(args.model)
    output_dir = args.output.resolve()
    if output_dir.exists() and not args.overwrite:
        raise ValueError(
            f"Output directory already exists: {output_dir}. Use --overwrite to reuse it."
        )

    spec, model_hash = _load(model_path)
    portable_spec = deepcopy(spec)
    _resolve_output_paths(spec, output_dir, trusted_paths=args.trusted_paths)
    _preflight_output_namespace(spec, output_dir, trusted_paths=args.trusted_paths)
    # the inputs are copied before the run, which reads the copies: the archive holds what ran (staged until
    # the model is built, so that a model that fails leaves no output directory)
    with tempfile.TemporaryDirectory(prefix="biolgca-") as staging:
        staging = Path(staging)
        copied = _with_copied_resources(portable_spec, model_path, staging, args.trusted_paths)
        resource_base = model_path.parent
        if copied is not portable_spec:
            spec = replace(spec, state=replace(spec.state, initializer=copied.state.initializer))
            resource_base = staging
        compiled = build_model(
            spec,
            resource_base=resource_base,
            trusted_paths=args.trusted_paths,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _move_resources(staging, output_dir)
    portable_spec = copied

    # Record the seed actually used, including one drawn for an unseeded model.
    portable_spec = replace(portable_spec, time=replace(portable_spec.time, seed=compiled.spec.time.seed))
    save_model_spec(portable_spec, output_dir / "model.resolved.json")
    result = compiled.run(showprogress=args.show_progress)
    measurements = {}
    for name, (data_name, steps_name, _, _) in RECORDED.items():  # stored under the model's attribute names
        if name in result.data:
            measurements[data_name] = result.data[name]
            measurements[steps_name] = result.data.steps(name)
    for observer in (result.spec.analysis.observers if result.spec.analysis is not None else ()):
        for name in observer.fields if isinstance(observer, FieldRecorder) else ():
            measurements[f"field_{name}"] = result.data[name]
            measurements[f"field_{name}_steps"] = result.data.steps(name)
    if measurements:
        np.savez_compressed(output_dir / "measurements.npz", **measurements)
        result.metadata["measurements_file"] = "measurements.npz"
    provenance = result.metadata.setdefault("provenance", {})
    provenance["model_file"] = {"path": str(model_path), "sha256": model_hash}
    provenance["vcs"] = vcs_commit()
    # the files of the archive, which `biolgca run` and `validate` compare when the archived model is run again
    archived = ["model.resolved.json", "model.resolved.arrays.npz", "resources/initial_state.npz"]
    provenance["archive"] = {name: sha256((output_dir / name).read_bytes()) for name in archived
                             if (output_dir / name).is_file()}
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(result.metadata), indent=2), encoding="utf-8"
    )
    print(output_dir)
    return 0


def _with_copied_resources(spec, model_path: Path, output_dir: Path, trusted_paths: bool):
    """The model with the file of its ``from_npz`` initializer copied to the output directory.

    The spec saved there (``model.resolved.json``, ``sweep.json``) then refers to the copy.
    """
    initializer = spec.state.initializer
    if initializer is None or initializer["name"] != "from_npz":
        return spec
    from .initializers import resolve_resource_path

    parameters = dict(initializer.get("parameters", {}))
    source = resolve_resource_path(parameters["path"], resource_base=model_path.parent,
                                   trusted_paths=trusted_paths)
    target = output_dir / "resources" / "initial_state.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target.resolve():
        shutil.copyfile(source, target)
    parameters["path"] = "resources/initial_state.npz"
    return replace(spec, state=replace(spec.state, initializer={"name": "from_npz", "parameters": parameters}))


def _with_copied_sweep_resources(spec, grid, model_path, output_dir, trusted_paths):
    """Archive the NPZ files the runs of a sweep read and keep their declarations portable.

    Returns the model and the grid, which refer to the copies, and archived path -> path as given.
    If the grid sets the initializer of every run, no run reads the model's own file: it is copied
    only if it exists (it may be a placeholder).
    """
    from .initializers import resolve_resource_path
    from .study import _canonical_path, vary

    initializers = {"state.initializer", "state.initializer.parameters", "state.initializer.parameters.path"}
    # Read every file before writing a copy: a file may be in resources/ of the output directory itself.
    varied, staged = False, []
    for path, options in grid.items():
        canonical = _canonical_path(spec, path)
        if canonical not in initializers:
            continue
        varied = True
        for index, value in enumerate(options):
            initializer = vary(spec, {path: value}).state.initializer
            if initializer is None or initializer.get("name") != "from_npz":
                continue
            parameters = dict(initializer.get("parameters", {}))
            source = resolve_resource_path(parameters["path"], resource_base=model_path.parent,
                                           trusted_paths=trusted_paths)
            staged.append((path, index, canonical, parameters, source.read_bytes()))
    portable_spec, sources = spec, {}
    base = spec.state.initializer
    if base is not None and base["name"] == "from_npz":
        given = base.get("parameters", {})["path"]
        if not varied or resolve_resource_path(given, resource_base=model_path.parent,
                                               trusted_paths=trusted_paths).is_file():
            portable_spec = _with_copied_resources(spec, model_path, output_dir, trusted_paths)
            sources["resources/initial_state.npz"] = given
    portable_grid = deepcopy(grid)
    for count, (path, index, canonical, parameters, data) in enumerate(staged, start=1):
        target = output_dir / "resources" / f"initial_state_{count}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        sources[f"resources/{target.name}"] = parameters["path"]
        parameters["path"] = f"resources/{target.name}"
        if canonical == "state.initializer":
            portable_grid[path][index] = {"name": "from_npz", "parameters": parameters}
        elif canonical == "state.initializer.parameters":
            portable_grid[path][index] = parameters
        else:
            portable_grid[path][index] = parameters["path"]
    return portable_spec, portable_grid, sources


def _show_given_values(table, spec, grid, portable_grid) -> None:
    """In the table, the values of the grid as given, not the paths of the copies that the runs read."""
    from .study import resolve_path

    columns = {path: column for column, path in table.attrs["paths"].items()}
    for path, values in grid.items():
        archived = portable_grid[path]
        column = columns.get(resolve_path(spec, path))
        if column is None or archived == values:
            continue
        table[column] = [values[archived.index(value)] if value in archived else value for value in table[column]]


def _move_resources(staging: Path, output_dir: Path) -> None:
    """Move the staged copies of the inputs (``resources/``) into the output directory."""
    staged = staging / "resources"
    if staged.is_dir():
        target = output_dir / "resources"
        target.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            shutil.copyfile(path, target / path.name)


def _load(model_path: Path):
    """The model of a file, and the hash of the file as read; warns if it is the archived model of an
    earlier run and its archive changed since (the run's metadata.json records the hashes)."""
    data = model_path.read_bytes()
    found = sha256(data)
    metadata = model_path.parent / "metadata.json"
    if metadata.is_file():
        try:
            archive = json.loads(metadata.read_text(encoding="utf-8")).get("provenance", {}).get("archive", {})
        except (ValueError, AttributeError):
            archive = {}
        if isinstance(archive, dict) and model_path.name in archive:
            changed = [name for name, recorded in archive.items() if (model_path.parent / name).is_file() and
                       sha256(data if name == model_path.name else (model_path.parent / name).read_bytes())
                       != recorded]
            for name in changed:
                current = found if name == model_path.name else sha256((model_path.parent / name).read_bytes())
                warn_user(f"{name} changed after the run that wrote it (sha256 {current[:16]}..., its "
                          f"metadata.json records {archive[name][:16]}...): the results in {model_path.parent} "
                          f"do not belong to it")
    return load_model_spec(model_path), found


def _existing_model_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find model spec file: {resolved}")
    return resolved


def _sweep(args) -> int:
    from .model import _package_version, model_spec_to_dict
    from .study import final_population, sweep

    plugins = _import_plugins(args)
    model_path = _existing_model_path(args.model)
    output_dir = args.output.resolve()
    if output_dir.exists() and not args.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}. Use --overwrite to reuse it.")
    spec, model_hash = _load(model_path)
    grid = {}
    for entry in args.vary:
        path, values = _parse_vary(entry)
        if path in grid:
            raise ValueError(f"--vary repeats {path!r}; give each parameter once")
        grid[path] = values
    seeds = None if args.seeds is None else _parse_seeds(args.seeds)
    measure = {name: name for name in args.measure} or {"population": final_population}
    # the inputs are copied before the first run, and the runs read the copies: a file that changes during
    # the sweep changes no run, and the archive holds what ran (staged until the sweep succeeds)
    with tempfile.TemporaryDirectory(prefix="biolgca-") as staging:
        staging = Path(staging)
        portable_spec, portable_grid, resources = _with_copied_sweep_resources(spec, grid, model_path, staging,
                                                                               args.trusted_paths)
        copied = bool(resources)
        running = deepcopy(portable_spec if copied else spec)
        if args.keep_files:  # the files of every run in a folder of its own in the output directory
            _resolve_output_paths(running, output_dir, trusted_paths=args.trusted_paths)
        table = sweep(running, grid=(portable_grid if copied else grid) or None, seeds=seeds, measure=measure,
                      n_jobs=args.n_jobs, long=args.long, plugins=plugins, showprogress=args.show_progress,
                      resource_base=staging if copied else model_path.parent, trusted_paths=args.trusted_paths,
                      keep_files=args.keep_files, errors=args.errors)
        output_dir.mkdir(parents=True, exist_ok=True)
        _move_resources(staging, output_dir)
    if copied:
        _show_given_values(table, spec, grid, portable_grid)
    table.map(_csv_cell).to_csv(output_dir / "table.csv", index=False)
    provenance = dict(table.attrs.get("provenance", {}))
    provenance["model_file"] = {"path": str(model_path), "sha256": model_hash}
    provenance["vcs"] = vcs_commit()
    description = {
        "model": model_spec_to_dict(portable_spec), "grid": _json_safe(portable_grid), "paths": table.attrs["paths"],
        "seeds": _json_safe(sorted(set(table["seed"].tolist()))), "measure": list(measure), "long": args.long,
        "errors": args.errors,
        "resources": resources, "plugins": plugins, "biolgca_version": _package_version(),
        "provenance": provenance,
    }
    (output_dir / "sweep.json").write_text(json.dumps(_json_safe(description), indent=2), encoding="utf-8")
    print(output_dir)
    return 0


def _parse_vary(entry):
    path, separator, values = entry.partition("=")
    if not separator or not path.strip() or not values.strip():
        raise ValueError(f"--vary needs PATH=VALUES, e.g. kappa=-4,0,4; got {entry!r}")
    return path.strip(), [_parse_value(value) for value in values.split(",")]


def _parse_value(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # a word, e.g. a channel set such as velocity


def _parse_seeds(text):
    if ":" in text:
        start, _, stop = text.partition(":")
        return list(range(int(start), int(stop)))
    return [int(seed) for seed in text.split(",")]


def _csv_cell(value):
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(_json_safe(value))
    return value


def _validate_output_declarations(spec, *, trusted_paths: bool) -> None:
    if spec.analysis is None:
        return
    check_observers(spec.analysis.observers)  # as the run would, but before anything is written
    for observer in spec.analysis.observers:
        if (observer.__class__.__name__ == "NodeRecorder" and spec.state.identity_based
                and not spec.state.volume_exclusion):
            raise ValueError("CLI NodeRecorder cannot persist identity NoVE list states; "
                             "use ChannelDensityRecorder or DensityRecorder instead")
        if isinstance(observer, CSVSnapshotObserver):
            _validate_relative_output(observer.output_dir, trusted_paths=trusted_paths)
            _validate_relative_output(
                observer.filename.format(kind=observer.kind, step=0),
                trusted_paths=trusted_paths,
            )
        elif isinstance(observer, ScalarTimeSeriesRecorder) and observer.output_path is not None:
            _validate_relative_output(observer.output_path, trusted_paths=trusted_paths)


def _resolve_output_paths(spec, output_dir: Path, *, trusted_paths: bool) -> None:
    _validate_output_declarations(spec, trusted_paths=trusted_paths)
    if spec.analysis is None:
        return
    for observer in spec.analysis.observers:
        if isinstance(observer, ScalarTimeSeriesRecorder) and observer.output_path is None:
            observer.output_path = Path("time_series.csv")  # the command line keeps what it records
        if isinstance(observer, CSVSnapshotObserver):
            observer.output_dir = _output_path(
                observer.output_dir, output_dir, trusted_paths=trusted_paths
            )
            observer._output_root = None if trusted_paths else output_dir
        elif isinstance(observer, ScalarTimeSeriesRecorder):
            observer.output_path = _output_path(
                observer.output_path, output_dir, trusted_paths=trusted_paths
            )


def _validate_relative_output(path, *, trusted_paths: bool) -> None:
    raw = Path(path)
    windows = PureWindowsPath(path)
    if not trusted_paths and (raw.anchor or windows.anchor or ".." in raw.parts or ".." in windows.parts):
        raise ValueError(
            "Generated output paths must be relative and may not contain '..'; "
            "use --trusted-paths only for trusted local models."
        )


def _preflight_output_namespace(spec, output_dir, *, trusted_paths):
    """Reject file collisions before any archive or observer output is written."""
    targets = {}
    directories = {}

    def reserve(path, owner):
        # Case folding also protects archives intended for case-insensitive OSes.
        resolved = path.resolve()
        key = resolved.as_posix().casefold()
        parents = [parent.as_posix().casefold() for parent in resolved.parents]
        previous_owner = targets.get(key) or directories.get(key)
        previous_owner = previous_owner or next((targets[parent] for parent in parents if parent in targets), None)
        if previous_owner is not None:
            raise ValueError(f"Output collision: {owner} and {previous_owner} at {path}")
        targets[key] = owner
        for parent in parents:
            directories[parent] = owner

    for name in ("model.resolved.json", "model.resolved.arrays.npz", "metadata.json", "measurements.npz",
                 "resources/initial_state.npz"):
        reserve(output_dir / name, "reserved archive " + name)
    if spec.analysis is None:
        return
    for index, observer in enumerate(spec.analysis.observers):
        owner = f"observer {index} ({type(observer).__name__})"
        if isinstance(observer, CSVSnapshotObserver):
            schedule = observer.schedule
            steps = (range(0, spec.time.steps + 1, schedule.every) if schedule.steps is None
                     else sorted(step for step in schedule.steps if step <= spec.time.steps))
            for step in steps:
                filename = observer.filename.format(kind=observer.kind, step=step)
                _validate_relative_output(filename, trusted_paths=trusted_paths)
                path = (observer.output_dir / filename).resolve()
                if not trusted_paths and not path.is_relative_to(output_dir):
                    raise ValueError("Snapshot output path escapes the run directory")
                reserve(path, f"{owner} step {step}")
        elif isinstance(observer, ScalarTimeSeriesRecorder):
            reserve(observer.output_path, owner)


def _output_path(path, output_dir: Path, *, trusted_paths: bool) -> Path:
    raw = Path(path)
    _validate_relative_output(raw, trusted_paths=trusted_paths)
    resolved = raw.resolve() if raw.is_absolute() else (output_dir / raw).resolve()
    if not trusted_paths and not resolved.is_relative_to(output_dir):
        raise ValueError(
            "Generated output path escapes the run directory; use --trusted-paths "
            "only for trusted local models."
        )
    return resolved


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
