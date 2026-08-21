"""Command-line interface for portable BioLGCA model workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .examples import describe_example, example_names, save_example_spec
from .model import build_model, load_model_spec, save_model_spec
from .simulation import CSVSnapshotObserver, ScalarTimeSeriesRecorder


def main(argv=None) -> int:
    """Run the ``biolgca`` command and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
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

    validate = commands.add_parser("validate", help="validate and compile a model")
    validate.add_argument("model", type=Path)
    validate.add_argument("--trusted-paths", action="store_true")
    validate.set_defaults(handler=_validate)

    run = commands.add_parser("run", help="run a model into an explicit directory")
    run.add_argument("model", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--trusted-paths", action="store_true")
    run.add_argument("--show-progress", action="store_true")
    run.set_defaults(handler=_run)
    return parser


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
    model_path = _existing_model_path(args.model)
    spec = load_model_spec(model_path)
    _validate_output_declarations(spec, trusted_paths=args.trusted_paths)
    build_model(
        spec,
        resource_base=model_path.parent,
        trusted_paths=args.trusted_paths,
    )
    print(f"Valid BioLGCA model: {model_path}")
    return 0


def _run(args) -> int:
    model_path = _existing_model_path(args.model)
    output_dir = args.output.resolve()
    if output_dir.exists() and not args.overwrite:
        raise ValueError(
            f"Output directory already exists: {output_dir}. Use --overwrite to reuse it."
        )

    spec = load_model_spec(model_path)
    _resolve_output_paths(spec, output_dir, trusted_paths=args.trusted_paths)
    compiled = build_model(
        spec,
        resource_base=model_path.parent,
        trusted_paths=args.trusted_paths,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_spec(compiled.spec, output_dir / "model.resolved.json")
    result = compiled.run(showprogress=args.show_progress)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(result.metadata), indent=2), encoding="utf-8"
    )
    print(output_dir)
    return 0


def _existing_model_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find model spec file: {resolved}")
    return resolved


def _validate_output_declarations(spec, *, trusted_paths: bool) -> None:
    if spec.analysis is None:
        return
    for observer in spec.analysis.observers:
        if isinstance(observer, CSVSnapshotObserver):
            _validate_relative_output(observer.output_dir, trusted_paths=trusted_paths)
        elif isinstance(observer, ScalarTimeSeriesRecorder):
            _validate_relative_output(observer.output_path, trusted_paths=trusted_paths)


def _resolve_output_paths(spec, output_dir: Path, *, trusted_paths: bool) -> None:
    if spec.analysis is None:
        return
    for observer in spec.analysis.observers:
        if isinstance(observer, CSVSnapshotObserver):
            observer.output_dir = _output_path(
                observer.output_dir, output_dir, trusted_paths=trusted_paths
            )
        elif isinstance(observer, ScalarTimeSeriesRecorder):
            observer.output_path = _output_path(
                observer.output_path, output_dir, trusted_paths=trusted_paths
            )


def _validate_relative_output(path, *, trusted_paths: bool) -> None:
    raw = Path(path)
    if not trusted_paths and (raw.is_absolute() or ".." in raw.parts):
        raise ValueError(
            "Generated output paths must be relative and may not contain '..'; "
            "use --trusted-paths only for trusted local models."
        )


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
