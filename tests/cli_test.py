import importlib
import json
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    save_model_spec,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import CSVSnapshotObserver


@pytest.mark.parametrize("target", ["model.resolved.json", "metadata.json", "measurements.npz",
                                    "resources/initial_state.npz", "MODEL.RESOLVED.JSON",
                                    "model.resolved.json/nested.csv"])
def test_output_collision_preflight_preserves_archive_sentinels(tmp_path, target, monkeypatch, capsys):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "model.resolved.json"
    sentinel.write_text("preserve me")
    model = _write_tiny_model(tmp_path / "input.json", observer=CSVSnapshotObserver(filename=target))

    def forbidden(*args, **kwargs):
        raise AssertionError("collision must be rejected before model construction")

    monkeypatch.setattr(importlib.import_module("lgca.cli"), "build_model", forbidden)
    assert _main(["run", str(model), "--output", str(output), "--overwrite"]) == 2
    assert "collision" in capsys.readouterr().err
    assert sentinel.read_text() == "preserve me"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("same_observer", [False, True])
def test_duplicate_observer_targets_rejected_before_output(tmp_path, same_observer, capsys):
    from lgca.simulation import ScalarTimeSeriesRecorder

    observers = [CSVSnapshotObserver(filename="shared.csv")]
    if not same_observer:
        observers.append(ScalarTimeSeriesRecorder(output_path="shared.csv"))
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(steps=1),
                     analysis=AnalysisSpec(observers=observers))
    path = save_model_spec(spec, tmp_path / "input.json")
    output = tmp_path / "run"
    assert _main(["run", str(path), "--output", str(output)]) == 2
    assert "collision" in capsys.readouterr().err
    assert not output.exists()


def test_cli_subprocess_persists_measurements_and_sample_steps(tmp_path):
    from lgca.simulation import NodeRecorder, DensityRecorder, PopulationRecorder, Schedule

    schedule = Schedule(steps=[0, 2])
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=3),
                     state=StateSpec(density=1), time=TimeSpec(steps=2, seed=114),
                     analysis=AnalysisSpec(observers=[NodeRecorder(schedule),
                         DensityRecorder(schedule), PopulationRecorder(schedule)]))
    path = save_model_spec(spec, tmp_path / "model.json")
    output = tmp_path / "run"
    result = subprocess.run([sys.executable, "-m", "lgca.cli", "run", str(path),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with np.load(output / "measurements.npz", allow_pickle=False) as data:
        for name in ("nodes_steps", "dens_steps", "n_steps"):
            np.testing.assert_array_equal(data[name], [0, 2])
        np.testing.assert_array_equal(data["nodes_t"].sum(axis=-1), data["dens_t"])
        np.testing.assert_array_equal(data["dens_t"].sum(axis=-1), data["n_t"])


def test_moved_cli_archive_replays_companion_resource_without_touching_first_run(tmp_path):
    nodes = np.array([[True, False], [False, True], [True, False]])
    np.savez(tmp_path / "input.npz", nodes=nodes)
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=3),
        state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": "input.npz"}}),
        time=TimeSpec(steps=2, seed=115),
        dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=[CSVSnapshotObserver(output_dir="snapshots")]))
    model = save_model_spec(spec, tmp_path / "model.json")
    first = tmp_path / "first"
    assert _main(["run", str(model), "--output", str(first)]) == 0
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    moved = tmp_path / "moved"
    shutil.copytree(first, moved)
    (tmp_path / "input.npz").unlink()
    archived = moved / "model.resolved.json"
    assert _main(["validate", str(archived)]) == 0
    second = tmp_path / "second"
    assert _main(["run", str(archived), "--output", str(second)]) == 0
    for relative, content in before.items():
        assert (first / relative).read_bytes() == content
    for original in (first / "snapshots").glob("*.csv"):
        assert (second / "snapshots" / original.name).read_bytes() == original.read_bytes()


def _main(argv):
    return importlib.import_module("lgca.cli").main(argv)


def _write_tiny_model(path, *, operator="random_walk", observer=None):
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 4), boundary="periodic"),
        state=StateSpec(density=1.0, restchannels=1),
        time=TimeSpec(steps=1, seed=51),
        dynamics=InteractionPipelineSpec(operators=[{"name": operator}]),
        analysis=AnalysisSpec(observers=[] if observer is None else [observer]),
    )
    save_model_spec(spec, path)
    return path


def test_examples_list_and_export_produce_a_valid_template(tmp_path, capsys):
    assert _main(["examples", "list"]) == 0
    assert "random_walk" in capsys.readouterr().out

    target = tmp_path / "random-walk.json"
    assert _main(["examples", "export", "random_walk", str(target)]) == 0
    exported = json.loads(target.read_text(encoding="utf-8"))
    assert exported["schema_version"] == 1
    assert exported["model"]["description"]["title"] == "Random walk example"


def test_validate_resolves_plugins_without_running_trajectory(tmp_path, capsys):
    model_path = _write_tiny_model(
        tmp_path / "model.json",
        observer=CSVSnapshotObserver(output_dir="snapshots"),
    )

    assert _main(["validate", str(model_path)]) == 0
    assert "valid" in capsys.readouterr().out.lower()
    assert not (tmp_path / "snapshots").exists()


def test_validate_rejects_unregistered_external_plugin(tmp_path, capsys):
    known_path = _write_tiny_model(tmp_path / "known.json")
    data = json.loads(known_path.read_text(encoding="utf-8"))
    data["model"]["dynamics"]["operators"][0]["name"] = "external.not_imported"
    model_path = tmp_path / "external.json"
    model_path.write_text(json.dumps(data), encoding="utf-8")

    assert _main(["validate", str(model_path)]) == 2
    assert "external.not_imported" in capsys.readouterr().err


def test_run_writes_resolved_spec_metadata_and_observer_outputs(tmp_path):
    model_path = _write_tiny_model(
        tmp_path / "model.json",
        observer=CSVSnapshotObserver(output_dir="snapshots"),
    )
    first_dir = tmp_path / "run-1"
    second_dir = tmp_path / "run-2"

    assert _main(["run", str(model_path), "--output", str(first_dir)]) == 0
    assert _main(["run", str(model_path), "--output", str(second_dir)]) == 0

    assert (first_dir / "model.resolved.json").exists()
    metadata = json.loads((first_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 51
    assert metadata["steps"] == 1
    first_csv = (first_dir / "snapshots" / "density_00001.csv").read_text()
    second_csv = (second_dir / "snapshots" / "density_00001.csv").read_text()
    assert first_csv == second_csv


@pytest.mark.parametrize("filename", ["../sentinel.csv", "..\\sentinel.csv", "/sentinel.csv",
                                      "C:\\sentinel.csv", "C:sentinel.csv", "\\sentinel.csv"])
def test_csv_snapshot_rejects_filename_escape_before_writing(tmp_path, filename):
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text("untouched", encoding="utf-8")
    model = _write_tiny_model(tmp_path / "model.json",
                              observer=CSVSnapshotObserver(filename=filename))
    assert _main(["run", str(model), "--output", str(tmp_path / "run")]) == 2
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (tmp_path / "run").exists()


def test_csv_snapshot_checks_final_resolved_target(tmp_path, monkeypatch):
    output = tmp_path / "run"
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text("untouched", encoding="utf-8")
    model = _write_tiny_model(tmp_path / "model.json",
                              observer=CSVSnapshotObserver(filename="snapshot.csv"))
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == output / "snapshot.csv":
            return sentinel
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert _main(["run", str(model), "--output", str(output)]) == 2
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_csv_snapshot_rejects_resolved_symlink_escape(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text("untouched", encoding="utf-8")
    try:
        (output / "snapshot.csv").symlink_to(sentinel)
    except OSError:
        pytest.skip("Creating symlinks requires OS privileges")
    model = _write_tiny_model(tmp_path / "model.json",
                              observer=CSVSnapshotObserver(filename="snapshot.csv"))
    assert _main(["run", str(model), "--output", str(output), "--overwrite"]) == 2
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_run_rejects_existing_output_without_explicit_overwrite(tmp_path, capsys):
    model_path = _write_tiny_model(tmp_path / "model.json")
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    assert _main(["run", str(model_path), "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_run_reuses_existing_output_only_with_explicit_overwrite(tmp_path):
    model_path = _write_tiny_model(tmp_path / "model.json")
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    assert _main(
        ["run", str(model_path), "--output", str(output), "--overwrite"]
    ) == 0
    assert (output / "model.resolved.json").exists()
    assert (output / "metadata.json").exists()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_run_rejects_output_path_escape(tmp_path, capsys):
    model_path = _write_tiny_model(
        tmp_path / "model.json",
        observer=CSVSnapshotObserver(output_dir="../escape"),
    )

    assert _main(["run", str(model_path), "--output", str(tmp_path / "run")]) == 2
    assert "trusted-paths" in capsys.readouterr().err
    assert not (tmp_path / "escape").exists()


def test_validate_rejects_initializer_resource_escape(tmp_path, capsys):
    outside = tmp_path.parent / "outside.npz"
    np.savez(outside, nodes=np.zeros((3, 4, 5), dtype=bool))
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 4)),
        state=StateSpec(
            restchannels=1,
            initializer={"name": "from_npz", "parameters": {"path": "../outside.npz"}},
        ),
        time=TimeSpec(steps=0, seed=1),
    )
    model_path = save_model_spec(spec, tmp_path / "unsafe.json")

    assert _main(["validate", str(model_path)]) == 2
    assert "trusted-paths" in capsys.readouterr().err


def test_validate_allows_explicit_trusted_initializer_resource(tmp_path):
    outside = tmp_path.parent / "trusted-outside.npz"
    np.savez(outside, nodes=np.zeros((3, 4, 5), dtype=bool))
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(3, 4)),
        state=StateSpec(
            restchannels=1,
            initializer={
                "name": "from_npz",
                "parameters": {"path": str(outside.resolve())},
            },
        ),
        time=TimeSpec(steps=0, seed=1),
    )
    model_path = save_model_spec(spec, tmp_path / "trusted.json")

    assert _main(["validate", str(model_path), "--trusted-paths"]) == 0
