"""Where a result came from: provenance in run and sweep metadata, inputs read once, and the hash pins in
files that the library writes (array files of model files, command-line archives)."""

import hashlib
import json

import numpy as np
import pytest

from lgca import interaction
from lgca.model import (
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    load_model_spec,
    save_model_spec,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.plugins import default_registry
from lgca.study import sweep


@pytest.fixture(autouse=True)
def _restore_registry():
    plugins, aliases = dict(default_registry._plugins), dict(default_registry._aliases)
    yield
    default_registry._plugins, default_registry._aliases = plugins, aliases


def _spec(steps=2, **state):
    state.setdefault("density", 0.3)
    return ModelSpec(space=SpaceSpec(geometry="lin", dims=20), state=StateSpec(restchannels=1, **state),
                     time=TimeSpec(steps=steps, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[
                         {"name": "birth_death", "parameters": {"birth_rate": 0.2}}, {"name": "random_walk"}]))


def _npz_spec(path, steps=2):
    return ModelSpec(space=SpaceSpec(geometry="lin", dims=4), time=TimeSpec(steps=steps, seed=1),
                     state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": path}}),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]))


def test_a_model_records_versions_operators_with_defaults_and_the_source_of_its_rules():
    @interaction(kind="birth_death", families="classical", name="thinning")
    def thinning(state, p=0.1):
        state.remove_cells(p)

    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=10), state=StateSpec(density=0.3),
                                  dynamics=InteractionPipelineSpec(operators=[
                                      {"name": "birth_death", "parameters": {"birth_rate": 0.2}}, thinning()])))
    provenance = model.metadata["provenance"]

    assert set(provenance["versions"]) >= {"biolgca", "python", "numpy", "scipy"}
    assert provenance["platform"] and provenance["inputs"] == {}
    birth_death, own = provenance["operators"]
    assert birth_death["parameters"]["birth_rate"] == 0.2 and "death_rate" in birth_death["parameters"]
    assert own == {"name": "thinning", "kind": "birth_death", "parameters": {"p": 0.1}}
    import inspect

    source = inspect.getsource(thinning.function).encode("utf-8")
    assert provenance["rules"]["thinning"]["sha256"] == hashlib.sha256(source).hexdigest()
    model.reconfigure({"dynamics.operators[1].parameters.p": 0.3})
    assert model.metadata["provenance"]["operators"][1]["parameters"] == {"p": 0.3}


def test_a_sweep_reads_its_input_files_once(tmp_path):
    np.savez(tmp_path / "start.npz", nodes=np.ones((4, 2), dtype=bool))
    first = []

    def cells_at_start(result):  # the file is rewritten after the first run: later runs must not see it
        if not first:
            first.append(True)
            np.savez(tmp_path / "start.npz", nodes=np.zeros((4, 2), dtype=bool))
        return int(result.data["population"][0])

    original = hashlib.sha256((tmp_path / "start.npz").read_bytes()).hexdigest()
    table = sweep(_npz_spec("start.npz"), seeds=range(3), resource_base=tmp_path,
                  measure={"start": cells_at_start, "population": "population"}, showprogress=False)

    assert table["start"].tolist() == [8, 8, 8]
    assert table.attrs["provenance"]["inputs"] == {"start.npz": original}
    assert "random_walk" in table.attrs["provenance"]["rules"]


def test_a_model_records_the_hash_of_the_file_it_started_from(tmp_path):
    np.savez(tmp_path / "start.npz", nodes=np.ones((4, 2), dtype=bool))
    model = build_model(_npz_spec("start.npz"), resource_base=tmp_path)

    expected = hashlib.sha256((tmp_path / "start.npz").read_bytes()).hexdigest()
    assert model.metadata["provenance"]["inputs"] == {"start.npz": expected}


def test_a_shared_array_file_that_another_model_overwrote_is_reported(tmp_path):
    nodes = np.zeros((300, 3), dtype=bool)
    nodes[:100, 0] = True
    big = ModelSpec(space=SpaceSpec(geometry="lin", dims=300), state=StateSpec(nodes=nodes))
    save_model_spec(big, tmp_path / "model.json")
    load_model_spec(tmp_path / "model.json")  # the pin matches
    nodes = nodes.copy()
    nodes[:200, 0] = True  # another model of the same shape, saved next to it under the same name
    save_model_spec(ModelSpec(space=big.space, state=StateSpec(nodes=nodes)), tmp_path / "model.yaml")

    with pytest.warns(UserWarning, match=r"array 'array_0' in model.arrays.npz changed after the model file"):
        load_model_spec(tmp_path / "model.json")


def test_the_command_line_archive_records_hashes_and_reports_edits(tmp_path):
    from lgca.cli import main

    save_model_spec(_spec(), tmp_path / "model.json")
    output = tmp_path / "run"
    assert main(["run", str(tmp_path / "model.json"), "--output", str(output)]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    provenance = metadata["provenance"]
    model_bytes = (tmp_path / "model.json").read_bytes()
    assert provenance["model_file"]["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    archived = (output / "model.resolved.json").read_bytes()
    assert provenance["archive"]["model.resolved.json"] == hashlib.sha256(archived).hexdigest()
    assert "operators" in provenance and "vcs" in provenance

    assert main(["validate", str(output / "model.resolved.json")]) == 0  # unchanged: no warning
    text = (output / "model.resolved.json").read_text().replace('"birth_rate": 0.2', '"birth_rate": 0.4')
    (output / "model.resolved.json").write_text(text)
    with pytest.warns(UserWarning, match="model.resolved.json changed after the run that wrote it"):
        assert main(["validate", str(output / "model.resolved.json")]) == 0


def test_the_command_line_runs_from_the_copies_of_its_inputs(tmp_path):
    from lgca.cli import main

    np.savez(tmp_path / "start.npz", nodes=np.ones((4, 2), dtype=bool))
    save_model_spec(_npz_spec("start.npz"), tmp_path / "model.json")
    output = tmp_path / "run"
    assert main(["run", str(tmp_path / "model.json"), "--output", str(output)]) == 0

    metadata = json.loads((output / "metadata.json").read_text())
    copy = (output / "resources" / "initial_state.npz").read_bytes()
    assert metadata["provenance"]["inputs"] == {"resources/initial_state.npz": hashlib.sha256(copy).hexdigest()}
    assert copy == (tmp_path / "start.npz").read_bytes()
    sweep_output = tmp_path / "sweep"
    assert main(["sweep", str(tmp_path / "model.json"), "--seeds", "0:2", "--output", str(sweep_output)]) == 0
    description = json.loads((sweep_output / "sweep.json").read_text())
    assert description["provenance"]["model_file"]["sha256"] == hashlib.sha256(
        (tmp_path / "model.json").read_bytes()).hexdigest()
    assert list(description["provenance"]["inputs"].values()) == [hashlib.sha256(copy).hexdigest()]
