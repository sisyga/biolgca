"""Large arrays of a model file go to an array file next to it; tuples are written as lists."""

import json

import numpy as np
import pytest

from lgca.model import (
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    load_model_spec,
    model_spec_from_json,
    run_model,
    save_model_spec,
)
from lgca.pipeline import InteractionPipelineSpec


def _spec(dims=(30, 30)):
    rng = np.random.default_rng(1)
    nodes = rng.random(dims + (5,)) < 0.3
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=dims),
        state=StateSpec(nodes=nodes, restchannels=1, fields={"signal": rng.random(dims), "small": np.ones(3)}),
        time=TimeSpec(steps=3, seed=2),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "chemotaxis", "parameters": {"beta": 1.0, "field": "signal"}}]))


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_large_arrays_go_to_an_array_file_and_come_back(tmp_path, suffix):
    pytest.importorskip("yaml") if suffix == ".yaml" else None
    spec = _spec()
    path = save_model_spec(spec, tmp_path / f"model{suffix}")
    array_file = tmp_path / "model.arrays.npz"
    assert array_file.exists()
    text = path.read_text()
    assert "__ndarray_file__" in text and len(text) < 5000  # the 4500 channel values are not in the text
    loaded = load_model_spec(path)
    np.testing.assert_array_equal(loaded.state.nodes, spec.state.nodes)
    np.testing.assert_array_equal(loaded.state.fields["signal"], spec.state.fields["signal"])
    np.testing.assert_array_equal(loaded.state.fields["small"], np.ones(3))
    assert loaded.space.dims == (30, 30)
    np.testing.assert_array_equal(run_model(loaded, showprogress=False).lgca.nodes,
                                  run_model(spec, showprogress=False).lgca.nodes)


def test_small_models_and_the_inline_option_write_one_file(tmp_path):
    save_model_spec(_spec(dims=(4, 4)), tmp_path / "small.json")
    assert not (tmp_path / "small.arrays.npz").exists()
    save_model_spec(_spec(), tmp_path / "model.json")
    assert (tmp_path / "model.arrays.npz").exists()
    save_model_spec(_spec(), tmp_path / "model.json", max_inline_array=None)  # replaces the stale array file
    assert not (tmp_path / "model.arrays.npz").exists()
    data = json.loads((tmp_path / "model.json").read_text())
    assert data["model"]["space"]["dims"] == [30, 30]  # a list, not {"__tuple__": ...}
    np.testing.assert_array_equal(load_model_spec(tmp_path / "model.json").state.nodes, _spec().state.nodes)


def test_references_need_the_array_file_next_to_the_model(tmp_path):
    path = save_model_spec(_spec(), tmp_path / "model.json")
    text = path.read_text()
    with pytest.raises(ValueError, match="load it from its file"):
        model_spec_from_json(text)
    (tmp_path / "model.arrays.npz").unlink()
    with pytest.raises(FileNotFoundError, match="keep the two files together"):
        load_model_spec(path)
    data = json.loads(text)
    data["model"]["state"]["nodes"]["__ndarray_file__"] = "../elsewhere.npz"
    (tmp_path / "moved.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="next to the model file"):
        load_model_spec(tmp_path / "moved.json")


def test_files_with_tuples_of_earlier_versions_still_load(tmp_path):
    save_model_spec(_spec(dims=(4, 4)), tmp_path / "model.json")
    data = json.loads((tmp_path / "model.json").read_text())
    data["model"]["space"]["dims"] = {"__tuple__": [4, 4]}
    (tmp_path / "old.json").write_text(json.dumps(data))
    assert load_model_spec(tmp_path / "old.json").space.dims == (4, 4)


def test_the_command_line_keeps_the_array_file_of_the_resolved_model(tmp_path):
    from lgca.cli import main

    save_model_spec(_spec(), tmp_path / "model.json")
    assert main(["run", str(tmp_path / "model.json"), "--output", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "model.resolved.arrays.npz").exists()
    resolved = load_model_spec(tmp_path / "run" / "model.resolved.json")
    np.testing.assert_array_equal(resolved.state.nodes, _spec().state.nodes)
