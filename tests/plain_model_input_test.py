"""Hand-authored model inputs exercise parser and CLI contracts."""

import json

import numpy as np
import pytest

from lgca.cli import main
from lgca.model import build_model, model_spec_from_json, model_spec_from_yaml


@pytest.mark.parametrize("file_format", ["json", "yaml"])
@pytest.mark.parametrize("backend", ["classical", "counts", "identity"])
def test_plain_node_arrays_build_and_execute_through_cli(file_format, backend, tmp_path):
    state = {"nodes": [[True, False], [False, True]]}
    if backend == "counts":
        state.update(volume_exclusion=False, nodes=[[2, 0], [0, 3]])
    elif backend == "identity":
        state.update(identity_based=True, nodes=[[7, 0], [0, 23]])
    document = {"schema_version": 1, "model": {"space": {"geometry": "lin"},
                "state": state, "time": {"steps": 1, "seed": 139}}}
    if file_format == "json":
        text = json.dumps(document)
        spec = model_spec_from_json(text)
    else:
        import yaml
        text = yaml.safe_dump(document)
        spec = model_spec_from_yaml(text)
    model = build_model(spec)
    np.testing.assert_array_equal(model.lgca.nodes[model.lgca.nonborder], state["nodes"])
    path = tmp_path / ("input." + file_format)
    path.write_text(text)
    assert main(["run", str(path), "--output", str(tmp_path / "run")]) == 0
    assert json.loads((tmp_path / "run" / "model.resolved.json").read_text())["schema_version"] == 1


@pytest.mark.parametrize("state", [{"nodes": [[True], [False, True]]},
                                  {"nodes": [[-1, 0]], "volume_exclusion": False},
                                  {"nodes": [[7, 7]], "identity_based": True}])
def test_invalid_plain_nodes_report_validation_errors(state, tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"schema_version": 1, "model": {
        "space": {"geometry": "lin"}, "state": state, "time": {"steps": 0},
    }}))
    assert main(["run", str(path), "--output", str(tmp_path / "run")]) != 0
    message = capsys.readouterr().err
    assert "nodes" in message
    assert "Traceback" not in message
