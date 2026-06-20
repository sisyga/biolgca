import csv

import numpy as np
import pytest

import lgca.model as model_api
import lgca.plugins as plugins_api
import lgca.simulation as simulation_api
from lgca import get_lgca
from lgca.model import (
    AnalysisSpec,
    Description,
    MODEL_SPEC_SCHEMA_VERSION,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    describe_model_graph,
    migrate_model_spec_dict,
    model_spec_from_json,
    model_spec_from_yaml,
    model_spec_to_dict,
    model_spec_to_json,
    model_spec_to_yaml,
    load_model_spec,
    run_model,
    save_model_spec,
)
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
from lgca.plugins import ParameterSpec, PluginInfo, describe_plugin, validate_plugin_parameters
from lgca.simulation import CSVSnapshotObserver, DensityRecorder, ScalarTimeSeriesRecorder


def _chemotaxis_spec():
    signal = np.arange(16, dtype=float).reshape(4, 4)
    return ModelSpec(
        description=Description(title="serializable chemotaxis"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1, fields={"signal": signal}),
        time=TimeSpec(steps=2, seed=7),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=1.5,
                            parameters={"field": "signal"},
                        )
                    ],
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[DensityRecorder()]),
    )


def test_model_spec_json_and_yaml_round_trip_numpy_fields(tmp_path):
    spec = _chemotaxis_spec()

    path = tmp_path / "model.json"
    model_spec_to_json(spec, path)
    loaded_json = model_spec_from_json(path)
    loaded_yaml = model_spec_from_yaml(model_spec_to_yaml(spec))

    assert loaded_json.description.title == "serializable chemotaxis"
    assert loaded_json.time.seed == 7
    assert isinstance(loaded_json.dynamics.operators[0], ReorientationSpec)
    assert isinstance(loaded_json.analysis.observers[0], DensityRecorder)
    np.testing.assert_array_equal(loaded_json.state.fields["signal"], spec.state.fields["signal"])
    np.testing.assert_array_equal(loaded_yaml.state.fields["signal"], spec.state.fields["signal"])


def test_model_spec_dict_contains_schema_version_and_migrates_current_version():
    spec_dict = model_spec_to_dict(_chemotaxis_spec())

    assert spec_dict["schema_version"] == MODEL_SPEC_SCHEMA_VERSION
    assert migrate_model_spec_dict(spec_dict) == spec_dict


def test_parameter_contracts_validate_required_allowed_and_builtin_probability():
    info = PluginInfo(
        name="test.operator",
        operator_kind="birth_death",
        backend_families=("classical",),
        parameters={
            "mode": ParameterSpec(
                required=True,
                type_label="string",
                allowed_values=("fast", "slow"),
            )
        },
    )

    with pytest.raises(ValueError, match="mode.*required"):
        validate_plugin_parameters(info, {})
    with pytest.raises(ValueError, match="mode.*fast, slow"):
        validate_plugin_parameters(info, {"mode": "medium"})

    birth_rate = describe_plugin("birth_death").parameter_specs["birth_rate"]
    assert birth_rate.type_label == "probability"

    spec = _chemotaxis_spec()
    bad_operator = {"name": "birth_death", "parameters": {"birth_rate": -0.1}}
    bad_spec = ModelSpec(
        space=spec.space,
        state=spec.state,
        time=spec.time,
        dynamics=InteractionPipelineSpec(operators=[bad_operator], propagation=False),
    )
    with pytest.raises(ValueError, match="birth_rate.*probability"):
        run_model(bad_spec, showprogress=False)


def test_runtime_metadata_and_csv_observers_capture_outputs(tmp_path):
    snapshot = CSVSnapshotObserver(kind="density", output_dir=tmp_path / "snapshots")
    series = ScalarTimeSeriesRecorder(output_path=tmp_path / "population.csv")
    spec = ModelSpec(
        description=Description(title="diagnostic run"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=1, seed=11),
        dynamics=InteractionPipelineSpec(operators=[{"name": "classical.random_walk"}]),
        analysis=AnalysisSpec(observers=[snapshot, series]),
    )

    result = run_model(spec, showprogress=False)

    runtime = result.metadata["runtime"]
    assert result.metadata["model_spec_schema_version"] == MODEL_SPEC_SCHEMA_VERSION
    assert result.metadata["biolgca_version"]
    assert runtime["elapsed_seconds"] >= 0.0
    assert any(entry["name"] == "classical.random_walk" for entry in runtime["operator_timings"])
    assert all(path in result.metadata["output_paths"] for path in [str(series.output_path), *map(str, snapshot.paths)])

    with series.output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [0, 1]
    assert "population" in rows[0]
    assert len(snapshot.paths) == 2
    assert all(path.exists() for path in snapshot.paths)


def test_describe_model_graph_exposes_fields_plugins_observers_and_outputs():
    graph = describe_model_graph(_chemotaxis_spec())
    node_ids = {node["id"] for node in graph["nodes"]}
    edges = {(edge["source"], edge["target"]) for edge in graph["edges"]}

    assert "field:signal" in node_ids
    assert "operator:0:reorientation.boltzmann" in node_ids
    assert "observer:0:DensityRecorder" in node_ids
    assert ("field:signal", "operator:0:reorientation.boltzmann") in edges
    assert ("operator:0:reorientation.boltzmann", "output:nodes") in edges
    assert ("observer:0:DensityRecorder", "output:dens_t") in edges


def test_curated_example_specs_smoke_run():
    from lgca.examples import all_example_specs

    examples = all_example_specs()
    assert set(examples) == {
        "alignment",
        "chemotaxis",
        "identity_tumor_growth",
        "multispecies_birth_death",
        "random_walk",
    }

    for name, spec in examples.items():
        result = run_model(spec, showprogress=False)
        assert result.metadata["title"]
        assert result.metadata["steps"] == spec.time.steps, name


def test_model_api_has_explicit_beginner_facing_public_exports():
    assert "ModelSpec" in model_api.__all__
    assert "run_model" in model_api.__all__
    assert "save_model_spec" in model_api.__all__
    assert "load_model_spec" in model_api.__all__
    assert "_to_jsonable" not in model_api.__all__


def test_plugin_and_simulation_apis_have_explicit_public_exports():
    assert "ParameterSpec" in plugins_api.__all__
    assert "validate_plugin_parameters" in plugins_api.__all__
    assert "_validate_parameter_value" not in plugins_api.__all__

    assert "CSVSnapshotObserver" in simulation_api.__all__
    assert "ScalarTimeSeriesRecorder" in simulation_api.__all__
    assert "_snapshot_values" not in simulation_api.__all__


def test_save_and_load_model_spec_choose_format_from_file_suffix(tmp_path):
    spec = _chemotaxis_spec()
    json_path = tmp_path / "model.json"
    yaml_path = tmp_path / "model.yaml"

    assert save_model_spec(spec, json_path) == json_path
    assert save_model_spec(spec, yaml_path) == yaml_path

    json_loaded = load_model_spec(json_path)
    yaml_loaded = load_model_spec(yaml_path)

    assert json_loaded.description.title == spec.description.title
    assert yaml_loaded.description.title == spec.description.title
    np.testing.assert_array_equal(json_loaded.state.fields["signal"], spec.state.fields["signal"])
    np.testing.assert_array_equal(yaml_loaded.state.fields["signal"], spec.state.fields["signal"])


def test_save_and_load_model_spec_report_friendly_format_errors(tmp_path):
    with pytest.raises(ValueError, match=r"Use a \.json, \.yaml, or \.yml file"):
        save_model_spec(_chemotaxis_spec(), tmp_path / "model.txt")

    with pytest.raises(FileNotFoundError, match="Could not find model spec file"):
        load_model_spec(tmp_path / "missing.json")


def test_example_lookup_api_lists_names_and_reports_unknown_examples():
    from lgca.examples import example_names, get_example_spec

    assert "random_walk" in example_names()
    assert get_example_spec("random_walk").description.title == "Random walk example"
    with pytest.raises(ValueError, match=r"Unknown example 'randm_walk'.*random_walk"):
        get_example_spec("randm_walk")


def test_original_get_lgca_quick_start_still_works():
    lgca = get_lgca(seed=1)
    lgca.timeevo(timesteps=1, showprogress=False)

    assert lgca.total_population() >= 0
