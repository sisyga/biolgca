import csv
import json

from profiling import BenchmarkScenario, default_scenarios, run_scenario, write_results


EXPECTED_RESULT_KEYS = {
    "scenario",
    "family",
    "geometry",
    "interaction",
    "dims",
    "density",
    "seed",
    "steps",
    "repeats",
    "wall_seconds",
    "wall_seconds_per_step",
    "peak_memory_bytes",
    "particles_final",
    "observer_policy",
    "python_version",
    "numpy_version",
    "biolgca_version",
}


def test_default_benchmarks_cover_maintained_state_families():
    assert {scenario.family for scenario in default_scenarios()} == {
        "classical_ve",
        "nove",
        "identity_ve",
        "identity_nove",
        "multispecies",
    }


def test_benchmark_smoke_emits_stable_machine_readable_schema(tmp_path):
    scenario = BenchmarkScenario(
        name="smoke_classical_square",
        family="classical_ve",
        geometry="square",
        dims=(4, 5),
        interaction="random_walk",
        density=0.3,
        steps=2,
        seed=17,
        kwargs={"ve": True, "restchannels": 1},
    )

    result = run_scenario(scenario, repeats=1)

    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["dims"] == [4, 5]
    assert result["steps"] == 2
    assert result["wall_seconds"] >= 0.0
    assert result["peak_memory_bytes"] >= 0

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    write_results([result], json_path)
    write_results([result], csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == [result]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["scenario"] == scenario.name


def test_benchmark_runs_native_modelspec_operator_path():
    scenario = BenchmarkScenario(
        name="smoke_native_birth_death",
        family="classical_ve",
        geometry="square",
        dims=(4, 5),
        interaction="birth_death",
        density=0.3,
        steps=2,
        seed=19,
        kwargs={"restchannels": 1},
        model_spec_operator={
            "name": "birth_death",
            "parameters": {"birth_rate": 0.2, "death_rate": 0.05, "capacity": 5},
        },
    )

    result = run_scenario(scenario, repeats=1)

    assert result["scenario"] == scenario.name
    assert result["interaction"] == "birth_death"
