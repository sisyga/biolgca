"""vary and sweep: changing a model by paths, and tables of runs over parameters and seeds."""

import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.study import final_population, resolve_path, sweep, vary


def _growth(steps=15, seed=1):
    return ModelSpec(
        space=SpaceSpec(geometry="lin", dims=60), state=StateSpec(density=0.2, restchannels=1),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 0.1}},
            {"name": "go_or_rest"},
            {"name": "random_walk", "parameters": {"channels": "velocity"}},
        ]))


def _movement():
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(10, 10)), state=StateSpec(density=0.3, restchannels=1),
        time=TimeSpec(steps=3, seed=1),
        dynamics=InteractionPipelineSpec(operators=[
            ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=1.0),
                                     ReorientationTermSpec("resting", parameters={"probability": {
                                         "cues": [{"name": "density", "kappa": 3.0, "theta": 0.5}]}})]),
        ]))


def test_vary_changes_a_copy_by_path():
    spec = _growth()
    variant = vary(spec, {
        "time.steps": 30,
        "dynamics.operators[0].parameters.birth_rate": 0.2,
        "dynamics.operators[go_or_rest].kappa": 2.0,  # by name, parameters implied
        "dynamics.operators[-1].parameters.channels": "all",
    })
    assert variant.time.steps == 30
    assert variant.dynamics.operators[0]["parameters"] == {"birth_rate": 0.2}
    assert variant.dynamics.operators[1] == {"name": "go_or_rest", "parameters": {"kappa": 2.0}}
    assert variant.dynamics.operators[2]["parameters"] == {"channels": "all"}
    # the original is unchanged
    assert spec.time.steps == 15 and spec.dynamics.operators[1] == {"name": "go_or_rest"}


def test_vary_reaches_terms_and_nested_parameters():
    spec = _movement()
    variant = vary(spec, {
        "dynamics.operators[0].terms[polar_alignment].beta": 2.5,
        "dynamics.operators[0].terms[resting].probability.cues[0].kappa": 6.0,
        "dynamics.operators[0].sweeps": 5,
    })
    terms = variant.dynamics.operators[0].terms
    assert terms[0].beta == 2.5
    assert terms[1].parameters["probability"]["cues"][0] == {"name": "density", "kappa": 6.0, "theta": 0.5}
    assert variant.dynamics.operators[0].parameters == {"sweeps": 5}
    assert spec.dynamics.operators[0].terms[1].parameters["probability"]["cues"][0]["kappa"] == 3.0
    run_model(variant, showprogress=False)


def test_short_names_find_the_one_place_with_that_name():
    spec = _growth()
    assert resolve_path(spec, "steps") == "time.steps"
    assert resolve_path(spec, "birth_rate") == "dynamics.operators[0].parameters.birth_rate"
    assert resolve_path(spec, "kappa") == "dynamics.operators[1].parameters.kappa"  # a default, not given
    assert resolve_path(spec, "restchannels") == "state.restchannels"
    with pytest.raises(KeyError, match=r"several places: state.density, dynamics.operators\[1\]"):
        resolve_path(spec, "density")  # the initial density, and the density cue of go_or_rest
    with pytest.raises(KeyError, match="several places"):
        resolve_path(spec, "channels")  # birth_death and random_walk
    with pytest.raises(KeyError, match="no field or parameter"):
        resolve_path(spec, "gamma")
    assert resolve_path(_movement(), "probability") == "dynamics.operators[0].terms[1].parameters.probability"
    with pytest.raises(KeyError, match="several places"):
        resolve_path(_movement(), "beta")  # both terms


@pytest.mark.parametrize("changes, message", [
    ({"time.stpes": 3}, "did you mean 'steps'"),
    ({"dynamics.operators[7].kappa": 1}, "has 3 entries"),
    ({"dynamics.operators[chemotaxis].beta": 1}, "no entry named 'chemotaxis'"),
    ({"time..steps": 3}, "invalid path"),
    ({"time.steps[0]": 3}, "not a list"),
])
def test_vary_explains_wrong_paths(changes, message):
    with pytest.raises((KeyError, ValueError), match=message):
        vary(_growth(), changes)


def test_a_sweep_has_one_row_per_run_in_order():
    table = sweep(_growth(), grid={"birth_rate": [0.0, 0.3], "steps": [5, 10]}, seeds=[4, 5],
                  showprogress=False)
    assert table.columns.tolist() == ["birth_rate", "steps", "seed", "population"]
    assert table[["birth_rate", "steps", "seed"]].values.tolist() == [
        [b, s, seed] for b in (0.0, 0.3) for s in (5, 10) for seed in (4, 5)]
    expected = final_population(run_model(vary(_growth(seed=5), {"birth_rate": 0.3, "steps": 10}),
                                          showprogress=False))
    assert table.population.iloc[-1] == expected
    assert table.attrs["paths"]["birth_rate"] == "dynamics.operators[0].parameters.birth_rate"
    assert table.attrs["biolgca_version"]


def test_explicit_combinations_and_the_seed_of_the_model():
    table = sweep(_growth(seed=9), grid=[{"birth_rate": 0.0}, {"birth_rate": 0.2, "kappa": 1.0}],
                  showprogress=False)
    assert table.seed.tolist() == [9, 9]
    assert np.isnan(table.kappa.iloc[0]) and table.kappa.iloc[1] == 1.0


@pytest.mark.parametrize("backend", ["processes", "threads"])
def test_results_do_not_depend_on_the_number_of_workers(backend):
    grid = {"birth_rate": [0.0, 0.2, 0.4]}
    measure = {"final": final_population, "population": "population"}
    serial = sweep(_growth(), grid=grid, seeds=range(3), measure=measure, showprogress=False)
    parallel = sweep(_growth(), grid=grid, seeds=range(3), measure=measure, n_jobs=3, backend=backend,
                     showprogress=False)
    pd.testing.assert_frame_equal(serial.drop(columns="population"), parallel.drop(columns="population"))
    for a, b in zip(serial.population, parallel.population):
        np.testing.assert_array_equal(a, b)


def test_a_recording_is_measured_as_a_time_series_and_long_tables_spread_it():
    measure = {"population": "population", "final": final_population}
    wide = sweep(_growth(steps=4), grid={"birth_rate": [0.0, 0.5]}, seeds=[1], measure=measure,
                 showprogress=False)
    assert len(wide) == 2 and len(wide.population.iloc[0]) == 5  # the recorder was added: steps 0..4
    assert wide.population.iloc[1][-1] == wide.final.iloc[1]

    long = sweep(_growth(steps=4), grid={"birth_rate": [0.0, 0.5]}, seeds=[1], measure=measure, long=True,
                 showprogress=False)
    assert long.columns.tolist() == ["birth_rate", "seed", "step", "final", "population"]
    assert len(long) == 10 and long.step.tolist() == [0, 1, 2, 3, 4] * 2
    np.testing.assert_array_equal(long.population.values[5:], wide.population.iloc[1])
    assert (long.final.values[5:] == wide.final.iloc[1]).all()


def test_functions_may_return_series_indexed_by_step():
    def every_other(result):
        steps = result.data.steps("population")
        return pd.Series(result.data["population"], index=steps).iloc[::2]

    long = sweep(_growth(steps=4), seeds=[1], measure={"population": "population", "thinned": every_other},
                 long=True, showprogress=False)
    assert long.step.tolist() == [0, 1, 2, 3, 4]
    assert long.thinned.isna().tolist() == [False, True, False, True, False]
    with pytest.raises(ValueError, match="gave an array"):
        sweep(_growth(steps=4), seeds=[1], measure={"population": "population",
                                                    "raw": lambda result: result.data["population"]},
              long=True, showprogress=False)


def test_processes_need_functions_they_can_import():
    with pytest.raises(TypeError, match="cannot be sent to worker processes"):
        sweep(_growth(), grid={"birth_rate": [0.0, 0.1]}, measure={"n": lambda result: 1}, n_jobs=2,
              showprogress=False)
    table = sweep(_growth(), grid={"birth_rate": [0.0, 0.1]}, measure={"n": lambda result: 1}, n_jobs=2,
                  backend="threads", showprogress=False)
    assert table.n.tolist() == [1, 1]


@pytest.mark.parametrize("method", ["default", "spawn"])  # spawn: as on Windows
def test_worker_processes_import_the_plugins(tmp_path, monkeypatch, method):
    import multiprocessing

    from lgca import study

    if method != "default":
        monkeypatch.setattr(study, "_start_method", lambda: multiprocessing.get_context(method))
    (tmp_path / "sweep_plugin_rules.py").write_text(textwrap.dedent('''
        from lgca import interaction

        @interaction(kind="birth_death", families=("classical",), name="sweep_test_cull")
        def cull(state, fraction=0.5):
            """Cells die with probability fraction."""
            state.remove_cells(fraction)
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = vary(_growth(), {"dynamics.operators[0]": {"name": "sweep_test_cull"}})
    table = sweep(spec, grid={"fraction": [0.0, 1.0]}, n_jobs=2, plugins=["sweep_plugin_rules"],
                  showprogress=False)
    assert table.population.iloc[1] == 0 and table.population.iloc[0] > 0
    sys.modules.pop("sweep_plugin_rules", None)


def test_a_failed_run_is_named():
    with pytest.raises(RuntimeError, match=r"the run with birth_rate=2\.0, seed=3 failed"):
        sweep(_growth(), grid={"birth_rate": [0.1, 2.0]}, seeds=[3], showprogress=False)
    with pytest.raises(ValueError, match="n_jobs"):
        sweep(_growth(), n_jobs=0, showprogress=False)
    with pytest.raises(ValueError, match="backend"):
        sweep(_growth(), n_jobs=2, backend="gpu", showprogress=False)


def test_every_run_has_its_own_observers_and_their_files_are_discarded(tmp_path, monkeypatch):
    from lgca.model import AnalysisSpec
    from lgca.simulation import PopulationRecorder, ScalarTimeSeriesRecorder, Schedule

    monkeypatch.chdir(tmp_path)
    series = ScalarTimeSeriesRecorder(metrics={"total": lambda lgca: int(lgca.cell_density[lgca.nonborder].sum())},
                                      schedule=Schedule(every=5),
                                      output_path="series.csv")
    spec = vary(_growth(), {"analysis": AnalysisSpec(observers=(PopulationRecorder(), series))})
    table = sweep(spec, grid={"steps": [5, 20]}, seeds=range(2), measure={"population": "population",
                  "total": "total"}, n_jobs=4, backend="threads", showprogress=False)
    assert [len(values) for values in table.population] == [6, 6, 21, 21]
    assert [len(values) for values in table.total] == [2, 2, 5, 5]
    assert not (tmp_path / "series.csv").exists() and series.records == []


def test_the_command_line_sweeps_into_a_table(tmp_path):
    import json

    from lgca.cli import main
    from lgca.model import save_model_spec

    save_model_spec(_growth(steps=4), tmp_path / "model.json")
    output = tmp_path / "runs"
    assert main([
        "sweep", str(tmp_path / "model.json"), "--vary", "birth_rate=0,0.5", "--vary", "channels=velocity,all",
        "--seeds", "0:2", "--output", str(output),
    ]) == 2  # channels occurs in two places
    assert main([
        "sweep", str(tmp_path / "model.json"), "--vary", "birth_rate=0,0.5",
        "--vary", "dynamics.operators[2].parameters.channels=velocity,all", "--seeds", "0:2",
        "--measure", "population", "--output", str(output),
    ]) == 0
    table = pd.read_csv(output / "table.csv")
    assert table.columns.tolist() == ["birth_rate", "channels", "seed", "population"]
    first = table.population.map(lambda values: json.loads(values)[0])
    assert len(table) == 8 and first.iloc[0] == first.iloc[2]  # seed 0 starts the same for every value
    description = json.loads((output / "sweep.json").read_text())
    assert description["seeds"] == [0, 1] and description["grid"]["birth_rate"] == [0, 0.5]
    long = tmp_path / "long"
    assert main(["sweep", str(tmp_path / "model.json"), "--vary", "birth_rate=0,0.5", "--measure", "population",
                 "--long", "--n-jobs", "2", "--output", str(long)]) == 0
    assert len(pd.read_csv(long / "table.csv")) == 10
