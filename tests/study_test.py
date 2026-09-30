"""vary and sweep: changing a model by paths, and tables of runs over parameters and seeds."""

import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
from lgca.pipeline import (
    BirthDeathSpec,
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


def test_vary_reaches_the_fields_of_a_pde():
    from lgca.fields import PDESpec

    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=20), state=StateSpec(density=0.3, restchannels=1, fields={"u": 0.0}),
        time=TimeSpec(steps=2, seed=1),
        dynamics=InteractionPipelineSpec(operators=[PDESpec(field="u", diffusion=1.0, cells=[{"uptake": 0.1}])]))
    variant = vary(spec, {"decay": 0.2, "dynamics.operators[pde].parameters.cells[0].uptake": 0.3,
                          "dynamics.operators[0].diffusion": 2.0})
    assert variant.dynamics.operators[0] == PDESpec(field="u", diffusion=2.0, decay=0.2, cells=[{"uptake": 0.3}])
    assert resolve_path(spec, "decay") == "dynamics.operators[0].parameters.decay"
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


@pytest.mark.parametrize("parameters", [{}, {"birth_rate": 0.3}])
def test_dataclass_and_mapping_operators_expose_the_same_parameter_values(parameters):
    from lgca.study import _get, _tokens

    values = []
    for operator in ({"name": "birth_death", "parameters": parameters},
                     BirthDeathSpec("birth_death", parameters=parameters)):
        spec = ModelSpec(dynamics=InteractionPipelineSpec(operators=[operator]))
        path = resolve_path(spec, "birth_rate")
        values.append(_get(spec, _tokens(path)))
    assert values[0] == values[1] == parameters.get("birth_rate", 0.0)


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


@pytest.mark.parametrize("name, long", [("birth_rate", False), ("seed", False), ("step", True)])
def test_measures_cannot_overwrite_sweep_coordinates(name, long):
    with pytest.raises(ValueError, match="measure names conflict with sweep columns"):
        sweep(_growth(), grid={"birth_rate": [0.1, 0.2]}, seeds=[1],
              measure={name: final_population}, long=long, showprogress=False)


@pytest.mark.parametrize("grid, seeds", [
    ({"time.seed": [11, 22]}, None),
    ({"seed": [11, 22]}, None),
    ({"birth_rate": [0.1], "seed": [11, 22]}, [1, 2]),
    ([{"birth_rate": 0.1, "time.seed": 11}, {"birth_rate": 0.2}], None),
])
def test_the_seed_is_not_varied_in_the_grid(grid, seeds):
    """A seed in the grid would be replaced by the seeds of the sweep, and the rows labelled with it
    would be identical runs."""
    with pytest.raises(ValueError, match=r"seed cannot be varied in the grid .*seeds="):
        sweep(_growth(steps=0), grid=grid, seeds=seeds, showprogress=False)


def test_the_command_line_takes_seeds_from_seeds_not_vary(tmp_path, capsys):
    from lgca.cli import main
    from lgca.model import save_model_spec

    save_model_spec(_growth(steps=0), tmp_path / "model.json")
    assert main(["sweep", str(tmp_path / "model.json"), "--vary", "seed=11,22",
                 "--output", str(tmp_path / "runs")]) == 2
    assert "--seeds 11,22" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


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


@pytest.mark.parametrize("state", [{}, {"volume_exclusion": False}, {"identity_based": True},
                                   {"identity_based": True, "volume_exclusion": False}],
                         ids=["classical", "nove", "ib", "nove_ib"])
def test_threads_sweeps_of_an_operator_object_equal_serial_sweeps(state):
    import threading

    from lgca import interaction
    from lgca.plugins import create_plugin
    from lgca.rules import FunctionInteractionOperator

    # every run builds its model from one operator object, which learns the model's capacity: the runs in
    # threads ran with each other's capacities
    def model(*operators):
        return ModelSpec(space=SpaceSpec(geometry="lin", dims=30), time=TimeSpec(steps=30, seed=7),
                         state=StateSpec(density=1.0, restchannels=1, capacity=2, **state),
                         dynamics=InteractionPipelineSpec(operators=list(operators), propagation=False))

    all_built = threading.Barrier(6, timeout=30)

    @interaction(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), register=False,
                 name="study_test.wait_until_all_runs_are_built")
    def wait_until_all_runs_are_built(state):
        """Changes nothing; the runs in threads start stepping together."""
        if state.step == 1:
            all_built.wait()

    operator = create_plugin("birth_death", {"birth_rate": 0.5})
    grid = {"state.capacity": [2, 50, 3]}
    named = sweep(model({"name": "birth_death", "parameters": {"birth_rate": 0.5}}), grid=grid, seeds=[1, 2],
                  showprogress=False)
    serial = sweep(model(operator), grid=grid, seeds=[1, 2], showprogress=False)
    threads = sweep(model(FunctionInteractionOperator(wait_until_all_runs_are_built), operator), grid=grid,
                    seeds=[1, 2], n_jobs=6, backend="threads", showprogress=False)
    pd.testing.assert_frame_equal(serial, named)
    pd.testing.assert_frame_equal(threads, named)
    assert operator.capacity is None  # the object in the spec is a template: it never runs


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


@pytest.mark.parametrize("recorder", ["PopulationRecorder", "ScalarTimeSeriesRecorder"])
def test_long_tables_keep_runs_that_recorded_nothing(recorder):
    from lgca import simulation
    from lgca.model import AnalysisSpec

    observer = getattr(simulation, recorder)(schedule=simulation.Schedule(steps=[10]))  # after a transient
    spec = vary(_growth(), {"analysis": AnalysisSpec(observers=[observer])})
    measure = {"population": "population", "final": final_population}
    long = sweep(spec, grid={"steps": [4, 12]}, seeds=[1, 2], measure=measure, long=True, showprogress=False)
    wide = sweep(spec, grid={"steps": [4, 12]}, seeds=[1, 2], measure=measure, showprogress=False)
    assert long.columns.tolist() == ["steps", "seed", "step", "final", "population"]
    assert long[["steps", "seed"]].values.tolist() == [[4, 1], [4, 2], [12, 1], [12, 2]]
    assert long.final.tolist() == wide.final.tolist()
    assert long.step.isna().tolist() == [True, True, False, False]
    assert long.population.isna().tolist() == [True, True, False, False]


def test_processes_sweeps_take_operator_objects_of_registered_rules():
    import pickle

    from lgca.plugins import create_plugin

    # a rule, a stack and a reorientation term pickle by reference, like functions: the workers import them
    for name, parameters in [("birth_death", {"birth_rate": 0.5}), ("go_or_grow_kappa", {}),
                             ("polar_alignment", {"beta": 2.0})]:
        operator = create_plugin(name, parameters)
        copied = pickle.loads(pickle.dumps(operator))
        assert type(copied) is type(operator) and copied.parameters == operator.parameters
        assert getattr(copied, "rule", None) is getattr(operator, "rule", None)
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(8, 8)), time=TimeSpec(steps=5, seed=1),
                     state=StateSpec(density=0.5, restchannels=1, volume_exclusion=False, capacity=4),
                     dynamics=InteractionPipelineSpec(operators=[
                         create_plugin("birth_death", {"birth_rate": 0.5}),
                         create_plugin("polar_alignment", {"beta": 2.0})]))
    grid = {"state.capacity": [2, 6]}
    serial = sweep(spec, grid=grid, seeds=[1, 2], showprogress=False)
    processes = sweep(spec, grid=grid, seeds=[1, 2], n_jobs=2, showprogress=False)
    pd.testing.assert_frame_equal(processes, serial)


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


def _from_npz(directory):
    from lgca.model import save_model_spec

    np.savez(directory / "state.npz", nodes=np.ones((4, 2), dtype=bool))
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=4),
                     state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": "state.npz"}}),
                     time=TimeSpec(steps=0, seed=1))
    save_model_spec(spec, directory / "model.json")
    return spec


@pytest.mark.parametrize("n_jobs, backend", [(1, "processes"), (2, "threads"), (2, "processes")])
def test_sweeps_read_the_resources_of_the_model(tmp_path, n_jobs, backend):
    spec = _from_npz(tmp_path)
    with pytest.raises(RuntimeError, match="require resource_base"):
        sweep(spec, showprogress=False)
    table = sweep(spec, seeds=[1, 2], n_jobs=n_jobs, backend=backend, resource_base=tmp_path, showprogress=False)
    assert table.population.tolist() == [8, 8]


def test_the_command_line_sweeps_a_model_with_resources(tmp_path):
    import json

    from lgca.cli import main
    from lgca.model import build_model, model_spec_from_dict

    _from_npz(tmp_path)
    output = tmp_path / "runs"
    assert main(["validate", str(tmp_path / "model.json")]) == 0
    assert main(["sweep", str(tmp_path / "model.json"), "--seeds", "1,2", "--output", str(output)]) == 0
    assert pd.read_csv(output / "table.csv").population.tolist() == [8, 8]
    description = json.loads((output / "sweep.json").read_text())
    assert description["model"]["model"]["state"]["initializer"]["parameters"]["path"] == "resources/initial_state.npz"
    compiled = build_model(model_spec_from_dict(description["model"]), resource_base=output)  # usable later
    assert compiled.lgca.cell_density[compiled.lgca.nonborder].sum() == 8


def _drawing(directory):
    from lgca.model import AnalysisSpec
    from lgca.plotting import AnimationObserver, PlotSnapshotObserver
    from lgca.simulation import CSVSnapshotObserver

    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(4, 4)), state=StateSpec(density=0.5),
        time=TimeSpec(steps=1, seed=1),
        analysis=AnalysisSpec(observers=[
            PlotSnapshotObserver(output_dir=directory / "snapshots", cbar=False),
            AnimationObserver(save_path=directory / "movies" / "density.gif", close=True, cbar=False),
            CSVSnapshotObserver(output_dir=directory / "csv")]))


def test_a_sweep_keeps_no_files_by_default(tmp_path):
    """Runs over several values would write to the same files; a sweep draws and writes none."""
    import matplotlib

    matplotlib.use("Agg")
    table = sweep(_drawing(tmp_path), grid={"density": [0.0, 2.0]}, showprogress=False)
    assert len(table) == 2 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("n_jobs, backend", [(1, "processes"), (2, "threads"), (2, "processes")])
def test_kept_files_go_to_a_folder_per_run(tmp_path, n_jobs, backend):
    import matplotlib

    matplotlib.use("Agg")
    sweep(_drawing(tmp_path), grid={"density": [0.0, 2.0]}, seeds=[1], n_jobs=n_jobs, backend=backend,
          keep_files=True, showprogress=False)
    runs = ["density=0.0_seed=1", "density=2.0_seed=1"]
    assert sorted(path.name for path in (tmp_path / "snapshots").iterdir()) == runs
    for run in runs:
        assert sorted(path.name for path in (tmp_path / "snapshots" / run).iterdir()) == [
            "density_00000.png", "density_00001.png"]
        assert (tmp_path / "movies" / run / "density.gif").is_file()
        assert (tmp_path / "csv" / run / "density_00001.csv").is_file()
    empty, full = (pd.read_csv(tmp_path / "csv" / run / "density_00000.csv").value.sum() for run in runs)
    assert empty == 0 and full > 0


def test_the_command_line_keeps_files_on_request(tmp_path):
    from lgca.cli import main
    from lgca.model import AnalysisSpec, save_model_spec
    from lgca.simulation import CSVSnapshotObserver, ScalarTimeSeriesRecorder

    spec = vary(_growth(steps=2), {"analysis": AnalysisSpec(observers=[
        CSVSnapshotObserver(output_dir="snapshots"), ScalarTimeSeriesRecorder(output_path="series.csv")])})
    save_model_spec(spec, tmp_path / "model.json")
    arguments = ["sweep", str(tmp_path / "model.json"), "--vary", "birth_rate=0,0.5"]
    assert main([*arguments, "--output", str(tmp_path / "plain")]) == 0
    assert sorted(path.name for path in (tmp_path / "plain").iterdir()) == ["sweep.json", "table.csv"]
    assert main([*arguments, "--keep-files", "--output", str(tmp_path / "kept")]) == 0
    for run in ("birth_rate=0_seed=1", "birth_rate=0.5_seed=1"):
        assert (tmp_path / "kept" / "snapshots" / run / "density_00002.csv").is_file()
        assert len(pd.read_csv(tmp_path / "kept" / run / "series.csv")) == 3
    assert not (tmp_path / "snapshots").exists() and not (tmp_path / "series.csv").exists()
