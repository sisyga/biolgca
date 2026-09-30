"""Cross-feature regressions found in the September 2026 code review."""

from dataclasses import replace

import jsonschema
import numpy as np
import pytest

from lgca import get_lgca, stack
from lgca.explorer import explore
from lgca.fields import PDESpec
from lgca.lattice_state import LatticeState
from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    load_model_spec_schema,
    model_spec_from_json,
    model_spec_from_yaml,
    model_spec_to_dict,
    model_spec_to_json,
    model_spec_to_yaml,
    run_model,
)
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import (
    FieldRecorder,
    NodeRecorder,
    PopulationRecorder,
    ScalarTimeSeriesRecorder,
    Schedule,
)
from lgca.study import sweep
from lgca.zoo._clones import family_trait


@pytest.mark.parametrize("sensed", [1, [1], (0, 1), np.array([1])])
@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_model_files_keep_sensed_species_and_seeded_dynamics(sensed, volume_exclusion):
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(6, 4)),
        state=StateSpec(density=2, n_species=2, volume_exclusion=volume_exclusion, capacity=12),
        time=TimeSpec(steps=3, seed=8),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
            ReorientationTermSpec(name="polar_alignment", beta=4, species=0, sensed_species=sensed),
        ])]),
    )
    jsonschema.validate(model_spec_to_dict(spec), load_model_spec_schema())
    expected = run_model(spec, showprogress=False).lgca.nodes
    for loaded in (model_spec_from_json(model_spec_to_json(spec)),
                   model_spec_from_yaml(model_spec_to_yaml(spec))):
        np.testing.assert_array_equal(loaded.dynamics.operators[0].terms[0].sensed_species, sensed)
        np.testing.assert_array_equal(run_model(loaded, showprogress=False).lgca.nodes, expected)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
def test_stacked_steady_fields_are_initialized_before_the_first_step(nested, boundary):
    @stack(kind="field", families="classical", name="review.steady_inner")
    def steady_inner(state):
        return [PDESpec(field="u", diffusion=1, decay=1, production=2, solver="steady")]

    @stack(kind="field", families="classical", name="review.steady_outer")
    def steady_outer(state):
        return [steady_inner()]

    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=4, boundary=boundary),
        state=StateSpec(density=0, fields={"u": 0}), time=TimeSpec(seed=1),
        dynamics=InteractionPipelineSpec(operators=[steady_outer() if nested else steady_inner()]),
    )
    model = build_model(spec)
    # The exact equilibrium of dc/dt = Laplace(c) + 2 - c is c = 2, including ghosts.
    np.testing.assert_allclose(model.lgca.u, 2)


@pytest.mark.parametrize("dtype", [np.uint64, np.int64, float])
@pytest.mark.parametrize("species", [1, 2])
def test_counts_reject_per_node_int64_overflow_before_conversion(dtype, species):
    lgca = get_lgca(geometry="lin", dims=2, ve=False, n_species=species, density=0,
                    interaction="only_propagation")
    state = LatticeState(lgca)
    # with two species, each species' 3 * 2**61 cells fit in int64, but not the node's 6 * 2**61
    values = np.full(state.counts.shape, 2**62 if species == 1 else 2**61, dtype=dtype)
    with pytest.raises(ValueError, match="signed int64"):
        state.counts = values
    assert not state.counts.any()


def test_counts_assignment_does_not_share_a_writable_buffer_with_the_caller():
    lgca = get_lgca(geometry="lin", dims=2, ve=False, density=0, interaction="only_propagation")
    state = LatticeState(lgca)
    values = np.ones(state.counts.shape, dtype=np.int64)
    state.counts = values
    values[...] = -1
    state.commit()
    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder], 1)


@pytest.mark.parametrize("number", [np.inf, np.nan, 2**63])
def test_adding_unrepresentable_cell_numbers_fails_without_changing_state(number):
    lgca = get_lgca(geometry="lin", dims=2, ve=False, density=0, interaction="only_propagation")
    state = LatticeState(lgca)
    with pytest.raises(ValueError, match="integers|signed int64"):
        state.add_cells(number)
    assert not state.counts.any()


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_species_channel_sets_survive_json_object_keys(volume_exclusion):
    lgca = get_lgca(geometry="lin", dims=50, restchannels=1, n_species=2, ve=volume_exclusion,
                    density=0, seed=1, interaction="only_propagation")
    state = LatticeState(lgca)
    state.add_cells(1, channels={"0": "rest", "1": "velocity"})
    assert np.all(state.counts[..., 0, :2] == 0)
    assert np.all(state.counts[..., 0, 2] == 1)
    assert np.all(state.counts[..., 1, 2] == 0)
    assert np.all(state.species_density == 1)


@pytest.mark.parametrize("value", [2.0, np.float64(2.0), np.int64(2)])
@pytest.mark.parametrize("option", ["substeps", "max_iterations"])
def test_integral_solver_options_are_normalized_to_integers(option, value):
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=4), state=StateSpec(density=0, fields={"u": 1}),
        time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[
            PDESpec(field="u", decay=1, solver_options={option: value}),
        ]),
    ))
    assert type(model.pipeline.operators[0].options[option]) is int
    model.step()
    expected = (1 + 1 / 2)**-2 if option == "substeps" else 0.5
    np.testing.assert_allclose(model.lgca.u[model.lgca.nonborder], expected)


def test_nonconvergent_nonlinear_fields_fail_without_publishing_an_inaccurate_solution():
    nodes = np.zeros((3, 3), dtype=int)
    nodes[:, 2] = 1
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=3), time=TimeSpec(seed=1),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=False, fields={"u": 2}),
        dynamics=InteractionPipelineSpec(operators=[
            PDESpec(field="u", cells=[{"uptake": 4, "saturation": 1, "n": 64}]),
        ]),
    )
    model = build_model(spec)
    with pytest.raises(RuntimeError, match="did not converge.*residual"):
        model.step()
    np.testing.assert_array_equal(model.lgca.u, 2)
    assert model._step == 0


@pytest.mark.parametrize("concentration,saturation", [(2e100, 1e100), (2e-100, 1e-100)])
def test_hill_uptake_is_finite_when_individual_powers_overflow_or_underflow(concentration, saturation):
    from lgca.fields import _hill_rate

    # Scaling c and K by a common factor scales the loss rate by its reciprocal.
    expected = np.array([0.0, 8 / 17 / saturation])
    np.testing.assert_allclose(_hill_rate(np.array([0.0, concentration]), saturation, 4), expected,
                               rtol=1e-12, atol=0)


def test_large_constant_field_auto_backend_does_not_force_sparse_lu(monkeypatch):
    from lgca import fields

    def unexpected_lu(*args, **kwargs):
        pytest.fail("auto must not factor a large lattice with sparse LU")

    monkeypatch.setattr(fields.spla, "splu", unexpected_lu)
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=fields._FACTOR_LIMIT[1] + 1),
        state=StateSpec(density=0, fields={"u": 1}), time=TimeSpec(seed=1),
        dynamics=InteractionPipelineSpec(operators=[PDESpec(field="u", diffusion=1, decay=1)]),
    ))
    model.step()
    np.testing.assert_allclose(model.lgca.u[model.lgca.nonborder], 0.5)


@pytest.mark.parametrize("source", [-1, np.nan, np.inf])
def test_pde_production_maps_are_validated_at_build_and_when_they_change(source):
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=4), state=StateSpec(density=0, fields={"u": 1, "source": source}),
        time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[PDESpec(field="u", production="source")]),
    )
    with pytest.raises(ValueError, match="production.*finite and non-negative"):
        build_model(spec)
    valid = replace(spec, state=replace(spec.state, fields={"u": 1, "source": 2}))
    model = build_model(valid)
    model.lgca.source[...] = source
    with pytest.raises(ValueError, match="production.*finite and non-negative"):
        model.step()
    np.testing.assert_array_equal(model.lgca.u, 1)


@pytest.mark.parametrize("points", [False, True])
def test_sweeps_reject_duplicate_paths_after_alias_resolution(points):
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), state=StateSpec(density=0), time=TimeSpec(steps=0, seed=1),
        dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}]),
    )
    grid = {"birth_rate": 0.1, "dynamics.operators[0].parameters.birth_rate": 0.2}
    grid = [grid] if points else {path: [value] for path, value in grid.items()}
    with pytest.raises(ValueError, match="same model parameter"):
        sweep(spec, grid=grid, showprogress=False)


@pytest.mark.parametrize("second", ["dynamics.operators[0].birth_rate",
                                   "dynamics.operators[birth_death].parameters.birth_rate",
                                   "dynamics.operators[-1].parameters.birth_rate"])
def test_sweeps_reject_equivalent_operator_selectors(second):
    spec = ModelSpec(time=TimeSpec(steps=0, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}]))
    with pytest.raises(ValueError, match="same model parameter"):
        sweep(spec, grid={"birth_rate": [0.1], second: [0.2]}, showprogress=False)


def test_named_sweep_selectors_keep_readable_columns_and_coalesce_point_aliases():
    spec = ModelSpec(time=TimeSpec(steps=0, seed=1), state=StateSpec(fields={"signal": 0}),
                     dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
                         ReorientationTermSpec("nematic_alignment"),
                         ReorientationTermSpec("chemotaxis", parameters={"field": "signal"}),
                     ])]))
    first = "dynamics.operators[0].terms[nematic_alignment].beta"
    second = "dynamics.operators[0].terms[chemotaxis].beta"
    table = sweep(spec, grid={first: [1], second: [2]}, showprogress=False)
    assert list(table) == ["nematic_alignment.beta", "chemotaxis.beta", "seed", "population"]
    table = sweep(spec, grid=[{first: 1}, {"dynamics.operators[0].terms[0].beta": 2}], showprogress=False)
    assert list(table) == ["beta", "seed", "population"]
    assert table.beta.tolist() == [1, 2]


def test_matching_population_recorders_can_also_stream_a_csv(tmp_path):
    from lgca import simulation

    scalar = ScalarTimeSeriesRecorder(output_path=tmp_path / "population.csv", schedule=Schedule(every=2))
    result = run_model(ModelSpec(time=TimeSpec(steps=2, seed=1), analysis=AnalysisSpec(observers=[
        PopulationRecorder(Schedule(every=2)), scalar,
    ])), showprogress=False)
    np.testing.assert_array_equal(result.data.steps("population"), [0, 2])
    np.testing.assert_array_equal(result.data["population"], [row["population"] for row in scalar.records])
    assert scalar.metrics["population"] is simulation._total_population
    assert scalar.output_path.is_file()


@pytest.mark.parametrize("seed", [1.9, True, -1, None])
def test_sweep_seeds_are_not_truncated_or_reinterpreted(seed):
    with pytest.raises(ValueError, match="seeds.*non-negative integers"):
        sweep(ModelSpec(time=TimeSpec(steps=0)), seeds=[seed], showprogress=False)


def test_empty_sweep_axes_fail_before_simulation():
    with pytest.raises(ValueError, match="empty"):
        sweep(ModelSpec(), grid={"steps": []}, showprogress=False)


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_family_trait_includes_the_first_real_cell(volume_exclusion):
    nodes = np.zeros((2, 3), dtype=bool if volume_exclusion else int)
    nodes[:, 2] = 1
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=volume_exclusion,
                        identity_based=True, traits={"fitness": [7.0, 9.0]}),
    ))
    model.lgca.init_families(type="heterogeneous")
    np.testing.assert_array_equal(family_trait(model.lgca, "fitness"), [7.0, 7.0, 9.0])


@pytest.mark.parametrize("name", ["step", "n"])
def test_scalar_metrics_cannot_overwrite_steps_or_use_unreachable_aliases(name):
    with pytest.raises(ValueError, match="Metric names"):
        ScalarTimeSeriesRecorder({name: lambda lgca: 42})


@pytest.mark.parametrize("other", [PopulationRecorder(),
                                    ScalarTimeSeriesRecorder({"population": lambda lgca: 42})])
def test_conflicting_recorder_outputs_fail_before_advancing(other):
    model = build_model(ModelSpec(time=TimeSpec(steps=1, seed=1), analysis=AnalysisSpec(observers=[
        ScalarTimeSeriesRecorder({"population": lambda lgca: 99}), other,
    ])))
    before = model.lgca.nodes.copy()
    with pytest.raises(ValueError, match="same output names"):
        model.run(showprogress=False)
    np.testing.assert_array_equal(model.lgca.nodes, before)
    assert model._step == 0


def test_field_and_scalar_output_names_cannot_collide():
    spec = ModelSpec(state=StateSpec(fields={"signal": 2}), time=TimeSpec(steps=0, seed=1),
                     analysis=AnalysisSpec(observers=[FieldRecorder(["signal"]),
                                                     ScalarTimeSeriesRecorder({"signal": lambda lgca: 42})]))
    with pytest.raises(ValueError, match="same output names"):
        run_model(spec, showprogress=False)


def test_subclassed_recorders_and_empty_scalar_schedules_are_in_result_data():
    class CustomNodes(NodeRecorder):
        pass

    result = run_model(ModelSpec(time=TimeSpec(steps=1, seed=1), analysis=AnalysisSpec(observers=[
        CustomNodes(), ScalarTimeSeriesRecorder({"custom": lambda lgca: 42}, schedule=Schedule(steps=[])),
    ])), showprogress=False)
    np.testing.assert_array_equal(result.data["nodes"], result.lgca.nodes_t)
    assert result.data["custom"].shape == (0,)
    assert result.data.steps("custom").shape == (0,)


def test_sweeps_use_existing_scalar_metrics_instead_of_adding_a_conflicting_recorder():
    spec = ModelSpec(time=TimeSpec(steps=1, seed=1), analysis=AnalysisSpec(observers=[
        ScalarTimeSeriesRecorder({"population": lambda lgca: 42}),
    ]))
    table = sweep(spec, measure={"recorded": "population"}, showprogress=False)
    np.testing.assert_array_equal(table.recorded.iloc[0], [42, 42])


@pytest.mark.parametrize("geometry,dims", [("lin", (3,)), ("square", (3, 4)), ("cubic", (3, 4, 2))])
def test_explorer_works_with_vector_fields_and_offers_scalar_field_views(geometry, dims):
    fields = {"wind": np.ones((*dims, len(dims))), "signal": 2,
              "conc": np.arange(np.prod(dims), dtype=float).reshape(dims)}
    spec = ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims), time=TimeSpec(seed=1),
                     state=StateSpec(density=0, fields=fields))
    explorer = explore(spec, window=3)
    try:
        assert "wind" not in explorer._view.options
        assert "signal" in explorer._view.options
        assert "conc" in explorer._view.options  # a scalar field given as an array
        explorer._view.value = "conc"
        explorer.advance(1)
        assert not explorer._status.value
    finally:
        explorer.close()


@pytest.mark.parametrize("positions", [[0.9], [np.nan], [np.uint64(2**63)], [np.uint64(2**64 - 1)]])
def test_cell_selection_does_not_truncate_positions_or_wrap_large_indices(positions):
    nodes = np.zeros((2, 3), dtype=bool)
    nodes[0, 2] = True
    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
                                 state=StateSpec(nodes=nodes, restchannels=1, identity_based=True)))
    state = LatticeState(model.lgca)
    with pytest.raises(ValueError, match="cell positions"):
        state.cells.kill(positions)
    assert len(state.cells) == 1


@pytest.mark.parametrize("operator", [{"name": "go_or_rest", "parameters": {"kappa": "value", "theta": 0.5}},
                                      {"name": "birth_death", "parameters": {"birth_rate": "value"}},
                                      {"name": "go_or_grow.growth", "parameters": {"r_b": "value"}},
                                      {"name": "go_or_rest", "parameters": {"probability": {
                                          "cues": [{"name": "density", "kappa": "value"}]}}}])
def test_dynamic_trait_parameters_cannot_turn_nan_into_silent_biological_events(operator):
    nodes = np.zeros((2, 3), dtype=bool)
    nodes[0, 2] = True
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, traits={"value": 0.5}),
        dynamics=InteractionPipelineSpec(operators=[operator]),
    ))
    state = LatticeState(model.lgca)
    state.cells.set_trait([0], "value", np.nan)
    with pytest.raises(ValueError, match="finite"):
        model.step()
    np.testing.assert_array_equal(model.lgca.nodes[model.lgca.nonborder] > 0, nodes)


def test_meltdown_scans_do_not_exhaust_generator_axes(monkeypatch):
    from lgca.zoo import mutational_meltdown

    monkeypatch.setattr(mutational_meltdown, "_scan_one", lambda arguments: "undecided")
    table = mutational_meltdown.scan(iter([0.25, 0.5]), iter([0, 1]), iter([1, 2]))
    assert len(table) == 8
    assert len(table[["n0", "gamma", "seed"]].drop_duplicates()) == 8


@pytest.mark.parametrize("cells", [0, 1])
def test_radial_profiles_handle_extinction_and_a_population_at_the_centre(cells):
    from lgca.zoo import phenotypic_plasticity

    lgca = build_model(phenotypic_plasticity.build_spec(size=3, cells=cells, kappa_0=(4, 4), steps=0)).lgca
    profile = phenotypic_plasticity.radial_profiles(lgca, bins=3)
    # the same profiles in either case, which the notebook reads
    names = {"r", "cells", "migrating", "resting", "migrating_fraction", "kappa_mean", "kappa_std"}
    assert set(profile) == names | {"front"}
    assert all(np.shape(profile[name]) == (3,) for name in names)
    assert np.ndim(profile["front"]) == 0
    assert np.isnan(profile["r"]).all()
    if cells:
        assert profile["front"] == 0
        assert profile["cells"][0] == 1
        assert profile["kappa_mean"][0] == 4
    else:
        assert np.isnan(profile["front"])
        assert not profile["cells"].any()
        assert not profile["migrating"].any()
        assert not profile["resting"].any()
        assert np.isnan(profile["kappa_mean"]).all()
        assert np.isnan(profile["kappa_std"]).all()


def test_rules_see_their_own_staged_field_updates_and_gradients():
    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=3), state=StateSpec(fields={"u": 0}),
                                 time=TimeSpec(seed=1)))
    state = LatticeState(model.lgca, kind="field")
    state.set_field("u", state.field("u") + np.arange(3))
    state.set_field("u", state.field("u") + 1)
    np.testing.assert_array_equal(state.field("u"), [1, 2, 3])
    gradient = state.gradient("u")
    np.testing.assert_array_equal(model.lgca.u, 0)
    state.commit()
    np.testing.assert_array_equal(state.gradient("u"), gradient)
    np.testing.assert_array_equal(model.lgca.u[model.lgca.nonborder], [1, 2, 3])


def test_command_line_rejects_repeated_grid_keys_before_writing_outputs(tmp_path, capsys):
    from lgca.cli import main
    from lgca.model import save_model_spec

    path = save_model_spec(ModelSpec(time=TimeSpec(steps=0, seed=1)), tmp_path / "model.json")
    output = tmp_path / "sweep"
    assert main(["sweep", str(path), "--vary", "steps=0", "--vary", "steps=1", "--output", str(output)]) == 2
    assert "repeats" in capsys.readouterr().err
    assert not output.exists()


def test_kept_sweep_files_have_valid_unique_folders_for_array_parameters(tmp_path):
    from lgca.simulation import CSVSnapshotObserver

    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=32), time=TimeSpec(steps=0, seed=1),
                     analysis=AnalysisSpec(observers=[CSVSnapshotObserver(output_dir=tmp_path)]))
    table = sweep(spec, grid={"nodes": [np.zeros((32, 2), bool), np.ones((32, 2), bool)]},
                  keep_files=True, showprogress=False)
    files = list(tmp_path.glob("*/density_00000.csv"))
    assert len(table) == len(files) == 2
    assert len({file.parent.name for file in files}) == 2
    assert all(len(file.parent.name) <= 120 for file in files)
    import pandas as pd

    assert sorted(pd.read_csv(file).value.sum() for file in files) == [0, 64]


@pytest.mark.parametrize("solver", ["implicit", "explicit"])
def test_numeric_solver_tolerances_are_normalized_after_validation(solver):
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=3), state=StateSpec(density=0, fields={"u": 1}),
        time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[
            PDESpec(field="u", decay=1, solver=solver, solver_options={"rtol": "0.0001"}),
        ]),
    ))
    assert type(model.pipeline.operators[0].options["rtol"]) is float
    model.step()
    expected = 0.5 if solver == "implicit" else np.exp(-1)
    np.testing.assert_allclose(model.lgca.u[model.lgca.nonborder], expected, rtol=1e-4)


def test_nonfinite_solver_output_is_rejected_before_changing_the_field():
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=3), state=StateSpec(density=0, fields={"u": 1e308}),
        time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[PDESpec(field="u", production=1e308)]),
    ))
    with np.errstate(over="ignore"), pytest.raises(RuntimeError, match="non-finite field"):
        model.step()
    np.testing.assert_array_equal(model.lgca.u, 1e308)


@pytest.mark.parametrize("path", ["state.initializer.parameters.path", "state.initializer.parameters",
                                   "state.initializer"])
def test_sweep_archives_keep_varied_initializer_files_and_can_be_replayed_without_originals(path, tmp_path):
    import json

    from lgca.cli import main
    from lgca.model import model_spec_from_dict, save_model_spec

    np.savez(tmp_path / "a.npz", nodes=np.array([[1, 0], [0, 0]], dtype=bool))
    np.savez(tmp_path / "b.npz", nodes=np.zeros((2, 2), dtype=bool))
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(steps=0, seed=1),
                     state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": "a.npz"}}))
    model_path = save_model_spec(spec, tmp_path / "model.json")
    output = tmp_path / "archive"
    if path.endswith(".path"):
        values = ["a.npz", "b.npz"]
    elif path.endswith(".parameters"):
        values = [{"path": name} for name in ("a.npz", "b.npz")]
    else:
        values = [{"name": "from_npz", "parameters": {"path": name}} for name in ("a.npz", "b.npz")]
    # A whole initializer contains a comma, which separates values on the command line; it goes to the
    # archive helper directly.
    if path == "state.initializer":
        from lgca.cli import _with_copied_sweep_resources

        output.mkdir()
        portable_spec, portable_grid, _ = _with_copied_sweep_resources(spec, {path: values}, model_path, output,
                                                                       False)
    else:
        options = ",".join(json.dumps(value) if isinstance(value, dict) else value for value in values)
        assert main(["sweep", str(model_path), "--vary", f"{path}={options}", "--output", str(output)]) == 0
        description = json.loads((output / "sweep.json").read_text())
        portable_spec = model_spec_from_dict(description["model"])
        portable_grid = description["grid"]
    (tmp_path / "a.npz").unlink()
    (tmp_path / "b.npz").unlink()
    table = sweep(portable_spec, grid=portable_grid, resource_base=output, showprogress=False)
    assert table.population.tolist() == [1, 0]


@pytest.mark.parametrize("switch", [
    {"traits": {"fitness": {"value": np.nan, "operation": "set"}}},
    {"traits": {"fitness": {"value": np.inf, "operation": "set"}}},
    {"when": {"fitness": [1, 0]}, "traits": {"fitness": {"value": 1}}},
    {"when": {"fitness": [np.nan, 1]}, "traits": {"fitness": {"value": 1}}},
])
def test_invalid_mutation_values_and_conditions_do_not_silently_change_or_disable_events(switch):
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=np.array([[1, 0, 0], [0, 0, 0]]), restchannels=1, volume_exclusion=False,
                        identity_based=True, capacity=4, traits={"fitness": 0.5}),
        dynamics=InteractionPipelineSpec(operators=[{"name": "trait_switch", "parameters": {"switch": switch}}]),
    )
    model = build_model(spec)
    with pytest.raises(ValueError, match="finite|low <= high"):
        model.step()
    np.testing.assert_array_equal(LatticeState(model.lgca).cells["fitness"], [0.5])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("redraw", [False, True])
def test_mutation_callbacks_cannot_write_nonfinite_effects_to_cell_traits(value, redraw):
    def effect(rng, size):
        return np.full(size, value)

    switch = {"traits": {"fitness": {"function": effect, "operation": "set",
                                     "at_bounds": "redraw" if redraw else "clip"}}}
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=np.array([[1, 0, 0], [0, 0, 0]]), restchannels=1, volume_exclusion=False,
                        identity_based=True, capacity=4, traits={"fitness": 0.5}),
        dynamics=InteractionPipelineSpec(operators=[{"name": "trait_switch", "parameters": {"switch": switch}}]),
    )
    model = build_model(spec)
    with pytest.raises(ValueError, match="finite"):
        model.step()
    np.testing.assert_array_equal(LatticeState(model.lgca).cells["fitness"], [0.5])


# Follow-up review, model


def test_the_cue_helper_takes_sensed_species_and_reserves_its_name():
    """polar_alignment(..., sensed_species=1) is the term of the single cue with these parameters."""
    from lgca.builtin_rules import polar_alignment
    from lgca.rules import reorientation_term

    assert polar_alignment(beta=2.0, species=1, sensed_species=1) == ReorientationTermSpec(
        name="polar_alignment", beta=2.0, species=1, sensed_species=1)

    def sensing(state, sensed_species=0):  # the term sets sensed_species, so the function would never get it
        return state.flux

    with pytest.raises(TypeError, match="sensed_species are set on the term"):
        reorientation_term(sensing, coupling="flux", register=False)


# Follow-up review, cli


@pytest.mark.parametrize("other", [PopulationRecorder(Schedule(every=2)),
                                   PopulationRecorder(Schedule(steps=range(5))),
                                   ScalarTimeSeriesRecorder(schedule=Schedule(every=2))])
def test_recorders_of_the_same_quantity_may_use_different_schedules(other, tmp_path):
    scalar = ScalarTimeSeriesRecorder(output_path=tmp_path / "population.csv")
    result = run_model(ModelSpec(time=TimeSpec(steps=4, seed=1),
                                 analysis=AnalysisSpec(observers=[other, scalar])), showprogress=False)
    steps = [step for step in range(5) if other.schedule.should_run(step)]
    np.testing.assert_array_equal(result.data.steps("population"), steps)
    np.testing.assert_array_equal(result.data["population"],
                                  [row["population"] for row in scalar.records if row["step"] in steps])
    assert [row["step"] for row in scalar.records] == [0, 1, 2, 3, 4]
    assert len(scalar.output_path.read_text().splitlines()) == 6


def test_cli_writes_a_time_series_next_to_a_sparse_population_recorder(tmp_path):
    from lgca.cli import main
    from lgca.model import save_model_spec

    spec = ModelSpec(time=TimeSpec(steps=4, seed=1), analysis=AnalysisSpec(observers=[
        PopulationRecorder(Schedule(every=2)), ScalarTimeSeriesRecorder()]))
    path = save_model_spec(spec, tmp_path / "model.json")
    assert main(["run", str(path), "--output", str(tmp_path / "run")]) == 0
    with np.load(tmp_path / "run" / "measurements.npz") as data:
        np.testing.assert_array_equal(data["n_steps"], [0, 2, 4])
    assert len((tmp_path / "run" / "time_series.csv").read_text().splitlines()) == 6


def test_scalar_recorders_need_a_metric():
    with pytest.raises(ValueError, match="at least one metric"):
        ScalarTimeSeriesRecorder(metrics={})


@pytest.mark.parametrize("observers,message", [
    ([PopulationRecorder(), PopulationRecorder(Schedule(every=2))], "Multiple PopulationRecorder"),
    ([FieldRecorder("u"), FieldRecorder("u", Schedule(every=2))], "several FieldRecorders"),
])
def test_command_line_rejects_conflicting_recorders_in_validate_and_before_writing_a_run(observers, message,
                                                                                          tmp_path, capsys):
    from lgca.cli import main
    from lgca.model import save_model_spec

    spec = ModelSpec(state=StateSpec(fields={"u": 0}), time=TimeSpec(steps=1, seed=1),
                     analysis=AnalysisSpec(observers=observers))
    path = save_model_spec(spec, tmp_path / "model.json")
    assert main(["validate", str(path)]) == 2
    assert message in capsys.readouterr().err
    output = tmp_path / "run"
    assert main(["run", str(path), "--output", str(output)]) == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_sweep_archive_rewritten_in_place_with_a_reordered_grid_keeps_the_inputs_of_its_runs(tmp_path):
    import json

    from lgca.cli import main
    from lgca.model import model_spec_from_dict, save_model_spec

    for cells, name in enumerate(("a.npz", "b.npz", "c.npz"), start=1):
        nodes = np.zeros((3, 2), dtype=bool)
        nodes.flat[:cells] = True
        np.savez(tmp_path / name, nodes=nodes)
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=3), time=TimeSpec(steps=0, seed=1),
                     state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": "a.npz"}}))
    model_path = save_model_spec(spec, tmp_path / "model.json")
    path, output = "state.initializer.parameters.path", tmp_path / "archive"
    assert main(["sweep", str(model_path), "--vary", f"{path}=a.npz,b.npz,c.npz", "--output", str(output)]) == 0
    first = json.loads((output / "sweep.json").read_text())
    # Sweep the archived model again into its own folder, reading the archived inputs in reverse order.
    archived = save_model_spec(model_spec_from_dict(first["model"]), output / "model.json")
    reordered = ",".join(reversed(first["grid"][path]))
    assert main(["sweep", str(archived), "--vary", f"{path}={reordered}", "--output", str(output),
                 "--overwrite"]) == 0
    description = json.loads((output / "sweep.json").read_text())
    table = sweep(model_spec_from_dict(description["model"]), grid=description["grid"], resource_base=output,
                  showprogress=False)
    assert table.population.tolist() == [3, 2, 1]


@pytest.mark.parametrize("vary", ["steps=0,1", "state.initializer.parameters.path=b.npz,sub/a.npz"])
def test_sweep_json_says_which_file_each_archived_copy_was_made_from(vary, tmp_path):
    import json

    import pandas as pd

    from lgca.cli import main
    from lgca.model import save_model_spec

    (tmp_path / "sub").mkdir()
    for cells, name in enumerate(("a.npz", "b.npz", "sub/a.npz")):
        nodes = np.zeros((3, 2), dtype=bool)
        nodes.flat[:cells] = True
        np.savez(tmp_path / name, nodes=nodes)
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=3), time=TimeSpec(steps=0, seed=1),
                     state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": "a.npz"}}))
    model_path = save_model_spec(spec, tmp_path / "model.json")
    output = tmp_path / "archive"
    assert main(["sweep", str(model_path), "--vary", vary, "--output", str(output)]) == 0
    resources = json.loads((output / "sweep.json").read_text())["resources"]
    if vary.startswith("steps"):
        assert resources == {"resources/initial_state.npz": "a.npz"}
    else:  # the model's own file is kept too, so that the model in sweep.json runs on its own
        assert resources == {"resources/initial_state.npz": "a.npz", "resources/initial_state_1.npz": "b.npz",
                             "resources/initial_state_2.npz": "sub/a.npz"}
        assert pd.read_csv(output / "table.csv").path.tolist() == ["b.npz", "sub/a.npz"]
    for archived, given in resources.items():
        assert (output / archived).read_bytes() == (tmp_path / given).read_bytes()


def test_sweep_does_not_need_a_missing_model_file_that_every_run_replaces(tmp_path):
    import json

    import pandas as pd

    from lgca.cli import main
    from lgca.model import model_spec_from_dict, save_model_spec

    np.savez(tmp_path / "a.npz", nodes=np.array([[1, 0], [0, 0]], dtype=bool))
    np.savez(tmp_path / "b.npz", nodes=np.ones((2, 2), dtype=bool))
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(steps=0, seed=1),
                     state=StateSpec(initializer={"name": "from_npz", "parameters": {"path": "placeholder.npz"}}))
    model_path = save_model_spec(spec, tmp_path / "model.json")
    output = tmp_path / "archive"
    assert main(["sweep", str(model_path), "--vary", "state.initializer.parameters.path=a.npz,b.npz",
                 "--output", str(output)]) == 0
    assert not (output / "resources" / "initial_state.npz").exists()
    (tmp_path / "a.npz").unlink()
    (tmp_path / "b.npz").unlink()
    description = json.loads((output / "sweep.json").read_text())
    table = sweep(model_spec_from_dict(description["model"]), grid=description["grid"], resource_base=output,
                  showprogress=False)
    assert table.population.tolist() == pd.read_csv(output / "table.csv").population.tolist() == [1, 4]


# Follow-up review, state

def _overflow_switch_model(switch):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=np.array([[1, 0, 0], [0, 0, 0]]), restchannels=1, volume_exclusion=False,
                        identity_based=True, capacity=4, traits={"fitness": 1e308, "other": 0.5}),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "trait_switch", "parameters": {"switch": switch}}]),
    ))


@pytest.mark.parametrize("effect", [
    {"value": 10.0, "operation": "multiply"},
    {"value": 10.0, "operation": "multiply", "bounds": [0, None]},
    {"value": 10.0, "operation": "multiply", "at_bounds": "redraw"},
    {"value": 1e308},
])
def test_mutations_that_overflow_do_not_write_infinite_traits(effect):
    model = _overflow_switch_model({"traits": {"fitness": effect}})
    with pytest.raises(ValueError, match="finite"):
        model.step()
    np.testing.assert_array_equal(LatticeState(model.lgca).cells["fitness"], [1e308])


def test_mutations_beyond_the_float_range_are_clipped_to_a_finite_bound():
    model = _overflow_switch_model({"traits": {"fitness": {"value": 10.0, "operation": "multiply",
                                                           "bounds": [0, 1e300]}}})
    model.step()
    np.testing.assert_array_equal(LatticeState(model.lgca).cells["fitness"], [1e300])


@pytest.mark.parametrize("switch", [
    # the second effect of an event is rejected: the first one is not written either
    {"traits": {"other": {"value": 2.0, "operation": "set"},
                "fitness": {"function": lambda rng, size: np.full(size, np.nan)}}},
    # an unknown trait in a later event: the earlier events do not run
    [{"traits": {"other": {"value": 2.0, "operation": "set"}}}, {"traits": {"missing": 0.1}}],
    [{"traits": {"other": {"value": 2.0, "operation": "set"}}},
     {"when": {"missing": 0}, "traits": {"other": {"value": 3.0, "operation": "set"}}}],
])
def test_rejected_mutation_events_change_no_trait(switch):
    model = _overflow_switch_model(switch)
    with pytest.raises(ValueError, match="finite|no trait 'missing'"):
        model.step()
    cells = LatticeState(model.lgca).cells
    assert (cells["other"].tolist(), cells["fitness"].tolist()) == ([0.5], [1e308])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_single_nonfinite_mutation_condition_is_named_as_such(value):
    with pytest.raises(ValueError, match="must be a finite value or a range"):
        _overflow_switch_model({"when": {"other": value}, "traits": {"other": 0.1}}).step()


def _single_cell_model(operators, volume_exclusion, traits=None):
    nodes = np.zeros((2, 3), dtype=int)
    nodes[0, 0] = 1
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=nodes.astype(bool) if volume_exclusion else nodes, restchannels=1,
                        identity_based=True, volume_exclusion=volume_exclusion,
                        traits=traits or {"value": 0.5}, **({} if volume_exclusion else {"capacity": 4})),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False),
    ))


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_trait_that_scales_a_reorientation_term_must_be_finite(volume_exclusion, value):
    model = _single_cell_model([ReorientationSpec(terms=[
        ReorientationTermSpec(name="resting_bias", beta=1.0, trait="value")])], volume_exclusion)
    cells = LatticeState(model.lgca).cells
    cells.set_trait([0], "value", value)
    with pytest.raises(ValueError, match="the trait 'value', which scales the term 'resting_bias', must be"):
        model.step()
    after = LatticeState(model.lgca).cells
    assert (after.label.tolist(), after.index.tolist(), after.channel.tolist()) == (
        cells.label.tolist(), cells.index.tolist(), cells.channel.tolist())


@pytest.mark.parametrize("operator,message", [
    ({"name": "birth_death", "parameters": {"birth_rate": "value"}},
     r"birth_rate \(the trait 'value'\) must be finite for every cell"),
    ({"name": "go_or_rest", "parameters": {"probability": {"cues": [{"name": "density", "kappa": "value"}]}}},
     "the trait 'value' of cue 'density' must be finite"),
    ({"name": "go_or_rest", "parameters": {"probability": {"hill": [{"name": "density", "K": "value"}]}}},
     "the trait 'value' of cue 'density' must be finite"),
    ({"name": "go_or_rest", "parameters": {"probability": {"cues": [{"name": "trait", "trait": "value"}]}}},
     "the trait 'value' of cue 'trait' must be finite"),
])
@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_nonfinite_trait_errors_name_the_trait(operator, message, volume_exclusion):
    model = _single_cell_model([operator], volume_exclusion)
    LatticeState(model.lgca).cells.set_trait([0], "value", np.nan)
    with pytest.raises(ValueError, match=message):
        model.step()


@pytest.mark.parametrize("operation", [lambda state: state.add_cells(2**61 + 2**60),
                                       lambda state: state.divide_cells(1.0, channels="same")])
def test_added_cells_cannot_overflow_the_int64_total_of_a_node(operation):
    lgca = get_lgca(geometry="lin", dims=2, restchannels=1, ve=False, density=0,
                    interaction="only_propagation")
    state = LatticeState(lgca)
    state.counts = np.full(state.counts.shape, 2**61, dtype=np.int64)  # 3 * 2**61 cells per node
    with pytest.raises(ValueError, match="signed int64"):
        operation(state)
    assert np.all(state.counts == 2**61)


def test_cells_spread_over_nodes_whose_sum_exceeds_int64():
    lgca = get_lgca(geometry="lin", dims=2, ve=False, density=0, seed=1, interaction="only_propagation")
    state = LatticeState(lgca)
    state.add_cells(2**62)  # a valid number per node; the lattice total is 2**63
    assert state.density.tolist() == [2**62, 2**62]


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_float_counts_are_checked_by_their_exact_node_totals(dtype):
    lgca = get_lgca(geometry="lin", dims=1, restchannels=0, ve=False, density=0,
                    interaction="only_propagation")
    state = LatticeState(lgca)
    state.counts = np.array([[[2**62, 2**62 - 512]]], dtype=dtype)  # float64 rounds the sum up to 2**63
    assert state.density.tolist() == [2**63 - 512]


@pytest.mark.parametrize("selected", [1, 0])  # also when it selects no cell: the error does not depend on it
def test_two_dimensional_cell_positions_are_named_by_their_shape(selected):
    nodes = np.zeros((2, 3), dtype=bool)
    nodes[:, 2] = True
    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
                                  state=StateSpec(nodes=nodes, restchannels=1, identity_based=True)))
    cells = LatticeState(model.lgca).cells
    with pytest.raises(ValueError, match=rf"1-D array, got shape \({selected}, 1\); use np.flatnonzero"):
        cells.kill(np.argwhere(cells.label != cells.label[0] if selected else cells.label < 0))
    assert len(cells) == 2


def test_negative_cell_positions_count_from_the_end():
    nodes = np.zeros((2, 3), dtype=bool)
    nodes[0, 2] = True
    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
                                  state=StateSpec(nodes=nodes, restchannels=1, identity_based=True)))
    cells = LatticeState(model.lgca).cells
    assert cells.kill([-1]) == 1
    assert len(cells) == 0


@pytest.mark.parametrize("species", [-1, np.uint64(2**64 - 1), True, 0.5])
def test_random_walk_species_are_checked_like_other_species_selections(species):
    with pytest.raises(ValueError, match="species"):
        build_model(ModelSpec(
            space=SpaceSpec(geometry="lin", dims=4), time=TimeSpec(seed=1),
            state=StateSpec(density=0.5, n_species=2),
            dynamics=InteractionPipelineSpec(operators=[
                {"name": "random_walk", "parameters": {"species": species}}]),
        )).step()


@pytest.mark.parametrize("channels", [{"0": "rest", 0: "velocity"}, {"01": "rest", "1": "velocity"},
                                      {True: "rest"}, {-1: "rest"}, {2: "rest"}, {"x": "rest"}])
def test_species_channel_sets_reject_ambiguous_or_unknown_species(channels):
    lgca = get_lgca(geometry="lin", dims=2, restchannels=1, n_species=2, ve=False, density=0,
                    interaction="only_propagation")
    state = LatticeState(lgca)
    with pytest.raises(ValueError, match="species"):
        state.add_cells(1, channels=channels)
    assert not state.counts.any()


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_a_failed_new_family_division_leaves_every_trait_one_value_per_label(volume_exclusion):
    model = _single_cell_model([], volume_exclusion, traits={"fitness": 0.5})
    lgca = model.lgca
    lgca.init_families(type="homogeneous", mutation=False)
    channels = "rest" if volume_exclusion else "same"
    state = LatticeState(lgca, kind="birth_death")
    with pytest.raises(ValueError, match="family tracking with mutations"):
        state.cells.divide([0], channels=channels, new_family=True)
    state = LatticeState(lgca, kind="birth_death")
    state.cells.divide([0], channels=channels)
    state.commit()
    assert len(LatticeState(lgca).cells) == 2
    assert all(len(lgca.props[name]) == lgca.maxlabel + 1 for name in lgca.props)


def _change_identities_but_not_counts(state):
    if state.volume_exclusion:  # the cell in channel 1 divides into the channel of the killed cell
        state.cells.kill([0])
        state.cells.divide([0], channels=[0])
    else:  # the daughter takes its mother's place
        state.cells.divide([0], channels="same")
        state.cells.kill([0])


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_field_states_keep_every_cell_in_its_channel(volume_exclusion):
    nodes = np.zeros((2, 3), dtype=int)
    nodes[0, :1 + volume_exclusion] = 1
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=2), time=TimeSpec(seed=1),
        state=StateSpec(nodes=nodes.astype(bool) if volume_exclusion else nodes, restchannels=1,
                        identity_based=True, volume_exclusion=volume_exclusion,
                        **({} if volume_exclusion else {"capacity": 4})),
    ))
    before = model.lgca.nodes.copy()
    state = LatticeState(model.lgca, kind="field")
    counts = state.counts.copy()
    _change_identities_but_not_counts(state)
    np.testing.assert_array_equal(state.counts, counts)  # the cells per channel are the same
    with pytest.raises(ValueError, match="a field must keep its cells"):
        state.commit()
    assert model.lgca.nodes.tolist() == before.tolist()


# Follow-up review, study

@pytest.mark.parametrize("grid", [
    {"birth_rate": [0.0, 0.9], "dynamics.operators[0].parameters": [{"death_rate": 0.0}]},
    {"dynamics.operators[birth_death]": [{"name": "birth_death"}], "birth_rate": [0.0, 0.9]},
    {"time.steps": [1, 2], "time": [TimeSpec(steps=0)]},
    [{"time": TimeSpec(steps=0), "steps": 1}],
])
def test_sweeps_reject_a_path_inside_another_varied_path(grid):
    # the later key replaced the earlier one: rows were labelled with values that never ran
    spec = ModelSpec(time=TimeSpec(steps=0, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}]))
    with pytest.raises(ValueError, match="same model parameter"):
        sweep(spec, grid=grid, showprogress=False)


@pytest.mark.parametrize("grid", [[], (), iter(())])
def test_sweeps_reject_an_empty_list_of_combinations(grid):
    with pytest.raises(ValueError, match="grid is empty"):
        sweep(ModelSpec(time=TimeSpec(steps=0, seed=1)), grid=grid, showprogress=False)


@pytest.mark.parametrize("path, message", [("space.dims[0]", "space.dims is not a list"),
                                           ("state.density[0]", "state.density is not a list"),
                                           ("dynamics.operators[1]", r"dynamics.operators\[1\] does not exist")])
def test_sweeps_name_the_path_of_an_index_that_does_not_fit(path, message):
    spec = ModelSpec(time=TimeSpec(steps=0, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}]))
    with pytest.raises(KeyError, match=message):
        sweep(spec, grid={path: [1]}, showprogress=False)


def test_explorer_controls_all_parameters_of_an_operator():
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=10), state=StateSpec(density=0.2), time=TimeSpec(seed=1),
                     dynamics=InteractionPipelineSpec(operators=[
                         {"name": "birth_death", "parameters": {"birth_rate": 0.3}}]))
    options = [{"birth_rate": 0.3}, {"death_rate": 0.1}]
    explorer = explore(spec, {"dynamics.operators[0].parameters": options})
    try:
        assert explorer._controls[0].widget.value == options[0]
        explorer.set(**{"dynamics.operators[0].parameters": options[1]})
        explorer.advance(1)
        assert explorer.spec.dynamics.operators[0]["parameters"] == options[1]
    finally:
        explorer.close()


def test_kept_sweep_folders_do_not_differ_only_in_letter_case():
    from lgca.study import _run_folders

    # Windows (and macOS) file systems ignore case: 'boundary=pbc' and 'boundary=PBC' are one folder
    folders = _run_folders([({"p": "pbc"}, 1), ({"p": "PBC"}, 1), ({"p": "x"}, 1)], {"p": "boundary"})
    assert len({folder.casefold() for folder in folders}) == 3
    assert folders[2] == "boundary=x_seed=1"


def test_threads_sweeps_give_every_run_its_own_operator_objects():
    from lgca.plugins import create_plugin
    from lgca.study import final_population

    # validate and setup store the model's capacity on the operator: runs at the same time overwrote it
    operator = create_plugin("birth_death", {"birth_rate": 0.5})
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=30),
                     state=StateSpec(density=1.0, restchannels=1, volume_exclusion=False, capacity=2),
                     time=TimeSpec(steps=30, seed=7),
                     dynamics=InteractionPipelineSpec(operators=[operator], propagation=False))
    measure = {"population": final_population, "shared": lambda result: result.pipeline.operators[0] is operator}
    grid = {"state.capacity": [2, 50]}
    serial = sweep(spec, grid=grid, seeds=[1, 2], measure=measure, showprogress=False)
    threads = sweep(spec, grid=grid, seeds=[1, 2], measure=measure, n_jobs=4, backend="threads",
                    showprogress=False)
    assert not threads.shared.any()
    assert threads.drop(columns="shared").equals(serial.drop(columns="shared"))


def test_threads_sweeps_explain_operator_objects_that_cannot_be_copied():
    import threading

    from lgca.plugins import create_plugin

    operator = create_plugin("birth_death", {"birth_rate": 0.5})
    operator.lock = threading.Lock()
    spec = ModelSpec(time=TimeSpec(steps=1, seed=1), dynamics=InteractionPipelineSpec(operators=[operator]))
    runs = []
    measure = {"population": lambda result: runs.append(1) or 0}
    # every model runs its own copy of an operator object, also one run at a time: before any run
    for n_jobs in (2, 1):
        with pytest.raises(ValueError, match=r"operators\[0\] \('birth_death'\) cannot be copied.*__deepcopy__"):
            sweep(spec, seeds=[1, 2], measure=measure, n_jobs=n_jobs, backend="threads", showprogress=False)
        with pytest.raises(ValueError, match=r"grid\['dynamics\.operators\[0\]'\] value \('birth_death'\) cannot be "
                                             r"copied"):
            sweep(replace(spec, dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}])),
                  grid={"dynamics.operators[0]": [{"name": "birth_death"}, operator]}, seeds=[1, 2],
                  measure=measure, n_jobs=n_jobs, backend="threads", showprogress=False)
        # an operator object in a list of operators, a value of the grid
        with pytest.raises(ValueError, match=r"grid\['dynamics\.operators'\] value\[1\] \('birth_death'\) cannot be "
                                             r"copied.*__deepcopy__"):
            sweep(replace(spec, dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}])),
                  grid={"dynamics.operators": [[{"name": "random_walk"}], [{"name": "random_walk"}, operator]]},
                  seeds=[1, 2], measure=measure, n_jobs=n_jobs, backend="threads", showprogress=False)
    assert runs == []


def test_explorer_warns_when_it_cannot_try_a_change_on_a_copy():
    import threading

    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=10), state=StateSpec(density=0.2), time=TimeSpec(seed=1),
                     dynamics=InteractionPipelineSpec(operators=[
                         {"name": "birth_death", "parameters": {"birth_rate": 0.2}}]))
    explorer = explore(spec, {"birth_rate": (0.0, 1.0)})
    try:
        explorer.lgca.lock = threading.Lock()  # the lattice cannot be copied
        with pytest.warns(UserWarning, match="could not be tried on a copy of the model"):
            explorer.set(birth_rate=0.5)
    finally:
        explorer.close()


def test_explorer_measures_only_scalar_fields():
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(3, 3)), time=TimeSpec(seed=1),
                     state=StateSpec(density=0, fields={"wind": np.ones((3, 3, 2)), "signal": 2}))
    with pytest.raises(ValueError, match="'wind' is a vector field"):
        explore(spec, measure="wind")
    explore(spec, measure="signal").close()


@pytest.mark.parametrize("seed", [1.9, True, -1])
def test_meltdown_scans_do_not_truncate_or_reinterpret_seeds(monkeypatch, seed):
    from lgca.zoo import mutational_meltdown

    monkeypatch.setattr(mutational_meltdown, "_scan_one", lambda arguments: "undecided")
    with pytest.raises(ValueError, match="seeds must be non-negative integers"):
        mutational_meltdown.scan([10], [0.5], seeds=[1, seed])


@pytest.mark.parametrize("seed", [2.0, -1, True])
def test_sweeps_blame_the_seed_of_the_model_not_the_seeds_argument(seed):
    with pytest.raises(ValueError, match=r"time\.seed must be a non-negative integer"):
        sweep(ModelSpec(time=TimeSpec(steps=0, seed=seed)), showprogress=False)


# Follow-up review, cli (review)


def test_command_line_sweep_rejects_two_names_of_one_parameter_before_keeping_files(tmp_path):
    from lgca.cli import main
    from lgca.model import save_model_spec

    spec = ModelSpec(time=TimeSpec(steps=1, seed=1), analysis=AnalysisSpec(observers=[ScalarTimeSeriesRecorder()]))
    path = save_model_spec(spec, tmp_path / "model.json")
    output = tmp_path / "runs"
    assert main(["sweep", str(path), "--vary", "steps=0", "--vary", "time.steps=1", "--keep-files",
                 "--output", str(output)]) == 2
    assert not output.exists()


# Follow-up review, study (review)

def test_threads_sweeps_also_copy_the_operator_objects_of_the_grid():
    from lgca.plugins import create_plugin
    from lgca.study import final_population

    # an operator object given as a value of the grid ran in all runs of its combination at the same time
    operator = create_plugin("birth_death", {"birth_rate": 0.5})
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=30),
                     state=StateSpec(density=1.0, restchannels=1, volume_exclusion=False, capacity=2),
                     time=TimeSpec(steps=30, seed=7),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death"}], propagation=False))
    measure = {"population": final_population, "shared": lambda result: result.pipeline.operators[0] is operator}
    grid = {"dynamics.operators[0]": [operator], "state.capacity": [2, 50]}
    serial = sweep(spec, grid=grid, seeds=[1, 2], measure=measure, showprogress=False)
    threads = sweep(spec, grid=grid, seeds=[1, 2], measure=measure, n_jobs=4, backend="threads",
                    showprogress=False)
    assert not threads.shared.any()
    assert threads.population.tolist() == serial.population.tolist()


# Follow-up review, lead

def test_a_reorientation_term_given_as_an_operator_is_rejected_instead_of_losing_its_beta():
    from lgca.builtin_rules import polar_alignment

    spec = ModelSpec(time=TimeSpec(seed=1), state=StateSpec(density=0.3),
                     dynamics=InteractionPipelineSpec(operators=[polar_alignment(beta=5.0)]))
    with pytest.raises(ValueError, match=r"operators\[0\].*ReorientationSpec\(terms="):
        build_model(spec)


def test_model_files_save_a_numpy_string_trait_of_a_reorientation_term():
    term = ReorientationTermSpec("resting_bias", beta=2.0, trait=np.str_("kappa"))
    spec = ModelSpec(state=StateSpec(identity_based=True, traits={"kappa": 1.0}), time=TimeSpec(seed=1),
                     dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])]))
    loaded = model_spec_from_yaml(model_spec_to_yaml(spec))
    assert loaded.dynamics.operators[0].terms[0].trait == "kappa"


def test_the_schema_rejects_a_negative_seed():
    data = model_spec_to_dict(ModelSpec(time=TimeSpec(seed=1)))
    data["model"]["time"]["seed"] = -1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, load_model_spec_schema())


def test_sweeps_reject_recorders_with_one_output_name_before_any_run():
    spec = ModelSpec(time=TimeSpec(steps=1, seed=1), analysis=AnalysisSpec(observers=[
        ScalarTimeSeriesRecorder({"population": lambda lgca: 99}), PopulationRecorder()]))
    with pytest.raises(ValueError, match="same output names"):  # not once per run, as RuntimeError
        sweep(spec, grid={"steps": [1, 2]}, showprogress=False)


def test_model_files_reject_a_reorientation_term_given_as_an_operator():
    from lgca.builtin_rules import polar_alignment

    spec = ModelSpec(time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[polar_alignment(beta=5.0)]))
    with pytest.raises(TypeError, match="not portable.*ReorientationSpec"):
        model_spec_to_dict(spec)


def test_short_names_skip_operator_objects_which_vary_cannot_change():
    from lgca.plugins import create_plugin
    from lgca.study import vary

    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(6, 6)), state=StateSpec(density=0.3),
                     time=TimeSpec(steps=2, seed=2), dynamics=InteractionPipelineSpec(operators=[
                         create_plugin("polar_alignment", {"beta": 2.0}),
                         {"name": "birth_death", "parameters": {"birth_rate": 0.1}}]))
    assert vary(spec, {"birth_rate": 0.2}).dynamics.operators[1]["parameters"]["birth_rate"] == 0.2
    assert vary(spec, {"density": 0.2}).state.density == 0.2
    table = sweep(spec, grid={"birth_rate": [0.1, 0.2]}, showprogress=False)
    assert table.birth_rate.tolist() == [0.1, 0.2]


def test_mutation_errors_name_the_trait():
    model = _overflow_switch_model({"traits": {"fitness": {"value": 10.0, "operation": "multiply"}}})
    with pytest.raises(ValueError, match="trait 'fitness'.*finite"):
        model.step()
