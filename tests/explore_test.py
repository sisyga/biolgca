"""lgca.explore: a model running live in a notebook, with controls for its parameters."""

import doctest
import time

import ipywidgets as widgets
import matplotlib
import numpy as np
import pytest

import lgca
import lgca.explorer
from lgca.explorer import explore
from lgca.fields import PDESpec
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model, run_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import _total_population

matplotlib.use("Agg")

PNG = b"\x89PNG"


def _spec(geometry="square", dims=(12, 12), operators=None, **state):
    state.setdefault("density", 0.3)
    state.setdefault("restchannels", 1)
    return ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims), state=StateSpec(**state),
                     time=TimeSpec(steps=5, seed=3),
                     dynamics=InteractionPipelineSpec(operators=operators or [{"name": "random_walk"}]))


def _growth(**state):
    return _spec(operators=[{"name": "birth_death", "parameters": {"birth_rate": 0.2}}, {"name": "random_walk"}],
                 **state)


def _chemotaxis(geometry="square", dims=(12, 12), solver="implicit"):
    return _spec(geometry, dims, fields={"signal": 0.0}, operators=[
        PDESpec(field="signal", diffusion=1.0, decay=0.1, cells=[{"production": 0.2}], solver=solver),
        ReorientationSpec(terms=[ReorientationTermSpec("random_walk"),
                                 ReorientationTermSpec("chemotaxis", beta=2.0, parameters={"field": "signal"})])])


@pytest.fixture
def close():
    explorers = []
    yield explorers.append
    for explorer in explorers:
        explorer.close()


def test_the_examples_of_the_module_run():
    assert doctest.testmod(lgca.explorer).failed == 0


def test_it_is_exported():
    assert lgca.explore is explore


@pytest.mark.parametrize("geometry, dims", [("lin", 30), ("square", (12, 10)), ("hex", (12, 10))])
@pytest.mark.parametrize("family", [
    {},
    {"volume_exclusion": False},
    {"identity_based": True},
    {"identity_based": True, "volume_exclusion": False},
    {"n_species": 2},
])
def test_every_view_draws_on_every_lattice_and_family(geometry, dims, family, close):
    spec = _spec(geometry, dims, operators=[{"name": "random_walk"}], fields={"signal": 0.5}, **family)
    explorer = explore(spec)
    close(explorer)
    expected = ["density"] + ([f"density: species {i}" for i in range(2)] if "n_species" in family else []) + [
        "flux", "signal"]
    assert list(explorer._view.options) == expected
    for view in expected:
        explorer._view.value = view
        explorer.advance(2)
        assert explorer.frame().startswith(PNG)
        assert explorer._image.value == explorer.frame()
    assert explorer.step == 2 * len(expected)
    assert explorer._status.value == ""


def test_advance_runs_the_model_as_run_model_does(close):
    spec = _growth()
    explorer = explore(spec, steps_per_frame=3)
    close(explorer)
    explorer.advance()  # the steps per frame
    explorer.advance(2)
    reference = run_model(spec, showprogress=False)  # its five steps
    np.testing.assert_array_equal(explorer.lgca.nodes, reference.lgca.nodes)
    assert explorer._steps == list(range(6))
    assert explorer._series["population"][-1] == _total_population(explorer.lgca)


def test_a_parameter_of_the_dynamics_changes_the_running_model(close):
    explorer = explore(_growth(), {"birth_rate": (0.0, 1.0), "death_rate": (0.0, 1.0)})
    close(explorer)
    explorer.advance(3)
    lattice, nodes = explorer.lgca, explorer.lgca.nodes.copy()
    cells = _total_population(lattice)
    explorer.set(birth_rate=0.0)  # no division, no death: the cells stay
    assert explorer.step == 3 and explorer.lgca is lattice
    np.testing.assert_array_equal(lattice.nodes, nodes)
    explorer.advance(5)
    assert _total_population(lattice) == cells
    explorer.set(death_rate=1.0)
    explorer.advance(1)
    assert _total_population(lattice) == 0
    assert explorer.spec.dynamics.operators[0]["parameters"] == {"birth_rate": 0.0, "death_rate": 1.0}
    assert explorer._changes == [3, 8]


def test_a_steady_field_is_solved_again_when_its_parameters_change(close):
    spec = _chemotaxis(solver="steady")
    explorer = explore(spec, {"decay": (0.01, 1.0)}, view="signal")
    close(explorer)
    explorer.advance(4)
    explorer.set(decay=0.5)
    # the same cells in a model built with the new decay
    nodes = explorer.lgca.nodes[explorer.lgca.nonborder]
    rebuilt = build_model(lgca.study.vary(spec, {"decay": 0.5, "state.nodes": nodes, "state.density": None}))
    np.testing.assert_allclose(explorer.lgca.signal, rebuilt.lgca.signal, rtol=1e-9)


def test_other_values_build_the_model_again(close):
    spec = _growth()
    explorer = explore(spec, {"density": (0.1, 2.0), "dims": [(12, 12), (20, 8)]})
    close(explorer)
    explorer.advance(4)
    explorer.set(density=1.5)
    assert explorer.step == 0 and explorer._steps == [0]
    np.testing.assert_array_equal(explorer.lgca.nodes,
                                  build_model(lgca.study.vary(spec, {"density": 1.5})).lgca.nodes)
    explorer.set(dims=(20, 8))
    assert explorer.lgca.dims == (20, 8) and explorer.frame().startswith(PNG)


def test_reset_starts_again_with_the_current_values(close):
    spec = _growth()
    explorer = explore(spec, {"birth_rate": (0.0, 1.0)})
    close(explorer)
    explorer.advance(3)
    explorer.set(birth_rate=0.5)
    explorer.reset()
    assert explorer.step == 0 and explorer._changes == []
    reference = build_model(lgca.study.vary(spec, {"birth_rate": 0.5}))
    np.testing.assert_array_equal(explorer.lgca.nodes, reference.lgca.nodes)
    for _ in range(3):
        reference.step()
    explorer.advance(3)
    np.testing.assert_array_equal(explorer.lgca.nodes, reference.lgca.nodes)


@pytest.mark.parametrize("name, control, value", [
    ("birth_rate", (-1.0, 1.0), -0.5),  # live: rejected by the operator
    ("density", (0.1, 20.0), 19.0),  # a new model: more cells per node than channels
])
def test_an_invalid_value_is_explained_and_the_control_goes_back(name, control, value, close):
    explorer = explore(_growth(), {name: control})
    close(explorer)
    explorer.advance(2)
    before, nodes = explorer.spec, explorer.lgca.nodes.copy()
    with pytest.raises(ValueError):
        explorer.set(**{name: value})
    assert "ValueError" in explorer._status.value
    assert explorer.spec is before and explorer.step == 2
    np.testing.assert_array_equal(explorer.lgca.nodes, nodes)
    assert explorer._controls[0].widget.value != value
    explorer.advance(1)


def test_controls_are_made_from_ranges_lists_and_widgets(close):
    log = widgets.FloatLogSlider(min=-3, max=0)
    explorer = explore(_chemotaxis(), {
        "dynamics.operators[1].terms[chemotaxis].beta": (0.0, 5.0),
        "restchannels": (0, 3),
        "geometry": ["square", "hex"],
        "decay": log,
    })
    close(explorer)
    beta, rest, geometry, decay = (control.widget for control in explorer._controls)
    assert isinstance(beta, widgets.FloatSlider) and beta.value == 2.0 and beta.description == "beta"
    assert beta.continuous_update and not rest.continuous_update  # a new model only when released
    assert isinstance(rest, widgets.IntSlider) and rest.value == 1
    assert isinstance(geometry, widgets.Dropdown) and geometry.value == "square"
    assert decay is log and decay.value == pytest.approx(0.1)
    explorer.set(geometry="hex")
    assert explorer.lgca.geometry == "hex"
    explorer.set(beta=4.0)  # by its label
    assert explorer.spec.dynamics.operators[1].terms[1].beta == 4.0


def test_a_control_starts_at_the_default_of_a_parameter_the_model_leaves_out(close):
    explorer = explore(_growth(), {"death_rate": (0.0, 1.0)})
    close(explorer)
    assert explorer._controls[0].widget.value == 0.0


def test_a_value_outside_its_range_is_moved_into_it_with_a_warning(close):
    with pytest.warns(UserWarning, match="lies outside its control"):
        explorer = explore(_growth(), {"birth_rate": (0.3, 1.0)})
    close(explorer)
    assert explorer.spec.dynamics.operators[0]["parameters"]["birth_rate"] == 0.3


def test_labels_tell_equal_names_apart(close):
    spec = _spec(operators=[ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=1.0),
                                                     ReorientationTermSpec("persistent_walk", beta=0.5)])])
    explorer = explore(spec, {"dynamics.operators[0].terms[0].beta": (0.0, 3.0),
                              "dynamics.operators[0].terms[1].beta": (0.0, 3.0)})
    close(explorer)
    assert [control.widget.description for control in explorer._controls] == ["terms[0].beta", "terms[1].beta"]


@pytest.mark.parametrize("controls, error, message", [
    ({"beta": (0.0, 1.0)}, KeyError, "no field or parameter"),
    ({"birth_rate": (1.0, 0.0)}, ValueError, "min < max"),
    ({"birth_rate": "fast"}, TypeError, "must be"),
    ({"birth_rate": []}, ValueError, "lists no values"),
    ({"birth_rate": (0.0, 1.0), "dynamics.operators[0].parameters.birth_rate": (0.0, 1.0)}, ValueError,
     "the same place"),
])
def test_invalid_controls_are_explained(controls, error, message):
    with pytest.raises(error, match=message):
        explore(_growth(), controls)


@pytest.mark.parametrize("arguments, error, message", [
    ({"view": "oxygen"}, ValueError, "not available"),
    ({"measure": "oxygen"}, ValueError, "unknown measure"),
    ({"measure": {"cells": 3}}, TypeError, "function of the LGCA object"),
    ({"steps_per_frame": 0}, ValueError, "positive integer"),
    ({"window": 1}, ValueError, "at least 2"),
])
def test_invalid_options_are_explained(arguments, error, message):
    with pytest.raises(error, match=message):
        explore(_growth(), **arguments)


def test_it_takes_a_model_spec():
    with pytest.raises(TypeError, match="ModelSpec"):
        explore(lgca.get_lgca(geometry="lin", dims=10))


@pytest.mark.parametrize("geometry", ["cubic", "moore"])
@pytest.mark.parametrize("family", [{}, {"volume_exclusion": False}, {"identity_based": True}, {"n_species": 2}])
def test_every_view_draws_in_3d_and_in_every_plane(geometry, family, close):
    spec = _spec(geometry, (6, 5, 4), fields={"signal": 0.5}, **family)
    explorer = explore(spec)
    close(explorer)
    for show in explorer._slice.options:
        explorer._slice.value = show
        for view in explorer._view.options:
            explorer._view.value = view
            explorer.advance(1)
            assert explorer.frame().startswith(PNG)
    assert explorer._status.value == ""


def test_a_plane_shows_the_values_of_its_nodes(close):
    explorer = explore(_chemotaxis("cubic", (6, 5, 4)), view="signal", slice=("y", 1))
    close(explorer)
    explorer.advance(2)
    assert explorer._panel.axes.get_title() == "$y = 1$"
    signal = explorer.lgca.signal[explorer.lgca.nonborder]
    np.testing.assert_allclose(explorer._panel.artist.get_array(), signal[:, 1, :].T)
    explorer._position.value = 3
    np.testing.assert_allclose(explorer._panel.artist.get_array(), signal[:, 3, :].T)

    explorer._view.value = "density"
    density = explorer.lgca.cell_density[explorer.lgca.nonborder]
    np.testing.assert_array_equal(explorer._panel.artist.get_array(), density[:, 3, :].T)


def test_a_plane_shows_the_flux_in_the_plane(close):
    explorer = explore(_spec("cubic", (4, 4, 4), density=0.0), view="flux", slice="x")
    close(explorer)
    nodes = np.zeros_like(explorer.lgca.nodes)
    inside = tuple(1 + np.array([2, 1, 3]))  # x = 2, y = 1, z = 3 with a border of one node
    nodes[inside + (0,)] = True  # +x: normal to the plane, no flux in it
    nodes[inside[:1] + (1, 1) + (2,)] = True  # +y at x = 2, y = 0, z = 0: flux in the plane
    explorer.lgca.nodes[...] = nodes
    explorer.lgca.update_dynamic_fields()
    explorer._redraw()

    assert explorer._position.value == 2 and explorer._position.description == "x"
    colours = explorer._panel.artist.get_facecolor().reshape(4, 4, 4)  # the plane's (y, z) nodes
    np.testing.assert_allclose(colours[1, 3], [0.5, 0.5, 0.5, 1])  # a cell, but no flux in the plane
    assert colours[0, 0, 3] == 1 and not np.allclose(colours[0, 0, :3], 0.5)  # coloured by its direction
    assert colours[2, 2, 3] == 0  # empty


def test_the_perspective_view_shows_the_nodes_with_cells_and_turns(close):
    explorer = explore(_spec("cubic", (6, 5, 4), volume_exclusion=False, capacity=8))
    close(explorer)
    explorer.advance(1)
    density = explorer.lgca.cell_density[explorer.lgca.nonborder]
    points = np.column_stack(explorer._panel.artist._offsets3d)
    np.testing.assert_array_equal(np.sort(points, axis=0), np.sort(np.argwhere(density > 0), axis=0))
    np.testing.assert_array_equal(np.sort(explorer._panel.artist.get_array()), np.sort(density[density > 0]))
    assert explorer._position.layout.display == "none" and explorer._turn.layout.display is None
    explorer._turn.value = 30
    assert explorer._panel.axes.azim == 30


def test_the_perspective_view_shows_where_a_field_is_high(close):
    explorer = explore(_chemotaxis("cubic", (6, 5, 4)), view="signal")
    close(explorer)
    assert explorer._panel.artist is None  # the field starts uniform
    assert "everywhere" in explorer._panel.axes.get_title()
    explorer.advance(3)
    signal = explorer.lgca.signal[explorer.lgca.nonborder]
    high = signal >= (signal.min() + signal.max()) / 2
    assert len(explorer._panel.artist.get_array()) == high.sum()


@pytest.mark.parametrize("option, message", [
    ("w", "must be 'x', 'y' or 'z'"),
    (("z", 1.5), "must be 'x', 'y' or 'z'"),
    (("z", 4), "outside the lattice"),
])
def test_invalid_planes_are_explained(option, message):
    with pytest.raises(ValueError, match=message):
        explore(_spec("cubic", (6, 5, 4)), slice=option)


def test_only_3d_lattices_have_planes(close):
    with pytest.raises(ValueError, match="cuts 3D lattices"):
        explore(_growth(), slice="z")
    explorer = explore(_growth())
    close(explorer)
    assert explorer._slice.layout.display == "none" and explorer._turn.layout.display == "none"


def test_measures_are_plotted_every_step(close):
    spec = _chemotaxis()
    explorer = explore(spec, measure=["population", "signal"])
    close(explorer)
    explorer.advance(3)
    signal = explorer.lgca.signal[explorer.lgca.nonborder]
    assert list(explorer._series) == ["population", "mean signal"]
    assert explorer._series["mean signal"][-1] == pytest.approx(signal.mean())
    assert len(explorer._series["population"]) == 4

    moving = explore(spec, measure={"moving": lambda lattice: lattice.nodes[lattice.nonborder][..., :4].sum()})
    close(moving)
    moving.advance(1)
    assert moving._series["moving"][-1] == moving.lgca.nodes[moving.lgca.nonborder][..., :4].sum()

    alone = explore(spec, measure=None)
    close(alone)
    assert alone._series_axes is None and alone.frame().startswith(PNG)


def test_the_colour_scale_grows_with_the_cells(close):
    explorer = explore(_growth(volume_exclusion=False, density=0.5), {"birth_rate": (0.0, 1.0)})
    close(explorer)
    explorer.set(birth_rate=1.0)
    top = explorer._panel.vmax
    explorer.advance(12)
    assert explorer._panel.vmax >= explorer.lgca.cell_density[explorer.lgca.nonborder].max() > top


def test_a_kymograph_keeps_every_view_when_switching(close):
    explorer = explore(_chemotaxis("lin", 40), window=10)
    close(explorer)
    explorer.advance(4)
    explorer._view.value = "signal"
    explorer.advance(8)
    history = explorer._histories["density"]
    np.testing.assert_array_equal(history[-1], explorer.lgca.cell_density[explorer.lgca.nonborder])
    assert not np.isnan(history).any()  # 13 steps fill the window of 10
    np.testing.assert_allclose(explorer._histories["signal"][-1],
                               explorer.lgca.signal[explorer.lgca.nonborder])


def _wait(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise TimeoutError
        time.sleep(0.01)


def test_play_runs_frames_until_paused_and_one_model_runs_at_a_time(close):
    first, second = explore(_growth(), interval=0.0), explore(_growth(), interval=0.0)
    close(first)
    close(second)
    first.play()
    _wait(lambda: first.step >= 3)
    second.play()
    assert not first._play.value
    first._thread.join()
    stopped = first.step
    _wait(lambda: second.step >= 3)
    second.pause()
    second._thread.join()
    assert first.step == stopped and second._play.description == "Play"


def test_a_failing_step_pauses_and_is_shown(close):
    explorer = explore(_growth(), interval=0.0)
    close(explorer)

    def fail(**timing):
        raise RuntimeError("the model broke")

    explorer.model.step = fail
    explorer.play()
    _wait(lambda: not explorer._play.value)
    assert "the model broke" in explorer._status.value


def test_the_current_spec_runs_on_its_own(close):
    explorer = explore(_growth(), {"birth_rate": (0.0, 1.0)})
    close(explorer)
    explorer.set(birth_rate=0.7)
    result = run_model(explorer.spec, showprogress=False)
    assert result.spec.dynamics.operators[0]["parameters"]["birth_rate"] == 0.7


@pytest.mark.parametrize("geometry, dims, volume_exclusion", [("hex", (12, 10), False), ("square", (10, 10), True),
                                                              ("lin", 30, False)])
def test_identity_based_models_show_the_mean_trait_per_node(geometry, dims, volume_exclusion, close):
    operators = [{"name": "go_or_grow_kappa", "parameters": {"kappa": 2.0}}]
    spec = _spec(geometry, dims, operators=operators, identity_based=True, volume_exclusion=volume_exclusion,
                 restchannels=1, **({} if volume_exclusion else {"capacity": 8}))
    explorer = explore(spec, view="mean kappa", measure=["population", "kappa"])
    close(explorer)
    assert "mean kappa" in explorer._view.options
    explorer.advance(3)
    assert explorer.frame().startswith(PNG)
    from lgca.lattice_state import LatticeState
    from lgca.plot_data import mean_trait

    cells = LatticeState(explorer.lgca).cells
    assert explorer._series["mean kappa"][-1] == pytest.approx(np.mean(cells["kappa"]))
    means = mean_trait(explorer.lgca, "kappa")
    occupied = explorer.lgca.cell_density[explorer.lgca.nonborder] > 0
    assert np.isnan(means[~occupied]).all() and np.isfinite(means[occupied]).all()
    first = tuple(np.argwhere(occupied)[0])
    here = np.all(np.stack(cells.node, -1) == np.array(first), axis=-1)
    assert means[first] == pytest.approx(np.mean(cells["kappa"][here]))


def test_an_unknown_trait_measure_is_explained():
    spec = _spec(operators=[{"name": "random_walk"}], identity_based=True)
    with pytest.raises(KeyError, match="no trait 'speed'"):
        explore(spec, measure="speed")
