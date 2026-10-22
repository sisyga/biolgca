"""What a newcomer sees while setting up and inspecting a model: messages, reprs and provenance.

The scenarios follow a student who moves from ``get_lgca`` to ``ModelSpec``: she misspells names,
prints results in a notebook, sweeps a parameter that becomes NaN, and writes her first rule with a
name that a built-in rule already has.
"""

import importlib.metadata
import threading
import warnings

import numpy as np
import pytest

from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    UncheckedChangeWarning,
    _package_version,
    build_model,
    run_model,
)
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec
from lgca.plugins import (
    ParameterSpec,
    PluginInfo,
    describe_plugin,
    validate_plugin_parameters,
)
from lgca.rules import interaction, reorientation_term
from lgca.simulation import DensityRecorder, PopulationRecorder
from lgca.study import final_population, vary


def _growth(n_species=1, steps=5, **state):
    return ModelSpec(space=SpaceSpec(geometry="square", dims=(8, 8)),
                     state=StateSpec(density=1.0, restchannels=1, n_species=n_species, **state),
                     time=TimeSpec(steps=steps, seed=7),
                     dynamics=InteractionPipelineSpec(operators=[
                         {"name": "birth_death", "parameters": {"birth_rate": 0.2, "death_rate": 0.05}}]),
                     analysis=AnalysisSpec(observers=[PopulationRecorder(), DensityRecorder()]))


def test_version_of_an_uninstalled_checkout_is_not_a_made_up_release(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert _package_version() == "unknown"


@pytest.mark.parametrize("n_species", [1, 2])
def test_population_is_the_sum_of_the_recorded_density(n_species):
    """Three reads of the population (recorder, sweep measure, recorded density) agree, for one species and
    for several, and are whole numbers."""
    result = run_model(_growth(n_species), showprogress=False)
    density = result.data["density"]
    population = result.data["population"]
    axes = tuple(range(1, density.ndim))
    np.testing.assert_array_equal(population, density.sum(axis=axes))
    final = final_population(result)
    assert type(final) is int and final == population[-1] > 0


def test_change_that_cannot_be_tried_on_a_copy_warns_with_its_own_category():
    model = build_model(_growth())
    model.lgca.lock = threading.Lock()  # e.g. a rule that keeps a lock on the lattice: no copy
    with pytest.warns(UncheckedChangeWarning, match="applied unchecked"):
        model.reconfigure({"birth_rate": 0.4})
    with warnings.catch_warnings():  # the category lets a test suite make it an error
        warnings.simplefilter("error", UncheckedChangeWarning)
        with pytest.raises(UncheckedChangeWarning):
            model.reconfigure({"birth_rate": 0.3})


def test_result_and_state_print_compactly_in_a_notebook():
    nodes = np.zeros((40, 40, 5), dtype=np.int64)
    nodes[20, 20, -1] = 3
    state = StateSpec(nodes=nodes, restchannels=1, volume_exclusion=False,
                      fields={"oxygen": np.ones((40, 40)), "signal": 0.5})
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(40, 40)), state=state, time=_growth().time,
                     dynamics=_growth().dynamics, analysis=_growth().analysis)
    result = run_model(spec, showprogress=False)
    text = repr(result)
    assert len(text) < 400 and "5 steps" in text and "seed 7" in text and "population" in text
    state = repr(spec.state)
    assert "array of shape (40, 40, 5), int64" in state and "'signal': 0.5" in state and "restchannels=1" in state
    assert repr(StateSpec()) == "StateSpec()"


def test_misspelled_short_name_in_vary_suggests_the_name():
    spec = _growth()
    with pytest.raises(KeyError, match=r"'birth_rat' \(did you mean 'birth_rate'\?\)"):
        vary(spec, {"birth_rat": 0.3})
    assert vary(spec, {"birth_rate": 0.3}).dynamics.operators[0]["parameters"]["birth_rate"] == 0.3


def test_unknown_parameter_lists_the_parameters_of_the_rule():
    with pytest.raises(ValueError, match=r"random_walk.beta; random_walk takes: "):
        build_model(ModelSpec(dynamics=InteractionPipelineSpec(operators=[
            {"name": "random_walk", "parameters": {"beta": 1.0}}])))

    @reorientation_term(coupling="channels", name="student.adhesion")
    def adhesion(state, sensitivity=1.0):
        return sensitivity * state.neighbor_values(state.density.astype(float))

    with pytest.raises(ValueError, match=r"student.adhesion.strength.*takes: sensitivity"):
        adhesion(beta=2.0, strength=1.0)


@pytest.mark.parametrize("decorator", ["interaction", "reorientation_term"])
def test_a_built_in_name_is_replaced_as_the_error_message_says(decorator):
    def chemotaxis(state, rate=0.1):
        return np.zeros(state.dims + (state.c.shape[0],))

    if decorator == "reorientation_term":
        with pytest.raises(ValueError, match="pass replace=True"):
            reorientation_term(coupling="flux", name="chemotaxis")(chemotaxis)
        term = reorientation_term(coupling="flux", name="chemotaxis", replace=True)(chemotaxis)
        run_model(ModelSpec(time=TimeSpec(steps=1, seed=1), dynamics=InteractionPipelineSpec(
            operators=[ReorientationSpec(terms=[term(beta=1.0)])])), showprogress=False)
    else:
        def walk(state):
            state.shuffle_cells()

        with pytest.raises(ValueError, match="pass replace=True"):
            interaction(kind="reorientation", families="classical", name="random_walk")(walk)
        rule = interaction(kind="reorientation", families="classical", name="random_walk", replace=True)(walk)
        assert describe_plugin("random_walk") is rule.info


def test_a_rate_that_became_nan_says_so():
    """A sweep over theta that reaches NaN, and a rate above 1, give messages that differ."""
    @interaction(kind="phenotype_switch", families="classical", n_species=2, name="student.switch")
    def switch(state, rate=0.1):
        state.switch_phenotype([[0, rate], [0, 0]])

    def run(rate):
        run_model(vary(ModelSpec(state=StateSpec(density=1.0, n_species=2), time=TimeSpec(steps=1, seed=1),
                                 dynamics=InteractionPipelineSpec(operators=[switch()])), {"rate": rate}),
                  showprogress=False)

    with pytest.raises(ValueError, match="not NaN or infinity"):
        run(float("nan"))
    with pytest.raises(ValueError, match="got values from 0 to 1.5"):
        run(1.5)


def test_forgotten_species_axis_is_named():
    @reorientation_term(coupling="channels", name="student.crowd")
    def crowd(state):
        return state.neighbor_values(state.counts.sum(-1))  # dims + (n_species,): the species axis is left

    with pytest.raises(ValueError, match=r"species axis.*state.density"):
        run_model(ModelSpec(time=TimeSpec(steps=1, seed=1), dynamics=InteractionPipelineSpec(
            operators=[ReorientationSpec(terms=[crowd(beta=1.0)])])), showprogress=False)


def test_parameter_type_labels_are_checked_and_unused_fields_deprecated():
    info = PluginInfo(name="student.decay", operator_kind="birth_death", backend_families=("classical",),
                      parameters={"p": {"default": 0.1, "type_label": "probability"},
                                  "n": {"default": 1, "type_label": "positive integer"}})
    validate_plugin_parameters(info, {"p": 0.5, "n": 3})
    with pytest.raises(ValueError, match="student.decay.p must be a probability"):
        validate_plugin_parameters(info, {"p": 1.5})
    with pytest.raises(ValueError, match="student.decay.n must be a positive integer"):
        validate_plugin_parameters(info, {"n": 0})
    with pytest.warns(DeprecationWarning, match="ParameterSpec.validator is deprecated"):
        ParameterSpec(default=0.1, validator="a probability")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ParameterSpec(default=0.1, type_label="probability", description="a rate")  # not deprecated


def test_example_plugin_describes_its_parameters():
    parameters = describe_plugin("custom.rest_or_align").parameter_specs
    assert parameters["alpha"].description == "Preference for resting over alignment."
    assert parameters["beta"].description.startswith("Sensitivity to the directional cue")
