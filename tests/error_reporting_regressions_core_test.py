"""Errors in a user's own rules keep their type and traceback; lgca's own messages say what to do.

A student who writes her first rule makes the usual mistakes: a typo in a dictionary key, a wrong
operand, a term put where an operator goes. lgca used to catch these errors and report them as its
own (an "unknown operator", a ``ValueError`` that names only the key), so the traceback no longer led
to her line. These tests write such rules and check what the user sees.
"""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model, run_model
from lgca.operator_base import BirthDeathOperator, PluginInfo
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.plugins import register_plugin
from lgca.rules import reorientation_term
from lgca.switching import switch_cue


def _spec(operators, n_species=1, steps=2):
    return ModelSpec(space=SpaceSpec(geometry="square", dims=(6, 6), boundary="periodic"),
                     state=StateSpec(density=1.0, restchannels=1, n_species=n_species),
                     time=TimeSpec(steps=steps, seed=3),
                     dynamics=InteractionPipelineSpec(operators=operators))


def test_keyerror_in_the_constructor_of_a_registered_operator_is_not_an_unknown_operator():
    class Grow(BirthDeathOperator):
        def __init__(self, info, parameters=None):
            super().__init__(info, parameters)
            self.rate = float(self.parameters["rtae"])  # the student's typo

        def apply(self, context, step):
            pass

    info = PluginInfo(name="student.grow", operator_kind="birth_death", backend_families=("classical",),
                      parameters={"rate": 0.1})
    register_plugin(info, lambda parameters: Grow(info, parameters))
    with pytest.raises(KeyError, match="rtae") as caught:
        build_model(_spec([{"name": "student.grow", "parameters": {"rate": 0.2}}]))
    assert "unknown operator" not in str(caught.value)
    assert caught.traceback[-1].name == "__init__"  # the traceback ends in the student's constructor


def test_misspelled_operator_name_suggests_the_registered_one():
    with pytest.raises(ValueError, match=r"unknown operator 'birth_deth' \(did you mean 'birth_death'\?\)"):
        build_model(_spec([{"name": "birth_deth"}]))


def test_rule_of_a_module_that_was_not_imported_says_to_import_it():
    """A model file that names a rule of the user's module, loaded without importing the module."""
    with pytest.raises(ValueError, match=r"import hypoxia_rules .*--plugins hypoxia_rules"):
        build_model(_spec([{"name": "hypoxia_rules.hypoxic_switch"}]))


def test_keyerror_in_a_reorientation_term_stays_a_keyerror():
    weights = {"rest": 1.0}

    @reorientation_term(coupling="rest", name="student.lazy")
    def lazy(state):
        return weights["resting"] * np.ones(state.dims)  # the key is "rest"

    with pytest.raises(KeyError, match="resting") as caught:
        run_model(_spec([ReorientationSpec(terms=[lazy(beta=1.0)])]))
    assert caught.traceback[-1].name == "lazy"


def test_term_that_reads_an_undeclared_field_says_how_to_declare_it():
    @reorientation_term(coupling="flux", name="student.up_oxygen")
    def up_oxygen(state):
        return state.gradient(state.field("oxygen"))

    with pytest.raises(ValueError, match=r"student.up_oxygen: .*declare the field in StateSpec.fields"):
        run_model(_spec([ReorientationSpec(terms=[up_oxygen(beta=1.0)])]))


@pytest.mark.parametrize("registered", [False, True])
def test_reorientation_term_given_as_an_operator_gets_advice_that_works(registered):
    """The error for a term in ``operators`` gives a form to use instead; that form must run."""
    @reorientation_term(coupling="channels", name="student.adhesion")
    def adhesion(state):
        return state.neighbor_values((state.density > 0).astype(float))

    name = "aggregation" if registered else "student.adhesion"  # aggregation is also a one-cue operator
    term = ReorientationTermSpec(name=name, beta=2.0)
    with pytest.raises(ValueError, match="reorientation term") as caught:
        build_model(_spec([term]))
    message = str(caught.value)
    if registered:
        assert "{'name': 'aggregation'" in message
        run_model(_spec([{"name": "aggregation", "parameters": {"beta": 2.0}}]))
    else:
        assert "{'name'" not in message  # the student's term is no operator of its own
        assert "ReorientationSpec(terms=[student.adhesion(beta=2.0)])" in message
        run_model(_spec([ReorientationSpec(terms=[adhesion(beta=2.0)])]))


def _switch_spec(cue):
    rates = [[0, {"max": 0.2, "cues": [cue]}], [0.05, 0]]
    return _spec([{"name": "phenotype_switch", "parameters": {"rates": rates}}], n_species=2)


def test_typeerror_inside_a_switch_cue_keeps_its_type():
    @switch_cue(name="student_crowded")
    def crowded(state):
        return state.density + None  # a bug in the cue's body

    with pytest.raises(TypeError, match="NoneType") as caught:
        run_model(_switch_spec({"name": "student_crowded", "kappa": 5, "theta": 0.5}))
    assert caught.traceback[-1].name == "crowded"


def test_wrong_parameter_of_a_switch_cue_is_named():
    @switch_cue(name="student_crowded")
    def crowded(state, radius=1):
        return state.density * 1.0

    with pytest.raises(ValueError, match=r"cue 'student_crowded': .*'radus'"):
        run_model(_switch_spec({"name": "student_crowded", "kappa": 5, "theta": 0.5, "radus": 2}))


def test_a_trait_name_given_as_a_resting_probability_says_how_to_write_it():
    from lgca.pipeline import (
        InteractionPipelineSpec,
        ReorientationSpec,
        ReorientationTermSpec,
    )

    term = ReorientationTermSpec("resting", parameters={"probability": "p_rest"})
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=10),
                     state=StateSpec(density=0.5, restchannels=1, identity_based=True, traits={"p_rest": 0.3}),
                     time=TimeSpec(steps=1, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])]))
    with pytest.raises(TypeError, match=r"\{'max': 'p_rest'\}"):
        build_model(spec)
