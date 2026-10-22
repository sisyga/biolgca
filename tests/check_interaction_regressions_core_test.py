"""check_interaction as a researcher uses it: the one-line test of a new rule must catch a broken rule.

Each test writes a rule the way the docs suggest, gets it wrong in a common way, and checks that
``check_interaction`` fails, and that its report leads to the mistake.
"""

import numpy as np
import pytest

from lgca.rules import interaction, reorientation_term
from lgca.testing import InteractionCheckError, check_interaction


def test_identity_based_trait_drift_is_checked_on_identity_based_models():
    """A phenotype switch of identity-based models used to be checked on no model at all ("0 models
    checked, all passed"), so a rule that kills cells, which a phenotype switch must not, passed."""
    @interaction(kind="phenotype_switch", families=("ib", "nove_ib"), name="evolution.drift")
    def drift(state, std=0.05):
        cells = state.cells
        new = np.clip(cells["r_b"] + std * state.rng.normal(size=len(cells)), 0, 1)
        cells.set_trait(np.ones(len(cells), dtype=bool), "r_b", new)

    report = check_interaction(drift, traits={"r_b": 0.2}, geometries=("square",))
    assert [case for case, _ in report.rows] == [f"square, {family}, 1 species, {boundary}"
                                                 for family in ("ib", "nove_ib")
                                                 for boundary in ("periodic", "reflecting")]

    @interaction(kind="phenotype_switch", families=("ib", "nove_ib"), name="evolution.culling_drift")
    def culling_drift(state):
        state.cells.kill(state.rng.random(len(state.cells)) < 0.5)  # only birth_death may remove cells

    with pytest.raises(InteractionCheckError, match="4 problems"):
        check_interaction(culling_drift, traits={"r_b": 0.2}, geometries=("square",))


def test_no_model_to_check_is_an_error_not_a_pass():
    # identity-based models have one species, so a rule for two species of them has no model
    @interaction(kind="birth_death", families=("ib",), n_species=2, name="evolution.death")
    def death(state):
        state.cells.kill(state.rng.random(len(state.cells)) < 0.1)

    with pytest.raises(ValueError, match="no model to check"):
        check_interaction(death)


def test_the_error_of_the_rule_is_the_cause_and_its_line_is_reported():
    @interaction(kind="birth_death", families=("classical",), name="student.grow")
    def grow(state, rate=0.1):
        parameters = {"rate": rate}
        state.divide_cells(parameters["rtae"])  # the typo

    with pytest.raises(InteractionCheckError) as caught:
        check_interaction(grow, geometries=("square",))
    cause = caught.value.__cause__
    assert isinstance(cause, KeyError) and cause.args == ("rtae",)
    assert cause.__traceback__.tb_next is not None  # the traceback leads into the rule
    assert f"({__name__.rpartition('.')[2]}.py:" in str(caught.value)  # this file and the rule's line


def test_rules_that_read_a_field_are_checked_with_the_fields_given():
    """The hypoxia response of the fields how-to: cells die where oxygen is scarce."""
    @interaction(kind="birth_death", families=("classical", "nove"), name="tumour.hypoxic_death")
    def hypoxic_death(state, r_d=0.3, threshold=0.2):
        state.remove_cells(r_d * (state.field("oxygen") < threshold))

    with pytest.raises(InteractionCheckError, match=r"oxygen.*check_interaction\(\.\.\., fields="):
        check_interaction(hypoxic_death, geometries=("square",))
    # a gradient along the first axis: a function of the lattice shape, so that every geometry has one
    check_interaction(hypoxic_death, fields={"oxygen": lambda dims: np.linspace(0, 1, dims[0])[:, None]},
                      geometries=("square", "hex"))
    # a constant field, with the growth of one step against the analytic value: all cells see oxygen 0.1
    check_interaction(hypoxic_death, fields={"oxygen": 0.1}, geometries=("square",), families=("nove",),
                      expected_growth=-0.3)


def test_a_missing_trait_is_given_to_check_interaction():
    @interaction(kind="birth_death", families=("ib",), name="evolution.tradeoff")
    def tradeoff(state):
        cells = state.cells
        cells.divide(state.rng.random(len(cells)) < cells["r_b"] * (1 - cells["p_rest"]))

    with pytest.raises(InteractionCheckError, match=r"no trait 'p_rest'.*check_interaction\(\.\.\., traits="):
        check_interaction(tradeoff, traits={"r_b": 0.2}, geometries=("square",))
    check_interaction(tradeoff, traits={"r_b": 0.2, "p_rest": 0.5}, geometries=("square",))


def test_reorientation_terms_are_checked_for_reproducibility():
    """A term that draws from numpy.random gives runs that differ for the same seed."""
    @reorientation_term(coupling="channels", name="student.adhesion")
    def adhesion(state):
        return state.neighbor_values((state.density > 0).astype(float))

    @reorientation_term(coupling="rest", name="student.jitter")
    def jitter(state):
        return np.random.random(state.dims)  # not state.rng

    report = check_interaction(adhesion, {"beta": 2.0}, geometries=("square",))
    assert len(report.rows) == 12  # classical and nove with 1 and 2 species, ib, nove_ib; two boundaries
    with pytest.raises(InteractionCheckError, match="the same seed gave different results"):
        check_interaction(jitter, geometries=("square",))
