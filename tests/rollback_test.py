"""A step is applied as a whole or not at all.

A step that raises, wherever in the step and also when interrupted, leaves the
model as it was before the step: cells (in their order), traits, labels,
families, fields with their ghost nodes, the random stream and the step
counter. Steps retried after the failure give the same run as steps that never
failed.
"""

import copy

import numpy as np
import pytest

from lgca import get_lgca, interaction
from lgca.lattice_state import LatticeState
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.plugins import default_registry

_FAMILIES = {  # (identity_based, volume_exclusion, n_species)
    "classical": (False, True, 1), "nove": (False, False, 1), "multispecies": (False, True, 2),
    "ib": (True, True, 1), "nove_ib": (True, False, 1)}
_FAIL = {"at": None, "error": RuntimeError}


@pytest.fixture(autouse=True)
def _rules():
    plugins, aliases = dict(default_registry._plugins), dict(default_registry._aliases)
    families = ("classical", "nove", "ib", "nove_ib")

    @interaction(kind="birth_death", families=families, name="turnover_with_families")
    def turnover_with_families(state):
        if state.identity_based:  # traits, labels and families change: all of it must come back
            cells = state.cells
            cells.set_trait(np.arange(len(cells)) % 3 == 0, "r_b", 0.3)
            cells.divide(state.rng.random(len(cells)) < 0.3, new_family=True)
            cells.kill(state.rng.random(len(cells)) < 0.2)
        else:
            state.divide_cells(0.3)
            state.remove_cells(0.2)

    @interaction(kind="field", families=families, name="mark")
    def mark(state):
        state.set_field("u", state.field("u") * 0.9 + state.density)

    @interaction(kind="birth_death", families=families, name="failing")
    def failing(state):
        state.remove_cells(0.1)  # draws random numbers and changes cells before it fails
        if state.step == _FAIL["at"]:
            _FAIL["at"] = None  # once: the retried step succeeds
            raise _FAIL["error"]("injected")

    yield
    default_registry._plugins, default_registry._aliases = plugins, aliases
    _FAIL["at"], _FAIL["error"] = None, RuntimeError


def _model(family, boundary="reflecting"):
    identity, volume_exclusion, n_species = _FAMILIES[family]
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(8, 6), boundary=boundary),
        state=StateSpec(density=0.4 if volume_exclusion else 2.0, restchannels=1, identity_based=identity,
                        volume_exclusion=volume_exclusion, n_species=n_species,
                        traits={"r_b": 0.2} if identity else {}, fields={"u": np.zeros((8, 6))}),
        time=TimeSpec(steps=3, seed=7),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "turnover_with_families"}, {"name": "mark"}, {"name": "failing"},
            {"name": "random_walk"}]),
    ))


def _snapshot(model):
    """Everything a step changes, in a comparable form."""
    lgca = model.lgca
    table = lgca._cell_table() if hasattr(lgca, "_cell_table") else None
    state = {"step": model._step, "random": lgca.rng.bit_generator.state,
             "cells": (table[0].tolist(), table[1].tolist()) if table is not None else lgca.nodes.tolist(),
             "u": np.asarray(lgca.u).tolist(), "density": np.asarray(lgca.cell_density).tolist()}
    if hasattr(lgca, "props"):
        state["traits"] = {name: (np.asarray(values).dtype.str, np.asarray(values).tolist())
                           for name, values in lgca.props.items()}
        state["labels"] = int(lgca.maxlabel)
        state["families"] = (getattr(lgca, "maxfamily", None), repr(getattr(lgca, "family_props", None)))
    return state


def _fail_once(model, name, call):
    """Make ``lgca.<name>`` fail at its ``call``-th call after doing its work; once only.

    The flag lives in the closure: the rollback restores the model's attributes, the wrapper included.
    """
    lgca = model.lgca
    method = getattr(lgca, name)
    calls = []

    def failing(*args, **kwargs):
        method(*args, **kwargs)
        calls.append(1)
        if len(calls) == call:
            raise RuntimeError("injected")

    setattr(lgca, name, failing)


_INJECTIONS = {
    "rule": lambda model: _FAIL.update(at=model._step + 1),
    "interrupt": lambda model: _FAIL.update(at=model._step + 1, error=KeyboardInterrupt),
    "propagation": lambda model: _fail_once(model, "propagation", 1),  # the cells moved, then it fails
    "boundaries": lambda model: _fail_once(model, "apply_boundaries", 2),  # after the operators
}


@pytest.mark.parametrize("injection", list(_INJECTIONS))
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_a_failed_step_leaves_the_model_as_before_and_retries_give_the_same_run(family, injection):
    model, clean = _model(family), _model(family)
    model.step()
    before = copy.deepcopy(_snapshot(model))
    _INJECTIONS[injection](model)

    with pytest.raises((RuntimeError, KeyboardInterrupt), match="injected") as error:
        model.step()

    assert "step 2 was rolled back" in "".join(getattr(error.value, "__notes__", []))
    assert _snapshot(model) == before
    for _ in range(3):  # the retried step and two more, as if nothing had failed
        model.step()
    for _ in range(4):
        clean.step()
    assert _snapshot(model) == _snapshot(clean)


def test_an_interrupted_reorientation_is_rolled_back(monkeypatch):
    import lgca.lattice_state

    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(6, 6)),
        state=StateSpec(density=0.8, restchannels=1, identity_based=True), time=TimeSpec(seed=5),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 0.3, "death_rate": 0.1}},
            ReorientationSpec(terms=[ReorientationTermSpec("polar_alignment", beta=1.0)])])))
    before = copy.deepcopy(_snapshot_without_fields(model))

    def interrupted(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(lgca.lattice_state, "place_labels", interrupted)
        with pytest.raises(KeyboardInterrupt):
            model.step()

    assert _snapshot_without_fields(model) == before
    model.step()


def _snapshot_without_fields(model):
    lgca = model.lgca
    return (model._step, lgca.rng.bit_generator.state, lgca.nodes.tolist(), int(lgca.maxlabel),
            {name: np.asarray(values).tolist() for name, values in lgca.props.items()})


def test_without_rollback_a_failed_step_stops_the_model():
    model = _model("classical")
    model.rollback = False
    _FAIL["at"] = 1
    with pytest.raises(RuntimeError, match="injected") as error:
        model.step()

    assert not getattr(error.value, "__notes__", [])
    with pytest.raises(RuntimeError, match="step 1 of this model failed .* rebuild"):
        model.step()


def test_a_function_given_to_get_lgca_is_rolled_back_too():
    def unreliable(lgca):
        lgca.nodes[lgca.nonborder] = False  # every cell is gone, then it fails
        raise RuntimeError("injected")

    lgca = get_lgca(geometry="square", dims=(6, 6), density=0.5, interaction=unreliable, seed=3)
    nodes, random = lgca.nodes.copy(), lgca.rng.bit_generator.state

    with pytest.raises(RuntimeError, match="injected"):
        lgca.timestep()

    np.testing.assert_array_equal(lgca.nodes, nodes)
    assert lgca.rng.bit_generator.state == random


def test_traits_written_and_grown_in_a_failed_step_are_restored():
    model = _model("ib")
    model.step()
    lgca = model.lgca
    traits = lgca.props["r_b"]
    values, size, buffer = traits.values.copy(), len(traits), traits._data
    _FAIL["at"] = 2

    with pytest.raises(RuntimeError, match="injected"):
        model.step()

    assert lgca.props["r_b"] is traits and traits._data is buffer and len(traits) == size
    np.testing.assert_array_equal(traits.values, values)
    assert traits._journal is None  # nothing is recorded between steps
    state = LatticeState(lgca, kind="birth_death")
    state.cells.set_trait(np.ones(len(state.cells), dtype=bool), "r_b", 0.5)  # no journal outside a step
    assert traits._journal is None
