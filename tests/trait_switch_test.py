"""trait_switch: cells of identity-based models change their traits by events, written as mutations."""

import numpy as np
import pytest

from lgca.builtin_rules import trait_switch
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.testing import check_interaction

DIMS = (60, 60)


def _model(ve, operators, traits, seed=1):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(density=0.3 if ve else 1.0, restchannels=1, identity_based=True, volume_exclusion=ve,
                        capacity=None if ve else 8, traits=traits),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False)))


def _traits(model, name):
    lgca = model.lgca
    labels = lgca.nodes[lgca.nonborder]
    labels = labels[labels > 0] if labels.dtype != object else np.array(
        [label for channel in labels.flat for label in channel], dtype=int)
    return np.asarray(lgca.props[name])[labels.astype(int)]


@pytest.mark.parametrize("ve", [True, False])
def test_two_states_switch_with_their_own_rates(ve):
    # a cell aligns (2) or not (0); it starts with rate 0.02 and stops with rate 0.05. An event with
    # probability 0.07 sets a new state drawn with probabilities (5/7, 2/7), whatever the old one was
    rates = {"on": 0.02, "off": 0.05}
    total = rates["on"] + rates["off"]
    switch = {"probability": total, "traits": {"alignment": {
        "distribution": "choice", "a": [0.0, 2.0], "p": [rates["off"] / total, rates["on"] / total],
        "operation": "set"}}}
    rng = np.random.default_rng(0)
    model = _model(ve, [{"name": "trait_switch", "parameters": {"switch": switch}}], {"alignment": 0.0})
    lgca = model.lgca
    lgca.props["alignment"][:] = rng.choice([0.0, 2.0], len(lgca.props["alignment"]))
    before = _traits(model, "alignment")
    model.step()
    after = _traits(model, "alignment")
    for start, end, p in ((0.0, 2.0, rates["on"]), (2.0, 0.0, rates["off"])):
        cells = before == start
        switched = (after[cells] == end).mean()
        assert abs(switched - p) < 4 * np.sqrt(p * (1 - p) / cells.sum())
    assert set(np.unique(after)) <= {0.0, 2.0}


def test_a_random_change_keeps_the_trait_within_its_bounds():
    switch = {"probability": 0.5, "traits": {"alignment": {"distribution": "normal", "scale": 0.3,
                                                          "bounds": [0, None]}}}
    model = _model(True, [{"name": "trait_switch", "parameters": {"switch": switch}}], {"alignment": 0.1})
    model.step()
    values = _traits(model, "alignment")
    assert values.min() >= 0
    changed = values != 0.1
    assert abs(changed.mean() - 0.5) < 4 * np.sqrt(0.25 / len(values))


def test_switched_traits_steer_the_cells():
    # cells that switch to a strong persistence keep their direction, the others turn at random
    nodes = np.zeros(DIMS + (5,), dtype=bool)
    nodes[..., 0] = True  # one cell per node, moving east
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=1, identity_based=True, traits={"persistence": 0.0}),
        time=TimeSpec(steps=1, seed=2),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "trait_switch", "parameters": {"switch": {"probability": 0.5, "traits": {
                "persistence": {"value": 5.0, "operation": "set"}}}}},
            ReorientationSpec(terms=[ReorientationTermSpec("persistent_walk", beta=1.0, trait="persistence")]),
        ], propagation=False)))
    model.step()
    lgca = model.lgca
    labels = lgca.nodes[lgca.nonborder]
    east = labels[..., 0][labels[..., 0] > 0].astype(int)
    persistence = np.asarray(lgca.props["persistence"])
    assert (persistence[east] == 5.0).mean() > 0.8  # mostly the switched cells kept moving east


def test_the_rule_conserves_the_cells_in_every_identity_family():
    report = check_interaction(trait_switch, parameters={"switch": {"alignment": 0.1}},
                               traits={"alignment": 1.0}, geometries=("square", "hex"))
    assert report.passed, report


def test_unknown_traits_and_empty_switches_are_refused():
    with pytest.raises(ValueError, match="no trait 'strength'"):
        _model(True, [{"name": "trait_switch", "parameters": {"switch": {"strength": 0.1}}}],
               {"alignment": 0.0}).step()
    with pytest.raises(ValueError, match="needs a switch"):
        _model(True, [{"name": "trait_switch", "parameters": {"switch": []}}], {"alignment": 0.0}).step()
