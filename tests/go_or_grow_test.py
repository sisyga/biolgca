"""Go-or-grow built from go_or_rest (one species) or go_or_grow.switch (two species) reproduces the
legacy rule in distribution."""

import numpy as np
import pytest

from lgca.builtin_rules import go_or_grow_layout
from lgca.model import (
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
)
from lgca.pipeline import InteractionPipelineSpec
from lgca.testing import check_interaction

DIMS = (60, 60)
REST = {True: 6, False: 1}
PARAMETERS = {"kappa": 4.0, "theta": 0.5, "r_b": 0.3, "r_d": 0.05}


def _legacy_nodes(ve, seed):
    """Random states over the whole density range: node i holds about i % 13 cells."""
    rng = np.random.default_rng(seed)
    channels = 6 + REST[ve]
    level = (np.arange(np.prod(DIMS)) % 13).reshape(DIMS) / 12
    if ve:
        return rng.random(DIMS + (channels,)) < level[..., None]
    return rng.poisson(1.5 * level[..., None], DIMS + (channels,))


def _two_species(nodes):
    split = np.zeros(nodes.shape[:-1] + (2, nodes.shape[-1]), dtype=nodes.dtype)
    split[..., 0, :6] = nodes[..., :6]
    split[..., 1, 6:] = nodes[..., 6:]
    return split


def _step(nodes, operators, **state):
    compiled = build_model(ModelSpec(
        description=Description(title="go-or-grow step"),
        space=SpaceSpec(geometry="hex", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=REST[nodes.dtype == bool],
                        volume_exclusion=nodes.dtype == bool, **state),
        time=TimeSpec(steps=1, seed=1),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False),
    ))
    compiled.step()
    return compiled.lgca.nodes[compiled.lgca.nonborder]


def _migrating_and_resting(nodes):
    if nodes.ndim == 4:
        return nodes[..., 0, :].sum(-1), nodes[..., 1, :].sum(-1)
    return nodes[..., :6].sum(-1), nodes[..., 6:].sum(-1)


def _pipeline(n_species, when_full="legacy"):
    switch = {key: PARAMETERS[key] for key in ("kappa", "theta")}
    growth = {key: PARAMETERS[key] for key in ("r_b", "r_d")}
    walk = {"channels": "velocity"} | ({"species": 0} if n_species == 2 else {})
    return [{"name": "go_or_rest" if n_species == 1 else "go_or_grow.switch",
             "parameters": switch | {"when_full": when_full}},
            {"name": "go_or_grow.growth", "parameters": growth | {"when_full": when_full}},
            {"name": "random_walk", "parameters": walk}]


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("ve", [True, False])
def test_one_step_matches_the_legacy_rule_at_every_density(ve, n_species):
    nodes = _legacy_nodes(ve, seed=4)
    legacy = _step(nodes, [{"name": "legacy.classical.go_or_grow" if ve else "legacy.nove.go_or_grow",
                            "parameters": PARAMETERS}])
    if n_species == 1:
        two = _step(nodes, _pipeline(1))
    else:
        two = _step(_two_species(nodes), _pipeline(2), n_species=2)

    density = nodes.sum(-1)
    for expected, measured in zip(_migrating_and_resting(legacy), _migrating_and_resting(two)):
        for level in np.unique(density):
            at_level = density == level
            if at_level.sum() < 50:
                continue
            error = np.sqrt((expected[at_level].var() + measured[at_level].var()) / at_level.sum())
            assert abs(expected[at_level].mean() - measured[at_level].mean()) < 4 * error + 1e-9, level


def test_the_reject_mode_is_a_different_model():
    nodes = _legacy_nodes(True, seed=4)
    legacy = _migrating_and_resting(_step(nodes, [{"name": "legacy.classical.go_or_grow", "parameters": PARAMETERS}]))
    reject = _migrating_and_resting(_step(nodes, _pipeline(1, when_full="reject")))
    crowded = nodes.sum(-1) >= 7

    assert reject[1][crowded].mean() > legacy[1][crowded].mean() + 0.1


@pytest.mark.parametrize("rule, parameters", [
    ("go_or_rest", {"kappa": 4.0, "theta": 0.5}),
    ("go_or_rest", {"kappa": -4.0, "theta": 0.5, "when_full": "reject"}),
    ("go_or_grow.switch", {"kappa": 4.0, "theta": 0.5}),
    ("go_or_grow.switch", {"kappa": -4.0, "theta": 0.5, "when_full": "reject"}),
    ("go_or_grow.growth", {"r_b": 0.3, "r_d": 0.05}),
    ("go_or_grow.growth", {"r_b": 0.3, "r_d": 0.05, "r_d_resting": 0.0, "when_full": "reject"}),
])
def test_go_or_grow_rules_keep_each_species_in_its_channels(rule, parameters):
    species = 1 if rule == "go_or_rest" else 2
    check_interaction(rule, parameters, families=("classical", "nove"), n_species=(species,),
                      prepare=go_or_grow_layout)
    if species == 1:
        return
    compiled = build_model(ModelSpec(
        description=Description(title="layout"),
        space=SpaceSpec(geometry="hex", dims=(20, 20)),
        state=StateSpec(nodes=_two_species(_legacy_nodes(True, 1)[:20, :20]), restchannels=6, n_species=2),
        time=TimeSpec(steps=5, seed=2),
        dynamics=InteractionPipelineSpec(operators=[{"name": rule, "parameters": parameters}]),
    ))
    for _ in range(5):
        compiled.step()
        nodes = compiled.lgca.nodes[compiled.lgca.nonborder]
        assert not nodes[..., 0, 6:].any() and not nodes[..., 1, :6].any()


def test_the_rules_need_two_species_in_their_channels():
    with pytest.raises(ValueError, match="phenotype switch.*one species"):
        build_model(ModelSpec(
            description=Description(title="one species"),
            space=SpaceSpec(geometry="hex", dims=(10, 10)),
            state=StateSpec(density=1, restchannels=1),
            dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_grow.switch"}]),
        ))
    nodes = np.zeros((10, 10, 2, 7), dtype=bool)
    nodes[..., 0, 6] = True  # a migrating cell in the rest channel
    compiled = build_model(ModelSpec(
        description=Description(title="misplaced"),
        space=SpaceSpec(geometry="hex", dims=(10, 10)),
        state=StateSpec(nodes=nodes, restchannels=1, n_species=2),
        time=TimeSpec(steps=1, seed=1),
        dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_grow.growth"}]),
    ))
    with pytest.raises(ValueError, match="velocity channels"):
        compiled.step()


# go_or_rest with a probability that responds to cues (lgca.switching)

def _rest_model(nodes, parameters, identity=False, fields=None):
    ve = nodes.dtype == bool
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=nodes.shape[:2]),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=ve, identity_based=identity,
                        capacity=None if ve else 8, fields=fields or {}),
        time=TimeSpec(steps=1, seed=3),
        dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_rest", "parameters": parameters}],
                                         propagation=False)))


@pytest.mark.parametrize("ve, identity", [(True, False), (False, False), (True, True)])
def test_kappa_and_theta_are_the_density_switch(ve, identity):
    # the same random numbers: the short form and the probability give the same cells
    rng = np.random.default_rng(5)
    counts = rng.random((30, 30, 5)) < 0.5 if ve else rng.poisson(0.8, (30, 30, 5))
    if identity:  # labels
        counts = np.where(counts, np.cumsum(counts).reshape(counts.shape), 0).astype(np.uint64)
    short = _rest_model(counts, {"kappa": 4.0, "theta": 0.4}, identity)
    cue = _rest_model(counts, {"probability": {"cues": [{"name": "density", "kappa": 4.0, "theta": 0.4}]}},
                      identity)
    short.step()
    cue.step()
    np.testing.assert_array_equal(cue.lgca.nodes, short.lgca.nodes)


def test_cells_rest_with_a_probability_that_responds_to_a_field():
    # no volume exclusion: every cell rests with p(field), whatever its channel before
    rng = np.random.default_rng(6)
    counts = rng.poisson(1.0, (100, 60, 5))
    level = np.repeat(np.arange(5), 20)[:, None] * np.ones((1, 60)) / 4
    probability = {"max": 0.9, "hill": [{"name": "field", "field": "oxygen", "K": 0.3, "n": -2}]}
    model = _rest_model(counts, {"probability": probability}, fields={"oxygen": level})
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]
    resting, cells = after[..., 4:].sum(-1), after.sum(-1)
    for value in np.unique(level):
        at = level == value
        n = cells[at].sum()
        p = 0.9 * 0.09 / (0.09 + value ** 2)
        assert abs(resting[at].sum() / n - p) < 4 * np.sqrt(p * (1 - p) / n), value


@pytest.mark.parametrize("parameters, message", [
    ({"probability": 0.5, "kappa": 2.0}, "not both"),
    ({"probability": 0.5, "density": "neighbourhood"}, "scope"),
    ({"probability": {"max": 2.0}}, "must be a probability"),
])
def test_go_or_rest_parameters_are_checked(parameters, message):
    model = _rest_model(np.zeros((5, 5, 5), dtype=bool), parameters)
    with pytest.raises(ValueError, match=message):
        model.step()
