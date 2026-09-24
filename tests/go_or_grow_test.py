"""Go-or-grow as a two-species model reproduces the legacy single-species rule in distribution."""

import numpy as np
import pytest

from lgca.builtin_rules import go_or_grow_layout
from lgca.model import Description, ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
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


@pytest.mark.parametrize("ve", [True, False])
def test_one_step_matches_the_legacy_rule_at_every_density(ve):
    nodes = _legacy_nodes(ve, seed=4)
    legacy = _step(nodes, [{"name": "classical.go_or_grow" if ve else "nove.go_or_grow",
                            "parameters": PARAMETERS}])
    split = {key: PARAMETERS[key] for key in ("kappa", "theta")}
    growth = {key: PARAMETERS[key] for key in ("r_b", "r_d")}
    two = _step(_two_species(nodes), [{"name": "go_or_grow.switch", "parameters": split},
                                      {"name": "go_or_grow.growth", "parameters": growth},
                                      {"name": "species_random_walk",
                                       "parameters": {"species": 0, "channels": "velocity"}}],
                n_species=2)

    density = nodes.sum(-1)
    for expected, measured in zip(_migrating_and_resting(legacy), _migrating_and_resting(two)):
        for level in np.unique(density):
            at_level = density == level
            if at_level.sum() < 50:
                continue
            error = np.sqrt((expected[at_level].var() + measured[at_level].var()) / at_level.sum())
            assert abs(expected[at_level].mean() - measured[at_level].mean()) < 4 * error + 1e-9, level


@pytest.mark.parametrize("rule, parameters", [
    ("go_or_grow.switch", {"kappa": 4.0, "theta": 0.5}),
    ("go_or_grow.switch", {"kappa": -4.0, "theta": 0.5, "capacity": "reject"}),
    ("go_or_grow.growth", {"r_b": 0.3, "r_d": 0.05}),
    ("go_or_grow.growth", {"r_b": 0.3, "r_d": 0.05, "r_d_resting": 0.0, "capacity": "reject"}),
])
def test_go_or_grow_rules_keep_each_species_in_its_channels(rule, parameters):
    check_interaction(rule, parameters, families=("classical", "nove"), n_species=(2,), prepare=go_or_grow_layout)
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
    with pytest.raises(ValueError, match="needs n_species=2"):
        build_model(ModelSpec(
            description=Description(title="one species"),
            space=SpaceSpec(geometry="hex", dims=(10, 10)),
            state=StateSpec(density=1, restchannels=1),
            dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_grow.growth"}]),
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
