"""Every built-in cue works alone as an operator in every family and reproduces the legacy operators.

The legacy classical operators draw their random numbers in another order, so they are
compared in distribution: the mean and spread of each node's score after one step, per
number of cells at the node. The NoVE alignment rules draw the same numbers and are compared
seed for seed (on hex the neighbour sums round differently, so hex is left out).

The nematic and contact-guidance terms use traceless tensors; in 2D these are the legacy
tensors c cᵀ - I/2, so the comparison includes a rest channel.
"""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec

DIMS = (60, 60)


def _model(nodes, operator, geometry="square", ve=True, fields=None, identity=False, seed=3, rest=1):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=nodes.shape[:-1]),
        state=StateSpec(nodes=nodes, restchannels=rest, volume_exclusion=ve, identity_based=identity,
                        fields=fields or {}, **({} if ve else {"capacity": 8})),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[operator], propagation=False)))


def _signal():
    x, y = np.meshgrid(np.arange(DIMS[0]), np.arange(DIMS[1]), indexing="ij")
    return np.exp(-((x - 30) ** 2 + (y - 20) ** 2) / 400.0)


def _director():
    angle = np.linspace(0, np.pi, DIMS[0])[:, None] * np.ones(DIMS[1])
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


CASES = {
    "alignment": ({"name": "classical.alignment", "parameters": {"beta": 1.5}},
                  {"name": "polar_alignment", "parameters": {"beta": 1.5}}, {}),
    "persistent_walk": ({"name": "classical.persistent_walk", "parameters": {"beta": 1.5}},
                        {"name": "persistent_walk", "parameters": {"beta": 1.5}}, {}),
    "aggregation": ({"name": "classical.aggregation", "parameters": {"beta": 1.5}},
                    {"name": "aggregation", "parameters": {"beta": 1.5}}, {}),
    "chemotaxis": (None, {"name": "chemotaxis", "parameters": {"beta": 20.0, "field": "signal"}},
                   {"signal": _signal()}),
    "nematic": ({"name": "classical.nematic", "parameters": {"beta": 1.5}},
                {"name": "nematic_alignment", "parameters": {"beta": 1.5}}, {}),
    "contact_guidance": ({"name": "classical.contact_guidance", "parameters": {"beta": 2.0}},
                         {"name": "contact_guidance", "parameters": {"beta": 2.0, "field": "director"}},
                         {"director": _director()}),
}


def _scores(model, weights):
    lgca = model.lgca
    nodes = lgca.nodes[lgca.nonborder].astype(float)
    return (nodes * weights).sum(-1)


@pytest.mark.parametrize("case", list(CASES))
def test_a_single_cue_reproduces_the_legacy_classical_operator(case):
    legacy_entry, new_entry, fields = CASES[case]
    nodes = np.random.default_rng(7).random(DIMS + (5,)) < 0.5
    new = _model(nodes, new_entry, fields=fields)
    term = new.pipeline.operators[0].terms[0]
    term.prepare(new.lgca)
    weights = term.weights
    if legacy_entry is None:  # chemotaxis: the legacy operator takes the gradient itself
        gradient = np.pad(term.field, [(1, 1), (1, 1), (0, 0)], mode="edge")
        legacy_entry = {"name": "classical.chemotaxis", "parameters": {"beta": 20.0, "gradient": gradient}}
    legacy = _model(nodes, legacy_entry, fields=fields, seed=4)
    new.step()
    legacy.step()
    a, b = _scores(new, weights), _scores(legacy, weights)
    density = nodes.sum(-1)
    for level in range(1, 5):  # empty and full nodes cannot change
        at = density == level
        for x, y in ((a[at], b[at]), (a[at] ** 2, b[at] ** 2)):
            error = np.sqrt((x.var() + y.var()) / at.sum())
            assert abs(x.mean() - y.mean()) < 4 * error + 1e-12, (case, level)


@pytest.mark.parametrize("parameters, legacy", [
    ({}, ("nove.dd_alignment", {})),
    ({"include_center": True}, ("nove.dd_alignment", {"include_center": True})),
    ({"normalize": True}, ("nove.di_alignment", {})),
    ({"normalize": True, "include_center": True}, ("nove.di_alignment", {"include_center": True})),
])
def test_polar_alignment_is_the_nove_alignment_seed_for_seed(parameters, legacy):
    nodes = np.random.default_rng(2).poisson(1.5, DIMS + (4,))
    new = _model(nodes, {"name": "polar_alignment", "parameters": {"beta": 0.8, **parameters}}, ve=False, rest=0)
    old = _model(nodes, {"name": legacy[0], "parameters": {"beta": 0.8, **legacy[1]}}, ve=False, rest=0)
    for _ in range(5):
        new.step()
        old.step()
    np.testing.assert_array_equal(new.lgca.nodes, old.lgca.nodes)


@pytest.mark.parametrize("ve, identity", [(True, False), (False, False), (True, True), (False, True)])
@pytest.mark.parametrize("name", ["polar_alignment", "nematic_alignment", "persistent_walk", "aggregation",
                                  "resting_bias"])
def test_single_cues_run_in_every_family(name, ve, identity):
    rng = np.random.default_rng(1)
    if ve:
        nodes = rng.random((10, 10, 5)) < 0.4
        if identity:
            nodes = np.where(nodes, np.cumsum(nodes).reshape(nodes.shape), 0).astype(np.uint64)
    else:
        nodes = rng.poisson(1.0, (10, 10, 5))
    model = _model(nodes, {"name": name, "parameters": {"beta": 1.0}}, ve=ve, identity=identity)
    cells = model.lgca.cell_density[model.lgca.nonborder].copy()
    model.step()
    np.testing.assert_array_equal(model.lgca.cell_density[model.lgca.nonborder], cells)


@pytest.mark.parametrize("geometry, dims", [("lin", (8,)), ("square", (6, 6)), ("hex", (6, 6)),
                                            ("cubic", (4, 4, 4)), ("moore", (4, 4, 4))])
@pytest.mark.parametrize("name", ["nematic_alignment", "contact_guidance"])
def test_axis_cues_score_resting_like_the_average_direction(geometry, dims, name):
    # traceless tensors: the velocity channels' scores sum to zero, the rest channel scores zero
    rng = np.random.default_rng(0)
    probe = build_model(ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims),
                                  state=StateSpec(density=0.3, restchannels=1)))
    d, K = probe.lgca.c.shape[0], probe.lgca.K
    nodes = rng.random(dims + (K,)) < 0.3
    fields = {"director": rng.normal(size=dims + (d,))}
    model = _model(nodes, {"name": name, "parameters": {"beta": 1.0}}, geometry=geometry, fields=fields)
    term = model.pipeline.operators[0].terms[0]
    term.prepare(model.lgca)
    weights = term.weights
    np.testing.assert_allclose(weights[..., -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(weights[..., :-1].sum(-1), 0.0, atol=1e-9)
    assert np.abs(weights).max() > 0.1 or geometry == "lin"
