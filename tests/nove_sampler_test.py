"""Count-state and sequential-operator NoVE regressions."""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.operator_base import InteractionOperator, PluginInfo


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("propagation", [False, True])
def test_sequential_reverse_then_align_refreshes_periodic_edges(normalize, propagation):
    class Reverse(InteractionOperator):
        def __init__(self):
            super().__init__(PluginInfo("reverse", "reorientation", ("nove",)))

        def apply(self, context, step):
            model = context.lgca
            model.nodes[model.nonborder] = model.nodes[model.nonborder][..., ::-1]

    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=4),
        state=StateSpec(volume_exclusion=False, nodes=np.array([[10, 0]] * 4)),
        time=TimeSpec(steps=1, seed=1),
        dynamics=InteractionPipelineSpec(operators=[Reverse(), {
            "name": "polar_alignment", "parameters": {"beta": 100, "normalize": normalize},
        }], propagation=propagation),
    ))
    model.run(False)
    np.testing.assert_array_equal(model.lgca.nodes[model.lgca.nonborder], [[0, 10]] * 4)


@pytest.mark.parametrize("operator", [{"name": "random_walk"}, {"name": "polar_alignment"},
                                      {"name": "polar_alignment", "parameters": {"normalize": True}}])
@pytest.mark.parametrize("source", ["explicit", "npz", "density"])
def test_native_nove_samplers_accept_all_count_sources(operator, source, tmp_path):
    state = dict(volume_exclusion=False, restchannels=0)
    if source == "explicit":
        state["nodes"] = np.array([[4, 0], [0, 3], [2, 5]], dtype=np.uint64)
    elif source == "npz":
        np.savez(tmp_path / "counts.npz", nodes=np.array([[4, 0], [0, 3], [2, 5]]))
        state["initializer"] = {"name": "from_npz", "parameters": {"path": "counts.npz"}}
    else:
        state["density"] = 4
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=3), state=StateSpec(**state),
        time=TimeSpec(steps=3, seed=130),
        dynamics=InteractionPipelineSpec(operators=[operator], propagation=False),
    ), resource_base=tmp_path)
    before = model.lgca.nodes[model.lgca.nonborder].sum(-1)
    model.run(False)
    np.testing.assert_array_equal(model.lgca.nodes[model.lgca.nonborder].sum(-1), before)


@pytest.mark.parametrize("operator", [{"name": "random_walk"}, {"name": "polar_alignment"},
                                      {"name": "polar_alignment", "parameters": {"normalize": True}}])
@pytest.mark.parametrize("counts", [[2**63, 0], [2**63 - 1, 1], [2**64 - 1, 1]])
def test_nove_sampler_rejects_unrepresentable_totals(operator, counts):
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", dims=1),
        state=StateSpec(volume_exclusion=False, nodes=np.array([counts], dtype=np.uint64)),
        dynamics=InteractionPipelineSpec(operators=[operator], propagation=False),
    )
    with pytest.raises(ValueError, match="sampler.*signed int64"):
        build_model(spec)
