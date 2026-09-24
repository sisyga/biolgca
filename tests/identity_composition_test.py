"""Independent checks of daughter properties across composed growth rules."""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec


@pytest.mark.parametrize("growth", ["ib.birth", "ib.birthdeath", "ib.birthdeath_discrete",
                                    "ib.go_and_grow_mutations"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.filterwarnings("ignore:The interaction name:FutureWarning")
def test_identity_growth_compositions_keep_all_living_property_rows(growth, reverse):
    parameters = {"r_b": 1.0}
    if growth != "ib.birth":
        parameters["r_d"] = 0.0
    operators = [{"name": growth, "parameters": parameters}, {
        "name": "ib.go_or_grow", "parameters": {
            "r_b": 0.0, "r_d": 0.0, "kappa": 5.0, "theta": 0.75,
        },
    }]
    if reverse:
        operators.reverse()
    compiled = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=1),
        state=StateSpec(identity_based=True, nodes=np.array([[1, 0, 0, 0]], dtype=np.uint64)),
        time=TimeSpec(steps=10, seed=42),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False),
    ))
    compiled.lgca.props["marker"] = [0, 17]
    for _ in range(10):
        compiled.step()
        labels = compiled.lgca.nodes[compiled.lgca.nonborder]
        living = labels[labels > 0]
        assert len(np.unique(living)) == living.size
        for values in compiled.lgca.props.values():
            assert len(values) == int(compiled.lgca.maxlabel) + 1
            assert np.isfinite(np.asarray(values)[living]).all()
        np.testing.assert_array_equal(np.asarray(compiled.lgca.props["marker"])[living], 17)
        np.testing.assert_array_equal(np.asarray(compiled.lgca.props["kappa"])[living], 5)
        np.testing.assert_array_equal(np.asarray(compiled.lgca.props["theta"])[living], .75)
    assert compiled.lgca.maxlabel > 1


def test_identity_growth_rules_without_volume_exclusion_combine():
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=4),
        state=StateSpec(identity_based=True, volume_exclusion=False, density=2, restchannels=1, capacity=8,
                        traits={"r_b": 0.5, "kappa": 2.0}),
        time=TimeSpec(seed=3),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": "r_b", "mutation": {"r_b": 0.01}}},
            {"name": "go_or_rest", "parameters": {"kappa": "kappa"}},
            {"name": "go_or_grow.growth", "parameters": {"r_b": 0.3, "mutation": {"kappa": 0.1}}},
        ]),
    ))
    for _ in range(5):
        model.step()
    for values in model.lgca.props.values():
        assert len(values) == int(model.lgca.maxlabel) + 1


@pytest.mark.parametrize("interaction", ["birth", "birthdeath", "birthdeath_discrete",
                                        "go_or_grow", "go_and_grow_mutations"])
def test_legacy_growth_inherits_additional_cell_property(interaction):
    from lgca import get_lgca

    parameters = {"kappa": 100, "theta": 0, "r_d": 0} if interaction == "go_or_grow" else {}
    model = get_lgca(geometry="lin", ib=True,
                     nodes=np.array([[0, 0, 1, 0, 0, 0]], dtype=np.uint64),
                     interaction=interaction, seed=42, r_b=1.0, propagation=False, **parameters)
    model.props["marker"] = [0, 17]
    for _ in range(10):
        model.timestep()
        assert len(model.props["marker"]) == int(model.maxlabel) + 1
    assert model.maxlabel > 1
    np.testing.assert_array_equal(np.asarray(model.props["marker"])[1:], 17)
