"""Traits of identity-based models change only through ``cells.set_trait``.

``state.cells["r_b"]`` is a copy. The pandas and NumPy idiom ``cells["r_b"][daughters] = new``
used to run without an error and change nothing: a model with mutations then had none.
"""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.rules import interaction


def _run(rule, volume_exclusion):
    return run_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(6, 6), boundary="periodic"),
        state=StateSpec(density=1.5, restchannels=1, identity_based=True, volume_exclusion=volume_exclusion,
                        traits={"r_b": 0.2}),
        time=TimeSpec(steps=2, seed=4),
        dynamics=InteractionPipelineSpec(operators=[{"name": rule.info.name}])))


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_writing_into_the_copy_of_a_trait_raises(volume_exclusion):
    @interaction(kind="phenotype_switch", families=("ib", "nove_ib"), name="evolution.raise_rb")
    def raise_rb(state):
        state.cells["r_b"][:] = 0.9

    with pytest.raises(ValueError, match="read-only"):
        _run(raise_rb, volume_exclusion)


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_set_trait_changes_the_trait_and_arithmetic_on_the_copy_works(volume_exclusion):
    @interaction(kind="phenotype_switch", families=("ib", "nove_ib"), name="evolution.raise_rb")
    def raise_rb(state):
        cells = state.cells
        cells.set_trait(np.ones(len(cells), dtype=bool), "r_b", cells["r_b"] + 0.1)

    result = _run(raise_rb, volume_exclusion)
    lgca = result.lgca
    labels = lgca.nodes[lgca.nonborder]
    living = np.unique(labels[labels > 0]) if labels.dtype != object else \
        np.unique([label for node in labels.reshape(-1, labels.shape[-1]) for channel in node for label in channel])
    np.testing.assert_allclose(np.asarray(lgca.props["r_b"])[living], 0.4)  # 0.2 and two steps of +0.1
