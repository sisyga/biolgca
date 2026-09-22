"""Independent scientific oracles complement the maintained parity matrix."""

from itertools import permutations

import numpy as np
import pytest

from lgca.pipeline import NativePhenotypeSwitchOperator


def test_phenotype_switch_metadata_matches_mass_and_momentum_observables():
    from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
    from lgca.pipeline import InteractionPipelineSpec, PhenotypeSwitchSpec

    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=1),
        state=StateSpec(nodes=np.array([[[True, False], [False, False]]]), n_species=2),
        time=TimeSpec(steps=1, seed=42),
        dynamics=InteractionPipelineSpec(operators=[PhenotypeSwitchSpec("phenotype_switch", {
            "rates": [[0, 1], [0, 0]],
        })], propagation=False),
    ))
    before = model.lgca.nodes[model.lgca.nonborder].copy()
    model.run(False)
    after = model.lgca.nodes[model.lgca.nonborder]
    assert before.sum() == after.sum() == 1
    assert (before[..., 0].sum() - before[..., 1].sum()) == 1
    assert (int(after[..., 0].sum()) - int(after[..., 1].sum())) == -1
    law = model.pipeline.operators[0].conservation_law
    assert law.conserves_total_particles is True
    assert law.conserves_momentum is False
    assert "changes momentum" in model.pipeline.describe_schedule()


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_all_small_states_respect_directed_transition_support(order):
    """Only 0 -> 1 is allowed; species 2 is isolated, including saturation."""
    order = np.asarray(order)
    rates = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    for mask in range(64):
        state = np.array([(mask >> bit) & 1 for bit in range(6)], dtype=bool).reshape(3, 2)
        before = state.sum(axis=1)
        for seed in range(4):
            result = NativePhenotypeSwitchOperator._sample_state(
                state[order], rates[np.ix_(order, order)], np.random.default_rng(seed)
            )
            counts = result.sum(axis=1)[np.argsort(order)]
            assert result.dtype == bool
            assert result.shape == state.shape
            assert counts.sum() == before.sum()
            assert np.all(counts <= 2)
            assert counts[2] == before[2]
            assert counts[0] <= before[0]
            assert counts[1] >= before[1]
