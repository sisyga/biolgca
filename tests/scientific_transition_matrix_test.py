"""Independent scientific oracles complement the maintained parity matrix."""

from itertools import permutations

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec


def test_phenotype_switch_metadata_matches_mass_and_momentum_observables():
    from lgca.pipeline import PhenotypeSwitchSpec

    nodes = np.zeros((50, 2, 2), dtype=bool)
    nodes[:, 0, 0] = True  # one cell of species 0 moving right at every node
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=50),
        state=StateSpec(nodes=nodes, n_species=2, restchannels=0),
        time=TimeSpec(steps=1, seed=42),
        dynamics=InteractionPipelineSpec(operators=[PhenotypeSwitchSpec("phenotype_switch", {
            "rates": [[0, 1], [0, 0]],
        })], propagation=False),
    ))
    model.run(False)
    after = model.lgca.nodes[model.lgca.nonborder]
    assert after.sum() == 50 and after[:, 1].sum() == 50  # every cell switched species, none was lost
    moved = after[:, 1, 1].sum()  # switched cells go to a random free channel: the momentum changes
    assert 0 < moved < 50
    law = model.pipeline.operators[0].conservation_law
    assert law.conserves_total_particles is True
    assert law.conserves_momentum is False
    assert "changes momentum" in model.pipeline.describe_schedule()


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_all_small_states_respect_directed_transition_support(order):
    """Only 0 -> 1 is allowed; species 2 is isolated, including saturation."""
    order = np.asarray(order)
    rates = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    states = np.array([[(mask >> bit) & 1 for bit in range(6)] for mask in range(64)], dtype=bool).reshape(64, 3, 2)
    before = states.sum(axis=2)
    for seed in range(4):
        model = build_model(ModelSpec(
            space=SpaceSpec(geometry="lin", dims=64),
            state=StateSpec(nodes=states[:, order], restchannels=0, n_species=3),
            time=TimeSpec(steps=1, seed=seed),
            dynamics=InteractionPipelineSpec(operators=[{"name": "phenotype_switch", "parameters": {
                "rates": rates[np.ix_(order, order)].tolist()}}], propagation=False)))
        model.step()
        result = model.lgca.nodes[model.lgca.nonborder]
        counts = result.sum(axis=2)[:, np.argsort(order)]
        assert result.dtype == bool
        np.testing.assert_array_equal(counts.sum(axis=1), before.sum(axis=1))
        assert np.all(counts <= 2)
        np.testing.assert_array_equal(counts[:, 2], before[:, 2])
        assert np.all(counts[:, 0] <= before[:, 0]) and np.all(counts[:, 1] >= before[:, 1])
