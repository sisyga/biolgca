"""Independent scientific oracles complement the maintained parity matrix."""

from itertools import permutations

import numpy as np
import pytest

from lgca.pipeline import NativePhenotypeSwitchOperator


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
