"""Trait buffers and the operations of the cell table."""

import numpy as np
import pytest

from lgca.cells import TraitArray
from lgca.lattice_state import LatticeState
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec


def test_trait_arrays_behave_like_lists_and_arrays():
    values = TraitArray([1, 2])
    values.append(3.5)  # widens integers to floats
    values.extend([4, 5])
    assert len(values) == 5 and values.dtype == float
    np.testing.assert_array_equal(values[np.array([4, 0])], [5, 1])
    values[1] = 7
    assert list(values) == [1, 7, 3.5, 4, 5]
    np.testing.assert_array_equal(np.concatenate([values, [6]]), [1, 7, 3.5, 4, 5, 6])
    for _ in range(100):
        values.append(0)
    assert len(values) == 105 and len(values._data) < 2 * 105 + 16


def _state(ve, density, dims=(30, 30), traits=None, seed=1):
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=dims),
        state=StateSpec(density=density, restchannels=2 if ve else 1, volume_exclusion=ve, identity_based=True,
                        traits=traits or {"r_b": 0.5}, **({} if ve else {"capacity": 8})),
        time=TimeSpec(steps=1, seed=seed), dynamics=InteractionPipelineSpec(operators=[])))
    return model.lgca, LatticeState(model.lgca)


@pytest.mark.parametrize("ve", [True, False])
def test_cells_match_the_lattice(ve):
    lgca, state = _state(ve, 2.0)
    cells = state.cells
    np.testing.assert_array_equal(np.bincount(cells.index, minlength=900).reshape(30, 30), state.density)
    np.testing.assert_array_equal(state.density[cells.node], np.bincount(cells.index)[cells.index])
    assert len(set(cells.label)) == len(cells) == state.density.sum()
    np.testing.assert_array_equal(cells["r_b"], 0.5)
    state.commit()  # writing back unchanged cells keeps the lattice
    labels = np.sort(state.cells.label)
    np.testing.assert_array_equal(labels, np.sort(LatticeState(lgca).cells.label))


@pytest.mark.parametrize("ve", [True, False])
def test_move_and_divide_respect_free_channels(ve):
    lgca, state = _state(ve, 4.0)
    cells = state.cells
    moving = cells.in_channels("velocity")
    before = state.counts.copy()
    moved = cells.move(moving, "rest")
    counts = state.counts
    if ve:
        assert counts.max() <= 1
        free_rest = 2 - before[..., 0, 4:].sum(-1)
        np.testing.assert_array_equal(counts[..., 0, 4:].sum(-1),
                                      before[..., 0, 4:].sum(-1) + np.minimum(free_rest, before[..., 0, :4].sum(-1)))
    else:
        assert moved.sum() == moving.sum()
    n = len(cells)
    daughters = cells.divide(np.ones(n, dtype=bool), channels="velocity")
    assert np.all(cells.label[daughters] > lgca.maxlabel - len(daughters))
    assert np.all(np.isin(cells.channel[daughters], range(4)))
    if ve:
        assert state.counts.max() <= 1
    else:
        assert len(daughters) == n
    state.commit()
    assert len(lgca.props["r_b"]) == lgca.maxlabel + 1


@pytest.mark.parametrize("ve", [True, False])
def test_remove_and_divide_cells_have_the_right_rates(ve):
    _, state = _state(ve, 1.0, dims=(100, 100))
    n = len(state.cells)
    removed = state.remove_cells(0.3).sum()
    assert abs(removed / n - 0.3) < 4 * np.sqrt(0.3 * 0.7 / n)
    m = len(state.cells)
    added = state.divide_cells(0.2, channels="all").sum()
    if not ve:  # with volume exclusion some divisions find no free channel
        assert abs(added / m - 0.2) < 4 * np.sqrt(0.2 * 0.8 / m)
    assert len(state.cells) == m + added


def test_pick_limits_the_cells_per_node():
    _, state = _state(True, 3.0)
    cells = state.cells
    chosen = cells.pick(np.ones(len(cells), dtype=bool), 1)
    per_node = np.bincount(cells.index[chosen], minlength=900)
    np.testing.assert_array_equal(per_node, np.minimum(1, state.density.ravel()))
