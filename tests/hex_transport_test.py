"""Hexagonal periodic transport requires reciprocal opposite steps."""

from copy import deepcopy

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import ModelSpec, SpaceSpec, StateSpec, build_model


@pytest.mark.parametrize("entry", ["legacy", "compiled", "propagation"])
def test_odd_periodic_hex_is_static_only(entry):
    if entry != "compiled":
        model = get_lgca(geometry="hex", dims=(3, 3), density=1, seed=141)
        advance = model.timestep if entry == "legacy" else model.propagation
    else:
        compiled = build_model(ModelSpec(space=SpaceSpec(geometry="hex", dims=(3, 3)),
                                         state=StateSpec(density=1)))
        model = compiled.lgca
        advance = compiled.step
    before = model.nodes.copy()
    rng = deepcopy(model.rng.bit_generator.state)
    assert model.xcoords.shape == (3, 3)  # construction remains available for static plots
    from matplotlib import pyplot as plt
    figure, *_ = model.plot_density(cbar=False)
    plt.close(figure)
    with pytest.raises(ValueError, match="even number of rows"):
        advance()
    np.testing.assert_array_equal(model.nodes, before)
    assert model.rng.bit_generator.state == rng


@pytest.mark.parametrize("channel", [1, 2, 4, 5])
@pytest.mark.parametrize("origin", [(0, 0), (1, 0), (2, 3), (1, 3)])
def test_even_hex_diagonal_seams_are_reciprocal(channel, origin):
    nodes = np.zeros((3, 4, 6), dtype=bool)
    nodes[origin + (channel,)] = True
    model = get_lgca(geometry="hex", nodes=nodes, interaction="only_propagation")
    model.timestep()
    model.nodes[model.nonborder] = np.roll(model.nodes[model.nonborder], 3, axis=-1)
    model.apply_boundaries()
    model.update_dynamic_fields()
    model.timestep()
    assert np.argwhere(model.nodes[model.nonborder]).tolist() == [[*origin, (channel + 3) % 6]]
