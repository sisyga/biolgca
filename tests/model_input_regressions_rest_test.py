"""Mistakes a newcomer makes when describing a lattice and its initial cells: they must fail loudly.

Each test writes the input as a student new to the package would (a channel axis too many in ``dims``, cell
counts in a volume-exclusion model, a density map instead of channel states) and checks that the model is
not silently built with another meaning, and that the message says what to do.
"""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model, run_model


def _model(nodes, geometry="square", dims=(10, 10), **state):
    return ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims), state=StateSpec(nodes=nodes, **state),
                     time=TimeSpec(steps=0, seed=1))


@pytest.mark.parametrize("geometry", ["square", "hex"])
def test_a_third_entry_in_dims_of_a_2d_lattice_is_an_error(geometry):
    # meant as 4 channels, or as a 3D lattice: either way a 10x10 model would not be what was asked for
    with pytest.raises(ValueError, match=r"restchannels=.*geometry='cubic'"):
        get_lgca(geometry=geometry, dims=(10, 10, 4), density=0.2, interaction="random_walk")
    with pytest.raises(ValueError, match=r"dims must be \(lx, ly\)"):
        get_lgca(geometry=geometry, dims=(5,), density=0.2, interaction="random_walk")
    lgca = get_lgca(geometry=geometry, dims=(10, 7), density=0.2, interaction="random_walk")
    assert lgca.dims == (10, 7)


@pytest.mark.parametrize("value", [2, -1, 0.5])
def test_volume_exclusion_nodes_other_than_0_or_1_are_an_error(value):
    # with volume exclusion a channel holds one cell: 2 cells would become 1, and -1 or 0.5 a whole cell
    nodes = np.zeros((10, 10, 5))
    nodes[3, 4, 4] = value
    with pytest.raises(ValueError, match="volume_exclusion=False"):
        run_model(_model(nodes, restchannels=1))


def test_volume_exclusion_accepts_zero_one_and_booleans_and_counts_every_cell():
    nodes = np.zeros((10, 10, 5), dtype=int)
    nodes[3, 4, :3] = 1
    model = build_model(_model(nodes, restchannels=1))
    assert model.lgca.cell_density[model.lgca.nonborder].sum() == 3
    model = build_model(_model(nodes.astype(bool), restchannels=1))
    assert model.lgca.cell_density[model.lgca.nonborder].sum() == 3


def test_cell_counts_go_to_a_model_without_volume_exclusion():
    nodes = np.zeros((10, 10, 5), dtype=int)
    nodes[3, 4, 4] = 2
    model = build_model(_model(nodes, restchannels=1, volume_exclusion=False))
    assert model.lgca.cell_density[model.lgca.nonborder].sum() == 2


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 2, 5)], ids=["density_map", "two_species_axis"])
def test_nodes_of_the_wrong_rank_say_which_shape_is_expected(shape):
    with pytest.raises(ValueError, match=r"nodes must have shape dims \+ \(K,\).*got shape"):
        build_model(_model(np.zeros(shape, dtype=int)))


def test_one_species_given_with_its_species_axis_is_the_same_model():
    # rules see channel states as dims + (n_species, K), also for one species
    nodes = np.zeros((10, 10, 5), dtype=int)
    nodes[2, 3, [0, 4]] = 1
    with_axis = build_model(_model(nodes[..., None, :], restchannels=1)).lgca
    without = build_model(_model(nodes, restchannels=1)).lgca
    np.testing.assert_array_equal(with_axis.nodes, without.nodes)


def test_custom_example_does_not_hide_errors_of_get_permutations(monkeypatch):
    # the example is the template of a custom interaction; a fallback would replace the real error
    from dataclasses import replace

    from lgca.base import LGCA_base
    from lgca.examples import get_example_spec

    def broken(self, n_particles):
        raise TypeError("bug in get_permutations")

    monkeypatch.setattr(LGCA_base, "get_permutations", broken)
    spec = get_example_spec("custom_rest_or_align")
    spec = replace(spec, space=replace(spec.space, dims=(6, 6)), time=replace(spec.time, steps=1))
    with pytest.raises(TypeError, match="bug in get_permutations"):
        build_model(spec).run()
