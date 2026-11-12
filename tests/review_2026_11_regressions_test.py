"""Regressions found in the November 2026 code review of version 2.0.0."""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model, run_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import OrderParameterRecorder

GEOMETRIES = [("lin", (30,)), ("square", (8, 8)), ("hex", (8, 8)), ("cubic", (5, 5, 5)), ("moore", (4, 4, 4))]


def _labels(lgca):
    """The labels of the cells on the lattice."""
    nodes = lgca.nodes[lgca.nonborder]
    if nodes.dtype == object:
        return np.array([label for channel in nodes.flat for label in channel], dtype=np.int64)
    return nodes[nodes > 0].astype(np.int64)


# ------------------------------------------------------------------ labels of identity-based cells


@pytest.mark.parametrize("volume_exclusion", [True, False])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_cells_grown_from_a_region_have_labels_of_their_own(geometry, dims, volume_exclusion):
    """The region initializer left ``maxlabel`` at 0 with volume exclusion, so daughters got the labels
    of living cells."""
    spec = ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims),
        state=StateSpec(identity_based=True, volume_exclusion=volume_exclusion, restchannels=1,
                        initializer={"name": "region", "parameters": {"extent": 3, "density": 2.0}}),
        time=TimeSpec(steps=6, seed=3),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 0.5}}, {"name": "random_walk"}]),
    )
    model = build_model(spec)
    initial = _labels(model.lgca)
    assert initial.size > 0 and model.lgca.maxlabel >= initial.max()
    if volume_exclusion:  # labelled along the lattice, as at construction
        np.testing.assert_array_equal(np.sort(initial), np.arange(1, initial.size + 1))

    model.run(showprogress=False)

    labels = _labels(model.lgca)
    assert labels.size > initial.size  # the population grew
    assert np.unique(labels).size == labels.size
    assert model.lgca.maxlabel >= labels.max()


@pytest.mark.parametrize("volume_exclusion", [True, False])
def test_traits_cover_the_cells_of_a_region(volume_exclusion):
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(8, 8)),
        state=StateSpec(identity_based=True, volume_exclusion=volume_exclusion, restchannels=1,
                        traits={"r_b": 0.4},
                        initializer={"name": "region", "parameters": {"extent": 3, "density": 2.0}}),
        time=TimeSpec(steps=5, seed=3),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": "r_b"}}, {"name": "random_walk"}]),
    )
    lgca = run_model(spec, showprogress=False).lgca
    labels = _labels(lgca)
    assert np.unique(labels).size == labels.size
    np.testing.assert_allclose(np.asarray(lgca.props["r_b"])[labels], 0.4)


@pytest.mark.parametrize("interaction", ["birthdeath", "go_or_grow", "random_walk"])
@pytest.mark.parametrize("ve", [True, False])
def test_random_reset_gives_the_new_cells_the_traits_of_a_new_model(ve, interaction):
    """``random_reset`` relabelled the cells but kept the trait tables of the old ones."""
    lgca = get_lgca(geometry="square", dims=(6, 6), ib=True, ve=ve, restchannels=2 if ve else 1, density=0.2,
                    interaction=interaction, seed=1)
    lgca.timeevo(3, showprogress=False)

    lgca.random_reset(2.0)

    labels = _labels(lgca)
    assert labels.size > 30 and np.unique(labels).size == labels.size
    assert lgca.maxlabel >= labels.max()
    for name, values in lgca.props.items():
        assert len(values) > labels.max(), name
    np.testing.assert_array_equal(lgca.cell_density[lgca.nonborder].sum(), labels.size)
    lgca.timeevo(5, showprogress=False)
    labels = _labels(lgca)
    assert np.unique(labels).size == labels.size


def test_random_reset_of_a_model_from_a_spec_with_traits_says_to_build_a_new_one():
    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=10),
                     state=StateSpec(identity_based=True, density=0.5, traits={"r_b": 0.2}))
    lgca = build_model(spec).lgca
    with pytest.raises(ValueError, match="build a new model"):
        lgca.random_reset(1.0)
