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


# ------------------------------------------------------------------------------ interaction radius

FAMILIES = [dict(), dict(ve=False), dict(ib=True), dict(ib=True, ve=False), dict(n_species=2),
            dict(n_species=2, ve=False)]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: "-".join(family) or "classical")
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_a_model_runs_on_after_its_interaction_radius_changed(geometry, dims, family):
    """Identity-based models without volume exclusion kept the lookup tables of the old padded shape."""
    lgca = get_lgca(geometry=geometry, dims=dims, density=1.0, restchannels=1, interaction="random_walk",
                    seed=4, **family)
    lgca.timeevo(2, showprogress=False)
    cells = int(lgca.cell_density[lgca.nonborder].sum())
    before = np.sort(_labels(lgca)) if family.get("ib") else None

    lgca.set_r_int(3)

    assert lgca.cell_density.shape == tuple(size + 6 for size in lgca.dims)
    assert int(lgca.cell_density[lgca.nonborder].sum()) == cells
    lgca.timeevo(3, showprogress=False)
    assert int(lgca.cell_density[lgca.nonborder].sum()) == cells
    if before is not None:
        np.testing.assert_array_equal(np.sort(_labels(lgca)), before)


# ----------------------------------------------------------------------------------- lattice size


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: "-".join(family) or "classical")
@pytest.mark.parametrize("geometry, ndim", [("lin", 1), ("square", 2), ("hex", 2), ("cubic", 3), ("moore", 3)])
@pytest.mark.parametrize("size", [np.int64(6), 6.0, 6, np.array(6)], ids=["numpy", "float", "int", "array"])
def test_one_integer_of_any_kind_gives_every_axis_that_length(geometry, ndim, family, size):
    """Without volume exclusion, a NumPy integer or a float gave a 50 x 50 lattice with a warning; with
    it, a ``TypeError``."""
    lgca = get_lgca(geometry=geometry, dims=size, density=0.5, interaction="only_propagation", seed=1, **family)
    assert lgca.dims == (6,) * ndim and all(type(length) is int for length in lgca.dims)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: "-".join(family) or "classical")
@pytest.mark.parametrize("geometry, sizes", [("lin", [7]), ("square", [6, 4]), ("hex", [4, 6]), ("cubic", [3, 4, 5])])
@pytest.mark.parametrize("kind", [tuple, list, np.array])
def test_a_sequence_of_any_kind_gives_one_length_per_axis(geometry, sizes, family, kind):
    lgca = get_lgca(geometry=geometry, dims=kind(sizes), density=0.5, interaction="only_propagation", seed=1,
                    **family)
    assert lgca.dims == tuple(sizes)


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("geometry, dims", [("lin", (6, 6)), ("square", (6,)), ("square", (6, 6, 6)),
                                            ("hex", (6, 6, 2)), ("cubic", (6, 6)), ("moore", (6,))])
def test_dims_with_another_number_of_axes_is_an_error_in_every_family(geometry, dims, ve):
    with pytest.raises(ValueError, match="dims must be"):
        get_lgca(geometry=geometry, dims=dims, ve=ve, density=0.5, interaction="only_propagation")


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("dims, error", [(0, ValueError), (-3, ValueError), (2.5, ValueError), ((4, 0), ValueError),
                                         ("6", TypeError), (True, TypeError), ([[4, 4]], TypeError)])
def test_dims_that_are_not_positive_integers_are_an_error(dims, error, ve):
    with pytest.raises(error, match="dims must"):
        get_lgca(geometry="square", dims=dims, ve=ve, density=0.5, interaction="only_propagation")


@pytest.mark.parametrize("volume_exclusion", [True, False])
@pytest.mark.parametrize("dims, expected", [(np.int64(6), (6, 6)), (6.0, (6, 6)), ([6, 4], (6, 4)),
                                            (np.array([6, 4]), (6, 4))])
def test_a_model_spec_reads_dims_as_get_lgca_does(dims, expected, volume_exclusion):
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=dims),
                     state=StateSpec(volume_exclusion=volume_exclusion, density=0.5), time=TimeSpec(steps=1, seed=1))
    result = run_model(spec, showprogress=False)
    assert result.lgca.dims == expected
    assert result.spec.space.dims in (expected, expected[0])
    assert type(np.atleast_1d(result.spec.space.dims).tolist()[0]) is int


def test_every_3d_lattice_has_the_documented_default_size():
    for family in ({}, {"ve": False}):  # without volume exclusion the default was 50 x 50 x 50
        assert get_lgca(geometry="cubic", density=0.1, interaction="only_propagation", **family).dims == (10, 10, 10)


# --------------------------------------------------------------------------------------- capacity


def _capacities(lgca):
    """The capacity the model reports, the one its interaction lists and the one its rules read."""
    return lgca.capacity, lgca.interaction_params["capacity"], lgca._compiled_model.spec.state.capacity


@pytest.mark.parametrize("interaction, default", [("birth", 8), ("birthdeath", 8), ("go_or_grow", 8),
                                                  ("evo_steric", 512), ("random_walk", 3)])
def test_a_model_reports_the_capacity_that_its_interaction_uses(interaction, default):
    """``lgca.capacity`` (and ``print(lgca)``) said 3 while growth ran with the interaction's 8."""
    lgca = get_lgca(geometry="lin", dims=10, ib=True, ve=False, interaction=interaction, seed=1)
    assert _capacities(lgca) == (default,) * 3
    assert f"Carrying capacity: {default}" in str(lgca)


def test_a_capacity_given_to_the_model_stays_when_the_interaction_changes():
    """``set_interaction`` without a capacity went back to the interaction's default of 8."""
    lgca = get_lgca(geometry="lin", dims=10, ib=True, ve=False, interaction="random_walk", capacity=20, seed=1)
    assert _capacities(lgca) == (20, 20, 20)
    lgca.set_interaction(interaction="birth")
    assert _capacities(lgca) == (20, 20, 20)
    lgca.set_interaction(interaction="birth", capacity=30)  # was ignored
    assert _capacities(lgca) == (30, 30, 30)
    lgca.set_interaction(interaction="evo_steric")
    assert _capacities(lgca) == (30, 30, 30)


def test_the_default_capacity_of_an_interaction_leaves_with_it():
    lgca = get_lgca(geometry="lin", dims=10, ib=True, ve=False, interaction="evo_steric", seed=1)
    assert _capacities(lgca) == (512, 512, 512)
    lgca.set_interaction(interaction="random_walk")
    assert _capacities(lgca) == (lgca.K,) * 3
    lgca.set_interaction(interaction="birth")
    assert _capacities(lgca) == (8, 8, 8)
