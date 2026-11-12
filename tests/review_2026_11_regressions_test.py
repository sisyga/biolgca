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


# --------------------------------------------------------------------------------------- families


def _model_with_families(ve, geometry="lin", dims=12):
    """A model whose every dividing cell founds a family."""
    spec = ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims),
        state=StateSpec(identity_based=True, volume_exclusion=ve, restchannels=2 if ve else 1, density=0.3),
        time=TimeSpec(steps=4, seed=2),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "birth_death", "parameters": {"birth_rate": 0.6, "new_family": True}}, {"name": "random_walk"}]),
    )
    return run_model(spec, showprogress=False).lgca


@pytest.mark.parametrize("ve", [True, False])
def test_the_generation_of_a_family_is_one_more_than_that_of_its_ancestor(ve):
    lgca = _model_with_families(ve)
    ancestor = lgca.family_props["ancestor"]
    assert max(ancestor) > 1  # families descend from families

    generation = lgca.calc_family_generations()

    assert len(generation) == lgca.num_families_total() + 1 and generation[0] == 0
    assert all(generation[family] == generation[ancestor[family]] + 1 for family in range(1, len(ancestor)))
    assert all(type(value) is int for value in generation)
    assert lgca.calc_generation_pop().sum() == _labels(lgca).size


@pytest.mark.parametrize("ve", [True, False])
def test_a_lineage_of_thousands_of_generations_has_generations(ve):
    """The generations were found by recursion, which Python ends after about a thousand calls."""
    lgca = _model_with_families(ve)
    first = lgca.num_families_total()
    for _ in range(3000):
        lgca.add_family(lgca.maxfamily)

    generation = lgca.calc_family_generations()

    assert generation[-1] == generation[first] + 3000
    assert lgca.calc_generation_pop().sum() == _labels(lgca).size


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("method, key", [("calc_family_generations", "generation"),
                                         ("propagate_ancestor_to_descendants", "init_ancestor")])
def test_family_tables_are_kept_until_a_family_is_founded(ve, method, key):
    """The tables were compared with a length that they never have, so every call built them anew."""
    lgca = _model_with_families(ve)
    table = getattr(lgca, method)()
    assert getattr(lgca, method)() is table and lgca.family_props[key] is table

    lgca.add_family(1)

    new = getattr(lgca, method)()
    assert new is not table and len(new) == len(table) + 1
    assert new[:-1] == table


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("method", ["calc_family_generations", "calc_generation_pop", "num_families_total",
                                    "propagate_ancestor_to_descendants", "calc_init_families_pop",
                                    "calc_family_pop_alive", "list_families_alive", "num_families_alive",
                                    "muller_plot"])
def test_family_analyses_of_a_model_without_families_say_how_to_record_them(ve, method):
    """Most of them ended in ``AttributeError: ... has no attribute 'family_props'``."""
    lgca = get_lgca(geometry="lin", dims=8, ib=True, ve=ve, density=0.3, interaction="random_walk", seed=1)
    with pytest.raises(RuntimeError, match="new_family=True"):
        getattr(lgca, method)()


@pytest.mark.parametrize("ve", [True, False])
def test_a_model_with_families_but_no_ancestry_says_that_it_has_no_family_tree(ve):
    if ve:
        lgca = get_lgca(geometry="lin", dims=8, ib=True, restchannels=2, density=0.3, interaction="birthdeath",
                        track_inheritance=True, seed=1)
    else:
        lgca = get_lgca(geometry="lin", dims=8, ib=True, ve=False, density=0.3, interaction="birthdeath", seed=1)
        lgca.init_families(type="heterogeneous", mutation=False)
    lgca.timeevo(4, recordfampop=True, showprogress=False)  # had no maxfamily: an AttributeError
    assert lgca.calc_family_pop_alive().sum() == _labels(lgca).size
    assert lgca.fam_pop_t[-1].sum() == _labels(lgca).size
    for method in ("calc_family_generations", "muller_plot", "num_families_total"):
        with pytest.raises(RuntimeError, match="no family tree"):
            getattr(lgca, method)()


# ------------------------------------------------------------------------------- order parameters


@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_mean_alignment_of_several_species_is_that_of_all_cells(geometry, dims):
    """``calc_mean_alignment`` could not broadcast the density to the flux of each species."""
    one = get_lgca(geometry=geometry, dims=dims, ve=False, density=2.0, restchannels=1,
                   interaction="only_propagation", seed=5)
    two = get_lgca(geometry=geometry, dims=dims, ve=False, n_species=2, density=1.0, restchannels=1,
                   interaction="only_propagation", seed=5)
    cells = one.nodes[one.nonborder]
    split = cells // 2  # the same cells, shared between two species
    two.nodes[two.nonborder] = np.stack([split, cells - split], axis=-2)
    two.apply_boundaries()
    two.update_dynamic_fields()

    assert two.calc_mean_alignment() == pytest.approx(one.calc_mean_alignment())
    assert -1 <= two.calc_mean_alignment() <= 1
    assert two.calc_polar_alignment_parameter() == pytest.approx(one.calc_polar_alignment_parameter())


@pytest.mark.parametrize("n_species", [1, 2, 3])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_order_parameters_are_recorded_for_any_number_of_species(geometry, dims, n_species):
    spec = ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims),
        state=StateSpec(volume_exclusion=False, n_species=n_species, density=1.5, restchannels=1),
        time=TimeSpec(steps=3, seed=6),
        dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
        analysis=AnalysisSpec(observers=[OrderParameterRecorder()]),
    )
    lgca = run_model(spec, showprogress=False).lgca
    for name in ("ent_t", "normEnt_t", "polAlParam_t", "meanAlign_t"):
        values = getattr(lgca, name)
        assert values.shape == (4,) and np.isfinite(values).all(), name


# ---------------------------------------------------------- rest channels without volume exclusion

VELOCITY_CHANNELS = {"lin": 2, "square": 4, "hex": 6, "cubic": 6, "moore": 26}


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_rest_channels_are_channels_of_the_state_on_every_lattice(geometry, dims, n_species):
    """In 1D and 2D, ``restchannels=3`` gave one rest channel and raised the capacity instead."""
    v = VELOCITY_CHANNELS[geometry]
    species = (n_species,) if n_species > 1 else ()
    lgca = get_lgca(geometry=geometry, dims=dims, ve=False, n_species=n_species, restchannels=3, density=1.0,
                    interaction="only_propagation", seed=1)
    assert (lgca.restchannels, lgca.K, lgca.capacity) == (3, v + 3, v + 3)
    assert lgca.nodes[lgca.nonborder].shape == dims + species + (v + 3,)

    lgca = get_lgca(geometry=geometry, dims=dims, ve=False, n_species=n_species, restchannels=3, capacity=20,
                    density=1.0, interaction="only_propagation", seed=1)
    assert (lgca.restchannels, lgca.K, lgca.capacity) == (3, v + 3, 20)

    spec = ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims),
                     state=StateSpec(volume_exclusion=False, n_species=n_species, restchannels=3, density=1.0))
    lgca = build_model(spec).lgca
    assert (lgca.restchannels, lgca.K, lgca.capacity) == (3, v + 3, v + 3)


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_nodes_with_several_rest_channels_are_a_state_on_every_lattice(geometry, dims, n_species):
    """In 1D and 2D, and with several species in 3D, they were refused; in 3D the capacity was that of
    one rest channel."""
    v = VELOCITY_CHANNELS[geometry]
    species = (n_species,) if n_species > 1 else ()
    nodes = np.random.default_rng(3).poisson(0.4, dims + species + (v + 3,))
    lgca = get_lgca(geometry=geometry, ve=False, n_species=n_species, nodes=nodes, interaction="only_propagation")
    assert (lgca.restchannels, lgca.K, lgca.capacity) == (3, v + 3, v + 3)
    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder], nodes)

    lgca.timeevo(4, showprogress=False)

    after = lgca.nodes[lgca.nonborder]
    np.testing.assert_array_equal(after[..., v:], nodes[..., v:])  # resting cells stay in their channel
    np.testing.assert_array_equal(after[..., :v].sum(axis=tuple(range(len(dims)))),
                                  nodes[..., :v].sum(axis=tuple(range(len(dims)))))


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
@pytest.mark.parametrize("interaction", ["random_walk", "go_or_rest", "go_or_grow", "di_alignment"])
def test_interactions_run_with_several_rest_channels_on_every_lattice(geometry, dims, n_species, interaction):
    v = VELOCITY_CHANNELS[geometry]
    lgca = get_lgca(geometry=geometry, dims=dims, ve=False, n_species=n_species, restchannels=3, density=2.0,
                    interaction=interaction, seed=2)
    cells = lgca.cell_density[lgca.nonborder].sum()
    lgca.timeevo(4, record=True, showprogress=False)
    assert lgca.nodes_t.shape[-1] == v + 3
    if interaction != "go_or_grow":
        assert lgca.cell_density[lgca.nonborder].sum() == cells
    if interaction == "random_walk":  # the cells of a node take any of its channels
        assert (lgca.nodes_t[1:, ..., v:].sum(axis=tuple(range(lgca.nodes_t.ndim - 1))) > 0).all()


@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_identity_based_models_without_volume_exclusion_keep_their_one_rest_channel(geometry, dims):
    v = VELOCITY_CHANNELS[geometry]
    with pytest.warns(UserWarning, match="have one rest channel"):
        lgca = get_lgca(geometry=geometry, dims=dims, ve=False, ib=True, restchannels=3, density=1.0,
                        interaction="only_propagation", seed=1)
    assert (lgca.restchannels, lgca.K, lgca.capacity) == (1, v + 1, v + 1)

    counts = np.ones(dims + (v + 1,), dtype=int)  # in 3D, cell counts as nodes ended in an AttributeError
    lgca = get_lgca(geometry=geometry, ve=False, ib=True, nodes=counts, interaction="only_propagation")
    assert (lgca.restchannels, lgca.K, lgca.capacity) == (1, v + 1, v + 1)
    assert _labels(lgca).size == counts.sum()

    with pytest.raises(RuntimeError, match="have one rest channel"):
        get_lgca(geometry=geometry, ve=False, ib=True, nodes=np.ones(dims + (v + 3,), dtype=int),
                 interaction="only_propagation")


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_rest_channels_beyond_those_of_the_nodes_add_to_the_capacity(geometry, dims, n_species):
    v = VELOCITY_CHANNELS[geometry]
    species = (n_species,) if n_species > 1 else ()
    for rest, capacity in ((1, v + 3), (3, v + 3), (4, v + 4)):
        nodes = np.ones(dims + species + (v + rest,), dtype=int)
        if rest == 3:
            lgca = get_lgca(geometry=geometry, ve=False, n_species=n_species, nodes=nodes, restchannels=3,
                            interaction="only_propagation")
        else:
            with pytest.warns(UserWarning, match=f"imply {rest} rest channels"):
                lgca = get_lgca(geometry=geometry, ve=False, n_species=n_species, nodes=nodes, restchannels=3,
                                interaction="only_propagation")
        assert (lgca.restchannels, lgca.K, lgca.capacity) == (rest, v + rest, capacity)


# ------------------------------------------------------------------------------------- recordings

RECORDINGS = ("nodes_t", "nodes_steps", "dens_t", "dens_steps", "n_t", "n_steps")


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: "-".join(family) or "classical")
@pytest.mark.parametrize("geometry, dims", GEOMETRIES[:3])
def test_the_recordings_on_a_model_are_those_of_its_last_run(geometry, dims, family):
    """A run that did not record the nodes left the nodes of the run before next to its own densities."""
    lgca = get_lgca(geometry=geometry, dims=dims, density=0.5, restchannels=1, interaction="random_walk", seed=1,
                    **family)
    lgca.timeevo(5, record=True, recordN=True, showprogress=False)
    nodes_t, n_t = lgca.nodes_t, lgca.n_t
    assert len(nodes_t) == len(lgca.dens_t) == len(n_t) == 6

    lgca.timeevo(7, showprogress=False)  # records the density

    assert len(lgca.dens_t) == len(lgca.dens_steps) == 8
    for name in ("nodes_t", "nodes_steps", "n_t", "n_steps"):
        assert not hasattr(lgca, name), name
    assert set(lgca.data) == {"density"}
    assert len(nodes_t) == len(n_t) == 6  # the arrays taken from the model are the caller's

    lgca.timeevo(2, record=True, recorddens=False, showprogress=False)

    assert len(lgca.nodes_t) == len(lgca.nodes_steps) == 3
    assert not hasattr(lgca, "dens_t") and not hasattr(lgca, "dens_steps")


def test_a_run_of_a_model_from_a_spec_forgets_the_recordings_of_the_run_before():
    from lgca.simulation import DensityRecorder, NodeRecorder, SimulationRunner

    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(5, 5)), state=StateSpec(density=0.5, restchannels=1),
                     time=TimeSpec(steps=3, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
                     analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]))
    lgca = run_model(spec, showprogress=False).lgca
    assert len(lgca.nodes_t) == len(lgca.dens_t) == 4

    SimulationRunner(lgca, timesteps=6, observers=[DensityRecorder()], showprogress=False).run()

    assert len(lgca.dens_t) == 7 and not hasattr(lgca, "nodes_t")


def test_means_of_recorded_nodes_leave_with_the_recorded_nodes():
    lgca = get_lgca(geometry="lin", dims=10, ib=True, ve=False, density=1.0, interaction="birthdeath", seed=1)
    lgca.timeevo(3, record=True, showprogress=False)
    assert set(lgca.calc_prop_mean_spatiotemp()) >= {"r_b"}

    lgca.timeevo(4, showprogress=False)

    assert lgca.mean_prop_t == {}
    assert not hasattr(lgca, "nodes_t") and not hasattr(lgca, "cells_t")


# ---------------------------------------------------------- the type of nodes without volume exclusion


def _nove_model(geometry, dims, n_species, route, tmp_path):
    """A model without volume exclusion whose nodes were filled on the given route."""
    v = VELOCITY_CHANNELS[geometry]
    shape = dims + ((n_species,) if n_species > 1 else ()) + (v + 1,)
    counts = np.random.default_rng(1).poisson(0.5, shape)
    common = dict(geometry=geometry, ve=False, n_species=n_species, interaction="random_walk", seed=1)
    if route == "density":
        return get_lgca(dims=dims, density=1.0, restchannels=1, **common)
    if route == "random_reset":
        lgca = get_lgca(nodes=counts.astype(np.uint64), **common)
        lgca.random_reset(1.0)
        return lgca
    if route in ("int64", "uint64", "int32", "float", "bool"):
        return get_lgca(nodes=counts.astype(route.replace("float", "float64")), **common)
    state = dict(volume_exclusion=False, n_species=n_species, restchannels=1)
    if route == "spec nodes":
        state["nodes"] = counts.astype(np.uint64)
    elif route == "region":
        state["initializer"] = {"name": "region", "parameters": {"extent": 2, "density": 3.0}}
    elif route == "from_npz":
        np.savez(tmp_path / "state.npz", nodes=counts.astype(np.uint64))
        state["initializer"] = {"name": "from_npz", "parameters": {"path": "state.npz"}}
    spec = ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims), state=StateSpec(**state),
                     time=TimeSpec(seed=1), dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]))
    return build_model(spec, resource_base=tmp_path).lgca


@pytest.mark.parametrize("route", ["density", "random_reset", "int64", "uint64", "int32", "float", "bool",
                                   "spec nodes", "region", "from_npz"])
@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_nodes_without_volume_exclusion_have_one_type_however_they_were_filled(geometry, dims, n_species, route,
                                                                              tmp_path):
    """Random nodes were ``int64`` and given ones ``uint64``, whose differences wrap around."""
    lgca = _nove_model(geometry, dims, n_species, route, tmp_path)
    assert lgca.nodes.dtype == np.int64
    assert lgca.cell_density.dtype == np.int64

    lgca.timeevo(2, record=True, showprogress=False)

    assert lgca.nodes.dtype == lgca.nodes_t.dtype == lgca.cell_density.dtype == np.int64
    assert lgca.dens_t.dtype.kind == "i"  # recorded in a smaller signed type while the counts fit
    assert (lgca.nodes_t[1] - lgca.nodes_t[0]).sum() == 0  # cells are conserved: the differences cancel
    assert (lgca.cell_density[lgca.nonborder] - 2 * lgca.cell_density[lgca.nonborder]).max() <= 0


@pytest.mark.parametrize("n_species", [1, 2])
@pytest.mark.parametrize("value", [np.uint64(2**63), float(2**63), np.uint64(2**64 - 1)])
def test_a_channel_holds_at_most_the_largest_signed_integer(value, n_species):
    shape = (3, 3) if n_species == 1 else (3, n_species, 3)
    nodes = np.zeros(shape, dtype=np.asarray(value).dtype)
    nodes.flat[0] = value
    with pytest.raises(ValueError, match="nodes counts exceed the representable range"):
        get_lgca(geometry="lin", ve=False, n_species=n_species, nodes=nodes, interaction="only_propagation")
    nodes.flat[0] = 2**62
    lgca = get_lgca(geometry="lin", ve=False, n_species=n_species, nodes=nodes, interaction="only_propagation")
    assert lgca.nodes.dtype == np.int64 and lgca.nodes.max() == 2**62


# ------------------------------------------------------------------------- errors that say what to do


def _spec(**sections):
    base = dict(space=SpaceSpec(geometry="lin", dims=6), state=StateSpec(density=0.3, restchannels=1),
                time=TimeSpec(steps=2, seed=1),
                dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]))
    return ModelSpec(**{**base, **sections})


@pytest.mark.parametrize("value", [None, "1", float("nan"), float("inf"), 1.5, True, [1]])
@pytest.mark.parametrize("section, name", [("state", "restchannels"), ("state", "n_species"), ("state", "capacity"),
                                           ("time", "steps"), ("time", "timing_trace")])
def test_a_number_of_a_spec_that_is_no_integer_is_named_with_its_value(section, name, value):
    """``None`` ended in ``TypeError: int() argument must be ...``, ``nan`` and ``inf`` in errors of ``int``."""
    if name == "capacity" and value is None:
        return  # no capacity
    kind = {"state": StateSpec, "time": TimeSpec}[section]
    with pytest.raises(ValueError) as error:
        build_model(_spec(**{section: kind(**{name: value})}))
    assert f"model.{section}.{name} must be a" in str(error.value) and f"got {value!r}" in str(error.value)


@pytest.mark.parametrize("density", ["a", [0.1, 0.2], float("nan"), -1, True])
def test_a_density_that_is_no_number_is_named_with_its_value(density):
    with pytest.raises(ValueError, match="model.state.density must be a finite non-negative scalar, got"):
        build_model(_spec(state=StateSpec(density=density)))


@pytest.mark.parametrize("section, value, write", [
    ("dynamics", [{"name": "random_walk"}], "InteractionPipelineSpec(operators="),
    ("dynamics", {"operators": [{"name": "random_walk"}]}, "model_spec_from_dict"),
    ("dynamics", None, "InteractionPipelineSpec(operators="),
    ("space", {"geometry": "lin"}, "SpaceSpec("), ("state", {"density": 0.1}, "StateSpec("),
    ("state", None, "StateSpec("), ("time", 10, "TimeSpec(steps="),
    ("analysis", [OrderParameterRecorder()], "AnalysisSpec(observers=")])
def test_a_section_of_a_spec_of_another_type_says_how_to_write_it(section, value, write):
    """A list of operators as ``dynamics`` ended in ``'list' object has no attribute 'propagation'``."""
    with pytest.raises(TypeError) as error:
        build_model(_spec(**{section: value}))
    assert f"model.{section} must be a" in str(error.value) and write in str(error.value)


@pytest.mark.parametrize("operators, message", [
    ("random_walk", "must be a list of operators.*write operators=\\[{'name': 'random_walk'}\\]"),
    ({"name": "random_walk"}, "must be a list of operators.*write operators=\\[{'name': 'random_walk'}\\]"),
    (["random_walk"], "operators\\[0\\] requires a 'name': write {'name': 'random_walk'}"),
    ([None], "operators\\[0\\] requires a 'name'.*got None")])
def test_operators_that_are_not_a_list_of_entries_say_how_to_write_them(operators, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_model(_spec(dynamics=InteractionPipelineSpec(operators=operators)))


@pytest.mark.parametrize("observers, message", [
    ([{"type": "DensityRecorder"}], "observers\\[0\\] must be an observer.*pass the object"),
    (["DensityRecorder"], "observers\\[0\\] must be an observer.*pass the object"),
    ([OrderParameterRecorder], "observers\\[0\\] must be an observer.*OrderParameterRecorder\\(\\)"),
    (OrderParameterRecorder(), "observers must be a sequence of observers")])
def test_observers_that_are_not_objects_say_what_to_pass(observers, message):
    """A mapping ended in ``'dict' object has no attribute 'on_step'`` when the run started."""
    with pytest.raises(TypeError, match=message):
        build_model(_spec(analysis=AnalysisSpec(observers=observers)))
    if isinstance(observers, list):
        from lgca.simulation import SimulationRunner

        lgca = get_lgca(geometry="lin", dims=6, density=0.3, interaction="random_walk")
        with pytest.raises(TypeError, match=message):
            SimulationRunner(lgca, timesteps=1, observers=observers, showprogress=False).run()


@pytest.mark.parametrize("family", [dict(), dict(identity_based=True), dict(n_species=2)],
                         ids=["classical", "identity", "species"])
@pytest.mark.parametrize("recorder", ["OrderParameterRecorder", "ChannelDensityRecorder"])
def test_recorders_of_models_without_volume_exclusion_say_so_for_other_models(recorder, family):
    """They ended in ``AttributeError: ... has no attribute 'calc_entropy'`` (``'channel_pop'``)."""
    import lgca.simulation as simulation

    spec = _spec(state=StateSpec(density=0.3, restchannels=1, **family),
                 analysis=AnalysisSpec(observers=[getattr(simulation, recorder)()]))
    with pytest.raises(RuntimeError, match=f"{recorder} records .* models without volume exclusion"):
        run_model(spec, showprogress=False)


@pytest.mark.parametrize("family", [dict(), dict(identity_based=True), dict(n_species=2)],
                         ids=["counts", "identity", "species"])
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_cells_per_channel_are_recorded_for_every_model_without_volume_exclusion(geometry, dims, family):
    """Models with one species and no identities had no ``channel_pop`` to record."""
    from lgca.simulation import ChannelDensityRecorder, DensityRecorder

    spec = ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims),
                     state=StateSpec(volume_exclusion=False, density=1.0, restchannels=1, **family),
                     time=TimeSpec(steps=3, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "random_walk"}]),
                     analysis=AnalysisSpec(observers=[ChannelDensityRecorder(), DensityRecorder()]))
    result = run_model(spec, showprogress=False)
    channels = result.data["channel_population"]
    assert channels.shape[0] == 4 and channels.shape[-1] == result.lgca.K
    density = result.data["density"]
    np.testing.assert_array_equal(channels.sum(-1), density)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: "-".join(family) or "classical")
def test_get_lgca_names_arguments_that_are_no_numbers(family):
    common = dict(geometry="lin", dims=6, density=0.3, interaction="only_propagation")
    with pytest.raises(ValueError, match="n_species must be a positive integer, got None"):
        get_lgca(**{**common, **family, "n_species": None})
    with pytest.raises(ValueError, match="restchannels must be a non-negative integer"):
        get_lgca(restchannels=float("inf"), **common, **family)
    if family.get("ve") is False:  # "3" passed as a number until the first comparison with one
        with pytest.raises(ValueError, match="capacity must be a positive integer, got '3'"):
            get_lgca(capacity="3", **common, **family)


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("n_species", [1, 2])
def test_nodes_given_as_lists_are_read_as_arrays(ve, n_species):
    """Lists ended in ``'list' object has no attribute 'shape'``."""
    nodes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if n_species == 2:
        nodes = [[site, site] for site in nodes]
    lgca = get_lgca(geometry="lin", ve=ve, n_species=n_species, nodes=nodes, interaction="only_propagation")
    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder], np.array(nodes))


def test_ragged_lists_as_nodes_say_how_to_give_lists_of_labels():
    with pytest.raises(ValueError, match="NumPy array of lists"):
        get_lgca(geometry="lin", ib=True, ve=False, nodes=[[[0], [], [1, 2]], [[], [3], []]],
                 interaction="only_propagation")
