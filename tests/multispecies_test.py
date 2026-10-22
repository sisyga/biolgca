import numpy as np
import pytest

from lgca import get_lgca


GEOMS = {
    "lin": ("lgca.ms_1d", "MSLGCA_1D", "MSLGCA_NoVE_1D"),
    "square": ("lgca.ms_square", "MSLGCA_Square", "MSLGCA_NoVE_Square"),
    "hex": ("lgca.ms_hex", "MSLGCA_Hex", "MSLGCA_NoVE_Hex"),
    "cubic": ("lgca.ms_cubic", "MSLGCA_Cubic", "MSLGCA_NoVE_Cubic"),
    "moore": ("lgca.ms_moore", "MSLGCA_Moore", "MSLGCA_NoVE_Moore"),
}


NODE_DIMS = {
    "lin": (3,),
    "square": (3, 4),
    "hex": (3, 4),
    "cubic": (3, 4, 5),
    "moore": (3, 4, 5),
}


def _load_ms_classes(geom):
    module_name, ve_name, nove_name = GEOMS[geom]
    module = __import__(module_name, fromlist=[ve_name, nove_name])
    return getattr(module, ve_name), getattr(module, nove_name)


@pytest.mark.parametrize("geom", GEOMS)
def test_get_lgca_multispecies_returns_geometry_class(geom):
    ve_cls, _ = _load_ms_classes(geom)

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        density=0,
        restchannels=1,
        interaction="only_propagation",
        dims=3,
    )

    assert isinstance(lgca, ve_cls)
    expected_shape = tuple(d + 2 * lgca.r_int for d in lgca.dims) + (2, lgca.K)
    assert lgca.nodes.shape == expected_shape
    assert lgca.species_density.shape == lgca.nodes.shape[:-1]
    assert lgca.cell_density.shape == lgca.nodes.shape[:-2]


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_propagation_preserves_species_axis(geom):
    ve_cls, _ = _load_ms_classes(geom)
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        density=0,
        restchannels=1,
        interaction="only_propagation",
        dims=4 if geom == "hex" else 3,
    )
    lgca.nodes.fill(False)
    center = (lgca.r_int,) * len(lgca.dims)
    channel = 0 if geom != "moore" else ve_cls._vels.index((1, 0, 0))
    lgca.nodes[center + (1, channel)] = True

    lgca.timestep()

    if geom == "lin":
        target = (lgca.r_int + 1,)
    elif geom in {"square", "hex"}:
        target = (lgca.r_int + 1, lgca.r_int)
    else:
        target = (lgca.r_int + 1, lgca.r_int, lgca.r_int)
    assert lgca.nodes[target + (1, channel)]
    assert not lgca.nodes[center + (1, channel)]
    assert not lgca.nodes[..., 0, :].any()


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_propagates_every_velocity_channel(geom):
    ve_cls, _ = _load_ms_classes(geom)
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        density=0,
        restchannels=1,
        interaction="only_propagation",
        dims=4 if geom == "lin" else (4, 4, 4) if geom in {"cubic", "moore"} else (4, 4),
        bc="absorbing",
    )
    center = tuple(r + 1 for r in (lgca.r_int,) * len(lgca.dims))

    for channel in range(ve_cls.velocitychannels):
        lgca.nodes.fill(False)
        lgca.nodes[center + (1, channel)] = True

        lgca.timestep()

        assert lgca.nodes[..., 1, :].sum() == 1
        assert not lgca.nodes[center + (1, channel)]
        assert not lgca.nodes[..., 0, :].any()


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_explicit_nodes_derive_geometry_from_species_axis(geom):
    ve_cls, _ = _load_ms_classes(geom)
    dims = NODE_DIMS[geom]
    nodes = np.zeros(dims + (2, ve_cls.velocitychannels + 1), dtype=bool)
    nodes[(0,) * len(dims) + (1, 0)] = True

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        nodes=nodes,
        interaction="only_propagation",
    )

    assert lgca.dims == dims
    assert lgca.n_species == 2
    assert lgca.K == ve_cls.velocitychannels + 1
    assert lgca.restchannels == 1
    assert lgca.nodes[lgca.nonborder].shape == nodes.shape
    assert lgca.nodes[lgca.nonborder].sum() == 1


@pytest.mark.parametrize("geom", GEOMS)
def test_get_lgca_multispecies_nove_returns_integer_lattice(geom):
    _, nove_cls = _load_ms_classes(geom)

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        restchannels=1,
        density=0,
        interaction="only_propagation",
        dims=3,
    )

    assert isinstance(lgca, nove_cls)
    assert lgca.nodes.dtype.kind in "iu"
    expected_shape = tuple(d + 2 * lgca.r_int for d in lgca.dims) + (2, lgca.K)
    assert lgca.nodes.shape == expected_shape


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_nove_propagation_preserves_counts(geom):
    _, nove_cls = _load_ms_classes(geom)
    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        restchannels=1,
        density=0,
        interaction="only_propagation",
        dims=4 if geom == "hex" else 3,
    )
    lgca.nodes.fill(0)
    center = (lgca.r_int,) * len(lgca.dims)
    channel = 0 if geom != "moore" else nove_cls._vels.index((1, 0, 0))
    lgca.nodes[center + (1, channel)] = 2

    lgca.timestep()

    if geom == "lin":
        target = (lgca.r_int + 1,)
    elif geom in {"square", "hex"}:
        target = (lgca.r_int + 1, lgca.r_int)
    else:
        target = (lgca.r_int + 1, lgca.r_int, lgca.r_int)
    assert lgca.nodes[target + (1, channel)] == 2
    assert lgca.nodes[center + (1, channel)] == 0
    assert lgca.nodes[..., 0, :].sum() == 0


@pytest.mark.parametrize("geom", GEOMS)
def test_multispecies_nove_explicit_nodes_derive_geometry_from_species_axis(geom):
    _, nove_cls = _load_ms_classes(geom)
    dims = NODE_DIMS[geom]
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    nodes[(0,) * len(dims) + (1, 0)] = 3

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="only_propagation",
    )

    assert lgca.dims == dims
    assert lgca.n_species == 2
    assert lgca.K == nove_cls.velocitychannels + 1
    assert lgca.restchannels == 1
    assert lgca.nodes[lgca.nonborder].shape == nodes.shape
    assert lgca.nodes[lgca.nonborder].sum() == 3


def test_multispecies_rejects_identity_based_mode_for_now():
    with pytest.raises(NotImplementedError, match="Multi-species identity-based"):
        get_lgca(
            geometry="square",
            n_species=2,
            ib=True,
            density=0,
            interaction="only_propagation",
        )


def test_excitable_medium_ms_rejects_single_species_models():
    with pytest.raises(ValueError, match="excitable_medium_ms.*multi-species"):
        get_lgca(
            geometry="square",
            restchannels=1,
            density=0,
            interaction="excitable_medium_ms",
        )


def test_excitable_medium_ms_requires_exactly_two_species():
    with pytest.raises(ValueError, match="exactly two species"):
        get_lgca(
            geometry="square",
            n_species=3,
            restchannels=1,
            density=0,
            interaction="excitable_medium_ms",
        )


def test_excitable_medium_ms_requires_volume_exclusion():
    with pytest.raises(ValueError, match="volume exclusion"):
        get_lgca(
            geometry="square",
            n_species=2,
            ve=False,
            restchannels=1,
            density=0,
            interaction="excitable_medium_ms",
        )


def test_excitable_medium_ms_requires_rest_channel():
    with pytest.raises(ValueError, match="rest channel"):
        get_lgca(
            geometry="square",
            n_species=2,
            restchannels=0,
            density=0,
            interaction="excitable_medium_ms",
        )


def test_excitable_medium_ms_follows_its_reaction_probabilities():
    # every node holds one resting inhibitor (species 0) and one moving activator (species 1): with
    # x = 1/4 of the velocity channels and y = 1 of the rest channel occupied, the documented rates give
    # an inhibitor gain x and loss y, and one fast activator reaction (N=1) a gain x^2 (1 + (y + beta) / alpha)
    # and a loss x^3 + x (y + beta) / alpha; the nodes do not interact
    beta, alpha = 0.05, 1.0
    with pytest.warns(UserWarning, match="local interactions"):
        lgca = get_lgca(geometry="square", n_species=2, restchannels=1, dims=(50, 50), density=0,
                        interaction="excitable_medium_ms", beta=beta, alpha=alpha, N=1, propagation=False,
                        seed=1)
    velocity = lgca.velocitychannels
    lgca.nodes[lgca.nonborder + (0, velocity)] = True
    lgca.nodes[lgca.nonborder + (1, 0)] = True
    lgca.timestep()

    nodes = lgca.nodes[lgca.nonborder]
    assert nodes[..., 0, :velocity].sum() == 0  # inhibitors only rest
    assert nodes[..., 1, velocity:].sum() == 0  # activators only move
    x, y = 1 / velocity, 1.0
    gain, loss = x * x * (1 + (y + beta) / alpha), x ** 3 + x * (y + beta) / alpha
    # one random number decides gain and loss, independently: +1 with gain (1 - loss), -1 with loss (1 - gain)
    expected = {"inhibitor": 1 + x * (1 - y) - y * (1 - x), "activator": 1 + gain - loss}
    measured = {"inhibitor": nodes[..., 0, velocity:].sum(-1), "activator": nodes[..., 1, :velocity].sum(-1)}
    for name, cells in measured.items():
        error = cells.std() / np.sqrt(cells.size)
        assert abs(cells.mean() - expected[name]) < 4 * error, name


def test_multispecies_nodes_shape_warning_uses_species_axis():
    nodes = np.zeros((3, 3, 2, 5), dtype=bool)

    with pytest.warns(UserWarning, match="Provided nodes"):
        get_lgca(
            geometry="square",
            n_species=2,
            nodes=nodes,
            dims=(4, 4),
            restchannels=1,
            interaction="only_propagation",
        )


@pytest.mark.parametrize("ve", [True, False])
def test_multispecies_random_density_is_total_density_across_species(ve):
    lgca = get_lgca(
        geometry="square",
        n_species=2,
        ve=ve,
        dims=(80, 80),
        restchannels=1,
        density=1.2,
        interaction="only_propagation",
        seed=123,
    )

    per_node = lgca.nodes[lgca.nonborder].sum(axis=(-2, -1))
    error = per_node.std() / np.sqrt(per_node.size)
    assert abs(per_node.mean() - 1.2) < 4 * error


@pytest.mark.parametrize("geom,dims", [("lin", (5000,)), ("hex", (70, 70)), ("moore", (16, 16, 16))])
def test_multispecies_nove_birth_routes_offspring_through_the_mutation_matrix(geom, dims):
    # two resting cells of each species per node: a cell of species s divides with r_b[s] (1 - 4 / capacity),
    # and its daughter belongs to species t with mutation_matrix[s, t]
    _, nove_cls = _load_ms_classes(geom)
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    nodes[..., nove_cls.velocitychannels] = 2
    r_b, capacity = np.array([0.8, 0.2]), 100
    mutation_matrix = np.array([[0.75, 0.25], [0.2, 0.8]])
    lgca = get_lgca(geometry=geom, n_species=2, ve=False, nodes=nodes, interaction="birth", r_b=list(r_b),
                    capacity=capacity, mutation_matrix=mutation_matrix, propagation=False, seed=1)

    lgca.timestep()

    daughters = lgca.nodes[lgca.nonborder].sum(-1).astype(int) - 2
    daughters = daughters.reshape(-1, 2)
    expected = (2 * r_b * (1 - 4 / capacity)) @ mutation_matrix
    error = daughters.std(axis=0) / np.sqrt(len(daughters))
    assert np.all(np.abs(daughters.mean(axis=0) - expected) < 4 * error), (daughters.mean(axis=0), expected)


def test_multispecies_nove_birth_growth_uses_total_density_for_capacity_limit():
    nodes = np.zeros((1, 2, 3), dtype=np.uint)
    nodes[0, 0, 2] = 1
    nodes[0, 1, 2] = 9

    lgca = get_lgca(
        geometry="lin",
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="birth",
        r_b=[1.0, 0.0],
        capacity=10,
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    assert lgca.nodes[lgca.nonborder].sum() == 10


def _stacked_mutation_matrix(lgca):
    """The mutation matrix of the growth rule that get_lgca stacked for a legacy interaction."""
    stacked = lgca._compiled_model.pipeline.operators[0].operators
    return next(np.asarray(operator.parameters["mutation_matrix"]) for operator in stacked
                if operator.parameters.get("mutation_matrix") is not None)


def test_multispecies_nove_birth_derives_mutation_matrix_from_birth_rate_bins():
    lgca = get_lgca(
        geometry="lin",
        n_species=3,
        ve=False,
        density=0,
        restchannels=1,
        interaction="birth",
        r_b=[0.1, 0.2, 0.4],
        std=0.01,
    )

    mutation_matrix = _stacked_mutation_matrix(lgca)

    assert mutation_matrix.shape == (3, 3)
    assert mutation_matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.all(np.diag(mutation_matrix) > 0.99)


@pytest.mark.parametrize("geom,dims", [("lin", (3,)), ("hex", (4, 4)), ("moore", (4, 4, 4))])
def test_multispecies_nove_birthdeath_applies_death_to_all_species(geom, dims):
    _, nove_cls = _load_ms_classes(geom)
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    center = tuple(dim // 2 for dim in dims)
    nodes[center + (0, 0)] = 5
    nodes[center + (1, nove_cls.velocitychannels)] = 7

    lgca = get_lgca(geometry=geom, n_species=2, ve=False, nodes=nodes, interaction="birthdeath",
                    r_b=[0.0, 0.0], r_d=1.0, propagation=False, seed=1)

    lgca.timestep()

    assert lgca.nodes[lgca.nonborder].sum() == 0


@pytest.mark.parametrize("geom,dims", [("lin", (3,)), ("hex", (4, 4)), ("moore", (4, 4, 4))])
def test_multispecies_nove_go_or_grow_uses_species_specific_kappa(geom, dims):
    # at density 0.8 > theta, kappa -100 makes every cell of species 0 move and kappa 100 every cell of
    # species 1 rest
    _, nove_cls = _load_ms_classes(geom)
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    center = tuple(dim // 2 for dim in dims)
    nodes[center + (0, 0)] = 40
    nodes[center + (1, 0)] = 40

    lgca = get_lgca(geometry=geom, n_species=2, ve=False, nodes=nodes, interaction="go_or_grow",
                    kappa=[-100.0, 100.0], theta=0.5, r_b=0.0, r_d=0.0, capacity=100, propagation=False, seed=1)

    lgca.timestep()

    rest_counts = lgca.nodes[lgca.nonborder][..., lgca.velocitychannels:].sum(axis=tuple(range(len(dims))) + (-1,))
    assert rest_counts[0] == 0
    assert rest_counts[1] == 40


def test_multispecies_nove_go_or_grow_derives_mutation_matrix_from_kappa_bins():
    lgca = get_lgca(
        geometry="lin",
        n_species=3,
        ve=False,
        density=0,
        restchannels=1,
        interaction="go_or_grow",
        kappa=[1.0, 2.0, 4.0],
        kappa_std=0.01,
    )

    mutation_matrix = _stacked_mutation_matrix(lgca)

    assert mutation_matrix.shape == (3, 3)
    assert mutation_matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.all(np.diag(mutation_matrix) > 0.99)


@pytest.mark.parametrize("ve,interaction", [(True, "go_or_grow_kappa"), (False, "evo_steric")])
def test_factory_rejects_interactions_without_multispecies_implementation(ve, interaction):
    with pytest.raises(ValueError, match="Unknown interaction"):
        get_lgca(geometry="square", dims=4, ve=ve, n_species=2, restchannels=1, seed=1, interaction=interaction)


@pytest.mark.parametrize(
    "ve,interaction",
    [(True, None), (True, "only_propagation"), (True, "alignment"), (False, None), (False, "random_walk"),
     (False, "dd_alignment")],
)
def test_the_motility_rules_keep_the_cells_of_every_species(ve, interaction):
    # the legacy multispecies classes had only a few interactions; the rules work for any number of species
    kwargs = {} if interaction is None else {"interaction": interaction}
    lgca = get_lgca(geometry="square", dims=6, ve=ve, n_species=2, restchannels=1, density=1, seed=1, **kwargs)
    lgca.timeevo(timesteps=3, record=True, recorddens=False, showprogress=False)
    per_species = lgca.nodes_t.sum(axis=(1, 2, 4))
    assert per_species[0].min() > 0
    np.testing.assert_array_equal(per_species, np.broadcast_to(per_species[0], per_species.shape))


def test_the_birth_rule_serves_several_species_with_volume_exclusion():
    # one cell per node and species: its daughter goes to a random channel of its species and survives in
    # the K - 1 empty ones, so each species grows by r_b (K - 1) / K per node
    nodes = np.zeros((60, 60, 2, 5), dtype=bool)
    nodes[..., 0] = True
    lgca = get_lgca(geometry="square", n_species=2, nodes=nodes, interaction="birth", r_b=0.4, seed=1)
    lgca.timestep()
    daughters = (lgca.nodes[lgca.nonborder].sum(-1) - 1).reshape(-1, 2)
    error = daughters.std(axis=0) / np.sqrt(len(daughters))
    assert np.all(np.abs(daughters.mean(axis=0) - 0.4 * 4 / 5) < 4 * error)


@pytest.mark.parametrize("geometry, dims", [("hex", (6, 6)), ("moore", (4, 4, 4)), ("square", (5, 4))])
def test_the_mean_species_property_of_a_node_weights_its_species_by_their_cells(geometry, dims):
    from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
    from lgca.plot_data import mean_species_property
    from lgca.simulation import DensityRecorder

    result = run_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims, boundary="reflecting"),
        state=StateSpec(density=0.3, n_species=3, volume_exclusion=False, capacity=4, restchannels=1),
        time=TimeSpec(steps=2, seed=4), analysis=AnalysisSpec(observers=[DensityRecorder()])), showprogress=False)
    lgca, values = result.lgca, [1.0, 0.5, 0.0]
    species = lgca.species_density[lgca.nonborder]
    occupied = species.sum(-1) > 0
    assert occupied.any() and not occupied.all()
    means = mean_species_property(lgca, values)
    np.testing.assert_allclose(means[occupied], (species @ values)[occupied] / species.sum(-1)[occupied])
    assert np.isnan(means[~occupied]).all()
    history = mean_species_property(lgca, values, result.data["density"])  # every recorded step
    assert history.shape == (3,) + dims
    np.testing.assert_array_equal(history[-1], means)
    with pytest.raises(ValueError, match="one number per species"):
        mean_species_property(lgca, [1.0, 0.0])


def test_the_mean_species_property_needs_several_species():
    from lgca.plot_data import mean_species_property

    with pytest.raises(ValueError, match="several species"):
        mean_species_property(get_lgca(geometry="square", dims=(4, 4), density=0.2), [1.0])
