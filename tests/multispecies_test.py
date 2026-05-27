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
        dims=3,
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
        dims=3,
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


def test_excitable_medium_ms_keeps_species_in_expected_channel_groups():
    with pytest.warns(UserWarning, match="local interactions"):
        lgca = get_lgca(
            geometry="square",
            n_species=2,
            restchannels=1,
            density=0,
            interaction="excitable_medium_ms",
            propagation=False,
            seed=1,
        )

    center = (lgca.r_int, lgca.r_int)
    lgca.nodes.fill(False)
    lgca.nodes[center + (0, lgca.velocitychannels)] = True
    lgca.nodes[center + (1, 0)] = True
    lgca.timestep()

    assert lgca.nodes[..., 0, : lgca.velocitychannels].sum() == 0
    assert lgca.nodes[..., 1, lgca.velocitychannels :].sum() == 0
    assert lgca.nodes[..., 0, lgca.velocitychannels :].sum() >= 0
    assert lgca.nodes[..., 1, : lgca.velocitychannels].sum() >= 0


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

    achieved_density = lgca.nodes[lgca.nonborder].sum() / np.prod(lgca.dims)
    assert achieved_density == pytest.approx(1.2, rel=0.15)


def test_multispecies_nove_birth_routes_offspring_through_explicit_mutation_matrix():
    nodes = np.zeros((1, 2, 3), dtype=np.uint)
    nodes[0, 0, 2] = 30
    mutation_matrix = np.array([[0.0, 1.0], [0.0, 1.0]])

    lgca = get_lgca(
        geometry="lin",
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="birth",
        r_b=[1.0, 0.0],
        capacity=1000,
        mutation_matrix=mutation_matrix,
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    species_counts = lgca.nodes[lgca.nonborder].sum(axis=(0, 2))
    assert species_counts[0] == 30
    assert species_counts[1] > 0


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

    mutation_matrix = lgca.interaction_params["mutation_matrix"]

    assert mutation_matrix.shape == (3, 3)
    assert mutation_matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.all(np.diag(mutation_matrix) > 0.99)


def test_multispecies_nove_birthdeath_applies_death_to_all_species():
    nodes = np.zeros((1, 2, 3), dtype=np.uint)
    nodes[0, 0, 0] = 5
    nodes[0, 1, 2] = 7

    lgca = get_lgca(
        geometry="lin",
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="birthdeath",
        r_b=[0.0, 0.0],
        r_d=1.0,
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    assert lgca.nodes[lgca.nonborder].sum() == 0


def test_multispecies_nove_go_or_grow_uses_species_specific_kappa():
    nodes = np.zeros((1, 2, 3), dtype=np.uint)
    nodes[0, 0, 0] = 40
    nodes[0, 1, 0] = 40

    lgca = get_lgca(
        geometry="lin",
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="go_or_grow",
        kappa=[-100.0, 100.0],
        theta=0.5,
        r_b=0.0,
        r_d=0.0,
        capacity=100,
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    rest_counts = lgca.nodes[lgca.nonborder][..., lgca.velocitychannels:].sum(axis=(0, 2))
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

    mutation_matrix = lgca.interaction_params["mutation_matrix"]

    assert mutation_matrix.shape == (3, 3)
    assert mutation_matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.all(np.diag(mutation_matrix) > 0.99)


@pytest.mark.parametrize("geom,dims", [("hex", (4, 4)), ("moore", (4, 4, 4))])
def test_multispecies_nove_birth_runs_on_multidimensional_lattices(geom, dims):
    _, nove_cls = _load_ms_classes(geom)
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    center = tuple(dim // 2 for dim in dims)
    nodes[center + (0, nove_cls.velocitychannels)] = 30

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="birth",
        r_b=[1.0, 0.0],
        capacity=1000,
        mutation_matrix=np.array([[0.0, 1.0], [0.0, 1.0]]),
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    species_counts = lgca.nodes[lgca.nonborder].sum(axis=tuple(range(len(dims))) + (-1,))
    assert species_counts[0] == 30
    assert species_counts[1] > 0


@pytest.mark.parametrize("geom,dims", [("hex", (4, 4)), ("moore", (4, 4, 4))])
def test_multispecies_nove_birthdeath_runs_on_multidimensional_lattices(geom, dims):
    _, nove_cls = _load_ms_classes(geom)
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    center = tuple(dim // 2 for dim in dims)
    nodes[center + (0, 0)] = 5
    nodes[center + (1, nove_cls.velocitychannels)] = 7

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="birthdeath",
        r_b=[0.0, 0.0],
        r_d=1.0,
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    assert lgca.nodes[lgca.nonborder].sum() == 0


@pytest.mark.parametrize("geom,dims", [("hex", (4, 4)), ("moore", (4, 4, 4))])
def test_multispecies_nove_go_or_grow_runs_on_multidimensional_lattices(geom, dims):
    _, nove_cls = _load_ms_classes(geom)
    nodes = np.zeros(dims + (2, nove_cls.velocitychannels + 1), dtype=np.uint)
    center = tuple(dim // 2 for dim in dims)
    nodes[center + (0, 0)] = 40
    nodes[center + (1, 0)] = 40

    lgca = get_lgca(
        geometry=geom,
        n_species=2,
        ve=False,
        nodes=nodes,
        interaction="go_or_grow",
        kappa=[-100.0, 100.0],
        theta=0.5,
        r_b=0.0,
        r_d=0.0,
        capacity=100,
        propagation=False,
        seed=1,
    )

    lgca.timestep()

    rest_counts = lgca.nodes[lgca.nonborder][..., lgca.velocitychannels:].sum(axis=tuple(range(len(dims))) + (-1,))
    assert rest_counts[0] == 0
    assert rest_counts[1] == 40
