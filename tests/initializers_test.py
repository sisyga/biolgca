from dataclasses import replace

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model


@pytest.mark.parametrize("n_species", [1, 2])
def test_nove_initialization_memory_scales_with_state_not_capacity(n_species):
    from lgca import get_lgca

    model = get_lgca(geometry="square", dims=(4, 4), ve=False, n_species=n_species,
                     restchannels=1, density=0, capacity=5, interaction="only_propagation")
    draws = []

    class TrackingRng:
        def poisson(self, lam, size):
            draws.append((lam, tuple(size)))
            return np.zeros(size, dtype=np.int64)

    model.rng = TrackingRng()
    model.capacity = 1_000_000
    model.random_reset(2)
    assert [shape for _, shape in draws] == [model.nodes.shape, model.nodes.shape[:-1]]
    assert draws[0][0] == pytest.approx(2 / n_species / model.capacity)
    assert draws[1][0] == pytest.approx(2 / n_species * (model.capacity - model.K) / model.capacity)
    # Independent Poisson means sum to the requested total site population.
    assert n_species * (model.K * draws[0][0] + draws[1][0]) == pytest.approx(2)


@pytest.mark.parametrize("value", [-1, .5, np.nan, np.inf, float(2**64)])
@pytest.mark.parametrize("n_species", [1, 2])
def test_invalid_count_inputs_rejected_across_entry_points(tmp_path, value, n_species):
    from lgca import get_lgca

    shape = (1, 2) if n_species == 1 else (1, n_species, 2)
    nodes = np.zeros(shape)
    nodes.flat[0] = value
    with pytest.raises(ValueError, match="nodes"):
        get_lgca(geometry="lin", ve=False, nodes=nodes, n_species=n_species,
                 interaction="only_propagation")
    state = StateSpec(volume_exclusion=False, nodes=nodes, n_species=n_species)
    with pytest.raises(ValueError, match="nodes"):
        build_model(ModelSpec(space=SpaceSpec(geometry="lin"), state=state))
    np.savez(tmp_path / "invalid.npz", nodes=nodes)
    state = replace(state, nodes=None, initializer={"name": "from_npz",
                    "parameters": {"path": "invalid.npz"}})
    with pytest.raises(ValueError, match="nodes"):
        build_model(ModelSpec(space=SpaceSpec(geometry="lin", dims=1), state=state),
                    resource_base=tmp_path)


def _initializer_spec(initializer, *, dims=(6, 5), seed=41):
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=dims, boundary="periodic"),
        state=StateSpec(restchannels=1, initializer=initializer),
        time=TimeSpec(steps=0, seed=seed),
    )


@pytest.mark.parametrize(
    "placement,expected_slices",
    [
        ("center", (slice(2, 4), slice(1, 4))),
        ("left", (slice(0, 2), slice(1, 4))),
        ("corner", (slice(0, 2), slice(0, 3))),
    ],
)
def test_region_initializer_populates_only_requested_region(placement, expected_slices):
    spec = _initializer_spec(
        {
            "name": "region",
            "parameters": {
                "placement": placement,
                "extent": [2, 3],
                "density": 5,
            },
        }
    )

    lgca = build_model(spec).lgca
    density = lgca.cell_density[lgca.nonborder]
    expected = np.zeros((6, 5), dtype=int)
    expected[expected_slices] = 5

    np.testing.assert_array_equal(density, expected)


@pytest.mark.parametrize("capacity", [2, 5, 10])
@pytest.mark.parametrize("family", ["ordinary", "identity", "multispecies"])
@pytest.mark.parametrize("region", [False, True])
def test_nove_density_is_independent_of_capacity(capacity, family, region):
    initializer = None
    if region:
        initializer = {"name": "region", "parameters": {
            "placement": "corner", "extent": [40, 80], "density": 10,
        }}
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(80, 80)),
        state=StateSpec(volume_exclusion=False, identity_based=family == "identity",
                        n_species=2 if family == "multispecies" else 1,
                        restchannels=1, capacity=capacity, density=None if region else 10,
                        initializer=initializer),
        time=TimeSpec(steps=0, seed=107),
    )
    lgca = build_model(spec).lgca
    density = lgca.cell_density[lgca.nonborder]
    if region:
        assert not density[40:].any()
        density = density[:40]
    # A sum of independent Poisson channels is Poisson(10); allow six SEs.
    assert density.mean() == pytest.approx(10, abs=6 * np.sqrt(10 / density.size))


def test_region_initializer_uses_model_rng_deterministically():
    initializer = {
        "name": "region",
        "parameters": {"placement": "center", "extent": 4, "density": 2.5},
    }

    first = build_model(_initializer_spec(initializer, seed=42)).lgca
    second = build_model(_initializer_spec(initializer, seed=42)).lgca

    np.testing.assert_array_equal(first.nodes, second.nodes)
    assert first.cell_density[first.nonborder].sum() > 0


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"placement": "middle", "extent": 2, "density": 1}, "placement"),
        ({"placement": "center", "extent": [2], "density": 1}, "extent"),
        ({"placement": "center", "extent": [7, 2], "density": 1}, "extent"),
    ],
)
def test_region_initializer_rejects_invalid_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        build_model(_initializer_spec({"name": "region", "parameters": parameters}))


def test_from_npz_initializer_loads_contained_relative_state(tmp_path):
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    nodes = np.zeros((3, 4, 5), dtype=bool)
    nodes[1, 2, 3] = True
    np.savez(state_dir / "initial.npz", nodes=nodes)
    spec = _initializer_spec(
        {"name": "from_npz", "parameters": {"path": "states/initial.npz"}},
        dims=(3, 4),
    )

    lgca = build_model(spec, resource_base=tmp_path).lgca

    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder], nodes)


@pytest.mark.parametrize(
    "path",
    ["../outside.npz", r"..\outside.npz", "C:/outside.npz", "C:outside.npz"],
)
def test_from_npz_initializer_rejects_unsafe_paths_by_default(tmp_path, path):
    spec = _initializer_spec(
        {"name": "from_npz", "parameters": {"path": path}}, dims=(3, 4)
    )

    with pytest.raises(ValueError, match="trusted_paths"):
        build_model(spec, resource_base=tmp_path)


def test_from_npz_initializer_rejects_incompatible_shape(tmp_path):
    np.savez(tmp_path / "initial.npz", nodes=np.zeros((2, 2, 5), dtype=bool))
    spec = _initializer_spec(
        {"name": "from_npz", "parameters": {"path": "initial.npz"}},
        dims=(3, 4),
    )

    with pytest.raises(ValueError, match=r"shape.*\(3, 4, 5\)"):
        build_model(spec, resource_base=tmp_path)


def test_from_npz_rejects_identity_labels_without_particle_properties(tmp_path):
    np.savez(tmp_path / "initial.npz", nodes=np.zeros((3, 4, 5), dtype=np.uint))
    spec = replace(
        _initializer_spec(
            {"name": "from_npz", "parameters": {"path": "initial.npz"}},
            dims=(3, 4),
        ),
        state=StateSpec(
            restchannels=1,
            identity_based=True,
            initializer={"name": "from_npz", "parameters": {"path": "initial.npz"}},
        ),
    )

    with pytest.raises(ValueError, match="identity-based.*properties"):
        build_model(spec, resource_base=tmp_path)
