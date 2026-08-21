from dataclasses import replace

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model


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


@pytest.mark.parametrize("path", ["../outside.npz", "C:/outside.npz"])
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
