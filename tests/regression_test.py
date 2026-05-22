import random

import numpy as np
import pytest

from lgca import get_lgca


def test_geometry_modules_do_not_use_base_wildcard_imports():
    module_paths = [
        "lgca/lgca_1d.py",
        "lgca/lgca_square.py",
        "lgca/lgca_hex.py",
        "lgca/lgca_cubic.py",
    ]
    for module_path in module_paths:
        with open(module_path, encoding="utf-8") as module_file:
            source = module_file.read()
        assert "from lgca.base import *" not in source


@pytest.mark.parametrize("alias", ["1D", "1d", "lin", "linear"])
def test_get_lgca_accepts_documented_1d_aliases(alias):
    lgca = get_lgca(
        geometry=alias,
        dims=4,
        density=0,
        interaction="only_propagation",
    )

    assert lgca.geometry == "lin"
    assert lgca.dims == (4,)


@pytest.mark.parametrize("ib, ve", [(False, True), (True, True), (False, False), (True, False)])
def test_unknown_interaction_raises_value_error(ib, ve):
    with pytest.raises(ValueError, match="Unknown interaction"):
        get_lgca(
            geometry="square",
            ib=ib,
            ve=ve,
            restchannels=0,
            density=0,
            interaction="not_an_interaction",
        )


def test_unknown_boundary_condition_raises_value_error():
    with pytest.raises(ValueError, match="Unknown boundary condition"):
        get_lgca(bc="not_a_boundary", interaction="only_propagation")


def test_unknown_constructor_kwarg_raises_with_suggestion():
    with pytest.raises(TypeError, match="densty.*density"):
        get_lgca(densty=0.5)


def test_known_constructor_and_interaction_kwargs_are_accepted():
    lgca = get_lgca(
        geometry="square",
        dims=3,
        density=0,
        restchannels=1,
        bc="periodic",
        seed=1,
        propagation=False,
        interaction="go_or_grow",
        r_b=0.2,
        r_d=0.01,
        kappa=5.0,
        theta=0.75,
    )

    assert lgca.dims == (3, 3)


def test_nove_1d_set_r_int_preserves_state():
    lgca = get_lgca(
        geometry="lin",
        ve=False,
        restchannels=0,
        dims=5,
        density=1,
        interaction="only_propagation",
        seed=1,
    )
    before = lgca.nodes[lgca.nonborder].copy()

    lgca.set_r_int(2)

    assert lgca.r_int == 2
    assert lgca.nodes.shape == (9, lgca.K)
    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder], before)


def test_nove_ib_moore_propagates_all_velocity_channels():
    lgca = get_lgca(
        geometry="moore",
        ib=True,
        ve=False,
        dims=3,
        density=0,
        interaction="only_propagation",
    )
    lgca.nodes[...] = np.empty(lgca.nodes.shape, dtype=object)
    for idx in np.ndindex(lgca.nodes.shape):
        lgca.nodes[idx] = []
    center = (lgca.r_int + 1, lgca.r_int + 1, lgca.r_int + 1)
    channel = 6
    dx, dy, dz = lgca._vels[channel]
    target = (center[0] + dx, center[1] + dy, center[2] + dz)
    lgca.nodes[center + (channel,)] = [1]

    lgca.propagation()

    assert lgca.nodes[target + (channel,)] == [1]
    assert lgca.nodes[center + (channel,)] == []


def test_contact_guidance_director_initializes_guiding_tensor():
    director = np.zeros((5, 5, 2))
    director[..., 0] = 1
    lgca = get_lgca(
        geometry="square",
        dims=3,
        density=0,
        interaction="contact_guidance",
        director=director,
    )

    assert hasattr(lgca, "guiding_tensor")
    lgca.timestep()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"geometry": "lin", "dims": 5, "density": -1},
        {"geometry": "lin", "dims": 5, "density": 99},
        {"geometry": "lin", "ve": False, "restchannels": 0, "density": -4},
    ],
)
def test_invalid_density_raises_value_error(kwargs):
    with pytest.raises(ValueError, match="density"):
        get_lgca(**kwargs)


def test_constructor_r_int_resizes_border():
    lgca = get_lgca(
        geometry="lin",
        dims=4,
        density=0,
        interaction="only_propagation",
        r_int=3,
    )

    assert lgca.r_int == 3
    assert lgca.nodes.shape == (10, lgca.K)


def test_constructor_rejects_invalid_r_int():
    with pytest.raises(ValueError, match="r_int"):
        get_lgca(geometry="lin", density=0, interaction="only_propagation", r_int=0)


@pytest.mark.parametrize(
    "param,value",
    [("r_b", -0.1), ("r_b", 1.1), ("r_d", -0.1), ("r_d", 1.1)],
)
def test_invalid_birthdeath_probabilities_raise(param, value):
    with pytest.raises(ValueError, match=param):
        get_lgca(
            geometry="lin",
            density=0,
            interaction="birthdeath",
            **{param: value},
        )


@pytest.mark.parametrize(
    "geometry,dims",
    [("lin", 0), ("lin", -1), ("square", (2, -1)), ("cubic", (2, 2, 0))],
)
def test_dims_must_be_positive(geometry, dims):
    with pytest.raises(ValueError, match="dims"):
        get_lgca(geometry=geometry, dims=dims, density=0, interaction="only_propagation")


def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        get_lgca(
            geometry="lin",
            ve=False,
            restchannels=0,
            capacity=0,
            density=0,
            interaction="only_propagation",
        )


def test_unsupported_geometry_interaction_raises_value_error():
    with pytest.raises(ValueError, match="contact_guidance.*not supported"):
        get_lgca(geometry="lin", density=0, interaction="contact_guidance")


def test_contact_guidance_rejects_bad_director_shape():
    with pytest.raises(ValueError, match="director.*shape"):
        get_lgca(
            geometry="square",
            dims=(2, 2),
            density=0,
            interaction="contact_guidance",
            director=np.zeros((2, 2)),
        )


def test_chemotaxis_rejects_bad_gradient_shape():
    with pytest.raises(ValueError, match="gradient.*shape"):
        get_lgca(
            geometry="square",
            dims=(2, 2),
            density=0,
            interaction="chemotaxis",
            gradient=np.zeros((2, 2)),
        )


def test_nove_ib_capacity_sets_lattice_and_interaction_capacity():
    lgca = get_lgca(
        geometry="lin",
        ib=True,
        ve=False,
        density=0,
        capacity=20,
        interaction="birth",
        seed=1,
    )

    assert lgca.capacity == 20
    assert lgca.interaction_params["capacity"] == 20


def _nove_ib_nodes(length=4):
    nodes = np.empty((length, 3), dtype=object)
    for idx in np.ndindex(nodes.shape):
        nodes[idx] = []
    for label in range(length):
        nodes[label, -1] = [label]
    return nodes


def _nove_ib_snapshot(lgca):
    return (
        tuple(int(value) for value in lgca.cell_density.ravel()),
        tuple(round(float(value), 12) for value in lgca.props.get("r_b", [])),
    )


def test_seeded_nove_ib_birth_is_independent_of_numpy_global_rng():
    def run_with_global_seed(global_seed):
        np.random.seed(global_seed)
        lgca = get_lgca(
            geometry="lin",
            ib=True,
            ve=False,
            nodes=_nove_ib_nodes(),
            interaction="birth",
            seed=42,
            r_b=1.0,
            capacity=1000,
            std=0.05,
            a_max=1.0,
        )
        lgca.timestep()
        return _nove_ib_snapshot(lgca)

    assert run_with_global_seed(1) == run_with_global_seed(999)


def test_seeded_nove_ib_cancerdfe_is_independent_of_numpy_global_rng():
    def run_with_global_seed(global_seed):
        np.random.seed(global_seed)
        lgca = get_lgca(
            geometry="lin",
            ib=True,
            ve=False,
            nodes=_nove_ib_nodes(),
            interaction="birthdeath_cancerdfe",
            seed=7,
            r_b=1.0,
            r_d=0.0,
            p_p=1.0,
            p_d=1.0,
            s_p=0.01,
            s_d=0.02,
            capacity=1000,
            a_max=1.0,
        )
        lgca.timestep()
        return _nove_ib_snapshot(lgca)

    assert run_with_global_seed(1) == run_with_global_seed(999)


def test_seeded_ib_birthdeath_discrete_is_independent_of_python_global_rng():
    def run_with_global_seed(global_seed):
        random.seed(global_seed)
        nodes = np.zeros((10, 3), dtype=np.uint)
        nodes[:, 0] = np.arange(1, 11)
        lgca = get_lgca(
            geometry="lin",
            ib=True,
            nodes=nodes,
            interaction="birthdeath_discrete",
            seed=11,
            r_b=1.0,
            r_d=0.0,
            drb=0.05,
            pmut=1.0,
            a_max=1.0,
        )
        lgca.timestep()
        return tuple(round(float(value), 12) for value in lgca.props["r_b"])

    assert run_with_global_seed(1) == run_with_global_seed(999)


def test_nove_ib_recorded_nodes_do_not_alias_live_lists():
    lgca = get_lgca(
        geometry="lin",
        ib=True,
        ve=False,
        nodes=_nove_ib_nodes(length=1),
        interaction="only_propagation",
    )

    lgca.timeevo(timesteps=0, record=True, recorddens=False, showprogress=False)
    recorded = lgca.nodes_t[0, 0, -1]
    live = lgca.nodes[lgca.nonborder][0, -1]

    assert recorded is not live
    live.append(99)
    assert recorded == [0]
