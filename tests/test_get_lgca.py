import pytest
import numpy as np

from lgca import get_lgca
from lgca.lgca_1d import LGCA_1D, IBLGCA_1D, NoVE_LGCA_1D, NoVE_IBLGCA_1D
from lgca.lgca_square import LGCA_Square
from lgca.square_ext import IBLGCA_Square, NoVE_LGCA_Square, NoVE_IBLGCA_Square
from lgca.lgca_hex import LGCA_Hex, IBLGCA_Hex, NoVE_LGCA_Hex, NoVE_IBLGCA_Hex
from lgca.lgca_cubic import LGCA_Cubic
from lgca.cubic_ext import (
    IBLGCA_Cubic,
    NoVE_LGCA_Cubic,
    NoVE_IBLGCA_Cubic,
)
from lgca.lgca_3dmoore import (
    LGCA_3dMoore,
    IBLGCA_Moore,
    NoVE_LGCA_Moore,
    NoVE_IBLGCA_Moore,
)

EXPECTED = {
    'lin': {
        (False, True): LGCA_1D,
        (True, True): IBLGCA_1D,
        (False, False): NoVE_LGCA_1D,
        (True, False): NoVE_IBLGCA_1D,
    },
    'square': {
        (False, True): LGCA_Square,
        (True, True): IBLGCA_Square,
        (False, False): NoVE_LGCA_Square,
        (True, False): NoVE_IBLGCA_Square,
    },
    'hex': {
        (False, True): LGCA_Hex,
        (True, True): IBLGCA_Hex,
        (False, False): NoVE_LGCA_Hex,
        (True, False): NoVE_IBLGCA_Hex,
    },
    'cubic': {
        (False, True): LGCA_Cubic,
        (True, True): IBLGCA_Cubic,
        (False, False): NoVE_LGCA_Cubic,
        (True, False): NoVE_IBLGCA_Cubic,
    },
    'moore': {
        (False, True): LGCA_3dMoore,
        (True, True): IBLGCA_Moore,
        (False, False): NoVE_LGCA_Moore,
        (True, False): NoVE_IBLGCA_Moore,
    },
}

geometries = ['lin', 'square', 'hex', 'cubic', 'moore']

PARAMS = [
    (g, ib, ve)
    for g in geometries
    for ib, ve in [
        (False, True),  # classical LGCA
        (True, True),   # identity based
        (False, False), # NoVE classical
        (True, False),  # NoVE identity based
    ]
]

@pytest.mark.parametrize("geom, ib, ve", PARAMS)
def test_get_lgca_returns_correct_subclass(geom, ib, ve):
    lgca = get_lgca(
        geometry=geom,
        ib=ib,
        ve=ve,
        density=0,
        dims=2,
        restchannels=1,
        interaction="only_propagation",
    )
    assert isinstance(lgca, EXPECTED[geom][(ib, ve)])


def test_warning_on_mismatched_dims():
    nodes = np.zeros((3, 4))
    with pytest.warns(UserWarning):
        get_lgca(
            geometry='lin',
            nodes=nodes,
            dims=10,
            interaction='only_propagation',
        )


def test_warning_on_mismatched_restchannels():
    nodes = np.zeros((3, 4))
    with pytest.warns(UserWarning):
        get_lgca(
            geometry='lin',
            nodes=nodes,
            restchannels=1,
            interaction='only_propagation',
        )


def test_warning_when_nodes_override_density():
    nodes = np.zeros((3, 2), dtype=bool)
    with pytest.warns(UserWarning, match="density"):
        get_lgca(
            geometry='lin',
            nodes=nodes,
            density=0.9,
            interaction='only_propagation',
        )


def test_warning_on_nonboolean_nodes():
    nodes = np.array([[2, 0], [3, 1]])
    with pytest.warns(UserWarning):
        lgca = get_lgca(
            geometry='lin',
            ib=False,
            ve=True,
            nodes=nodes,
            interaction='only_propagation',
        )
    assert set(np.unique(lgca.nodes[lgca.nonborder])) <= {0, 1}


@pytest.mark.parametrize(
    "family",
    [
        {},
        {"ib": True},
        {"ve": False},
        {"ve": False, "ib": True},
        {"n_species": 2},
        {"n_species": 2, "ve": False},
    ],
    ids=["classical", "ib", "nove", "nove_ib", "multispecies", "multispecies_nove"],
)
def test_user_defined_interaction_function_runs_every_step_with_its_parameters(family):
    calls = []

    def my_rule(lgca):
        calls.append(lgca.interaction_params["my_rate"])

    lgca = get_lgca(geometry="lin", dims=8, density=1, seed=1, interaction=my_rule, my_rate=0.3, **family)
    lgca.timeevo(timesteps=3, record=False, showprogress=False)

    assert calls == [0.3, 0.3, 0.3]


def test_user_defined_interaction_function_changes_the_state():
    def remove_all_cells(lgca):
        lgca.nodes[...] = False

    lgca = get_lgca(geometry="square", dims=(5, 5), density=1, seed=1, interaction=remove_all_cells)
    lgca.timeevo(timesteps=1, record=False, showprogress=False)

    assert lgca.nodes.sum() == 0


@pytest.mark.parametrize("family", [{}, {"ib": True}, {"ve": False}, {"ve": False, "ib": True}])
def test_factory_and_simulation_print_nothing_but_log_chosen_defaults(family, capsys, caplog):
    with caplog.at_level("INFO", logger="lgca"):
        lgca = get_lgca(geometry="lin", dims=8, interaction="go_or_grow", restchannels=2, seed=1, **family)
        lgca.timeevo(timesteps=2, record=True, showprogress=False)

    assert capsys.readouterr().out == ""
    assert any("r_b" in record.getMessage() for record in caplog.records)


def test_too_few_rest_channels_for_go_or_grow_warn_at_the_caller():
    with pytest.warns(UserWarning, match="die out") as record:
        get_lgca(geometry="lin", dims=8, interaction="go_or_grow", restchannels=1, seed=1)

    assert record[0].filename == __file__


@pytest.mark.parametrize("geometry", ["lin", "square", "hex", "cubic", "moore"])
def test_default_nove_model_builds_and_runs(geometry):
    lgca = get_lgca(geometry=geometry, ve=False, seed=1)
    lgca.timeevo(timesteps=2, record=False, showprogress=False)

    assert lgca.restchannels == 0


def test_deprecation_warnings_point_at_the_callers_line():
    from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
    from lgca.pipeline import InteractionPipelineSpec

    spec = ModelSpec(space=SpaceSpec(geometry="lin", dims=5),
                     state=StateSpec(density=1, volume_exclusion=False, parameters={"capacity": 4}),
                     time=TimeSpec(steps=1, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "nove.random_walk"}]))

    with pytest.warns(DeprecationWarning, match="state.capacity") as record:
        run_model(spec, showprogress=False)

    assert record[0].filename == __file__
