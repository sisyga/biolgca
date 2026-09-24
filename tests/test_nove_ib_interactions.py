"""Identity-based interactions without volume exclusion reproduce their stated rates.

Most tests use many isolated nodes (propagation switched off), so every node is
an independent sample of the local rule, and compare the measured rate with its
exact expectation within four binomial standard deviations.
"""

import numpy as np
import pytest

from lgca import get_lgca
from lgca.builtin_rules import tanh_switch


N_NODES = 400


def _isolated_nodes(interaction, cells_per_node, *, channel=-1, **params):
    """1D lattice of N_NODES nodes that each start with ``cells_per_node`` cells in ``channel``."""
    counts = np.zeros((N_NODES, 3), dtype=int)
    counts[:, channel] = cells_per_node
    return get_lgca(geometry="lin", ib=True, ve=False, nodes=counts, restchannels=1,
                    interaction=interaction, propagation=False, seed=7, **params)


def _node_cells(lgca):
    return [np.concatenate([np.asarray(c, dtype=int) for c in node]) for node in lgca.nodes[lgca.nonborder]]


def _population(lgca):
    return int(lgca.cell_density[lgca.nonborder].sum())


def _assert_binomial_mean(observed, trials, probability):
    expected = trials * probability
    sd = np.sqrt(trials * probability * (1 - probability))
    assert abs(observed - expected) < 4 * sd, (observed, expected, sd)


def test_tanh_switch_is_a_sigmoid_centred_on_theta():
    rho = np.linspace(0, 1, 11)
    values = tanh_switch(rho, kappa=5.0, theta=0.3)

    assert values[3] == pytest.approx(0.5)
    assert np.all(np.diff(values) > 0)
    assert np.all((values > 0) & (values < 1))
    assert tanh_switch(0.4, kappa=20.0, theta=0.3) > tanh_switch(0.4, kappa=5.0, theta=0.3)


def test_random_walk_moves_cells_but_keeps_every_identity():
    lgca = get_lgca(geometry="lin", ib=True, ve=False, dims=50, density=3, restchannels=1,
                    interaction="random_walk", seed=1)
    before = np.sort(np.concatenate(_node_cells(lgca)))
    positions_before = [set(cells) for cells in _node_cells(lgca)]

    lgca.timeevo(timesteps=10, showprogress=False)

    np.testing.assert_array_equal(np.sort(np.concatenate(_node_cells(lgca))), before)
    assert [set(cells) for cells in _node_cells(lgca)] != positions_before


def test_birth_is_logistic_in_the_local_density():
    lgca = _isolated_nodes("birth", 10, r_b=0.5, capacity=40, std=0.01, a_max=1.0)

    lgca.timestep()

    _assert_binomial_mean(_population(lgca) - 10 * N_NODES, 10 * N_NODES, 0.5 * (1 - 10 / 40))


def test_birth_stops_at_capacity():
    lgca = _isolated_nodes("birth", 10, r_b=1.0, capacity=10, std=0.01, a_max=1.0)

    lgca.timestep()

    assert _population(lgca) == 10 * N_NODES


def test_daughters_inherit_the_birth_rate_of_their_own_mother():
    lgca = _isolated_nodes("birth", 5, r_b=0.5, capacity=1000, std=0.01, a_max=1.0)
    node_rates = np.linspace(0.2, 0.9, N_NODES)
    for rate, cells in zip(node_rates, _node_cells(lgca)):
        for cell in cells:
            lgca.props["r_b"][cell] = rate
    first_daughter = lgca.maxlabel + 1

    lgca.timestep()

    rates = np.asarray(lgca.props["r_b"])
    daughters_seen = 0
    for rate, cells in zip(node_rates, _node_cells(lgca)):
        daughters = cells[cells >= first_daughter]
        daughters_seen += daughters.size
        np.testing.assert_allclose(rates[daughters], rate, atol=0.05)
    assert daughters_seen > N_NODES


def test_death_removes_the_expected_fraction_of_cells():
    lgca = _isolated_nodes("birthdeath", 10, r_b=0.0, r_d=0.3, capacity=40, std=0.01, a_max=1.0)

    lgca.timestep()

    _assert_binomial_mean(_population(lgca), 10 * N_NODES, 0.7)


def test_go_or_grow_switches_cells_to_rest_with_the_density_sigmoid():
    lgca = _isolated_nodes("go_or_grow", 10, channel=0, r_b=0.0, r_d=0.0, capacity=20,
                           kappa=5.0, theta=0.3, kappa_std=0.0, theta_std=0.0)

    lgca.timestep()

    resting = int(lgca.channel_pop[lgca.nonborder][:, -1].sum())
    _assert_binomial_mean(resting, 10 * N_NODES, tanh_switch(10 / 20, kappa=5.0, theta=0.3))


@pytest.mark.parametrize("driver,sign", [(True, 1), (False, -1)])
def test_cancer_dfe_mutations_shift_daughter_rates_by_their_mean_effect(driver, sign):
    lgca = _isolated_nodes("birthdeath_cancerdfe", 10, r_b=0.3, r_d=0.0, capacity=1000, a_max=1.0,
                           p_d=1.0 if driver else 0.0, p_p=0.0 if driver else 1.0, s_d=0.05, s_p=0.05)
    first_daughter = lgca.maxlabel + 1

    lgca.timestep()

    shifts = np.asarray(lgca.props["r_b"][first_daughter:]) - 0.3
    assert shifts.size > 500
    assert np.all(sign * shifts >= 0)
    assert sign * shifts.mean() == pytest.approx(0.05, abs=0.01)


def _single_resting_cell_glioblastoma_lgca(**kw):
    nodes = np.zeros((3, 3), dtype=int)
    nodes[1, -1] = 1
    params = {
        "ve": False,
        "ib": True,
        "geometry": "lin",
        "nodes": nodes,
        "restchannels": 1,
        "interaction": "go_or_grow_glioblastoma",
        "capacity": 100,
        "r_b": 1.0,
        "r_d": 0.0,
        "r_m": 1.0,
        "fitness_increase": 1.5,
        "kappa": 2.0,
        "kappa_std": 0.0,
        "theta": -10.0,
        "seed": 0,
    }
    params.update(kw)
    return get_lgca(**params)


def test_glioblastoma_initializes_one_family_per_founder_cell():
    lgca = _single_resting_cell_glioblastoma_lgca(r_b=0.3, kappa=4.0)

    founder = int(lgca.maxlabel)  # the only cell
    assert lgca.props["family"][founder] == 1
    assert lgca.props["r_b"][founder] == pytest.approx(0.3)
    assert lgca.props["kappa"][founder] == pytest.approx(4.0)


def test_glioblastoma_mutation_creates_a_fitter_family_with_inherited_traits():
    lgca = _single_resting_cell_glioblastoma_lgca()

    lgca.interaction(lgca)
    lgca.update_dynamic_fields()

    daughter = int(lgca.maxlabel)
    assert lgca.cell_density[lgca.nonborder].sum() == 2
    assert lgca.maxfamily == 2
    assert lgca.props["family"][daughter] == 2
    assert lgca.family_props["ancestor"][2] == 1
    assert lgca.props["r_b"][daughter] == pytest.approx(1.5)
    assert lgca.props["kappa"][daughter] == pytest.approx(2.0)


def test_glioblastoma_family_populations_are_recorded_as_families_appear():
    lgca = _single_resting_cell_glioblastoma_lgca()

    lgca.timeevo(timesteps=1, recordfampop=True, showprogress=False)

    assert lgca.fam_pop_t.shape == (2, lgca.maxfamily + 1)
    assert lgca.fam_pop_t[0].sum() == 1
    assert lgca.fam_pop_t[1].sum() == 2
