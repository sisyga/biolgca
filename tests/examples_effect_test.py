"""Each curated example shows the effect its docstring promises at its own settings."""

import numpy as np

from lgca.examples import run_example


def _polarization(lgca, nodes):
    flux = lgca.calc_flux(nodes.astype(float)).sum(axis=(0, 1))
    return float(np.linalg.norm(flux) / nodes.sum())


def _nematic_order(lgca, nodes):
    velocities = nodes[..., :lgca.velocitychannels].astype(float)
    angles = np.arctan2(lgca.c[1], lgca.c[0])
    return float(abs((velocities * np.exp(2j * angles)).sum()) / velocities.sum())


def test_alignment_orders_random_headings_into_streams():
    lgca = run_example("alignment").lgca

    assert _polarization(lgca, lgca.nodes_t[0]) < 0.1
    assert _polarization(lgca, lgca.nodes_t[-1]) > 0.3


def test_nematic_alignment_orders_axes_but_not_directions():
    lgca = run_example("nematic_interaction").lgca
    final = lgca.nodes_t[-1]

    assert _nematic_order(lgca, final) > 0.2
    assert _polarization(lgca, final) < 0.1


def test_aggregation_concentrates_cells():
    lgca = run_example("aggregation").lgca

    assert lgca.dens_t[-1].var() > 2 * lgca.dens_t[0].var()


def test_chemotaxis_gathers_cells_at_high_signal():
    density = run_example("chemotaxis").lgca.dens_t[-1]

    assert density[40:].sum() / density.sum() > 0.4


def test_go_or_grow_shows_an_allee_effect_that_negative_kappa_removes():
    from lgca.examples.go_or_grow import build_spec
    from lgca.model import run_model

    allee = run_example("go_or_grow").lgca.n_t
    invasion = run_model(build_spec(kappa=-4.0), showprogress=False).lgca.n_t

    assert allee[0] == invasion[0] == 12
    assert allee[-1] < allee[0]
    assert invasion[-1] > 1000


def test_identity_tumor_grows_and_its_trait_varies():
    lgca = run_example("identity_tumor_growth", steps=60).lgca

    assert lgca.n_t[-1] > 5 * lgca.n_t[0]
    assert np.std(lgca.props["kappa"][1:]) > 0


def test_faster_dividing_species_dominates():
    lgca = run_example("multispecies_birth_death", steps=40).lgca
    per_species = lgca.nodes[lgca.nonborder].sum(axis=(0, 1, 3))

    assert per_species[0] > 2 * per_species[1]


def test_birth_rate_evolves_upwards():
    lgca = run_example("evolutionary_go_and_grow").lgca
    birth_rates = np.asarray(lgca.props["r_b"])

    def mean_birth_rate(nodes):
        return birth_rates[nodes[nodes > 0]].mean()

    assert mean_birth_rate(lgca.nodes_t[-1]) > mean_birth_rate(lgca.nodes_t[0]) + 0.05
