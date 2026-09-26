"""lgca.zoo: every entry builds, its parameters are where its card says, and it reproduces its paper."""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from lgca import zoo
from lgca.model import build_model, run_model
from lgca.study import _get, _tokens, vary
from lgca.zoo import allee_effect as allee

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source" / "zoo"


@pytest.mark.parametrize("name", zoo.ENTRIES)
def test_every_entry_has_a_card_parameters_and_a_notebook(name):
    module = zoo.load(name)
    card = module.CARD
    assert card.name == name and card.question.endswith("?")
    assert card.fidelity.startswith(("same rules", "simplified", "new model"))
    assert card.citation.startswith(card.authors) and (card.doi is None or card.doi in card.citation)
    spec = module.build_spec()
    for key, parameter in module.PARAMETERS.items():
        if parameter.path is not None:  # the card's value is the default of build_spec
            assert _get(spec, _tokens(parameter.path)) == parameter.value, key
    assert (DOCS / f"{name}.ipynb").exists()
    assert name in (DOCS / "index.rst").read_text(encoding="utf-8")
    assert zoo.parameter_table(name).shape[0] == len(module.PARAMETERS)


def test_the_catalogue_lists_the_entries_in_order():
    assert [card.name for card in zoo.catalogue()] == list(zoo.ENTRIES)
    with pytest.raises(KeyError, match="has no entry"):
        zoo.load("unicorns")


@pytest.mark.parametrize("name", zoo.ENTRIES)
def test_full_builds_the_papers_model(name):
    spec = zoo.load(name).build_spec(full=True)
    assert build_model(spec).lgca is not None


# ------------------------------------------------------------------ Allee effect

def test_allee_parameters_and_size():
    spec = allee.build_spec(full=True)
    assert spec.space.dims == (100, 100) and spec.time.steps == 5000
    assert allee.build_spec().space.dims == (50, 50)
    lattice = build_model(allee.build_spec(density=0.25)).lgca
    assert lattice.K == allee.K == 8 and lattice.restchannels == 4
    # every channel occupied with probability ϱ₀: 50 × 50 × 8 × 0.25 cells, within 4 standard errors
    cells = lattice.cell_density[lattice.nonborder].sum()
    assert abs(cells - 5000) < 4 * np.sqrt(20000 * 0.25 * 0.75)
    with pytest.raises(ValueError, match="between 0 and 1"):
        allee.build_spec(density=2.0)


def test_allee_thresholds_of_the_mean_field():
    # r_s(ϱ*) r_b = r_d: ϱ* = θ + artanh(2 r_d / r_b - 1) / κ
    exact = 0.75 + np.arctanh(2 * 0.01 / 0.2 - 1) / 4.4
    assert brentq(allee.per_capita_growth, 0.05, 0.95) == pytest.approx(exact)
    assert brentq(allee.per_capita_growth_nodes, 0.05, 0.95) == pytest.approx(0.2403, abs=1e-3)
    # a lone cell in an empty lattice, and the repulsive switch, which always grows
    assert allee.per_capita_growth_nodes(0.0) == pytest.approx(allee.switch(1 / 8) * 0.2 - 0.01)
    assert np.all(allee.per_capita_growth(np.linspace(0, 1, 11), kappa=-4.4) > 0)


def test_allee_node_average_matches_sampled_nodes():
    rng = np.random.default_rng(3)
    rho = 0.3
    n = rng.binomial(8, rho, size=400_000)  # cells of nodes with every channel occupied with probability rho
    sampled = np.sum(n * allee.per_capita_growth(n / 8)) / n.sum()
    assert allee.per_capita_growth_nodes(rho) == pytest.approx(sampled, abs=3e-4)


def _grown(density, kappa=4.4, seeds=range(4), steps=600):
    runs = [run_model(allee.build_spec(density=density, kappa=kappa, seed=seed, steps=steps, record_every=steps),
                      showprogress=False) for seed in seeds]
    return [allee.final_density(run) > density for run in runs]


def test_below_the_threshold_populations_decline_above_they_grow():
    """The simulated threshold lies near 0.25 (Fig 3), far below the mean-field 0.42."""
    assert not any(_grown(0.18))
    assert all(_grown(0.34))


def test_with_repulsion_small_populations_grow():
    assert all(_grown(0.1, kappa=-4.4))


def test_allee_parameters_can_be_varied_by_their_paths():
    spec = vary(allee.build_spec(), {allee.PARAMETERS["kappa"].path: -1.0, allee.PARAMETERS["r_b"].path: 0.3})
    assert spec.dynamics.operators[1]["parameters"]["kappa"] == -1.0
    assert spec.dynamics.operators[0]["parameters"]["r_b"] == 0.3
