"""lgca.zoo: every entry builds, its parameters are where its card says, and it reproduces its paper."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from lgca import zoo
from lgca.lattice_state import LatticeState
from lgca.model import build_model, run_model
from lgca.pipeline import InteractionPipelineSpec
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


# ------------------------------------------------------------------ phenotypic plasticity

from lgca.zoo import phenotypic_plasticity as plasticity

SMALL = {"hex": {"size": 60, "steps": 120}, "lin": {"geometry": "lin", "size": 401, "steps": 400}}


def test_plasticity_starts_with_capacity_cells_of_uniform_kappa():
    for geometry, capacity, channels in (("hex", 50, 7), ("lin", 100, 3)):
        spec = plasticity.build_spec(geometry=geometry)
        assert spec.state.capacity == capacity and spec.state.nodes.shape[-1] == channels
        assert spec.state.nodes.sum() == capacity and spec.state.nodes[..., -1].sum() == capacity  # resting
        kappa = spec.state.traits["kappa"]
        assert len(kappa) == capacity and -4 <= kappa.min() and kappa.max() <= 4
    assert plasticity.build_spec(full=True).space.dims == (250, 250)
    assert plasticity.build_spec(full=True).time.steps == 300
    with pytest.raises(ValueError, match="'hex'"):
        plasticity.build_spec(geometry="square")
    with pytest.raises(KeyError, match="regimes 1, 2 and 3"):
        plasticity.regime(4)


@pytest.mark.parametrize("geometry", ["hex", "lin"])
@pytest.mark.parametrize("seed", [3, 4])
def test_the_three_regimes_of_the_paper(geometry, seed):
    summary = {number: plasticity.core_and_rim(run_model(plasticity.regime(number, seed=seed, **SMALL[geometry]),
                                                         showprogress=False).lgca)
               for number in (1, 2, 3)}
    # 1: no death; independent cells (κ ≈ 0) and the most migration
    assert -1.5 < summary[1]["all"] < 0.5
    assert summary[1]["migrating"] > 0.5 and summary[1]["migrating"] > summary[2]["migrating"]
    # 2: attractive core (κ > 0), repulsive rim (κ < 0)
    assert summary[2]["core"] > 0.5 and summary[2]["rim"] < -0.5
    # 3: repulsive throughout, weaker at the rim
    assert summary[3]["all"] < -0.5 and summary[3]["core"] < summary[3]["rim"] < 0


def test_node_maps_and_profiles_count_the_cells():
    run = run_model(plasticity.regime(2, size=40, steps=60), showprogress=False)
    maps = plasticity.node_maps(run.lgca)
    np.testing.assert_array_equal(maps["cells"], run.lgca.cell_density[run.lgca.nonborder])
    assert np.isnan(maps["kappa"][maps["cells"] == 0]).all()
    profile = plasticity.radial_profiles(run.lgca, bins=6)
    assert profile["r"].shape == (6,) and np.all(np.diff(profile["r"]) > 0)
    np.testing.assert_allclose(profile["cells"], profile["migrating"] + profile["resting"])
    record = plasticity.kymographs(plasticity.regime(2, geometry="lin", size=101, steps=30), every=10)
    assert record["cells"].shape == (4, 101) and list(record["steps"]) == [0, 10, 20, 30]
    assert record["cells"][0].sum() == 100


# ------------------------------------------------------------------ clones (entries 4 and 5)

from lgca.zoo import _clones
from lgca.zoo import clonal_go_or_grow as clonal


class _Tree:
    """A stand-in model with a family tree and cells, for the clone measures."""

    def __init__(self, ancestor, families, nodes, dims):
        self.family_props = {"ancestor": ancestor}
        self.maxfamily = len(ancestor) - 1
        self.props = {"family": families}
        self.dims = dims
        self._nodes = nodes


def test_clonal_indices_count_drivers_and_diversity(monkeypatch):
    # families: 1 the initial clone, 2 and 3 its mutants, 4 a mutant of 2
    tree = _Tree([0, 0, 1, 1, 2], None, None, (3,))
    np.testing.assert_array_equal(_clones.family_depth(tree), [0, 1, 2, 2, 3])
    families = np.array([1, 1, 2, 4, 4, 4])
    monkeypatch.setattr(_clones, "cell_families", lambda lgca: families)
    indices = _clones.clonal_indices(tree)
    p = np.array([2, 1, 3]) / 6
    assert indices["n"] == pytest.approx((2 * 1 + 1 * 2 + 3 * 3) / 6)
    assert indices["D"] == pytest.approx(1 / np.sum(p ** 2)) and indices["clones"] == 3
    bounds = _clones.diversity_bounds([1.0, 1.5, 2.5])
    np.testing.assert_allclose(bounds["upper"], [1.0, 4.0, np.inf])
    np.testing.assert_allclose(bounds["sweeps"], [1.0, 2.0, 2.0])


def test_dominant_and_largest_clones():
    run = run_model(clonal.build_spec(size=30, steps=60, r_m=0.2, seed=2), showprogress=False)
    lattice = run.lgca
    dominant = _clones.dominant_clone(lattice)
    cells = LatticeState(lattice).cells
    families = _clones.cell_families(lattice)
    node = np.ravel_multi_index(cells.node, lattice.dims)
    for index in np.unique(node)[:40]:  # the dominant family is the most frequent one at the node
        counts = np.bincount(families[node == index])
        assert counts[dominant.flat[index]] == counts.max()
    assert np.all(dominant[lattice.cell_density[lattice.nonborder] == 0] == -1)
    ranks = _clones.largest_clones(dominant, number=3)
    assert np.isnan(ranks[dominant < 0]).all() and set(np.unique(ranks[dominant >= 0])) <= {0, 1, 2, 3}
    # every cell of a clone has the clone's κ, which the Muller plot is coloured by
    kappa = np.asarray(_clones.family_trait(lattice, "kappa"))
    np.testing.assert_allclose(kappa[families], np.asarray(cells["kappa"], dtype=float))


def test_clonal_go_or_grow_setup_and_research_model():
    spec = clonal.build_spec()
    assert spec.space.dims == (150, 150) and spec.time.steps == 400 and spec.state.nodes.sum() == 50
    assert spec.state.nodes[75, 75, -1] == 50 and spec.state.traits == {"r_b": 0.2, "kappa": 0.0}
    assert clonal.build_spec(full=True).space.dims == (250, 250)
    fixed = clonal.build_spec(evolve_kappa=False).dynamics.operators[2]["parameters"]["mutation"]
    assert set(fixed["traits"]) == {"r_b"}
    # the entry writes out the research model go_or_grow_glioblastoma: the same runs
    small = clonal.build_spec(size=30, steps=40, r_m=0.1, seed=4)
    research = replace(small, dynamics=InteractionPipelineSpec(operators=[{
        "name": "go_or_grow_glioblastoma", "parameters": {"r_b": 0.2, "r_d": 0.05, "r_m": 0.1, "fitness_increase": 1.1,
                                                          "theta": 0.5, "kappa": 0.0, "kappa_std": 1.0}}]))
    runs = [run_model(s, showprogress=False).lgca for s in (small, research)]
    cells = [LatticeState(lattice).cells for lattice in runs]
    assert len(cells[0]) == len(cells[1]) > 50
    for trait in ("r_b", "kappa"):
        np.testing.assert_array_equal(np.sort(np.asarray(cells[0][trait])), np.sort(np.asarray(cells[1][trait])))


@pytest.mark.parametrize("seed", [3, 4])
def test_an_evolving_switch_turns_repulsive_and_speeds_up_growth(seed):
    runs = {evolve: clonal.core_and_rim(run_model(clonal.build_spec(size=80, steps=250, evolve_kappa=evolve,
                                                                    seed=seed), showprogress=False).lgca)
            for evolve in (True, False)}
    assert runs[True]["kappa"]["all"] < -0.4 and runs[False]["kappa"]["all"] == 0
    assert runs[True]["cells"] > 1.1 * runs[False]["cells"]
    assert runs[True]["r_b"]["all"] > 0.21 and runs[False]["r_b"]["all"] > 0.2  # drivers are selected


from lgca.zoo import evolution_modes as modes


def test_evolution_modes_setup():
    spec = modes.build_spec()
    assert spec.space.dims == (40, 40) and spec.state.capacity == 64 and spec.time.steps == 1000
    assert spec.state.nodes.sum() == 64 and spec.state.nodes[20, 20, -1] == 64
    assert spec.dynamics.operators[0]["parameters"]["crowding"] is False  # a hard capacity
    full = modes.build_spec(full=True)
    assert full.space.dims == (60, 60) and full.state.capacity == 512
    assert np.all(modes.build_spec(start="full").state.nodes[..., -1] == 64)
    with pytest.raises(ValueError, match="'full' or 'centre'"):
        modes.build_spec(start="edge")
    assert modes.moving_fraction(0.0) == pytest.approx(6 / 7)
    assert modes.moving_fraction(5.0) == pytest.approx(modes.moving_fraction(0.0, gamma=5.0))


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_contact_inhibition_sets_the_moving_fraction_in_a_full_tissue(alpha):
    spec = modes.build_spec(alpha=alpha, start="full", size=12, r_b=0.0, r_d=0.0, steps=1)
    spec = replace(spec, space=replace(spec.space, boundary="periodic"))  # every neighbour is full
    lattice = run_model(spec, showprogress=False).lgca
    counts = lattice.channel_pop[lattice.nonborder]
    moving = counts[..., :6].sum() / counts.sum()
    expected = modes.moving_fraction(alpha)
    assert abs(moving - expected) < 4 * np.sqrt(expected * (1 - expected) / counts.sum())


@pytest.mark.parametrize("seed", [1, 2])
def test_contact_inhibition_holds_evolution_back(seed):
    """Drivers spread through a mixing tissue, but hardly beyond their gland when cells stop moving."""
    final = {}
    for alpha in (0.0, 10.0):
        _, record = modes.record(modes.build_spec(alpha=alpha, size=20, capacity=32, steps=600, r_m=1e-3,
                                                  seed=seed), every=600)
        final[alpha] = {key: record[key][-1] for key in ("n", "D", "r_b")}
        assert record["cells"][-1] > 0.9 * 20 * 20 * 32 * 0.5  # the tissue is full
    assert final[0.0]["n"] > 2 and final[10.0]["n"] < 1.5
    assert final[0.0]["D"] > final[10.0]["D"] and final[0.0]["r_b"] > final[10.0]["r_b"] + 0.01
    n, D = final[0.0]["n"], final[0.0]["D"]
    assert n >= 2 or D <= 1 / (2 - n) ** 2  # the bound of Noble et al.


# ------------------------------------------------------------------ jamming

from lgca.zoo import jamming


def _random_hex(seed=5, size=8):
    from lgca import get_lgca

    rng = np.random.default_rng(seed)
    nodes = rng.random((size, size, jamming.K)) < 0.45
    return get_lgca(geometry="hex", dims=(size, size), nodes=nodes, restchannels=3, bc="reflecting")


def _neighbours(lattice):
    """For every node, its neighbours and the unit vectors to them, from the coordinates alone."""
    x, y = lattice.xcoords.ravel(), lattice.ycoords.ravel()
    distance = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    return [(np.flatnonzero(np.isclose(row, 1.0)), x, y) for row in distance]


def test_adhesion_and_pressure_follow_the_model_definition():
    lattice = _random_hex()
    state = LatticeState(lattice)
    b, rho_0, K = 6, jamming.RHO_0, jamming.K
    counts = state.counts[..., 0, :].reshape(-1, K)
    n = counts.sum(-1).astype(float)
    c = state.c.T  # (b, 2)
    flux = counts[:, :b] @ c
    resting = counts[:, b:].sum(-1)
    near = _neighbours(lattice)
    n_nb = np.array([n[index].sum() for index, _, _ in near])
    n_crit = (b + 1) * rho_0
    u = n_nb * np.clip(1 - n_nb / n_crit, 0, None) / (2 * n_crit)
    excess = np.clip(n - rho_0, 0, None) / (K - rho_0)
    adhesion = jamming.adhesion.function(state).reshape(-1, K)
    pressure = jamming.pressure.function(state).reshape(-1, 2)
    for node, (index, x, y) in enumerate(near):
        directions = np.stack([x[index] - x[node], y[index] - y[node]], -1)
        gradient_u = directions.T @ u[index]
        flux_nb = flux[index].sum(0)
        expected = c @ (gradient_u + flux_nb / (2 * b))
        np.testing.assert_allclose(adhesion[node, :b], expected, atol=1e-12)
        np.testing.assert_allclose(adhesion[node, b:], resting[index].sum() / (b * rho_0))
        np.testing.assert_allclose(pressure[node], -(directions.T @ excess[index]), atol=1e-12)


def test_matrix_degradation_and_influx():
    spec = jamming.build_spec(ecm=2.0, rate=0.5)
    only = lambda *names: vary(spec, {"dynamics.operators": [op for op in spec.dynamics.operators
                                                            if isinstance(op, dict) and op["name"] in names],
                                      "dynamics.propagation": False})
    model = build_model(only("jamming.degradation"))
    lattice = model.lgca
    n = lattice.cell_density[lattice.nonborder].copy()
    model.step()
    np.testing.assert_allclose(lattice.ecm[lattice.nonborder], 2.0 * (1 - n / jamming.K))
    # influx: every free channel of the two lowest rows is filled with probability 0.5
    model = build_model(only("jamming.influx"))
    lattice = model.lgca
    before = lattice.cell_density[lattice.nonborder].copy()
    free = (jamming.K - before[:, :2]).sum()
    model.step()
    after = lattice.cell_density[lattice.nonborder]
    np.testing.assert_array_equal(after[:, 2:], before[:, 2:])
    added = (after - before).sum()
    assert abs(added - 0.5 * free) < 4 * np.sqrt(free * 0.25)


def test_the_invasion_modes():
    """Weak adhesion and a sparse matrix release single cells; a dense matrix or strong adhesion stops
    it; only adhesion correlates the movement of neighbours (Fig 5d, e)."""
    def mode(beta, ecm):
        runs = [jamming.invasion_mode(run_model(jamming.build_spec(beta=beta, ecm=ecm, seed=seed),
                                                showprogress=False)) for seed in (1, 2)]
        return {key: np.mean([run[key] for run in runs]) for key in runs[0]}

    free, confined, adhesive = mode(0.2, 0.2), mode(0.2, 5.0), mode(10.0, 0.2)
    assert free["single_cells"] > 4 * confined["single_cells"]
    assert free["single_cells"] > 4 * adhesive["single_cells"]
    assert adhesive["correlation"] > free["correlation"] + 0.2
    assert abs(confined["correlation"] - free["correlation"]) < 0.1


def test_the_spheroid_is_a_disc_that_supplies_cells():
    spec = jamming.build_spec(setup="spheroid", radius=5)
    source = spec.state.fields["source"]
    assert spec.space.dims == (80, 80) and 70 < source.sum() < 100  # about π 5² nodes
    assert spec.state.nodes.sum() == 3 * source.sum()
    with pytest.raises(ValueError, match="'sheet'"):
        jamming.build_spec(setup="ring")


# ------------------------------------------------------------------ evolving front

from lgca.zoo import evolving_front


def test_gamma_sets_the_diffusion_coefficient():
    gamma = evolving_front.gamma_for(0.14)
    assert 4 / (4 + np.exp(gamma)) == pytest.approx(4 * 0.14)
    with pytest.raises(ValueError, match="between 0 and 1/4"):
        evolving_front.gamma_for(0.3)
    np.testing.assert_allclose(evolving_front.predicted_front([0, 10, 20], 0.2, 5.0),
                               5.0 + evolving_front.wave_speed(0.2) * np.array([0, 10, 20]))
    # the paper's correction for the discreteness of the front: 1 - 4 / ln²K, 0.81 at K = 100
    assert evolving_front.discreteness(100) == pytest.approx(0.8114, abs=1e-4)
    np.testing.assert_allclose(evolving_front.predicted_front([0, 10, 20], 0.2, 5.0, capacity=100),
                               5.0 + 0.8114 * evolving_front.wave_speed(0.2) * np.array([0, 10, 20]), rtol=1e-4)


def test_without_mutation_the_front_moves_near_the_fisher_kpp_speed():
    """Slower by the discreteness of the leading edge: 0.27 instead of 0.33 at K = 30."""
    record = evolving_front.record(evolving_front.build_spec(p_mu=0.0, length=200, width=4, capacity=30,
                                                             steps=400, seed=3), every=20)
    speed = np.polyfit(record["steps"][5:], record["front"][5:], 1)[0]
    assert 0.7 * evolving_front.wave_speed(0.2) < speed < evolving_front.wave_speed(0.2)
    assert np.all(record["alpha_mean"] == pytest.approx(0.2))


def test_the_fastest_cells_gather_at_the_front():
    record = evolving_front.record(evolving_front.build_spec(length=200, width=4, capacity=30, steps=400, seed=3),
                                   every=400)
    alpha, position = record["alpha"][-1], record["front"][-1]
    assert np.nanmean(alpha[position - 20:position]) > np.nanmean(alpha[:20]) + 0.1
    assert record["alpha_mean"][-1] > 0.25 and record["alpha_top"][-1] > record["alpha_mean"][-1]


# ------------------------------------------------------------------ excitable media

from lgca.zoo import excitable_media


def test_the_reaction_terms_and_their_nullclines():
    rho = np.linspace(0, 1, 11)
    f, _ = excitable_media.reaction(rho, 0.75 * rho - 0.02)  # the middle branch of f = 0
    np.testing.assert_allclose(f, 0, atol=1e-15)
    np.testing.assert_allclose(excitable_media.reaction(rho, rho)[1], 0)
    nodes = excitable_media.quadrants(10)
    x, y = excitable_media.fractions(nodes)
    assert x[7, 2] == 1 and y[7, 2] == 0 and x[2, 7] == 0 and y[2, 7] == 1 and x[7, 7] == y[7, 7] == 1
    assert x[2, 2] == y[2, 2] == 0


def test_spirals_persist_in_the_lgca_and_its_mean_field():
    run = run_model(excitable_media.build_spec(size=60, steps=300, record_every=100), showprogress=False)
    x, y = excitable_media.fractions(np.asarray(run.data["nodes"])[-1])
    assert 0.1 < x.mean() < 0.6 and 0.1 < y.mean() < 0.6  # still excited after 300 steps
    x_0, y_0 = excitable_media.fractions(np.asarray(run.data["nodes"])[0])
    pde = excitable_media.barkley(run.lgca, x_0, y_0, steps=300, record_every=100, probe=(30, 30))
    assert pde["x"].shape == (4, 60, 60) and pde["probe"].shape == (301, 2)
    assert 0.1 < pde["x"][-1].mean() < 0.6
    assert np.ptp(pde["probe"][150:, 0]) > 0.8  # the node goes around the excitation loop


def test_mean_return_time_of_a_periodic_node():
    nodes = np.zeros((20, 2, 23), dtype=bool)
    for frame in range(20):  # n_X runs through 0, 1, 2, 3 and repeats: period 4
        nodes[frame, :, :frame % 4] = True
    assert excitable_media.mean_return_time(nodes) == 4
    nodes[1:, 1] = True  # a node that never returns to its first state is left out
    assert excitable_media.mean_return_time(nodes) == 4
