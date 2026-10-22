"""End-to-end tests of what users new to the package do with it.

Every test plays a scenario of a student, researcher or reviewer: build a model from the documented APIs
(``ModelSpec`` or ``get_lgca``), write a rule of their own, run, read the results, save and reload a model
file, sweep a parameter, and check a scientific expectation. The oracles are independent of the library
code they test: analytic probabilities with tolerances of four standard errors, cell conservation,
models stepped by hand, and fixed seeds.
"""
import dataclasses
import importlib
import os
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from lgca import get_lgca, interaction, reorientation_term
from lgca.fields import PDESpec
from lgca.lattice_state import LatticeState
from lgca.model import (
    AnalysisSpec,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    load_model_spec,
    model_spec_to_dict,
    run_model,
    save_model_spec,
)
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.simulation import (
    DensityRecorder,
    NodeRecorder,
    PopulationRecorder,
    Schedule,
    SimulationRunner,
)
from lgca.study import sweep
from lgca.testing import InteractionCheckError, check_interaction

matplotlib.use("Agg")


def _sem(values):
    values = np.asarray(values, dtype=float)
    return values.std(ddof=1) / np.sqrt(values.size)


# ---------------------------------------------------------------------------------------------------------
# A student writes an adhesion term and asks whether it makes clusters
# ---------------------------------------------------------------------------------------------------------

@reorientation_term(coupling="channels")
def student_adhesion(state):
    """Cells prefer velocity channels that point to occupied neighbouring nodes."""
    return state.neighbor_values((state.density > 0).astype(float))


def _adhesion_spec(beta, seed, steps=60, size=30, operators=None):
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(size, size), boundary="periodic"),
        state=StateSpec(density=0.5, restchannels=1),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(
            operators=operators or [ReorientationSpec(terms=[student_adhesion(beta=beta)])]),
        analysis=AnalysisSpec(observers=[DensityRecorder(), PopulationRecorder()]),
    )


def _clustering(density):
    """Cells on the four neighbours of a cell, relative to a uniform layout (1 for no clustering)."""
    rho = density.astype(float)
    neighbours = sum(np.roll(rho, s, axis=a) for a in (0, 1) for s in (1, -1))
    return (rho * neighbours).sum() / rho.sum() / (4 * rho.mean())


def test_adhesion_workflow_build_run_read_and_rerun_from_file(tmp_path):
    """The student runs the term, reads the recordings and reruns the saved model file to identical data;
    a reorientation keeps the cells, and the recordings hold the interior only."""
    spec = _adhesion_spec(beta=2.0, seed=4, steps=30, size=20)
    result = run_model(spec, showprogress=False)

    population = result.data["population"]
    density = result.data["density"]
    assert density.shape == (31, 20, 20)  # initial state plus one map per step, no ghost nodes
    assert np.all(population == population[0])
    np.testing.assert_array_equal(density.sum(axis=(1, 2)), population)

    path = save_model_spec(spec, tmp_path / "adhesion.json")
    again = run_model(load_model_spec(path), showprogress=False)
    np.testing.assert_array_equal(again.data["density"], density)


@pytest.mark.parametrize("beta", [1.0, 2.0])
def test_lone_pair_cell_turns_to_its_partner_with_boltzmann_probability(beta):
    """What the term means: a cell with one occupied neighbour enters the channel towards it with
    probability e^beta / (e^beta + 4) (five channels, score 1 for one of them)."""
    size = 40
    left = np.zeros((size, size), bool)
    left[0::4, 0::4] = True
    right = np.roll(left, 1, axis=0)  # partner one node east (+x)
    nodes = np.zeros((size, size, 5), dtype=int)
    nodes[left | right, 4] = 1
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(size, size)),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=50, seed=3),
        dynamics=InteractionPipelineSpec(
            operators=[ReorientationSpec(terms=[student_adhesion(beta=beta)])], propagation=False),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )
    recorded = run_model(spec, showprogress=False).data["nodes"][1:]
    towards = np.concatenate([recorded[:, left, 0].ravel(), recorded[:, right, 2].ravel()])
    p = np.exp(beta) / (np.exp(beta) + 4)
    assert abs(towards.mean() - p) < 4 * np.sqrt(p * (1 - p) / towards.size)


def test_adhesion_forms_clusters_that_a_random_walk_does_not():
    """The research question: the clustering index is 1 for a random walk and rises with beta, by more
    than four combined standard errors over eight seeds."""
    seeds = range(8)

    def index(spec):
        return _clustering(run_model(spec, showprogress=False).data["density"][-1])

    walk = np.array([index(_adhesion_spec(0.0, s, operators=[{"name": "random_walk"}])) for s in seeds])
    weak = np.array([index(_adhesion_spec(1.0, s)) for s in seeds])
    strong = np.array([index(_adhesion_spec(2.0, s)) for s in seeds])
    assert abs(walk.mean() - 1) < 4 * _sem(walk)
    assert weak.mean() - walk.mean() > 4 * np.hypot(_sem(weak), _sem(walk))
    assert strong.mean() - weak.mean() > 4 * np.hypot(_sem(strong), _sem(weak))


@pytest.mark.xfail(strict=True, reason="cue-by-name-suggestion-fails: the operator mapping that the error "
                                       "message suggests for a single custom cue is not accepted")
def test_single_custom_cue_by_name_as_the_error_message_suggests():
    """The student puts the term straight into the operators; following the advice of the error message
    must work and run the same model as ``ReorientationSpec(terms=[term])``."""
    with pytest.raises(ValueError, match="give one cue by name") as info:
        run_model(_adhesion_spec(2.0, 1, steps=2, operators=[student_adhesion(beta=2.0)]), showprogress=False)
    assert "'name': 'student_adhesion'" in str(info.value)
    by_name = _adhesion_spec(2.0, 1, steps=20,
                             operators=[{"name": "student_adhesion", "parameters": {"beta": 2.0}}])
    np.testing.assert_array_equal(run_model(by_name, showprogress=False).data["density"],
                                  run_model(_adhesion_spec(2.0, 1, steps=20), showprogress=False).data["density"])


# ---------------------------------------------------------------------------------------------------------
# A cancer researcher models a glioma with an oxygen-dependent go-or-grow switch in a module of rules
# ---------------------------------------------------------------------------------------------------------

GLIOMA_RULES = textwrap.dedent('''
    import numpy as np
    from lgca import interaction


    @interaction(kind="phenotype_switch", families=("classical", "nove"), n_species=2)
    def hypoxic_switch(state, theta=0.3, kappa=10.0, rate=0.5):
        """Cells migrate (species 0) under hypoxia and proliferate (species 1) under normoxia."""
        go = (1 - np.tanh(kappa * (state.field("oxygen") - theta))) / 2
        rates = np.zeros(state.dims + (2, 2))
        rates[..., 1, 0] = rate * go
        rates[..., 0, 1] = rate * (1 - go)
        state.switch_phenotype(rates, channels={0: "velocity", 1: "rest"})
''')
REST = 4


@pytest.fixture
def glioma_rules(tmp_path, monkeypatch):
    """The user's rule module, written next to the model file as tutorial 6 suggests."""
    (tmp_path / "glioma_rules.py").write_text(GLIOMA_RULES)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("glioma_rules", None)
    yield importlib.import_module("glioma_rules")
    sys.modules.pop("glioma_rules", None)


def _glioma_spec(rules, size=31, steps=40, seed=1, theta=0.3):
    nodes = np.zeros((size, size, 2, 4 + REST), dtype=int)
    c = size // 2
    nodes[c - 2:c + 3, c - 2:c + 3, 1, 4:] = 1  # proliferative core
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(size, size), boundary="reflecting"),
        state=StateSpec(n_species=2, restchannels=REST, nodes=nodes, fields={"oxygen": 1.0}),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[
            PDESpec(field="oxygen", diffusion=100.0, cells=[{"uptake": 5.0}], boundary={"value": 1.0},
                    solver="steady"),
            rules.hypoxic_switch(theta=theta),
            {"name": "birth_death", "parameters": {"birth_rate": 0.2, "death_rate": 0.01, "species": 1,
                                                  "channels": "rest"}},
            {"name": "random_walk", "parameters": {"species": 0, "channels": "velocity"}},
        ]),
        analysis=AnalysisSpec(observers=[PopulationRecorder(), DensityRecorder(schedule=Schedule(every=10))]),
    )


def _front_radius(density):
    """95th percentile of the distance of the cells from the centre."""
    d = np.asarray(density).sum(-1)
    x, y = np.indices(d.shape)
    r = np.hypot(x - (d.shape[0] - 1) / 2, y - (d.shape[1] - 1) / 2).ravel()
    order = np.argsort(r)
    cumulative = np.cumsum(d.ravel()[order])
    return r[order][np.searchsorted(cumulative, 0.95 * cumulative[-1])]


def test_model_file_reruns_identically_and_keeps_phenotypes_in_their_channels(glioma_rules, tmp_path):
    """The model file reruns to identical results; the tumour grows, its hypoxic core releases migratory
    cells, and each phenotype stays in its channel type."""
    spec = _glioma_spec(glioma_rules)
    direct = run_model(spec, showprogress=False)
    save_model_spec(spec, tmp_path / "glioma.json")
    again = run_model(load_model_spec(tmp_path / "glioma.json"), showprogress=False)

    np.testing.assert_array_equal(again.data["population"], direct.data["population"])
    np.testing.assert_array_equal(again.lgca.nodes, direct.lgca.nodes)
    assert direct.data["population"][-1] > direct.data["population"][0]
    nodes = direct.lgca.nodes[direct.lgca.nonborder]
    assert nodes[..., 0, :].sum() > 0  # migratory cells
    assert nodes[..., 0, 4:].sum() == 0  # migratory cells never sit in rest channels
    assert nodes[..., 1, :4].sum() == 0  # proliferative cells never move


@pytest.mark.parametrize("oxygen", [0.2, 0.3, 0.5])
def test_realised_switching_probability_matches_the_oxygen_response(glioma_rules, oxygen):
    """In uniform oxygen c a proliferative cell becomes migratory with probability
    rate (1 - tanh(kappa (c - theta))) / 2 (all velocity channels are free), and switching keeps every
    cell."""
    theta, kappa, rate, size = 0.3, 10.0, 0.5, 60
    nodes = np.zeros((size, size, 2, 4 + REST), dtype=int)
    nodes[..., 1, 4:] = 1
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(size, size), boundary="reflecting"),
        state=StateSpec(n_species=2, restchannels=REST, nodes=nodes, fields={"oxygen": oxygen}),
        time=TimeSpec(steps=1, seed=7),
        dynamics=InteractionPipelineSpec(
            operators=[glioma_rules.hypoxic_switch(theta=theta, kappa=kappa, rate=rate)]),
    ))
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]
    n = nodes.sum()
    p = rate * (1 - np.tanh(kappa * (oxygen - theta))) / 2
    assert after.sum() == n
    assert abs(after[..., 0, :].sum() / n - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_higher_switching_threshold_pushes_the_invasion_front_further(glioma_rules):
    """The sweep: a higher oxygen threshold for migration makes more cells migratory and moves the front
    farther out, by more than three combined standard errors over four seeds."""
    def front(result):
        return _front_radius(result.data["density"][-1])

    def migratory(result):
        density = result.data["density"][-1]
        return density[..., 0].sum() / density.sum()

    table = sweep(_glioma_spec(glioma_rules, size=41, steps=60), grid={"theta": [0.1, 0.6]}, seeds=range(4),
                  measure={"front": front, "migratory": migratory}, backend="threads", n_jobs=1,
                  showprogress=False)
    stats = table.groupby("theta")[["front", "migratory"]].agg(["mean", "sem"])
    for column in ("front", "migratory"):
        low, high = stats.loc[0.1, column], stats.loc[0.6, column]
        assert high["mean"] - low["mean"] > 3 * np.hypot(low["sem"], high["sem"]) + 1e-9, stats


@pytest.mark.xfail(strict=True, reason="cli-plugins-cwd: `biolgca run --plugins` cannot import a module "
                                       "from the working directory")
def test_command_line_runs_model_with_rules_module_next_to_it(glioma_rules, tmp_path):
    """Tutorial 6's layout: model.json and the rules module in one folder; ``biolgca run`` from that
    folder gives the population of the Python run."""
    spec = _glioma_spec(glioma_rules, steps=10)
    save_model_spec(spec, tmp_path / "glioma.json")
    expected = run_model(spec, showprogress=False).data["population"]
    biolgca = shutil.which("biolgca", path=str(Path(sys.executable).parent)) or "biolgca"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([biolgca, "run", "glioma.json", "--plugins", "glioma_rules", "--output", "run"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    with np.load(tmp_path / "run" / "measurements.npz") as data:
        np.testing.assert_array_equal(data["n_t"], expected)


# ---------------------------------------------------------------------------------------------------------
# An evolutionary biologist follows a heritable proliferation trait with a motility trade-off
# ---------------------------------------------------------------------------------------------------------

R_MAX = 0.5


@interaction(kind="birth_death", families=("ib", "nove_ib"))
def tradeoff_growth(state, r_d=0.05, std=0.03, r_max=R_MAX):
    """Cells die with r_d and divide with r_b (1 - crowding); a daughter's r_b mutates and sets p_rest."""
    cells = state.cells
    cells.kill(state.rng.random(len(cells)) < r_d)
    crowding = np.clip(state.density[cells.node] / state.capacity, 0, 1)
    dividing = state.rng.random(len(cells)) < cells["r_b"] * (1 - crowding)
    daughters = cells.divide(dividing, channels="all")
    r_b = np.clip(cells["r_b"][daughters] + state.rng.normal(0, std, len(daughters)), 0, r_max)
    cells.set_trait(daughters, "r_b", r_b)
    cells.set_trait(daughters, "p_rest", r_b / r_max)


RESTING = ReorientationSpec(terms=[ReorientationTermSpec("resting", parameters={"probability": {"max": "p_rest"}})])


def _tradeoff_spec(seed=1, steps=40, std=0.03, r_d=0.05, capacity=8, operators=None, density=0.5):
    return ModelSpec(
        space=SpaceSpec(geometry="square", dims=(20, 20), boundary="periodic"),
        state=StateSpec(density=density, restchannels=1, volume_exclusion=False, capacity=capacity,
                        identity_based=True, traits={"r_b": 0.1, "p_rest": 0.1 / R_MAX}),
        time=TimeSpec(steps=steps, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators or [tradeoff_growth(r_d=r_d, std=std), RESTING]),
        analysis=AnalysisSpec(observers=[NodeRecorder(), PopulationRecorder()]),
    )


def _with_traits(spec, **traits):
    return dataclasses.replace(spec, state=dataclasses.replace(spec.state, traits=traits))


def _traits_over_time(lgca, name):
    values = np.asarray(lgca.props[name])
    return [values[cells.label] for cells in lgca.cells_t]


def _initial_cell_count(spec):
    lgca = build_model(spec).lgca
    return int(lgca.cell_density[lgca.nonborder].sum())


def test_trait_distribution_workflow_is_recorded_and_reproducible():
    """One trait value per living cell at every recorded step, the trade-off p_rest = r_b / r_max for
    every cell, variation from mutation, and the same seed gives the same trait distribution."""
    result = run_model(_tradeoff_spec(), showprogress=False)
    r_b = _traits_over_time(result.lgca, "r_b")
    p_rest = _traits_over_time(result.lgca, "p_rest")
    assert [len(values) for values in r_b] == list(result.data["population"])
    assert all(np.all((values >= 0) & (values <= R_MAX)) for values in r_b)
    np.testing.assert_allclose(p_rest[-1], r_b[-1] / R_MAX)
    assert np.ptp(r_b[0]) == 0 and np.std(r_b[-1]) > 0.01

    again = run_model(_tradeoff_spec(), showprogress=False)
    other = run_model(_tradeoff_spec(seed=2), showprogress=False)
    np.testing.assert_array_equal(_traits_over_time(again.lgca, "r_b")[-1], r_b[-1])
    assert not np.array_equal(other.data["population"], result.data["population"])


def test_selection_without_mutation_matches_branching_process():
    """Two clones (r_b 0.1 and 0.3) without crowding and mutation each grow like a Galton-Watson process
    with mean (1 - r_d)(1 + r_b); fails if daughters do not inherit r_b or death is ignored."""
    steps, r_d, slow, fast = 10, 0.05, 0.1, 0.3
    spec = _tradeoff_spec(steps=steps, r_d=r_d, std=0.0, capacity=10**6, density=2,
                          operators=[tradeoff_growth(r_d=r_d, std=0.0), {"name": "random_walk"}])
    r_b = np.where(np.arange(_initial_cell_count(spec)) % 2 == 0, slow, fast)
    lgca = run_model(_with_traits(spec, r_b=r_b, p_rest=r_b), showprogress=False).lgca
    final = np.asarray(lgca.props["r_b"])[lgca.cells_t[-1].label]
    for value in (slow, fast):
        start = int(np.sum(r_b == value))
        mean = (1 - r_d) * (1 + value)  # a survivor stays and divides with r_b
        variance = (1 - r_d) * (1 + 3 * value) - mean ** 2
        expected = start * mean ** steps
        se = np.sqrt(start * variance * mean ** (steps - 1) * (mean ** steps - 1) / (mean - 1))
        assert abs(np.sum(np.isclose(final, value)) - expected) < 4 * se


def test_cells_rest_with_their_own_probability():
    """Motility cost: without volume exclusion every cell rests with its own trait p_rest (two groups,
    0.2 and 0.8, pooled over 20 steps); the reorientation keeps the population."""
    spec = _tradeoff_spec(steps=20, operators=[RESTING])
    n0 = _initial_cell_count(spec)
    p = np.where(np.arange(n0) % 2 == 0, 0.2, 0.8)
    result = run_model(_with_traits(spec, r_b=p * R_MAX, p_rest=p), showprogress=False)
    lgca = result.lgca
    props = np.asarray(lgca.props["p_rest"])
    assert list(result.data["population"]) == [n0] * 21
    for value in (0.2, 0.8):
        resting = total = 0
        for cells in lgca.cells_t[1:]:
            mine = np.isclose(props[cells.label], value)
            resting += np.sum(cells.channel[mine] >= lgca.velocitychannels)
            total += np.sum(mine)
        assert abs(resting / total - value) < 4 * np.sqrt(value * (1 - value) / total)


@pytest.mark.xfail(strict=True, reason="check-interaction-ib-phenotype-switch-zero-models: check_interaction "
                                       "builds no model for an identity-based phenotype switch")
def test_check_interaction_checks_identity_based_phenotype_switch():
    """A trait-drift rule declared as phenotype_switch, tested with the documented one-liner: a broken
    rule must be reported."""
    @interaction(kind="phenotype_switch", families=("ib", "nove_ib"))
    def broken_drift(state, std=0.01):
        raise RuntimeError("this rule always fails")

    with pytest.raises(InteractionCheckError, match="always fails"):
        check_interaction(broken_drift, traits={"r_b": 0.2})


# ---------------------------------------------------------------------------------------------------------
# A student starts with get_lgca, the quick start of the tutorials
# ---------------------------------------------------------------------------------------------------------

def test_a_tiny_example_run_conserves_the_cells_of_a_random_walk():
    """A curated example run for a few steps records one population per step, conserved by the walk."""
    from lgca.examples import run_example

    result = run_example("random_walk", steps=3, showprogress=False)
    assert result.data.steps("population").tolist() == [0, 1, 2, 3]
    population = result.data["population"].tolist()
    assert population[0] > 0 and population == [population[0]] * 4


@pytest.mark.parametrize("geom, dims", [("lin", (20000,)), ("square", (100, 100)), ("hex", (100, 100)),
                                        ("cubic", (20, 20, 20)), ("moore", (20, 20, 20))])
@pytest.mark.parametrize("ve, ib, density", [(True, False, 0.5), (True, True, 0.5), (False, False, 1.5)])
def test_random_reset_reaches_the_requested_mean_cells_per_node(geom, dims, ve, ib, density):
    """``get_lgca(density=...)`` gives that many cells per node on average."""
    lgca = get_lgca(geometry=geom, ve=ve, ib=ib, dims=dims, density=density, restchannels=2,
                    interaction="only_propagation", seed=1)
    per_node = (lgca.nodes[lgca.nonborder] > 0).sum(-1) if ib else lgca.nodes[lgca.nonborder].sum(-1)
    assert abs(per_node.mean() - density) < 4 * per_node.std() / np.sqrt(per_node.size)


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("geom, dims", [("lin", (7,)), ("square", (5, 4)), ("hex", (4, 6)), ("cubic", (3, 3, 3))])
def test_recordings_hold_the_states_of_every_step(geom, dims, ve):
    """``timeevo(record=True, recordN=True, recordpertype=True)`` records the actual states: periodic
    propagation keeps the cells, and the densities follow from the recorded nodes."""
    lgca = get_lgca(geometry=geom, ve=ve, dims=dims, density=0.5, restchannels=1, seed=5,
                    interaction="only_propagation")
    initial = lgca.nodes[lgca.nonborder].copy()
    lgca.timeevo(timesteps=3, record=True, recordN=True, recordpertype=True, showprogress=False)

    np.testing.assert_array_equal(lgca.n_t, [initial.sum()] * 4)
    np.testing.assert_array_equal(lgca.nodes_t[0], initial)
    np.testing.assert_array_equal(lgca.nodes_t[-1], lgca.nodes[lgca.nonborder])
    np.testing.assert_array_equal(lgca.dens_t, lgca.nodes_t.sum(-1))
    np.testing.assert_array_equal(lgca.velcells_t + lgca.restcells_t, lgca.dens_t)
    np.testing.assert_array_equal(lgca.restcells_t, lgca.nodes_t[..., lgca.velocitychannels:].sum(-1))


def _random_walker(seed):
    return get_lgca(geometry="square", dims=(5, 6), density=0.4, restchannels=1, interaction="random_walk",
                    bc="periodic", seed=seed)


def test_every_recorded_frame_is_the_state_at_its_step():
    """Frame k of a recording is the lattice at step k: the oracle is the same seeded model stepped by
    hand."""
    recorded, replay = _random_walker(3), _random_walker(3)
    SimulationRunner(recorded, timesteps=4, observers=[NodeRecorder(), DensityRecorder()],
                     showprogress=False).run()
    for step in range(5):
        np.testing.assert_array_equal(recorded.nodes_t[step], replay.nodes[replay.nonborder])
        np.testing.assert_array_equal(recorded.dens_t[step], replay.cell_density[replay.nonborder])
        replay.timestep()
    assert not np.array_equal(recorded.nodes_t[0], recorded.nodes_t[-1])  # the cells did move


CONSERVING = ["aggregation", "alignment", "chemotaxis", "contact_guidance", "go_or_rest", "nematic",
              "only_propagation", "persistent_motion", "persistent_walk", "random_walk"]


@pytest.mark.parametrize("bc", ["pbc", "rbc"])
@pytest.mark.parametrize("name", CONSERVING)
def test_motility_interactions_neither_create_nor_lose_cells(name, bc):
    """Motility models keep the cells; a broken volume exclusion in a boolean lattice shows as lost
    cells."""
    nodes = np.random.default_rng(1).random((6, 6, 6)) < 0.5
    lgca = get_lgca(geometry="square", nodes=nodes, interaction=name, bc=bc, seed=2)
    lgca.timeevo(timesteps=5, record=True, recorddens=False, showprogress=False)
    np.testing.assert_array_equal(lgca.nodes_t.sum(axis=(1, 2, 3)), nodes.sum())


@pytest.mark.parametrize("bc", ["reflecting", "periodic"])
def test_multispecies_nove_moore_conserves_each_species(bc):
    """A random walk of two species on the 3D Moore lattice keeps the cells of every species."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgca = get_lgca(geometry="moore", ve=False, n_species=2, dims=(4, 4, 4), density=2, bc=bc, seed=2,
                        interaction="random_walk")
    before = lgca.nodes[lgca.nonborder].sum(axis=(0, 1, 2, -1))
    lgca.timeevo(5, showprogress=False)
    np.testing.assert_array_equal(lgca.nodes[lgca.nonborder].sum(axis=(0, 1, 2, -1)), before)


def test_go_or_grow_rests_with_the_documented_switch_probability():
    """Go-or-grow: the resting fraction is (1 + tanh(kappa (rho - theta))) / 2, written out at
    rho = 10 / 20, kappa = 5, theta = 0.3."""
    expected = 0.5 * (1 + np.tanh(1.0))
    counts = np.zeros((400, 3), dtype=int)
    counts[:, 0] = 10
    lgca = get_lgca(geometry="lin", ib=True, ve=False, nodes=counts, restchannels=1, interaction="go_or_grow",
                    propagation=False, seed=7, r_b=0.0, r_d=0.0, capacity=20, kappa=5.0, theta=0.3,
                    kappa_std=0.0, theta_std=0.0)
    lgca.timestep()
    resting = int(lgca.channel_pop[lgca.nonborder][:, -1].sum())
    n = counts.sum()
    assert abs(resting - n * expected) < 4 * np.sqrt(n * expected * (1 - expected))


@pytest.mark.parametrize("geometry, dims", [("hex", (40, 40)), ("square", (40, 40)), ("lin", (1600,))])
def test_nove_polar_alignment_of_uniform_movers_follows_the_analytic_neighbour_flux(geometry, dims):
    """Polar alignment without volume exclusion: n cells per node move along channel 0, so the neighbour
    flux is (number of neighbours) n c_0, computed from the velocities alone; the channel probabilities
    are its Boltzmann weights."""
    beta, n = 0.15, 3
    probe = build_model(ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims),
                                  state=StateSpec(density=0, volume_exclusion=False, capacity=8))).lgca
    velocities = np.asarray(probe.c, dtype=float).T  # (b, d)
    b = len(velocities)
    nodes = np.zeros(dims + (b,), dtype=int)
    nodes[..., 0] = n
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=dims, boundary="periodic"),
        state=StateSpec(nodes=nodes, volume_exclusion=False, capacity=8, restchannels=0),
        time=TimeSpec(steps=1, seed=11),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
            ReorientationTermSpec("polar_alignment", beta=beta)])], propagation=False)))
    model.step()
    weights = np.exp(beta * velocities @ (b * n * velocities[0]))
    expected = weights / weights.sum()
    interior = model.lgca.nodes[model.lgca.nonborder]
    # the shifted rows of the hex lattice see their neighbours differently: check each row parity
    for parity in (0, 1):
        after = (interior[:, parity::2] if len(dims) > 1 else interior[parity::2]).reshape(-1, b)
        trials = after.sum()
        assert trials == n * after.shape[0]
        observed = after.sum(0) / trials
        assert np.all(np.abs(observed - expected) <= 4 * np.sqrt(expected * (1 - expected) / trials)), parity


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_a_saved_reorientation_model_is_the_same_model(tmp_path, suffix):
    """A chemotaxis model saved for a reviewer reloads with the term's beta and field and runs the same."""
    signal = np.arange(16, dtype=float).reshape(4, 4)
    spec = ModelSpec(
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1, fields={"signal": signal}),
        time=TimeSpec(steps=5, seed=7),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
            ReorientationTermSpec(name="chemotaxis", beta=1.5, parameters={"field": "signal"})])]))
    loaded = load_model_spec(save_model_spec(spec, tmp_path / f"model{suffix}"))
    assert model_spec_to_dict(loaded)["model"]["dynamics"] == model_spec_to_dict(spec)["model"]["dynamics"]
    term = loaded.dynamics.operators[0].terms[0]
    assert (term.beta, dict(term.parameters)) == (1.5, {"field": "signal"})
    np.testing.assert_array_equal(run_model(loaded, showprogress=False).lgca.nodes,
                                  run_model(spec, showprogress=False).lgca.nodes)


# ---------------------------------------------------------------------------------------------------------
# A researcher writes rules against the state operations, or against the label lists of older code
# ---------------------------------------------------------------------------------------------------------

def test_rules_see_the_cells_that_code_writing_label_lists_moved():
    """Code that edits the label lists between steps, as legacy rules and notebooks do: the next rule
    sees the cells where the code put them."""
    seen = {}

    def to_rest(lgca):
        nodes = lgca.nodes
        for index in np.ndindex(nodes.shape[:-1]):
            channels = nodes[index]
            channels[-1] = [label for channel in channels for label in channel]
            for k in range(len(channels) - 1):
                channels[k] = []

    @interaction(kind="reorientation", families=("nove_ib",), name="workflow.look")
    def look(state):
        seen["channels"] = np.asarray(state.cells.channel).copy()

    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(6, 6), boundary="periodic"),
        state=StateSpec(density=2, restchannels=1, volume_exclusion=False, identity_based=True, capacity=8),
        time=TimeSpec(steps=1, seed=1),
        dynamics=InteractionPipelineSpec(operators=[{"name": "workflow.look"}], propagation=False)))
    model.step()  # the cells now live in the table
    _ = model.lgca.nodes  # read once, as a plot would
    to_rest(model.lgca)
    model.step()
    assert len(seen["channels"]) == model.lgca.cell_density[model.lgca.nonborder].sum()
    assert np.all(seen["channels"] == model.lgca.K - 1)


def test_removal_probabilities_apply_per_species_and_per_channel():
    """Per-species and per-channel death probabilities given to ``state.remove_cells`` act on their axis."""
    lgca = get_lgca(geometry="square", dims=(6, 4), density=0, restchannels=1, n_species=2, seed=1)
    lgca.nodes[lgca.nonborder] = True
    state = LatticeState(lgca)
    state.remove_cells([0.0, 1.0])  # per species
    np.testing.assert_array_equal(state.counts[..., 0, :], 1)
    np.testing.assert_array_equal(state.counts[..., 1, :], 0)

    state = LatticeState(lgca)
    p = np.zeros((6, 4, 2, 5))
    p[..., 4] = 1.0  # per channel: only the rest channel empties
    state.remove_cells(p)
    np.testing.assert_array_equal(state.counts[..., :4], 1)
    np.testing.assert_array_equal(state.counts[..., 4], 0)


@pytest.mark.parametrize("geometry, dims", [("lin", (8,)), ("square", (6, 6)), ("hex", (6, 6)),
                                            ("cubic", (5, 5, 5))])
def test_neighbor_sum_of_one_cell_marks_the_nodes_its_channels_point_to(geometry, dims):
    """A crowding rule's ``state.neighbor_sum`` counts each neighbour once: one cell is seen by exactly
    as many nodes as there are velocity channels, and a uniform field sums to the number of neighbours."""
    lgca = get_lgca(geometry=geometry, dims=dims, density=0, restchannels=1, bc="periodic")
    state = LatticeState(lgca)
    values = np.zeros(dims)
    centre = tuple(d // 2 for d in dims)
    values[centre] = 1
    total = state.neighbor_sum(values)
    assert total.sum() == lgca.velocitychannels and total.max() == 1 and total[centre] == 0
    np.testing.assert_array_equal(total, state.neighbor_values(values).sum(-1))
    np.testing.assert_array_equal(state.neighbor_sum(np.ones(dims)), lgca.velocitychannels)


# ---------------------------------------------------------------------------------------------------------
# A researcher animates a run for a talk
# ---------------------------------------------------------------------------------------------------------

def _shown_density(animation, frame):
    animation._func(frame)
    return np.asarray(animation._fig.axes[0].images[0].get_array()).T


def _close(animation):
    from matplotlib import pyplot as plt

    animation._draw_was_started = True  # no "animation was deleted without rendering" warning
    plt.close("all")


@pytest.mark.parametrize("entry", ["method", "facade"])
def test_each_animation_frame_shows_its_own_density(entry):
    """Frame n of an animated density shows the data of step n, not only its title."""
    from lgca.plotting import animate

    model = get_lgca(geometry="square", dims=(4, 3), density=0.5, seed=2, interaction="only_propagation")
    density_t = np.arange(3 * 12).reshape(3, 4, 3) % 5  # three different frames
    animation = (animate(model, kind="density", data=density_t, steps=[0, 1, 2]) if entry == "facade"
                 else model.animate_density(density_t=density_t, steps=[0, 1, 2]))
    for frame in (1, 2):
        np.testing.assert_array_equal(_shown_density(animation, frame), density_t[frame])
    _close(animation)


def test_the_animation_of_a_run_shows_the_density_of_each_frame():
    """``AnimationObserver``: every frame shows the state at its time stamp (oracle: a replay by hand)."""
    from lgca.plotting import AnimationObserver

    lgca, replay = _random_walker(5), _random_walker(5)
    observer = AnimationObserver(kind="density", cbar=False)
    SimulationRunner(lgca, timesteps=3, observers=[observer], showprogress=False).run()
    for frame in range(4):
        np.testing.assert_array_equal(_shown_density(observer.animation, frame),
                                      replay.cell_density[replay.nonborder])
        replay.timestep()
    _close(observer.animation)


def test_a_two_species_animation_shows_all_cells_by_default():
    """Animating a two-species model without choosing a species shows all cells."""
    from lgca.plotting import AnimationObserver

    lgca = get_lgca(geometry="square", dims=(4, 4), density=0.5, n_species=2, interaction="only_propagation",
                    seed=11)
    observer = AnimationObserver(kind="density", cbar=False, close=True)
    SimulationRunner(lgca, timesteps=0, observers=[observer], showprogress=False).run()
    np.testing.assert_array_equal(_shown_density(observer.animation, 0),
                                  lgca.species_density[lgca.nonborder].sum(-1))
    assert lgca.species_density[lgca.nonborder][..., 1].any()  # the second species is there to be shown
    _close(observer.animation)


# ---------------------------------------------------------------------------------------------------------
# A reviewer checks what the examples and the model classes promise
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="tumour_spheroid._positions does not use the hex lattice coordinates")
def test_tumour_spheroid_hex_positions_match_lattice_neighbours():
    """Distances measured in the tumour-spheroid example are the lattice's: one node spacing apart."""
    from lgca.examples import tumour_spheroid

    dims = (8, 8)
    lgca = get_lgca(geometry="hex", dims=dims, density=0)
    x, y = tumour_spheroid._positions("hex", dims)
    np.testing.assert_allclose(x, lgca.xcoords)
    np.testing.assert_allclose(y, lgca.ycoords)


@pytest.mark.xfail(strict=True, reason="NoVE_IBLGCA_base.get_prop ignores the props table it is given")
def test_nove_ib_get_prop_reads_the_given_property_table():
    """A property table passed to ``get_prop`` of an identity-based model without volume exclusion is the
    one read, as in the model with volume exclusion."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgca = get_lgca(geometry="square", ib=True, ve=False, dims=(4, 4), density=1, seed=1,
                        interaction="random_walk")
    values = lgca.get_prop(props={"r_b": np.full(lgca.maxlabel + 1, 0.3)})
    assert np.all(values == 0.3)


# ---------------------------------------------------------------------------------------------------------
# Students make the usual mistakes: the error says what happened and where
# ---------------------------------------------------------------------------------------------------------

def _square(operators, steps=1, n_species=1):
    return ModelSpec(space=SpaceSpec(geometry="square", dims=(5, 5), boundary="periodic"),
                     state=StateSpec(density=0.3, restchannels=1, n_species=n_species),
                     time=TimeSpec(steps=steps, seed=1),
                     dynamics=InteractionPipelineSpec(operators=operators))


@pytest.mark.xfail(strict=True, reason="the CLI reports only the message of an error in a plugin rule, "
                                       "without its type and traceback")
def test_cli_shows_where_a_bug_in_a_plugin_rule_is(tmp_path, monkeypatch, capsys, request):
    """A typo in a dict key of the student's rule, run with ``biolgca run --plugins``: the error names
    the KeyError and the student's file."""
    (tmp_path / "workflow_student_rules.py").write_text(textwrap.dedent("""
        from lgca.rules import interaction

        @interaction(kind="birth_death", families=("classical",), name="workflow_student.grow")
        def grow(state, rate=0.1):
            params = {"rate": rate}
            return params["rtae"]
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: sys.modules.pop("workflow_student_rules", None))
    importlib.import_module("workflow_student_rules")
    model = save_model_spec(_square([{"name": "workflow_student.grow"}]), tmp_path / "model.json")
    cli = importlib.import_module("lgca.cli")
    code = cli.main(["run", str(model), "--plugins", "workflow_student_rules", "--output", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code != 0
    assert "KeyError" in err and "rtae" in err
    assert "workflow_student_rules.py" in err


@pytest.mark.xfail(strict=True, reason="build_model reports an operator whose constructor raises as an "
                                       "unknown operator")
def test_registered_operator_whose_constructor_fails_is_not_called_unknown():
    """A plugin author's operator class with a typo in its constructor is reported with that error, not
    as an unknown operator."""
    from lgca.operator_base import BirthDeathOperator, PluginInfo
    from lgca.plugins import register_plugin

    class Grow(BirthDeathOperator):
        def __init__(self, info, parameters=None):
            super().__init__(info, parameters)
            self.rate = float(self.parameters["rtae"])

        def apply(self, context, step):
            pass

    info = PluginInfo(name="workflow_student.grow_op", operator_kind="birth_death",
                      backend_families=("classical",), parameters={"rate": 0.1})
    register_plugin(info, lambda parameters: Grow(info, parameters))
    with pytest.raises(Exception) as caught:
        build_model(_square([{"name": "workflow_student.grow_op", "parameters": {"rate": 0.2}}]))
    assert "unknown operator" not in str(caught.value)
    assert "rtae" in str(caught.value) or isinstance(caught.value, KeyError)


@pytest.mark.xfail(strict=True, reason="a KeyError raised inside a reorientation term becomes a ValueError")
def test_keyerror_in_a_user_reorientation_term_stays_a_keyerror():
    """A wrong dict key inside the student's term surfaces as the KeyError it is."""
    weights = {"rest": 1.0}

    @reorientation_term(coupling="rest", name="workflow_student.lazy")
    def lazy(state, strength=1.0):
        return strength * weights["resting"] * np.ones(state.dims)

    with pytest.raises(KeyError, match="resting"):
        build_model(_square([ReorientationSpec(terms=[ReorientationTermSpec("workflow_student.lazy")])])).run()


def test_unknown_field_read_by_a_term_says_where_to_declare_it():
    """A term that reads a field the model does not have tells the student to declare it in
    ``StateSpec.fields``."""
    @reorientation_term(coupling="rest", name="workflow_student.reads_oxygen")
    def reads_oxygen(state):
        return state.field("oxygen")

    with pytest.raises((KeyError, ValueError), match="StateSpec.fields"):
        build_model(_square([ReorientationSpec(terms=[
            ReorientationTermSpec("workflow_student.reads_oxygen")])])).run()


@pytest.mark.xfail(strict=True, reason="a TypeError raised inside a switch cue becomes a ValueError about "
                                       "the cue")
def test_typeerror_inside_a_user_cue_is_not_reported_as_a_parameter_error():
    """A bug inside the student's switch cue surfaces as the TypeError it is, not as a wrong parameter."""
    from lgca.switching import switch_cue

    @switch_cue(name="workflow_crowded")
    def crowded(state):
        return state.density + None

    rates = [[0, {"max": 0.2, "cues": [{"name": "workflow_crowded", "kappa": 5, "theta": 0.5}]}], [0.05, 0]]
    with pytest.raises(TypeError, match="NoneType"):
        build_model(_square([{"name": "phenotype_switch", "parameters": {"rates": rates}}], n_species=2)).run()


def test_wrong_cue_parameter_is_a_clear_value_error():
    """A misspelt parameter of a switch cue is named in the error."""
    from lgca.switching import switch_cue

    @switch_cue(name="workflow_crowded2")
    def crowded(state, radius=1):
        return state.density * 1.0

    rates = [[0, {"max": 0.2, "cues": [{"name": "workflow_crowded2", "kappa": 5, "theta": 0.5, "radus": 2}]}],
             [0.05, 0]]
    with pytest.raises(ValueError, match="radus"):
        build_model(_square([{"name": "phenotype_switch", "parameters": {"rates": rates}}], n_species=2)).run()


@pytest.mark.xfail(strict=True, reason="square and hex lattices accept dims with three entries")
@pytest.mark.parametrize("geometry", ["square", "hex"])
def test_square_and_hex_reject_dims_with_three_entries(geometry):
    """A 2D lattice given three dimensions says that ``dims`` is wrong instead of running another model."""
    with pytest.raises(ValueError, match="dims"):
        get_lgca(geometry=geometry, dims=(10, 10, 4), density=0.2, seed=1)


@pytest.mark.xfail(strict=True, reason="check_interaction drops the original exception of the rule")
def test_check_interaction_keeps_the_error_of_the_rule():
    """``check_interaction`` of a broken rule keeps the rule's exception and traceback for debugging."""
    @interaction(kind="birth_death", families=("classical",), name="workflow_student.grow2")
    def grow(state, rate=0.1):
        params = {"rate": rate}
        return params["rtae"]

    with pytest.raises(InteractionCheckError) as caught:
        check_interaction("workflow_student.grow2", geometries=("square",))
    assert isinstance(caught.value.__cause__, KeyError)
    assert caught.value.__cause__.__traceback__ is not None
