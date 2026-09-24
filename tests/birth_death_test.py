"""birth_death: one growth rule for every model family, validated against the legacy rules.

Cells die and divide at the same time; with volume exclusion a daughter goes to a random channel
(distinct for the dividing cells of a node) and survives if it was empty, without it the division
probability is scaled by 1 - n / capacity. The legacy rules are compared in
distribution after one step: the mean (and where the legacy rule draws the same distribution,
the spread) of the births per node, per number of cells at the node.
"""

import numpy as np
import pytest

from lgca.builtin_rules import birth_death
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.testing import check_interaction

DIMS = (100, 100)


def _model(nodes, operator, geometry="square", ve=True, identity=False, seed=1, capacity=None, traits=None,
           n_species=1, rest=1):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=nodes.shape[:len(nodes.shape) - 1 - (n_species > 1)]),
        state=StateSpec(nodes=nodes, restchannels=rest, volume_exclusion=ve, identity_based=identity,
                        capacity=capacity, traits=traits or {}, n_species=n_species),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[operator], propagation=False)))


def _counts(model, n_species=1):
    lgca = model.lgca
    if n_species > 1:
        return lgca.nodes[lgca.nonborder].sum(-1).astype(int)
    return lgca.cell_density[lgca.nonborder].astype(int)


def _births(nodes, operator, n_species=1, **options):
    model = _model(nodes, operator, n_species=n_species, **options)
    before = _counts(model, n_species)
    model.step()
    return before, _counts(model, n_species) - before


def _labels(occupied):
    return np.where(occupied, np.cumsum(occupied).reshape(occupied.shape), 0).astype(np.uint64)


def _lists(counts):
    nodes = np.empty(counts.shape, dtype=object)
    first = 0
    for index in np.ndindex(counts.shape):
        nodes[index] = list(range(first, first + counts[index]))
        first += counts[index]
    return nodes


def _assert_same_distribution(new, legacy, spread=True):
    (density, a), (density_legacy, b) = new, legacy
    np.testing.assert_array_equal(density, density_legacy)
    total = density.sum(-1) if density.ndim > len(DIMS) else density
    compared = 0
    for level in np.unique(total):
        at = total == level
        if at.sum() < 100:
            continue
        for x, y in ((a[at], b[at]), (a[at] ** 2, b[at] ** 2))[:1 + spread]:
            error = np.sqrt((x.var(0) + y.var(0)) / at.sum())
            assert np.all(np.abs(x.mean(0) - y.mean(0)) <= 4 * error + 1e-12), level
        compared += 1
    assert compared >= 3


def test_classical_births_match_the_legacy_rule_in_the_mean():
    # legacy: every empty channel is filled with probability r_b * n / K, so the variance differs
    nodes = np.random.default_rng(0).random(DIMS + (5,)) < 0.4
    new = _births(nodes, {"name": "birth_death", "parameters": {"birth_rate": 0.4, "death_rate": 0.2}})
    legacy = _births(nodes, {"name": "classical.birthdeath", "parameters": {"r_b": 0.4, "r_d": 0.2}}, seed=2)
    _assert_same_distribution(new, legacy, spread=False)


def test_identity_births_match_the_legacy_rule():
    labels = _labels(np.random.default_rng(0).random(DIMS + (5,)) < 0.4)
    new = _births(labels, {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": 0.2}},
                  identity=True, traits={"r_b": 0.4})
    legacy = _births(labels, {"name": "ib.birthdeath", "parameters": {"r_b": 0.4, "r_d": 0.2, "std": 1e-9,
                                                                      "a_max": 1.0}}, identity=True, seed=2)
    _assert_same_distribution(new, legacy)


def test_identity_births_without_volume_exclusion_match_the_legacy_rule():
    lists = _lists(np.random.default_rng(0).poisson(0.8, DIMS + (5,)))
    options = {"ve": False, "identity": True, "capacity": 10}
    new = _births(lists, {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": 0.2}},
                  traits={"r_b": 0.4}, **options)
    legacy = _births(lists, {"name": "nove_ib.birthdeath", "parameters": {"r_b": 0.4, "r_d": 0.2, "std": 1e-9,
                                                                         "a_max": 1.0}}, seed=2, **options)
    _assert_same_distribution(new, legacy)


def test_species_births_with_mutations_match_the_legacy_rule():
    nodes = np.random.default_rng(0).poisson(0.4, DIMS + (2, 5))
    matrix = [[0.9, 0.1], [0.2, 0.8]]
    options = {"ve": False, "capacity": 8, "n_species": 2}
    new = _births(nodes, {"name": "birth_death", "parameters": {"birth_rate": [0.3, 0.6], "death_rate": 0.2,
                                                                "mutation_matrix": matrix}}, **options)
    legacy = _births(nodes, {"name": "multispecies.birthdeath", "parameters": {"r_b": [0.3, 0.6], "r_d": 0.2,
                                                                              "mutation_matrix": matrix}},
                     seed=2, **options)
    _assert_same_distribution(new, legacy)


@pytest.mark.parametrize("identity", [False, True])
def test_volume_exclusion_gives_logistic_growth(identity):
    # a node with n of K cells changes by r_b n (K - n) / K - r_d n on average, also when nearly full
    occupied = np.random.default_rng(4).random((200, 200, 5)) < 0.7
    nodes = _labels(occupied) if identity else occupied
    parameters = {"birth_rate": "r_b" if identity else 0.4, "death_rate": 0.1}
    density, births = _births(nodes, {"name": "birth_death", "parameters": parameters}, identity=identity,
                              traits={"r_b": 0.4} if identity else None)
    for n in range(1, 5):
        at = density == n
        error = births[at].std() / np.sqrt(at.sum())
        assert abs(births[at].mean() - (0.4 * n * (5 - n) / 5 - 0.1 * n)) < 4 * error, n


FAMILIES = [("classical", True, False), ("nove", False, False), ("ib", True, True), ("nove_ib", False, True)]
GEOMETRIES = [("lin", (4000,)), ("square", (60, 60)), ("hex", (60, 60)), ("cubic", (16, 16, 16)),
              ("moore", (16, 16, 16))]


@pytest.mark.parametrize("family, ve, identity", FAMILIES)
@pytest.mark.parametrize("geometry, dims", GEOMETRIES)
def test_rates_in_every_family_and_geometry(family, ve, identity, geometry, dims):
    # one cell per node: it dies with r_d and, independently, divides with r_b (1 - 1 / capacity)
    K = build_model(ModelSpec(space=SpaceSpec(geometry=geometry, dims=dims),
                              state=StateSpec(density=0, restchannels=1))).lgca.K
    occupied = np.zeros(dims + (K,), dtype=bool)
    occupied[..., 0] = True
    if identity:
        nodes = _labels(occupied) if ve else _lists(occupied.astype(int))
    else:
        nodes = occupied if ve else occupied.astype(int)
    capacity = K if ve else 4
    r_b, r_d = 0.5, 0.2
    parameters = {"birth_rate": "r_b" if identity else r_b, "death_rate": "r_d" if identity else r_d}
    model = _model(nodes, {"name": "birth_death", "parameters": parameters}, geometry=geometry, ve=ve,
                   identity=identity, capacity=None if ve else capacity,
                   traits={"r_b": r_b, "r_d": r_d} if identity else None)
    model.step()
    after = model.lgca.cell_density[model.lgca.nonborder].astype(int).ravel()
    q = r_b * (1 - 1 / capacity)
    expected = np.array([r_d * (1 - q), (1 - r_d) * (1 - q) + r_d * q, (1 - r_d) * q])
    observed = np.bincount(after, minlength=3) / after.size
    np.testing.assert_allclose(observed, expected, atol=4 * np.sqrt(0.25 / after.size))


@pytest.mark.parametrize("family, ve, identity", FAMILIES)
def test_without_crowding_capacity_is_a_hard_limit(family, ve, identity):
    counts = np.random.default_rng(1).integers(0, 2, (30, 30, 5))
    if identity:
        nodes = _labels(counts.astype(bool)) if ve else _lists(counts)
    else:
        nodes = counts.astype(bool) if ve else counts
    model = _model(nodes, {"name": "birth_death", "parameters": {"birth_rate": 1.0, "crowding": False}},
                   ve=ve, identity=identity, capacity=3)
    before = model.lgca.cell_density[model.lgca.nonborder].astype(int)
    model.step()
    after = model.lgca.cell_density[model.lgca.nonborder].astype(int)
    np.testing.assert_array_equal(after, np.maximum(before, np.minimum(2 * before, 3)))


def test_without_crowding_species_share_the_room_at_random():
    nodes = np.zeros((20000, 2, 4), dtype=bool)
    nodes[..., 0] = True  # one cell of each species; room for one more
    model = _model(nodes, {"name": "birth_death", "parameters": {"birth_rate": 1.0, "crowding": False}},
                   geometry="lin", n_species=2, capacity=3, rest=2)
    model.step()
    births = model.lgca.nodes[model.lgca.nonborder].sum(-1) - 1
    np.testing.assert_array_equal(births.sum(-1), 1)
    assert abs(births[:, 0].mean() - 0.5) < 4 * np.sqrt(0.25 / len(births))


def _identity_model(ve, parameters, traits, seed=3):
    counts = np.zeros((100, 100, 5), dtype=int)
    counts[..., 0] = 1
    nodes = _labels(counts.astype(bool)) if ve else _lists(counts)
    return _model(nodes, {"name": "birth_death", "parameters": parameters}, ve=ve, identity=True,
                  capacity=None if ve else 100, traits=traits, seed=seed)


def _daughter_values(model, name, first):
    labels = np.array([label for channel in model.lgca.nodes[model.lgca.nonborder].flat
                       for label in (channel if isinstance(channel, list) else [channel]) if label >= first])
    return np.asarray(model.lgca.props[name])[labels]


@pytest.mark.parametrize("ve", [True, False])
def test_traits_mutate_within_bounds(ve):
    model = _identity_model(ve, {"birth_rate": 1.0, "mutation": {"r": {"std": 0.1, "bounds": [0.0, 0.5]}}},
                            {"r": 0.45})
    first = model.lgca.maxlabel + 1
    model.step()
    values = _daughter_values(model, "r", first)
    assert len(values) > 5000 and values.min() >= 0 and values.max() <= 0.5
    from scipy.stats import truncnorm

    law = truncnorm((0 - 0.45) / 0.1, (0.5 - 0.45) / 0.1, loc=0.45, scale=0.1)
    assert abs(values.mean() - law.mean()) < 4 * law.std() / np.sqrt(len(values))


@pytest.mark.parametrize("ve", [True, False])
def test_traits_mutate_in_steps(ve):
    model = _identity_model(ve, {"birth_rate": 1.0, "mutation": {"r": {"step": 0.1, "probability": 0.3,
                                                                       "bounds": [None, 0.5]}}},
                            {"r": 0.45})
    first = model.lgca.maxlabel + 1
    model.step()
    values = _daughter_values(model, "r", first)
    changed = np.round(values - 0.45, 6)
    assert set(np.unique(changed)) <= {-0.1, 0.0, 0.05}  # a step up is clipped to 0.5
    for step, p in ((-0.1, 0.15), (0.05, 0.15)):
        assert abs((changed == step).mean() - p) < 4 * np.sqrt(p * (1 - p) / len(values))


@pytest.mark.parametrize("ve", [True, False])
def test_daughters_found_families_with_a_probability(ve):
    model = _identity_model(ve, {"birth_rate": 1.0, "new_family": 0.2}, {})
    lgca = model.lgca
    lgca.init_families(type="homogeneous", mutation=True)
    first, families = lgca.maxlabel + 1, lgca.maxfamily
    model.step()
    family = _daughter_values(model, "family", first)
    founded = family > families
    assert abs(founded.mean() - 0.2) < 4 * np.sqrt(0.16 / len(family))
    assert len(np.unique(family[founded])) == founded.sum() == lgca.maxfamily - families
    assert set(family[~founded]) == {0} or set(family[~founded]) == {1}  # the mother's family


def test_the_rule_passes_the_interaction_checks():
    check_interaction(birth_death, {"birth_rate": 0.3, "death_rate": 0.1})
    check_interaction(birth_death, {"birth_rate": 0.3, "crowding": False})
    check_interaction(birth_death, {"birth_rate": "r_b", "death_rate": 0.1, "mutation": {"r_b": 0.01}},
                      families=("ib", "nove_ib"), traits={"r_b": 0.3})


@pytest.mark.parametrize("parameters, identity, message", [
    ({"birth_rate": "r_b"}, False, "names a cell trait"),
    ({"mutation": {"r_b": 0.1}}, False, "only identity-based models"),
    ({"birth_rate": [0.1, 0.2, 0.3]}, False, "one per species"),
    ({"mutation_matrix": [[0.5, 0.4], [0, 1]]}, False, "sum to 1"),
    ({"mutation_matrix": [[0, 1], [1, 0]]}, True, "classical model with several species"),
    ({"mutation": {"r_b": {"std": 0.1, "step": 0.1}}}, True, "must be a standard deviation"),
    ({"new_family": 1.5}, True, "True, False or a probability"),
])
def test_invalid_parameters_are_explained(parameters, identity, message):
    counts = np.ones((4, 4, 5), dtype=int)
    nodes = _labels(counts.astype(bool)) if identity else counts.astype(bool)
    n_species = 2 if "mutation_matrix" in parameters and not identity else 1
    if n_species == 2:
        nodes = np.stack([nodes, nodes], axis=-2)
    model = _model(nodes, {"name": "birth_death", "parameters": {"birth_rate": 0.5, **parameters}},
                   identity=identity, n_species=n_species, traits={"r_b": 0.2} if identity else None)
    with pytest.raises(ValueError, match=message):
        model.step()
