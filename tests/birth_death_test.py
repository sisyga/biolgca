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
    legacy = _births(nodes, {"name": "legacy.classical.birthdeath", "parameters": {"r_b": 0.4, "r_d": 0.2}}, seed=2)
    _assert_same_distribution(new, legacy, spread=False)


def test_identity_births_match_the_legacy_rule():
    labels = _labels(np.random.default_rng(0).random(DIMS + (5,)) < 0.4)
    new = _births(labels, {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": 0.2}},
                  identity=True, traits={"r_b": 0.4})
    legacy = _births(labels, {"name": "legacy.ib.birthdeath", "parameters": {"r_b": 0.4, "r_d": 0.2, "std": 1e-9,
                                                                      "a_max": 1.0}}, identity=True, seed=2)
    _assert_same_distribution(new, legacy)


def test_identity_births_without_volume_exclusion_match_the_legacy_rule():
    lists = _lists(np.random.default_rng(0).poisson(0.8, DIMS + (5,)))
    options = {"ve": False, "identity": True, "capacity": 10}
    new = _births(lists, {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": 0.2}},
                  traits={"r_b": 0.4}, **options)
    legacy = _births(lists, {"name": "legacy.nove_ib.birthdeath", "parameters": {"r_b": 0.4, "r_d": 0.2, "std": 1e-9,
                                                                         "a_max": 1.0}}, seed=2, **options)
    _assert_same_distribution(new, legacy)


def test_species_births_with_mutations_match_the_legacy_rule():
    nodes = np.random.default_rng(0).poisson(0.4, DIMS + (2, 5))
    matrix = [[0.9, 0.1], [0.2, 0.8]]
    options = {"ve": False, "capacity": 8, "n_species": 2}
    new = _births(nodes, {"name": "birth_death", "parameters": {"birth_rate": [0.3, 0.6], "death_rate": 0.2,
                                                                "mutation_matrix": matrix}}, **options)
    legacy = _births(nodes, {"name": "legacy.multispecies.birthdeath", "parameters": {"r_b": [0.3, 0.6], "r_d": 0.2,
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
def test_a_redrawn_normal_change_is_a_truncated_normal(ve):
    effect = {"distribution": "normal", "scale": 0.1, "bounds": [0.0, 0.5], "at_bounds": "redraw"}
    model = _identity_model(ve, {"birth_rate": 1.0, "mutation": {"r": effect}}, {"r": 0.45})
    first = model.lgca.maxlabel + 1
    model.step()
    values = _daughter_values(model, "r", first)
    assert len(values) > 5000 and values.min() >= 0 and values.max() <= 0.5
    from scipy.stats import truncnorm

    law = truncnorm((0 - 0.45) / 0.1, (0.5 - 0.45) / 0.1, loc=0.45, scale=0.1)
    assert abs(values.mean() - law.mean()) < 4 * law.std() / np.sqrt(len(values))
    assert abs(values.std() - law.std()) < 0.05 * law.std()


@pytest.mark.parametrize("ve", [True, False])
def test_kinds_of_mutation_apply_independently_in_order(ve):
    # passengers subtract an exponential effect, then drivers add 0.02; values stay <= 0.5
    mutation = [
        {"probability": 0.3, "traits": {"r": {"distribution": "exponential", "scale": 0.01,
                                              "operation": "subtract", "bounds": [None, 0.5]}}},
        {"probability": 0.2, "traits": {"r": {"value": 0.02, "bounds": [None, 0.5]}}},
    ]
    model = _identity_model(ve, {"birth_rate": 1.0, "mutation": mutation}, {"r": 0.49})
    first = model.lgca.maxlabel + 1
    model.step()
    change = np.round(_daughter_values(model, "r", first) - 0.49, 12)
    n = len(change)
    assert change.max() <= 0.01
    # at 0.5: a driver alone, or after a passenger effect below 0.01 (probability 1 - 1/e)
    at_bound = 0.2 * (0.7 + 0.3 * (1 - np.exp(-1)))
    below = 0.3 * (0.8 + 0.2 * np.exp(-1))  # a passenger, and no driver that brings the value back
    for observed, p in (((change == 0.01).mean(), at_bound), ((change == 0).mean(), 0.7 * 0.8),
                        ((change < 0).mean() + ((change > 0) & (change < 0.01)).mean(), below)):
        assert abs(observed - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_a_fixed_effect_can_multiply_and_a_registered_function_can_draw_it():
    from lgca import mutation_effect

    @mutation_effect
    def two_values(rng, size, low=0.5, high=2.0):
        return np.where(rng.random(size) < 0.5, low, high)

    model = _identity_model(False, {"birth_rate": 1.0, "mutation": {
        "r": {"function": "two_values", "operation": "multiply"}, "s": {"value": 3.0, "operation": "multiply"}}},
        {"r": 0.4, "s": 2.0})
    first = model.lgca.maxlabel + 1
    model.step()
    r, s = _daughter_values(model, "r", first), _daughter_values(model, "s", first)
    assert set(np.round(r, 9)) == {0.2, 0.8} and abs((r < 0.5).mean() - 0.5) < 0.03
    np.testing.assert_allclose(s, 6.0)


@pytest.mark.parametrize("ve", [True, False])
def test_mutated_daughters_found_families(ve):
    model = _identity_model(ve, {"birth_rate": 1.0, "new_family": True,
                                 "mutation": {"probability": 0.2, "traits": {}}}, {})
    lgca = model.lgca
    lgca.init_families(type="homogeneous", mutation=True)
    first, families = lgca.maxlabel + 1, lgca.maxfamily
    model.step()
    family = _daughter_values(model, "family", first)
    founded = family > families
    assert abs(founded.mean() - 0.2) < 4 * np.sqrt(0.16 / len(family))
    assert len(np.unique(family[founded])) == founded.sum() == lgca.maxfamily - families
    ancestors = np.asarray(lgca.family_props["ancestor"])[family[founded]]
    mothers = set(np.unique(family[~founded]))
    assert len(mothers) == 1 and set(ancestors) == mothers  # descend from the mother's family


def test_the_rule_passes_the_interaction_checks():
    check_interaction(birth_death, {"birth_rate": 0.3, "death_rate": 0.1})
    check_interaction(birth_death, {"birth_rate": 0.3, "crowding": False})
    check_interaction(birth_death, {"birth_rate": "r_b", "death_rate": 0.1, "mutation": {"r_b": 0.01},
                                    "new_family": True}, families=("ib", "nove_ib"), traits={"r_b": 0.3})


@pytest.mark.parametrize("parameters, identity, message", [
    ({"birth_rate": "r_b"}, False, "names a cell trait"),
    ({"mutation": {"r_b": 0.1}}, False, "only identity-based models"),
    ({"birth_rate": [0.1, 0.2, 0.3]}, False, "one per species"),
    ({"mutation_matrix": [[0.5, 0.4], [0, 1]]}, False, "sum to 1"),
    ({"mutation_matrix": [[0, 1], [1, 0]]}, True, "classical model with several species"),
    ({"mutation": {"r_b": {"distribution": "normal", "value": 0.1}}}, True, "exactly one of"),
    ({"mutation": {"r_b": {"distribution": "bit_generator"}}}, True, "must name a distribution"),
    ({"mutation": {"r_b": {"function": "nowhere"}}}, True, "is not registered"),
    ({"mutation": {"probability": 2, "traits": {}}}, True, "must be a probability"),
    ({"mutation": {"r_b": {"value": 1, "operation": "divide"}}}, True, "must be one of"),
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


def _rest_counts(model):
    lgca = model.lgca
    counts = lgca._channel_counts(lgca.nodes[lgca.nonborder]).astype(int)
    return counts[..., :lgca.velocitychannels], counts[..., lgca.velocitychannels:]


@pytest.mark.parametrize("ve, identity", [(True, False), (False, False), (True, True), (False, True)])
def test_a_channel_set_limits_who_dies_and_divides_and_where_daughters_go(ve, identity):
    # go-or-grow: only resting cells divide, into rest channels; the logistic factor counts the cells of
    # the set with volume exclusion (4 rest channels) and all cells of the node without (one rest channel)
    rng = np.random.default_rng(5)
    rest = 4 if ve else 1
    if ve:
        occupied = rng.random(DIMS + (4 + rest,)) < 0.4
        nodes = _labels(occupied) if identity else occupied
    else:
        counts = rng.poisson(0.6, DIMS + (4 + rest,))
        nodes = _lists(counts) if identity else counts
    model = _model(nodes, {"name": "birth_death", "parameters": {
        "birth_rate": 0.5, "death_rate": 0.1, "channels": "rest"}}, ve=ve, identity=identity, rest=rest,
        capacity=None if ve else 12, seed=3)
    moving, resting = _rest_counts(model)
    model.step()
    moving_after, resting_after = _rest_counts(model)
    np.testing.assert_array_equal(moving_after, moving)  # moving cells neither die nor divide
    n_rest = resting.sum(-1)
    change = resting_after.sum(-1) - n_rest
    crowding = 1 - n_rest / rest if ve else 1 - (moving.sum(-1) + n_rest) / 12
    expected = 0.5 * n_rest * np.clip(crowding, 0, 1) - 0.1 * n_rest
    error = np.sqrt(np.maximum(change.var(), 1e-12) / change.size)
    assert abs(change.mean() - expected.mean()) < 4 * error


def test_a_channel_set_without_volume_exclusion_changes_nothing_else():
    counts = np.random.default_rng(6).poisson(1.0, DIMS + (5,))
    model = _model(counts, {"name": "birth_death", "parameters": {"birth_rate": 0.3, "channels": [0, 1]}},
                   ve=False, capacity=20)
    before = model.lgca.nodes[model.lgca.nonborder].copy()
    model.step()
    after = model.lgca.nodes[model.lgca.nonborder]
    np.testing.assert_array_equal(after[..., 2:], before[..., 2:])
    assert np.all(after[..., :2] >= before[..., :2]) and after[..., :2].sum() > before[..., :2].sum()


# birth_rate and death_rate that respond to cues (lgca.switching)

def _levels_model(family_ve, identity, parameters, n_nodes=20000, levels=5, n_species=1, traits=None):
    """One cell per node (and species) in channel 0 of a 1D lattice; the field u has ``levels`` values."""
    K = 3
    occupied = np.zeros((n_nodes,) + ((n_species,) if n_species > 1 else ()) + (K,), dtype=bool)
    occupied[..., 0] = True
    if identity:
        nodes = _labels(occupied) if family_ve else _lists(occupied.astype(int))
    else:
        nodes = occupied if family_ve else occupied.astype(int)
    level = np.arange(n_nodes) * levels // n_nodes
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="lin", dims=(n_nodes,)),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=family_ve, identity_based=identity,
                        capacity=None if family_ve else 4, n_species=n_species, traits=traits or {},
                        fields={"u": level / (levels - 1)}),
        time=TimeSpec(steps=1, seed=21),
        dynamics=InteractionPipelineSpec(operators=[{"name": "birth_death", "parameters": parameters}],
                                         propagation=False)))
    return model, level, (K if family_ve else 4) * (n_species if family_ve and not identity else 1)


def _assert_one_cell_per_level(after, level, p_birth, p_death, capacity):
    """One cell per node: after one step 0, 1 or 2 cells, with the probabilities of the node's level."""
    for value in np.unique(level):
        u = value / level.max()
        q, d = p_birth(u) * (1 - 1 / capacity), p_death(u)
        expected = np.array([d * (1 - q), (1 - d) * (1 - q) + d * q, (1 - d) * q])
        at = after[level == value]
        observed = np.bincount(at, minlength=3)[:3] / at.size
        np.testing.assert_allclose(observed, expected, atol=4 * np.sqrt(0.25 / at.size), err_msg=str(u))


FORMS = {
    "tanh": ({"max": 0.8, "cues": [{"name": "field", "field": "u", "kappa": 4.0, "theta": 0.5}]},
             lambda u: 0.8 * (1 + np.tanh(4.0 * (u - 0.5))) / 2),
    "boltzmann": ({"rate": 0.5, "cues": [{"name": "field", "field": "u", "beta": 2.0}]},
                  lambda u: 0.5 * np.exp(2 * u) / (1 + 0.5 * np.exp(2 * u))),
    "hill": ({"max": 0.7, "hill": [{"name": "field", "field": "u", "K": 0.4, "n": 2}]},
             lambda u: 0.7 * u ** 2 / (0.16 + u ** 2)),
}
HYPOXIC_DEATH = ({"max": 0.4, "hill": [{"name": "field", "field": "u", "K": 0.3, "n": -2}]},
                 lambda u: 0.4 * 0.09 / (0.09 + u ** 2))


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("family, ve, identity", FAMILIES)
def test_birth_and_death_rates_respond_to_cues(family, ve, identity, form):
    (birth, p_birth), (death, p_death) = FORMS[form], HYPOXIC_DEATH
    model, level, capacity = _levels_model(ve, identity, {"birth_rate": birth, "death_rate": death})
    model.step()
    after = model.lgca.cell_density[model.lgca.nonborder].astype(int)
    _assert_one_cell_per_level(after, level, p_birth, p_death, capacity)


def test_a_rate_that_responds_to_cues_may_read_traits():
    # every cell divides with its own maximal rate and half-saturation constant
    model, level, capacity = _levels_model(False, True, {"birth_rate": {"max": "top", "hill": [
        {"name": "field", "field": "u", "K": "K"}]}}, traits={"K": 0.5, "top": 0.9})
    model.step()
    after = model.lgca.cell_density[model.lgca.nonborder].astype(int)
    _assert_one_cell_per_level(after, level, lambda u: 0.9 * u / (0.5 + u), lambda u: 0.0, capacity)


@pytest.mark.parametrize("ve", [True, False])
def test_species_rates_that_respond_to_cues(ve):
    # species 0 divides by the Hill form, species 1 at a constant rate and dies where u is low; with
    # species=[0] species 1 is left alone
    hill, p_hill = FORMS["hill"]
    parameters = {"birth_rate": [hill, 0.3], "death_rate": [0.0, HYPOXIC_DEATH[0]]}
    # with volume exclusion the room is K = 3 channels per species; without, the capacity 4 for the two
    # cells of the node, which _assert_one_cell_per_level (one cell, 1 - 1/4) gets as a factor on the rate
    room, capacity = (1.0, 3) if ve else ((1 - 2 / 4) / (1 - 1 / 4), 4)
    for species in (None, [0]):
        model, level, _ = _levels_model(ve, False, {**parameters, "species": species}, n_species=2)
        model.step()
        after = model.lgca.nodes[model.lgca.nonborder].sum(-1).astype(int)
        _assert_one_cell_per_level(after[:, 0], level, lambda u: p_hill(u) * room, lambda u: 0.0, capacity)
        if species is None:
            _assert_one_cell_per_level(after[:, 1], level, lambda u: 0.3 * room, HYPOXIC_DEATH[1], capacity)
        else:
            np.testing.assert_array_equal(after[:, 1], 1)


@pytest.mark.parametrize("parameters, identity, message", [
    ({"birth_rate": {"max": 0.5, "cues": [{"name": "field", "field": "u", "kappa": "k"}]}}, False,
     "reads cell traits"),
    ({"birth_rate": [{"max": 0.5, "hill": []}]}, False, r"one per species \(2\)"),
    ({"death_rate": {"max": 1.5, "hill": []}}, False, "must be a probability"),
    ({"death_rate": {"hill": [{"name": "field", "field": "u"}]}}, True, "needs 'K'"),
])
def test_rates_that_respond_to_cues_are_checked(parameters, identity, message):
    model, _, _ = _levels_model(True, identity, parameters, n_nodes=100, n_species=1 if identity else 2)
    with pytest.raises((ValueError, TypeError), match=message):
        model.step()


def test_the_model_graph_shows_the_fields_a_rate_reads():
    from lgca.model import describe_model_graph

    model, _, _ = _levels_model(True, False, {"birth_rate": FORMS["hill"][0], "death_rate": 0.1}, n_nodes=10)
    edges = describe_model_graph(model.spec)["edges"]
    assert {"source": "field:u", "target": "operator:0:birth_death"} in edges
