"""The research models, rebuilt as stacks of generic rules, reproduce the legacy interactions.

One step from the same state is compared in distribution: the mean number of cells and of
resting cells per node, per number of cells at the node, and the traits and families of
daughters. Nodes at the lattice edge are left out: in a pipeline, the legacy operators see
empty ghost nodes where the rules see the periodic neighbours.
"""

import numpy as np
import pytest

from lgca.model import (
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    model_spec_from_dict,
    model_spec_to_dict,
)
from lgca.pipeline import InteractionPipelineSpec

DIMS = (100, 100)


def _initial(ve, density, seed=0, K=7):
    rng = np.random.default_rng(seed)
    if ve:
        occupied = rng.random(DIMS + (K,)) < density
        return np.where(occupied, np.cumsum(occupied).reshape(occupied.shape), 0).astype(np.uint64)
    counts = rng.poisson(density, DIMS + (K,))
    nodes = np.empty(counts.shape, dtype=object)
    first = 0
    for index in np.ndindex(counts.shape):
        nodes[index] = list(range(first, first + counts[index]))
        first += counts[index]
    return nodes


def _model(name, parameters, nodes, ve, seed, propagation=False):
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="hex", dims=DIMS, boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1, volume_exclusion=ve, identity_based=True,
                        capacity=None if ve else 8),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=[{"name": name, "parameters": parameters}],
                                         propagation=propagation)))


def _step(name, parameters, nodes, ve, seed):
    model = _model(name, parameters, nodes, ve, seed)
    lgca = model.lgca
    before = lgca.cell_density[lgca.nonborder].astype(int).copy()
    first = int(lgca.maxlabel) + 1
    model.step()
    interior = lgca.nodes[lgca.nonborder]
    if ve:
        cells = np.asarray(interior > 0, dtype=int)
        labels = interior[interior > 0].astype(int)
    else:
        cells = np.vectorize(len)(interior)
        labels = np.array([label for channel in interior.flat for label in channel], dtype=int)
    return {"lgca": lgca, "before": before, "total": cells.sum(-1), "rest": cells[..., -1],
            "labels": labels, "daughters": labels[labels >= first]}


def _assert_same_nodes(new, legacy):
    inner = np.zeros(DIMS, dtype=bool)
    inner[1:-1, 1:-1] = True
    compared = 0
    for level in np.unique(new["before"]):
        at = (new["before"] == level) & inner
        if at.sum() < 200:
            continue
        for key in ("total", "rest"):
            x, y = new[key][at], legacy[key][at]
            error = np.sqrt((x.var() + y.var()) / at.sum())
            assert abs(x.mean() - y.mean()) < 4 * error + 1e-12, (key, level)
        compared += 1
    assert compared >= 3


def _trait(result, name, cells):
    lgca = result["lgca"]
    if name in lgca.props:
        return np.asarray(lgca.props[name])[cells]
    families = np.asarray(lgca.props["family"])[cells]  # legacy: the trait of the cell's family
    return np.asarray(lgca.family_props[name])[families]


def _assert_same_mean(a, b):
    error = np.sqrt(a.var() / len(a) + b.var() / len(b))
    assert abs(a.mean() - b.mean()) < 4 * error + 1e-12


def _founded(result):
    """Fraction of daughters in a family that none of the initial cells belongs to."""
    lgca = result["lgca"]
    family = np.asarray(lgca.props["family"])
    initial = family[result["labels"][~np.isin(result["labels"], result["daughters"])]]
    return np.mean(~np.isin(family[result["daughters"]], initial)), len(result["daughters"])


NOVE = {
    "go_or_grow_kappa": {"r_b": 0.3, "r_d": 0.05, "kappa": 4.0, "theta": 0.4},
    "go_or_grow_glioblastoma": {"r_b": 0.3, "r_d": 0.05, "kappa": 4.0, "theta": 0.4, "r_m": 0.2,
                                "fitness_increase": 1.5},
    # kappa = 0: half of the cells rest at any density, which the legacy normalization does not change
    "go_or_grow_kappa_chemo": {"r_b": 0.3, "r_d": 0.05, "kappa": 0.0, "theta": 0.4, "beta": 5.0},
    "birthdeath_cancerdfe": {"r_b": 0.3, "r_d": 0.05, "p_d": 0.2, "p_p": 0.3, "s_d": 0.05, "s_p": 0.03,
                             "a_max": 0.35, "gamma": 1.0},
    "evo_steric": {"r_b": 0.3, "r_d": 0.05, "r_m": 0.2, "fitness_increase": 1.5, "alpha": 1.0, "gamma": 1.0},
}


TRAITS = {"go_or_grow_kappa": ("kappa",), "go_or_grow_glioblastoma": ("r_b", "kappa"), "go_or_grow_kappa_chemo": (),
          "birthdeath_cancerdfe": ("r_b",), "evo_steric": ("r_b",)}


@pytest.mark.parametrize("name", list(NOVE))
def test_models_without_volume_exclusion_match_the_legacy_rules(name):
    nodes = _initial(False, 0.8)
    new = _step(name, NOVE[name], nodes, False, 1)
    legacy = _step(f"legacy.nove_ib.{name}", NOVE[name], nodes, False, 2)
    _assert_same_nodes(new, legacy)
    for trait in TRAITS[name]:
        _assert_same_mean(_trait(new, trait, new["daughters"]), _trait(legacy, trait, legacy["daughters"]))
    if "r_m" in NOVE[name]:
        (p_new, n_new), (p_legacy, n_legacy) = _founded(new), _founded(legacy)
        assert abs(p_new - NOVE[name]["r_m"]) < 4 * np.sqrt(0.16 / n_new)
        assert abs(p_legacy - NOVE[name]["r_m"]) < 4 * np.sqrt(0.16 / n_legacy)


def test_chemotaxis_of_moving_cells_matches_the_legacy_rule():
    from lgca.lattice_state import LatticeState

    nodes = _initial(False, 0.8)
    parameters = {"r_b": 0.0, "r_d": 0.0, "kappa": 0.0, "beta": 8.0}
    scores = []
    for name, seed in (("go_or_grow_kappa_chemo", 1), ("legacy.nove_ib.go_or_grow_kappa_chemo", 2)):
        model = _model(name, parameters, nodes, False, seed)
        lgca = model.lgca
        gradient = LatticeState(lgca).gradient(lgca.cell_density[lgca.nonborder])
        model.step()
        moving = np.vectorize(len)(lgca.nodes[lgca.nonborder])[..., :6]
        scores.append((moving * (gradient @ lgca.c)).sum(-1)[1:-1, 1:-1].ravel())
    _assert_same_mean(*scores)
    assert scores[0].mean() > 0.5  # cells do move up the gradient


VE = {
    "go_and_grow_mutations": {"r_b": 0.4, "r_d": 0.1, "r_m": 0.2, "effect": "driver_mutation",
                              "fitness_increase": 1.5},
    # a_max above every step up: the legacy rule lets r_b + drb exceed a_max, the stack clips it
    "birthdeath_discrete": {"r_b": 0.4, "r_d": 0.1, "drb": 0.05, "pmut": 0.5, "a_max": 1.0},
}


@pytest.mark.parametrize("name", list(VE))
def test_models_with_volume_exclusion_match_the_legacy_rules(name):
    nodes = _initial(True, 0.45)
    new = _step(name, VE[name], nodes, True, 1)
    legacy = _step(f"legacy.ib.{name}", VE[name], nodes, True, 2)
    _assert_same_nodes(new, legacy)
    _assert_same_mean(_trait(new, "r_b", new["daughters"]), _trait(legacy, "r_b", legacy["daughters"]))


@pytest.mark.parametrize("n_species, legacy", [(1, "legacy.classical.excitable_medium"),
                                               (2, "legacy.multispecies.excitable_medium_ms")])
def test_the_excitable_medium_matches_the_legacy_rule(n_species, legacy):
    rng = np.random.default_rng(0)
    nodes = rng.random((150, 150) + ((2,) if n_species == 2 else ()) + (8,)) < 0.3
    if n_species == 2:  # inhibitors (species 0) rest, activators (species 1) move
        nodes[..., 0, :4] = nodes[..., 1, 4:] = False
    counts = []
    for name, seed in (("excitable_medium", 1), (legacy, 2)):
        model = build_model(ModelSpec(
            space=SpaceSpec(geometry="square", dims=(150, 150), boundary="periodic"),
            state=StateSpec(nodes=nodes, restchannels=4, n_species=n_species), time=TimeSpec(steps=1, seed=seed),
            dynamics=InteractionPipelineSpec(operators=[{"name": name, "parameters": {"N": 10}}],
                                             propagation=False)))
        model.step()
        after = model.lgca.nodes[model.lgca.nonborder]
        if n_species == 2:
            counts.append((after[..., 1, :4].sum(-1), after[..., 0, 4:].sum(-1)))
        else:
            counts.append((after[..., :4].sum(-1), after[..., 4:].sum(-1)))
    for new, old in zip(*counts):
        for x, y in ((new, old), (new ** 2, old ** 2)):
            _assert_same_mean(x.ravel().astype(float), y.ravel().astype(float))


@pytest.mark.parametrize("ve", [True, False])
@pytest.mark.parametrize("name", ["go_or_grow_glioblastoma", "evo_steric", "birthdeath_cancerdfe"])
def test_stacks_run_in_both_identity_families_and_are_saved_by_name(name, ve):
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(20, 20)),
                     state=StateSpec(density=0.3 if ve else 1.0, restchannels=1, volume_exclusion=ve,
                                     identity_based=True, capacity=None if ve else 8),
                     time=TimeSpec(steps=5, seed=4),
                     dynamics=InteractionPipelineSpec(operators=[{"name": name, "parameters": {"r_b": 0.25}}]))
    model = build_model(model_spec_from_dict(model_spec_to_dict(spec)))
    assert model.pipeline.operator_names == [name]
    np.testing.assert_allclose(np.asarray(model.lgca.props["r_b"]), 0.25)  # the parameter sets the trait
    model.run(False)


def test_a_state_trait_takes_precedence_over_the_parameter():
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(10, 10)),
                     state=StateSpec(density=1.0, restchannels=1, volume_exclusion=False, identity_based=True,
                                     capacity=8, traits={"kappa": 2.0}),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_grow_kappa",
                                                                  "parameters": {"kappa": 7.0}}]))
    np.testing.assert_allclose(np.asarray(build_model(spec).lgca.props["kappa"]), 2.0)


def test_a_stack_explains_where_it_does_not_apply():
    spec = ModelSpec(state=StateSpec(density=0.3, restchannels=1),
                     dynamics=InteractionPipelineSpec(operators=[{"name": "go_or_grow_kappa"}]))
    with pytest.raises(ValueError, match="written for models identity-based"):
        build_model(spec)


def test_own_stacks_combine_rules_under_one_name():
    from lgca import stack

    @stack(kind="birth_death", families=("ib", "nove_ib"), traits="r_b")
    def grow_and_walk(state, r_b=0.3, r_d=0.1):
        """Cells with their own birth rate grow and walk."""
        return [{"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": r_d}},
                {"name": "random_walk"}]

    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(20, 20)),
                     state=StateSpec(density=0.2, restchannels=1, identity_based=True),
                     time=TimeSpec(steps=10, seed=1),
                     dynamics=InteractionPipelineSpec(operators=[grow_and_walk(r_d=0.0)]))
    model = build_model(spec)
    before = model.lgca.cell_density.sum()
    assert model.pipeline.operators[0].stacked_names == ["birth_death", "random_walk"]
    model.run(False)
    assert model.lgca.cell_density[model.lgca.nonborder].sum() > before
