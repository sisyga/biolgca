"""Go-or-grow with a switch per cell (go_or_rest with kappa="kappa") matches the legacy
identity-based rules in distribution, and the cell operations behind it keep labels and traits."""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.testing import check_interaction

DIMS = (60, 60)
REST = {True: 6, False: 1}
CAPACITY = 12


def _initial(ve, seed=4):
    """Random states over the whole density range: node i holds about i % 13 cells."""
    rng = np.random.default_rng(seed)
    channels = 6 + REST[ve]
    level = (np.arange(np.prod(DIMS)) % 13).reshape(DIMS) / 12
    if ve:
        nodes = rng.random(DIMS + (channels,)) < level[..., None]
    else:
        nodes = rng.poisson(1.5 * level[..., None], DIMS + (channels,))
    kappa = rng.uniform(-4, 6, int(nodes.sum()))  # one value per cell, in label order
    return nodes, kappa


def _model(nodes, kappa, operators, traits=None, seed=1):
    ve = nodes.dtype == bool
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="hex", dims=DIMS),
        state=StateSpec(nodes=nodes, restchannels=REST[ve], volume_exclusion=ve, identity_based=True,
                        traits=traits or {}, **({} if ve else {"capacity": CAPACITY})),
        time=TimeSpec(steps=1, seed=seed),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=False),
    ))


def _pipeline(r_b, r_d, when_full="legacy"):
    return [{"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": 0.5, "when_full": when_full}},
            {"name": "go_or_grow.growth", "parameters": {"r_b": r_b, "r_d": r_d, "when_full": when_full}},
            {"name": "random_walk", "parameters": {"channels": "velocity"}}]


def _summary(lgca):
    """Moving cells, resting cells and the summed kappa of resting cells per node."""
    inner = lgca.nodes[lgca.nonborder]
    kappa = np.asarray(lgca.props["kappa"], dtype=float)
    velocity = lgca.velocitychannels
    if inner.dtype == object:
        counts = lgca._channel_counts(inner)
        rest_kappa = np.array([kappa[[label for channel in node[velocity:] for label in channel]].sum()
                               for node in inner.reshape(-1, lgca.K).tolist()]).reshape(DIMS)
    else:
        counts = inner > 0
        rest_kappa = np.where(counts[..., velocity:], kappa[inner[..., velocity:].astype(int)], 0).sum(-1)
    return counts[..., :velocity].sum(-1), counts[..., velocity:].sum(-1), rest_kappa


def _assert_same_by_density(density, expected, measured):
    for a, b in zip(expected, measured):
        for level in np.unique(density):
            at_level = density == level
            if at_level.sum() < 50:
                continue
            error = np.sqrt((a[at_level].var() + b[at_level].var()) / at_level.sum())
            assert abs(a[at_level].mean() - b[at_level].mean()) < 4 * error + 1e-9, level


@pytest.mark.parametrize("ve", [True, False])
def test_one_step_with_a_kappa_per_cell_matches_the_legacy_rule(ve):
    # without death, the legacy order (death first) and the classical order agree
    nodes, kappa = _initial(ve)
    legacy = _model(nodes, kappa, [{
        "name": "legacy.ib.go_or_grow" if ve else "legacy.nove_ib.go_or_grow",
        "parameters": {"r_b": 0.3, "r_d": 0.0, "kappa": kappa.tolist(), "theta": 0.5,
                       "kappa_std": 0.0, "theta_std": 0.0}}])
    new = _model(nodes, kappa, _pipeline(0.3, 0.0), traits={"kappa": kappa})
    legacy.step()
    new.step()
    density = nodes.sum(-1)
    _assert_same_by_density(density, _summary(legacy.lgca), _summary(new.lgca))


def _cells(lgca):
    """Label, channel and node density of every cell."""
    inner = lgca.nodes[lgca.nonborder].reshape(-1, lgca.K)
    density = lgca._channel_counts(inner).sum(-1)
    labels, channels, nodes = [], [], []
    for node, channels_of_node in enumerate(inner.tolist()):
        for channel, content in enumerate(channels_of_node):
            for label in (content if isinstance(content, list) else [content] if content else []):
                labels.append(label), channels.append(channel), nodes.append(node)
    order = np.argsort(labels)
    return (np.asarray(labels)[order], np.asarray(channels)[order],
            density[np.asarray(nodes)[order]])


@pytest.mark.parametrize("ve", [True, False])
def test_every_cell_rests_with_its_own_probability(ve):
    # where no channel is full, a cell rests after the switch with probability
    # (1 + tanh(kappa (rho - theta))) / 2 for its own kappa, whether it rested before or not
    from lgca.builtin_rules import tanh_switch

    nodes, kappa = _initial(ve)
    model = _model(nodes, kappa, [_pipeline(0.0, 0.0)[0]], traits={"kappa": kappa})
    lgca = model.lgca
    labels, _, density = _cells(lgca)
    capacity = lgca.K if ve else CAPACITY
    model.step()
    after_labels, channel, _ = _cells(lgca)
    np.testing.assert_array_equal(after_labels, labels)
    resting = channel >= lgca.velocitychannels
    cell_kappa = np.asarray(lgca.props["kappa"])[labels]
    p = tanh_switch(density / capacity, cell_kappa, 0.5)
    sparse = density <= 3 if ve else density >= 0  # with VE: at most 3 cells, no channel set is full
    for low, high in ((-4, -1), (-1, 2), (2, 6)):
        cells = sparse & (cell_kappa >= low) & (cell_kappa < high)
        expected = p[cells].mean()
        error = np.sqrt((p[cells] * (1 - p[cells])).sum()) / cells.sum()
        assert abs(resting[cells].mean() - expected) < 4 * error, (low, high)


@pytest.mark.parametrize("ve", [True, False])
def test_daughters_inherit_and_mutate_traits_and_can_found_families(ve):
    nodes, _ = _initial(ve)
    model = _model(nodes, None, [
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": 0.5}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": 0.5, "r_d": 0.0, "mutation": {"kappa": 0.1},
                                                     "new_family": True}},
    ], traits={"kappa": 2.0, "theta": 0.5})
    assert type(model.lgca.maxlabel) is int  # a NumPy integer + 1 is a float with NumPy 1.x
    first_new = model.lgca.maxlabel + 1
    model.step()
    lgca = model.lgca
    born = lgca.maxlabel + 1 - first_new
    assert born > 200
    kappa, theta = np.asarray(lgca.props["kappa"]), np.asarray(lgca.props["theta"])
    family = np.asarray(lgca.props["family"])
    assert len(kappa) == len(theta) == len(family) == lgca.maxlabel + 1
    np.testing.assert_array_equal(theta[first_new:], 0.5)  # inherited unchanged
    change = kappa[first_new:] - 2.0  # mutated: normal with standard deviation 0.1
    assert abs(change.mean()) < 4 * 0.1 / np.sqrt(born)
    assert abs(change.std() - 0.1) < 4 * 0.1 / np.sqrt(2 * born)
    assert len(set(family[first_new:])) == born  # every daughter founded a family
    ancestors = np.asarray(lgca.family_props["ancestor"])[family[first_new:]]
    assert np.all(ancestors == 1)  # all initial cells belong to family 1
    assert lgca.num_families_alive() == born + 1


@pytest.mark.parametrize("rule, parameters", [
    ("go_or_rest", {"kappa": "kappa", "theta": 0.5}),
    ("go_or_rest", {"kappa": 4.0, "theta": "theta", "when_full": "reject"}),
    ("go_or_grow.growth", {"r_b": "r_b", "r_d": 0.05, "mutation": {"r_b": 0.01}}),
    ("go_or_grow.growth", {"r_b": 0.3, "r_d": "r_d", "when_full": "reject", "new_family": True}),
    ("random_walk", {"channels": "velocity"}),
])
def test_rules_pass_the_interaction_check_on_identity_models(rule, parameters):
    check_interaction(rule, parameters, families=("ib", "nove_ib"), n_species=(1,),
                      traits={"kappa": 4.0, "theta": 0.5, "r_b": 0.3, "r_d": 0.05})


def test_a_trait_name_needs_an_identity_based_model():
    with pytest.raises(ValueError, match="names a cell trait"):
        build_model(ModelSpec(
            space=SpaceSpec(geometry="square", dims=(4, 4)), state=StateSpec(density=1, restchannels=1),
            dynamics=InteractionPipelineSpec(operators=[
                {"name": "go_or_rest", "parameters": {"kappa": "kappa"}}]))).step()


@pytest.mark.parametrize("ve", [True, False])
def test_new_families_are_recorded_and_plotted(ve):
    import matplotlib

    matplotlib.use("Agg")
    from lgca.model import AnalysisSpec, run_model
    from lgca.simulation import FamilyPopulationRecorder

    operators = [
        {"name": "go_or_rest", "parameters": {"kappa": "kappa", "theta": 0.5}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.02, "mutation": {"kappa": 0.3},
                                                     "new_family": True}},
        {"name": "random_walk", "parameters": {"channels": "velocity"}},
    ]
    result = run_model(ModelSpec(
        space=SpaceSpec(geometry="hex", dims=(10, 10)),
        state=StateSpec(density=2, restchannels=3 if ve else 1, volume_exclusion=ve, identity_based=True,
                        traits={"kappa": 2.0}, **({} if ve else {"capacity": 8})),
        time=TimeSpec(steps=10, seed=2), dynamics=InteractionPipelineSpec(operators=operators),
        analysis=AnalysisSpec(observers=[FamilyPopulationRecorder()])), showprogress=False)
    lgca = result.lgca
    assert np.asarray(lgca.fam_pop_t).shape == (11, lgca.maxfamily + 1)
    np.testing.assert_array_equal(np.asarray(lgca.fam_pop_t)[-1], lgca.calc_family_pop_alive())
    assert lgca.num_families_total() == lgca.maxfamily > 10
    lgca.muller_plot()


@pytest.mark.filterwarnings("ignore:The interaction name:FutureWarning")
def test_decorated_growth_combines_with_a_legacy_identity_growth_name():
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(20, 20)),
        state=StateSpec(density=1, restchannels=2, identity_based=True),
        time=TimeSpec(steps=1, seed=3),
        dynamics=InteractionPipelineSpec(operators=[
            {"name": "ib.birthdeath"},
            {"name": "go_or_grow.growth", "parameters": {"r_b": 0.1, "r_d": 0.01}},
            {"name": "random_walk"}])))
    for _ in range(10):
        model.step()
    lgca = model.lgca
    assert len(lgca.props["r_b"]) == lgca.maxlabel + 1


def test_traits_need_one_value_per_initial_cell():
    nodes = np.zeros((4, 4, 5), dtype=bool)
    nodes[0, 0, :3] = True
    spec = ModelSpec(space=SpaceSpec(geometry="square", dims=(4, 4)),
                     state=StateSpec(nodes=nodes, restchannels=1, identity_based=True,
                                     traits={"kappa": [1.0, 2.0, 3.0]}),
                     dynamics=InteractionPipelineSpec(operators=[]))
    np.testing.assert_array_equal(np.asarray(build_model(spec).lgca.props["kappa"])[1:], [1, 2, 3])
    with pytest.raises(ValueError, match="2 values, but there are 3 initial cells"):
        build_model(ModelSpec(space=spec.space, dynamics=spec.dynamics, state=StateSpec(
            nodes=nodes, restchannels=1, identity_based=True, traits={"kappa": [1.0, 2.0]})))
    with pytest.raises(ValueError, match="needs an identity-based model"):
        build_model(ModelSpec(space=spec.space, dynamics=spec.dynamics, state=StateSpec(
            density=1, traits={"kappa": 1.0})))
