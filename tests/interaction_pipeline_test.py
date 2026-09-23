import numpy as np
import pytest

from lgca import get_lgca
from lgca.model import (
    AnalysisSpec,
    Description,
    ModelSpec,
    SpaceSpec,
    StateSpec,
    TimeSpec,
    build_model,
    run_model,
)
from lgca.pipeline import (
    BirthDeathSpec,
    InteractionPipelineSpec,
    NativePhenotypeSwitchOperator,
    PhenotypeSwitchSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)
from lgca.plugins import PluginInfo, ReorientationOperator
from lgca.pipeline import NativeBirthDeathOperator
from lgca.simulation import NodeRecorder
from lgca.simulation import DensityRecorder, PopulationRecorder


def test_native_uniform_reorientation_preserves_total_mass():
    spec = ModelSpec(
        description=Description(title="native uniform reorientation"),
        space=SpaceSpec(geometry="square", dims=(5, 5), boundary="periodic"),
        state=StateSpec(density=0.55, restchannels=1),
        time=TimeSpec(steps=1, seed=23),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[ReorientationTermSpec(name="random_walk", beta=1.0)]
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)

    assert result.lgca.nodes_t[0].sum() == result.lgca.nodes_t[1].sum()
    assert result.metadata["reorientation_term_names"] == ["random_walk"]


def test_native_birth_death_changes_total_mass_within_capacity():
    nodes = np.zeros((3, 3, 5), dtype=bool)
    nodes[1, 1, 0] = True
    spec = ModelSpec(
        description=Description(title="native birth death"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=30),
        dynamics=InteractionPipelineSpec(
            operators=[
                BirthDeathSpec(
                    name="birth_death",
                    parameters={"birth_rate": 1.0, "death_rate": 0.0, "capacity": 2},
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)

    assert result.metadata["operator_names"] == ["birth_death"]
    assert result.lgca.nodes_t[0].sum() == 1
    assert result.lgca.nodes_t[1].sum() == 2
    assert result.lgca.nodes_t[1].sum(axis=-1).max() <= 2


def test_single_species_birth_death_lattice_kernel_preserves_local_capacity():
    nodes = np.zeros((2, 3, 5), dtype=bool)
    nodes[..., 0] = True
    operator = NativeBirthDeathOperator(
        {"birth_rate": 1.0, "death_rate": 0.0, "capacity": 3}
    )

    result = operator._apply_single_species_lattice(
        nodes,
        birth_rate=1.0,
        death_rate=0.0,
        rng=np.random.default_rng(37),
        capacity=3,
    )

    assert result.shape == nodes.shape
    assert result.dtype == nodes.dtype
    np.testing.assert_array_equal(result.sum(axis=-1), np.full((2, 3), 2))
    assert result.sum(axis=-1).max() <= 3


def test_native_birth_death_supports_species_specific_rates():
    nodes = np.zeros((3, 3, 2, 5), dtype=bool)
    nodes[1, 1, 0, 0] = True
    nodes[1, 1, 1, 1] = True
    spec = ModelSpec(
        description=Description(title="native multispecies birth death"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1, n_species=2),
        time=TimeSpec(steps=1, seed=31),
        dynamics=InteractionPipelineSpec(
            operators=[
                BirthDeathSpec(
                    name="birth_death",
                    parameters={
                        "birth_rate": [1.0, 0.0],
                        "death_rate": [0.0, 1.0],
                        "capacity": 4,
                    },
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    species_counts = result.lgca.nodes_t[1].sum(axis=(0, 1, 3))

    assert species_counts.tolist() == [2, 0]
    assert result.lgca.nodes_t[1].sum(axis=(2, 3)).max() <= 4


def test_native_reorientation_preserves_multispecies_mass_by_species():
    spec = ModelSpec(
        description=Description(title="native multispecies reorientation"),
        space=SpaceSpec(geometry="square", dims=(5, 5), boundary="periodic"),
        state=StateSpec(density=0.75, restchannels=1, n_species=2),
        time=TimeSpec(steps=1, seed=24),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(name="random_walk", beta=1.0),
                        ReorientationTermSpec(name="resting_bias", beta=0.25),
                    ]
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    before = result.lgca.nodes_t[0].sum(axis=(0, 1, 3))
    after = result.lgca.nodes_t[1].sum(axis=(0, 1, 3))

    np.testing.assert_array_equal(after, before)
    assert result.metadata["reorientation_term_names"] == ["random_walk", "resting_bias"]


@pytest.mark.parametrize("geometry,dims,neighbors", [
    ("lin", (5,), 2), ("square", (5, 5), 4),
    ("hex", (5, 6), 6), ("cubic", (5, 5, 5), 6),
])
def test_nematic_scores_count_neighbors_and_ignore_empty_extra_species(geometry, dims, neighbors):
    from lgca.pipeline import _NematicAlignmentTerm

    lgca = get_lgca(geometry=geometry, dims=dims, density=0,
                    interaction="only_propagation")
    lgca.nodes[..., 0] = True
    coord = tuple(lgca.r_int + 2 for _ in dims)
    neighbor = (coord[0] + 1,) + coord[1:]
    lgca.nodes[neighbor + (1,)] = True
    candidates = np.eye(lgca.K, dtype=bool)
    term = _NematicAlignmentTerm(ReorientationTermSpec(name="nematic_alignment"))
    expected = neighbors * (lgca.c.T @ lgca.c[:, 0]) ** 2
    expected += (lgca.c.T @ lgca.c[:, 1]) ** 2
    term.prepare(lgca, lgca.nodes)
    single = term.score(candidates, lgca.nodes[coord], lgca, coord)
    np.testing.assert_allclose(single, expected)
    lgca._reorientation_source_nodes = np.stack(
        [lgca.nodes, np.zeros_like(lgca.nodes)], axis=-2
    )
    lgca.n_species = 2
    term.prepare(lgca, lgca._reorientation_source_nodes.sum(axis=-2))
    multiple = term.score(candidates, lgca.nodes[coord], lgca, coord)
    np.testing.assert_allclose(multiple, expected)


@pytest.mark.parametrize("propagation", [False, True])
def test_custom_rest_or_align_conserves_empty_partial_and_full_sites(propagation):
    nodes = np.array([[False, False, False], [True, False, False],
                      [True, True, True], [False, True, True]])
    spec = ModelSpec(
        space=SpaceSpec(geometry="lin", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=4, seed=12),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "custom.rest_or_align"}], propagation=propagation,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )
    result = run_model(spec, showprogress=False)
    history = result.lgca.nodes_t
    np.testing.assert_array_equal(history.sum(axis=(1, 2)), 6)
    if not propagation:
        np.testing.assert_array_equal(history.sum(axis=-1), np.tile([0, 1, 3, 2], (5, 1)))


@pytest.mark.parametrize("name", ["persistent_walk", "resting_bias", "nematic_alignment", "aggregation"])
def test_single_species_scoped_terms_match_unscoped_seeded_evolution(name):
    results = []
    for species in (None, 0):
        result = run_model(ModelSpec(
            space=SpaceSpec(geometry="square", dims=(4, 4)),
            state=StateSpec(density=1, restchannels=1),
            time=TimeSpec(steps=4, seed=117),
            dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
                ReorientationTermSpec(name, beta=5, species=species)
            ])]),
            analysis=AnalysisSpec(observers=[NodeRecorder()]),
        ), showprogress=False)
        results.append(result.lgca.nodes_t)
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("term,field", [
    (ReorientationTermSpec("persistent_walk", parameters={"betta": 100}), "parameters"),
    (ReorientationTermSpec("persistent_walk", species=-1), "species"),
    (ReorientationTermSpec("persistent_walk", species=True), "species"),
    (ReorientationTermSpec("persistent_walk", species=.5), "species"),
    (ReorientationTermSpec("persistent_walk", beta=np.nan), "beta"),
    (ReorientationTermSpec("persistent_walk", beta=np.inf), "beta"),
    (ReorientationTermSpec("persistent_walk", beta="2"), "beta"),
])
def test_composed_term_invalid_fields_fail_independently(term, field):
    with pytest.raises(ValueError, match=rf"terms\[0\].*{field}"):
        build_model(ModelSpec(dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[term])])))


@pytest.mark.parametrize("propagation", ["flase", 1, 0, [], {}])
def test_pipeline_rejects_invalid_propagation(propagation):
    with pytest.raises(ValueError, match="dynamics.propagation"):
        build_model(ModelSpec(dynamics=InteractionPipelineSpec(propagation=propagation)))


def test_opposite_neighbors_distinguish_polar_and_nematic_scores():
    from lgca.pipeline import _PolarAlignmentTerm, _NematicAlignmentTerm

    lgca = get_lgca(geometry="square", dims=(3, 3), density=0, interaction="only_propagation")
    lgca.nodes[1, 2, 0] = True  # east
    lgca.nodes[3, 2, 2] = True  # west
    candidates = np.eye(4, dtype=bool)
    for kind, name, expected in (
        (_PolarAlignmentTerm, "polar_alignment", [0, 0, 0, 0]),
        (_NematicAlignmentTerm, "nematic_alignment", [2, 0, 2, 0]),
    ):
        term = kind(ReorientationTermSpec(name))
        term.prepare(lgca, lgca.nodes)
        np.testing.assert_allclose(term.score(candidates, lgca.nodes[2, 2], lgca, (2, 2)), expected)


def test_nematic_alignment_term_favors_neighbor_axis_in_one_sampler():
    nodes = np.zeros((3, 3, 5), dtype=bool)
    nodes[1, 1, 1] = True
    nodes[0, 1, 0] = True
    nodes[2, 1, 2] = True
    spec = ModelSpec(
        description=Description(title="native nematic alignment"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=33),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(name="random_walk", beta=1.0),
                        ReorientationTermSpec(name="nematic_alignment", beta=50.0),
                    ]
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    center_after = result.lgca.nodes_t[1, 1, 1]

    assert center_after[:4].sum() == 1
    assert center_after[0] or center_after[2]
    assert result.metadata["reorientation_term_names"] == ["random_walk", "nematic_alignment"]


def test_persistent_walk_term_favors_previous_local_flux():
    nodes = np.zeros((3, 3, 5), dtype=bool)
    nodes[1, 1, 0] = True
    spec = ModelSpec(
        description=Description(title="native persistent walk"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=35),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(name="random_walk", beta=1.0),
                        ReorientationTermSpec(name="persistent_walk", beta=50.0),
                    ]
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    center_after = result.lgca.nodes_t[1, 1, 1]

    assert center_after[:4].sum() == 1
    assert center_after[0]
    assert result.metadata["reorientation_term_names"] == ["random_walk", "persistent_walk"]


def test_aggregation_term_favors_density_gradient():
    nodes = np.zeros((3, 3, 5), dtype=bool)
    nodes[1, 1, 1] = True
    nodes[2, 1, 0] = True
    nodes[2, 1, 1] = True
    spec = ModelSpec(
        description=Description(title="native aggregation"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=36),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(name="random_walk", beta=1.0),
                        ReorientationTermSpec(name="aggregation", beta=50.0),
                    ]
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    center_after = result.lgca.nodes_t[1, 1, 1]

    assert center_after[:4].sum() == 1
    assert center_after[0]
    assert result.metadata["reorientation_term_names"] == ["random_walk", "aggregation"]


def test_contact_guidance_term_requires_director_field():
    spec = ModelSpec(
        description=Description(title="missing contact guidance director"),
        space=SpaceSpec(geometry="square", dims=(3, 3), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=1, seed=37),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[ReorientationTermSpec(name="contact_guidance", beta=1.0)]
                )
            ],
            propagation=False,
        ),
    )

    with pytest.raises(ValueError, match=r"state\.fields\.director"):
        build_model(spec)


def test_contact_guidance_term_favors_director_axis():
    nodes = np.zeros((3, 3, 5), dtype=bool)
    nodes[1, 1, 1] = True
    director = np.zeros((3, 3, 2), dtype=float)
    director[..., 0] = 1.0
    spec = ModelSpec(
        description=Description(title="native contact guidance"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1, fields={"director": director}),
        time=TimeSpec(steps=1, seed=38),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(name="random_walk", beta=1.0),
                        ReorientationTermSpec(name="contact_guidance", beta=50.0),
                    ]
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    center_after = result.lgca.nodes_t[1, 1, 1]

    assert center_after[:4].sum() == 1
    assert center_after[0] or center_after[2]
    assert result.metadata["reorientation_term_names"] == ["random_walk", "contact_guidance"]


def test_phenotype_switch_preserves_total_mass_and_changes_species():
    nodes = np.zeros((3, 3, 2, 5), dtype=bool)
    nodes[1, 1, 0, 0] = True
    nodes[1, 2, 0, 4] = True
    spec = ModelSpec(
        description=Description(title="phenotype switch"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1, n_species=2),
        time=TimeSpec(steps=1, seed=25),
        dynamics=InteractionPipelineSpec(
            operators=[
                PhenotypeSwitchSpec(
                    name="phenotype_switch",
                    parameters={"rates": [[0.0, 1.0], [0.0, 0.0]]},
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    before = result.lgca.nodes_t[0].sum(axis=(0, 1, 3))
    after = result.lgca.nodes_t[1].sum(axis=(0, 1, 3))

    assert before.tolist() == [2, 0]
    assert after.tolist() == [0, 2]
    assert result.lgca.nodes_t[1].sum() == result.lgca.nodes_t[0].sum()


def test_phenotype_switch_collision_is_one_atomic_conserved_transition():
    nodes = np.zeros((1, 1, 2, 4), dtype=bool)
    nodes[0, 0, :, 0] = True
    spec = ModelSpec(
        description=Description(title="collision-heavy phenotype switch"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, n_species=2),
        time=TimeSpec(steps=1, seed=0),
        dynamics=InteractionPipelineSpec(
            operators=[
                PhenotypeSwitchSpec(
                    name="phenotype_switch",
                    parameters={"rates": [[0.0, 1.0], [0.0, 0.0]]},
                )
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)
    final_state = result.lgca.nodes_t[1, 0, 0]

    assert final_state.sum() == 2
    assert final_state.sum(axis=1).tolist() == [0, 2]


@pytest.mark.parametrize("mask", range(16))
@pytest.mark.parametrize(
    "rates",
    [
        np.zeros((2, 2)),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.35], [0.65, 0.0]]),
    ],
)
def test_phenotype_switch_preserves_every_two_species_two_channel_ve_state(mask, rates):
    state = np.array([(mask >> bit) & 1 for bit in range(4)], dtype=bool).reshape(2, 2)

    for seed in range(4):
        result = NativePhenotypeSwitchOperator._sample_state(
            state, rates, np.random.default_rng(seed)
        )

        assert result.shape == state.shape
        assert result.dtype == state.dtype
        assert result.sum() == state.sum()


@pytest.mark.parametrize("order", [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
def test_saturated_phenotype_switch_preserves_forbidden_species_under_relabeling(order):
    state = np.array([[True, False], [True, True], [False, False]])
    rates = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    order = np.asarray(order)
    for seed in range(32):
        result = NativePhenotypeSwitchOperator._sample_state(
            state[order], rates[np.ix_(order, order)], np.random.default_rng(seed)
        )
        # The sole permitted destination is full: every attempted switch stays.
        np.testing.assert_array_equal(result, state[order])
        assert result.dtype == bool
        assert result.sum() == 3


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_shared_birth_capacity_has_exchangeable_competition(order):
    from lgca.pipeline import NativeBirthDeathOperator

    operator = NativeBirthDeathOperator()
    operator.birth_rate = np.array([1., 1., 0.])[list(order)]
    operator.death_rate = np.zeros(3)
    operator.capacity = 3
    node = np.array([[True, False], [True, False], [False, False]])[list(order)]
    lattice = np.broadcast_to(node, (2000,) + node.shape).copy()

    result = operator._apply_multispecies_lattice(lattice, np.random.default_rng(108))

    assert result.dtype == bool
    assert (result.sum(axis=(-2, -1)) == 3).all()
    counts = result.sum(axis=-1)[:, np.argsort(order)]
    assert (counts[:, 2] == 0).all()
    winners = (counts == 2).sum(axis=0)
    # Each identical species wins with probability 1/2 (six standard errors).
    assert winners[0] / 2000 == pytest.approx(.5, abs=6 * np.sqrt(.25 / 2000))
    assert winners[1] == 2000 - winners[0]


def test_phenotype_switch_zero_rates_leave_complete_state_unchanged():
    for state in (
        np.array([[True, False], [False, True]]),
        np.array([[2, 0], [1, 3]], dtype=np.int64),
    ):
        result = NativePhenotypeSwitchOperator._sample_state(
            state, np.zeros((2, 2)), np.random.default_rng(7)
        )

        assert np.array_equal(result, state)


def test_phenotype_switch_forced_transition_changes_species_without_losing_particle():
    state = np.array([[True, False], [False, False]])
    rates = np.array([[0.0, 1.0], [0.0, 0.0]])

    result = NativePhenotypeSwitchOperator._sample_state(
        state, rates, np.random.default_rng(9)
    )

    assert result.sum(axis=1).tolist() == [0, 1]


def test_phenotype_switch_nove_samples_one_complete_conserved_state():
    state = np.array([[3, 1], [2, 4]], dtype=np.int64)
    rates = np.array([[0.0, 1.0], [1.0, 0.0]])

    result = NativePhenotypeSwitchOperator._sample_state(
        state, rates, np.random.default_rng(11)
    )

    assert result.shape == state.shape
    assert result.dtype == state.dtype
    assert result.sum() == state.sum()
    assert result.sum(axis=1).tolist() == [6, 4]


def test_phenotype_switch_one_particle_frequency_matches_rate_matrix():
    state = np.array([[True, False], [False, False]])
    rates = np.array([[0.0, 0.3], [0.0, 0.0]])
    rng = np.random.default_rng(1234)

    switched = sum(
        NativePhenotypeSwitchOperator._sample_state(state, rates, rng)[1].sum()
        for _ in range(5000)
    )

    assert switched / 5000 == pytest.approx(0.3, abs=0.03)


def test_native_phenotype_switch_can_be_created_from_registry_name():
    nodes = np.zeros((3, 3, 2, 5), dtype=bool)
    nodes[1, 1, 0, 0] = True
    spec = ModelSpec(
        description=Description(title="registry phenotype switch"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1, n_species=2),
        time=TimeSpec(steps=1, seed=32),
        dynamics=InteractionPipelineSpec(
            operators=[
                {
                    "name": "phenotype_switch",
                    "parameters": {"rates": [[0.0, 1.0], [0.0, 0.0]]},
                }
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)

    assert result.metadata["operator_names"] == ["phenotype_switch"]
    assert result.lgca.nodes_t[1].sum(axis=(0, 1, 3)).tolist() == [0, 1]


def test_operator_order_is_validated_before_simulation():
    spec = ModelSpec(
        description=Description(title="bad order"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=1, seed=26),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(terms=[ReorientationTermSpec(name="random_walk")]),
                BirthDeathSpec(name="classical.birth"),
            ],
            propagation=False,
        ),
    )

    with pytest.raises(ValueError, match=r"dynamics\.operators\[1\].*order"):
        build_model(spec)


def test_pipeline_refreshes_density_between_birth_and_go_or_rest():
    nodes = np.zeros((1, 1, 5), dtype=bool)
    nodes[0, 0, 1] = True
    spec = ModelSpec(
        description=Description(title="dynamic field freshness"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=0),
        dynamics=InteractionPipelineSpec(
            operators=[
                {"name": "classical.birth", "parameters": {"r_b": 1.0}},
                {
                    "name": "classical.go_or_rest",
                    "parameters": {"kappa": 50.0, "theta": 0.3},
                },
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(),
    )
    composed = build_model(spec)
    explicitly_refreshed = build_model(spec)

    composed.pipeline.execute_step(composed.context, 1)
    explicitly_refreshed.pipeline.operators[0].apply(explicitly_refreshed.context, 1)
    explicitly_refreshed.lgca.update_dynamic_fields()
    explicitly_refreshed.pipeline.operators[1].apply(explicitly_refreshed.context, 1)
    explicitly_refreshed.lgca.apply_boundaries()
    explicitly_refreshed.lgca.update_dynamic_fields()

    actual = composed.lgca.nodes[composed.lgca.nonborder].reshape(-1)
    reference = explicitly_refreshed.lgca.nodes[
        explicitly_refreshed.lgca.nonborder
    ].reshape(-1)
    np.testing.assert_array_equal(actual, reference)
    assert actual.astype(int).tolist() == [0, 1, 0, 0, 1]


def test_pipeline_refreshes_species_density_after_phenotype_switch():
    class SpeciesDensityReader(ReorientationOperator):
        def __init__(self):
            super().__init__(
                PluginInfo(
                    name="test.species_density_reader",
                    operator_kind="reorientation",
                    backend_families=("multispecies",),
                )
            )
            self.observed = None
            self.observed_boundary = None

        def dependencies(self):
            return {"boundary_nodes", "species_density"}

        def outputs(self):
            return set()

        def apply(self, context, step):
            self.observed = context.lgca.species_density[
                context.lgca.nonborder
            ].copy()
            self.observed_boundary = context.lgca.nodes[-1, -1, 1].copy()

    nodes = np.zeros((1, 1, 2, 4), dtype=bool)
    nodes[0, 0, 0, 0] = True
    reader = SpeciesDensityReader()
    spec = ModelSpec(
        description=Description(title="species density freshness"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, n_species=2),
        time=TimeSpec(steps=1, seed=0),
        dynamics=InteractionPipelineSpec(
            operators=[
                PhenotypeSwitchSpec(
                    name="phenotype_switch",
                    parameters={"rates": [[0.0, 1.0], [0.0, 0.0]]},
                ),
                reader,
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(),
    )

    compiled = build_model(spec)
    compiled.pipeline.execute_step(compiled.context, 1)

    assert reader.observed.reshape(-1, 2).tolist() == [[0, 1]]
    assert reader.observed_boundary.sum() == 1


def test_unknown_reorientation_term_reports_modelspec_path():
    spec = ModelSpec(
        description=Description(title="unknown term"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=1, seed=27),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(terms=[ReorientationTermSpec(name="missing_term")]),
            ],
            propagation=False,
        ),
    )

    with pytest.raises(ValueError, match=r"dynamics\.operators\[0\]\.terms\[0\].*missing_term"):
        build_model(spec)


def test_chemotaxis_term_requires_named_static_field():
    spec = ModelSpec(
        description=Description(title="missing chemotaxis field"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1),
        time=TimeSpec(steps=1, seed=28),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=1.0,
                            parameters={"field": "signal"},
                        )
                    ]
                ),
            ],
            propagation=False,
        ),
    )

    with pytest.raises(ValueError, match=r"state\.fields\.signal"):
        build_model(spec)


def test_chemotaxis_term_uses_static_field_dependency_in_schedule():
    signal = np.arange(16, dtype=float).reshape(4, 4)
    spec = ModelSpec(
        description=Description(title="chemotaxis field"),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(density=0.25, restchannels=1, fields={"signal": signal}),
        time=TimeSpec(steps=1, seed=29),
        dynamics=InteractionPipelineSpec(
            operators=[
                ReorientationSpec(
                    terms=[
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=1.0,
                            parameters={"field": "signal"},
                        )
                    ]
                ),
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)

    assert result.lgca.nodes_t[0].sum() == result.lgca.nodes_t[1].sum()
    assert "inputs=signal" in result.pipeline.describe_schedule()


def test_target_example_shape_runs_with_species_specific_reorientation_terms():
    signal = np.arange(16, dtype=float).reshape(4, 4)
    spec = ModelSpec(
        description=Description(
            title="Two-phenotype migration model",
            details=(
                "Birth/death and phenotype switching followed by species-specific "
                "Boltzmann reorientation terms, then deterministic propagation."
            ),
        ),
        space=SpaceSpec(geometry="square", dims=(4, 4), boundary="periodic"),
        state=StateSpec(
            density=0.4,
            restchannels=1,
            n_species=2,
            fields={"signal": signal},
        ),
        time=TimeSpec(steps=2, seed=34),
        dynamics=InteractionPipelineSpec(
            operators=[
                BirthDeathSpec(
                    name="birth_death",
                    parameters={"birth_rate": 0.1, "death_rate": 0.0},
                ),
                PhenotypeSwitchSpec(
                    name="phenotype_switch",
                    parameters={"rates": [[0.0, 0.1], [0.0, 0.0]]},
                ),
                ReorientationSpec(
                    sampler="boltzmann",
                    terms=[
                        ReorientationTermSpec(
                            name="nematic_alignment",
                            beta=2.0,
                            species=0,
                        ),
                        ReorientationTermSpec(
                            name="chemotaxis",
                            beta=1.5,
                            species=1,
                            parameters={"field": "signal"},
                        ),
                    ],
                ),
            ],
            propagation="default",
        ),
        analysis=AnalysisSpec(
            observers=[
                NodeRecorder(),
                DensityRecorder(),
                PopulationRecorder(),
            ],
        ),
    )

    result = run_model(spec, showprogress=False)

    assert result.metadata["operator_names"] == [
        "birth_death",
        "phenotype_switch",
        "reorientation.boltzmann",
    ]
    assert result.metadata["reorientation_term_names"] == ["nematic_alignment", "chemotaxis"]
    assert result.lgca.nodes_t.shape[0] == 3
    assert result.lgca.dens_t.shape[0] == 3
    assert result.lgca.n_t.shape[0] == 3


def test_native_moore_nematic_uses_safe_lazy_tensor_permutations():
    nodes = np.zeros((2, 2, 2, 26), dtype=bool)
    nodes[0, 0, 0, 0] = True
    spec = ModelSpec(
        description=Description(title="lazy Moore nematic"),
        space=SpaceSpec(geometry="moore", boundary="periodic"),
        state=StateSpec(nodes=nodes),
        time=TimeSpec(steps=1, seed=3),
        dynamics=InteractionPipelineSpec(
            operators=[{"name": "classical.nematic", "parameters": {"beta": 1.0}}],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)

    assert result.lgca.nodes_t.shape == (2, 2, 2, 2, 26)
    assert result.lgca.nodes_t[1].sum() == 1


def test_native_contact_guidance_supports_safe_lazy_square_configuration():
    nodes = np.zeros((2, 2, 16), dtype=bool)
    nodes[0, 0, 0] = True
    spec = ModelSpec(
        description=Description(title="lazy square contact guidance"),
        space=SpaceSpec(geometry="square", boundary="periodic"),
        state=StateSpec(nodes=nodes, restchannels=12),
        time=TimeSpec(steps=1, seed=4),
        dynamics=InteractionPipelineSpec(
            operators=[
                {"name": "classical.contact_guidance", "parameters": {"beta": 1.0}}
            ],
            propagation=False,
        ),
        analysis=AnalysisSpec(observers=[NodeRecorder()]),
    )

    result = run_model(spec, showprogress=False)

    assert result.lgca.nodes_t[1].sum() == 1


def test_combinatorial_permutation_request_fails_before_allocation(monkeypatch):
    import lgca.base as base_module

    lgca = get_lgca(
        geometry="moore",
        dims=(2, 2, 2),
        density=0,
        interaction="only_propagation",
    )
    lgca.calc_permutations()
    called = False

    def fail_if_called(*args):
        nonlocal called
        called = True
        raise AssertionError("unsafe permutation generation was attempted")

    monkeypatch.setattr(base_module, "_generate_permutations", fail_if_called)

    with pytest.raises(ValueError, match="combinatorial permutation request"):
        lgca.get_permutations(13)
    assert not called


def test_lazy_permutation_cache_evicts_by_bytes(monkeypatch):
    import lgca.base as base_module

    monkeypatch.setattr(base_module, "_MAX_LAZY_CACHE_BYTES", 10)
    cache = {0: np.zeros(8, dtype=np.uint8)}

    base_module.LGCA_base._store_bounded_cache(
        cache, 1, np.ones(8, dtype=np.uint8)
    )

    assert list(cache) == [1]


def test_multispecies_turnover_matches_its_rates():
    from lgca.pipeline import NativeBirthDeathOperator

    rng = np.random.default_rng(3)
    operator = NativeBirthDeathOperator()
    operator.capacity = 100  # never limiting

    operator.death_rate, operator.birth_rate = np.array([0.2, 0.5]), np.zeros(2)
    full = np.ones((4000, 2, 4), dtype=bool)
    survivors = operator._apply_multispecies_lattice(full, rng).mean(axis=(0, 2))
    np.testing.assert_allclose(survivors, [0.8, 0.5], atol=0.01)

    operator.death_rate, operator.birth_rate = np.zeros(2), np.array([0.1, 0.4])
    one_cell = np.zeros((20000, 2, 4), dtype=bool)
    one_cell[..., 0] = True
    births = operator._apply_multispecies_lattice(one_cell, rng).sum(axis=-1).mean(axis=0) - 1
    np.testing.assert_allclose(births, [0.1, 0.4], atol=0.01)
