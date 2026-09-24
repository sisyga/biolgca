"""Batched categorical sampling preserves scientific and RNG contracts."""

import numpy as np
import pytest

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec


def scalar_reference(lgca, operator):
    source = lgca._reorientation_source_nodes
    for spatial in np.ndindex(lgca.dims):
        coord = tuple(index + lgca.r_int for index in spatial)
        state = source[coord]
        species_states = state if getattr(lgca, "n_species", 1) > 1 else state[None]
        result = []
        for species, node in enumerate(species_states):
            count = int(node.sum())
            if not count:
                result.append(np.zeros_like(node))
                continue
            candidates = lgca.get_permutations(count)
            scores = np.zeros(len(candidates))
            for term in operator.terms:
                if term.species is None or term.species == species:
                    scores += term.beta * term.score(candidates, node, lgca, coord)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            result.append(candidates[lgca.rng.choice(len(candidates), p=probabilities)])
        lgca.nodes[coord] = result if getattr(lgca, "n_species", 1) > 1 else result[0]


@pytest.mark.parametrize("species", [1, 2])
@pytest.mark.parametrize("geometry", ["square", "hex"])
def test_batched_composition_matches_scalar_rng_order_and_small_batches(species, geometry, monkeypatch):
    import lgca.pipeline as pipeline

    channels = 5 if geometry == "square" else 7
    shape = (4, 4, channels) if species == 1 else (4, 4, species, channels)
    nodes = np.random.default_rng(143).random(shape) < .5
    terms = [ReorientationTermSpec(name, beta=.7) for name in (
        "uniform", "resting_bias", "nematic_alignment", "polar_alignment", "persistent_walk", "aggregation")]
    terms += [ReorientationTermSpec("chemotaxis", parameters={"field": "signal"}, species=species - 1),
              ReorientationTermSpec("contact_guidance", parameters={"field": "director"})]
    spec = ModelSpec(space=SpaceSpec(geometry=geometry, dims=(4, 4)),
        state=StateSpec(nodes=nodes, n_species=species, restchannels=1, fields={
            "signal": np.arange(16).reshape(4, 4), "director": np.ones((4, 4, 2))}),
        time=TimeSpec(steps=3, seed=143),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=terms)]))
    expected = build_model(spec)
    operator = expected.pipeline.operators[0]
    monkeypatch.setattr(operator, "_sample_batches", lambda lgca: scalar_reference(lgca, operator))
    expected.run(False)
    for budget in (32 * 1024**2, 35 * 8 * 16):
        monkeypatch.setattr(pipeline, "_MAX_CANDIDATE_BATCH_BYTES", budget)
        actual = build_model(spec)
        actual.run(False)
        np.testing.assert_array_equal(actual.lgca.nodes, expected.lgca.nodes)
        assert actual.lgca.rng.bit_generator.state == expected.lgca.rng.bit_generator.state


def test_rest_bias_has_hand_derived_categorical_distribution():
    nodes = np.zeros((10000, 3), dtype=bool)
    nodes[:, 0] = True
    model = build_model(ModelSpec(space=SpaceSpec(geometry="lin"), state=StateSpec(nodes=nodes, restchannels=1),
        time=TimeSpec(steps=1, seed=143), dynamics=InteractionPipelineSpec(propagation=False,
            operators=[ReorientationSpec(terms=[ReorientationTermSpec("resting_bias", beta=np.log(3))])])) )
    model.run(False)
    result = model.lgca.nodes[model.lgca.nonborder]
    np.testing.assert_array_equal(result.sum(-1), 1)
    observed = result.mean(axis=0)
    expected = np.array([.2, .2, .6])  # Boltzmann weights (1, 1, 3).
    assert np.all(np.abs(observed - expected) < 6 * np.sqrt(expected * (1 - expected) / len(nodes)))


def test_composed_scores_are_bounded_and_candidate_features_reused(monkeypatch):
    import lgca.pipeline as pipeline

    model = build_model(ModelSpec(space=SpaceSpec(geometry="square", dims=(8, 8)),
        state=StateSpec(nodes=np.tile([True, True, False, False], (8, 8, 1))),
        dynamics=InteractionPipelineSpec(operators=[ReorientationSpec(terms=[
            ReorientationTermSpec("aggregation"), ReorientationTermSpec("polar_alignment")])], propagation=False)))
    monkeypatch.setattr(pipeline, "_MAX_CANDIDATE_BATCH_BYTES", 6 * 8 * 6 * 2)
    feature_calls = []
    original_features = pipeline._candidate_features

    def counted(*args):
        feature_calls.append(1)
        return original_features(*args)

    monkeypatch.setattr(pipeline, "_candidate_features", counted)
    batches = []
    term = model.pipeline.operators[0].terms[0]
    original_score = term.score_batch

    def checked(features, lgca, coords):
        batches.append(len(coords[0]))
        assert len(coords[0]) <= 2
        return original_score(features, lgca, coords)

    monkeypatch.setattr(term, "score_batch", checked)
    model.step()
    assert len(feature_calls) == 1
    assert sum(batches) == 64


def test_candidate_feature_budget_rejects_before_allocation(monkeypatch):
    import lgca.pipeline as pipeline

    model = build_model(ModelSpec(space=SpaceSpec(geometry="square", dims=(2, 2)),
        state=StateSpec(density=0), dynamics=InteractionPipelineSpec(operators=[])))
    candidates = model.lgca.get_permutations(2)
    monkeypatch.setattr(pipeline, "_MAX_CANDIDATE_BATCH_BYTES", 1)
    with pytest.raises(ValueError, match="Candidate features require.*reduce channels"):
        pipeline._candidate_features(candidates, model.lgca)
