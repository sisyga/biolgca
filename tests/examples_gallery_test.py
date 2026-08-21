import pytest
import jsonschema
import numpy as np
from dataclasses import replace

import lgca.model as model_api
from lgca.model import load_model_spec, model_spec_from_json, model_spec_to_dict, model_spec_to_json, run_model
from lgca.examples import example_names


def test_example_gallery_lists_beginner_model_cards_by_category():
    from lgca.examples import describe_example, example_gallery

    cards = example_gallery()
    names = {card.name for card in cards}

    assert {"random_walk", "alignment", "chemotaxis"}.issubset(names)
    assert all(card.title and card.question and card.category for card in cards)
    assert all(card.concepts for card in cards)

    guidance = example_gallery(category="guidance")
    assert [card.name for card in guidance] == ["chemotaxis", "contact_guidance"]

    chemotaxis = describe_example("chemotaxis")
    assert chemotaxis.category == "guidance"
    assert "signal" in chemotaxis.question.lower()

    with pytest.raises(ValueError, match=r"Unknown example category 'misguided'.*guidance"):
        example_gallery(category="misguided")


def test_example_names_preserves_sorted_api_order():
    from lgca.examples import example_names

    assert example_names() == tuple(sorted(example_names()))


def test_run_example_uses_stable_name_and_allows_tiny_live_runs():
    from lgca.examples import run_example

    result = run_example("random_walk", steps=1, showprogress=False)

    assert result.metadata["title"] == "Random walk example"
    assert result.metadata["steps"] == 1
    assert result.lgca.total_population() >= 0


def test_save_example_spec_writes_loadable_model_config(tmp_path):
    from lgca.examples import save_example_spec

    path = save_example_spec("alignment", tmp_path / "alignment.json")
    loaded = load_model_spec(path)

    assert path.exists()
    assert loaded.description.title == "Alignment example"
    assert loaded.time.seed == 102


def test_every_curated_example_is_explicitly_portable():
    from lgca.examples import example_gallery

    assert all(card.portable is True for card in example_gallery())


def test_model_spec_schema_is_valid_and_accepts_every_curated_example():
    from lgca.examples import all_example_specs

    schema = model_api.load_model_spec_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    for name, spec in all_example_specs().items():
        errors = list(validator.iter_errors(model_spec_to_dict(spec)))
        assert errors == [], f"{name}: {[error.message for error in errors]}"


@pytest.mark.parametrize("example_name", example_names())
def test_portable_examples_round_trip_and_reproduce_seeded_run(example_name):
    from lgca.examples import get_example_spec

    spec = get_example_spec(example_name)
    spec = replace(spec, time=replace(spec.time, steps=1))
    loaded = model_spec_from_json(model_spec_to_json(spec))

    original = run_model(spec, showprogress=False)
    replay = run_model(loaded, showprogress=False)

    np.testing.assert_array_equal(replay.lgca.nodes, original.lgca.nodes)
