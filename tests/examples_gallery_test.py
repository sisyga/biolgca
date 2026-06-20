import pytest

from lgca.model import load_model_spec


def test_example_gallery_lists_beginner_model_cards_by_category():
    from lgca.examples import describe_example, example_gallery

    cards = example_gallery()
    names = {card.name for card in cards}

    assert {"random_walk", "alignment", "chemotaxis"}.issubset(names)
    assert all(card.title and card.question and card.category for card in cards)
    assert all(card.concepts for card in cards)

    guidance = example_gallery(category="guidance")
    assert [card.name for card in guidance] == ["chemotaxis"]

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
