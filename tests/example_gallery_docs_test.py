from pathlib import Path

from lgca.examples import example_gallery


DOCS_SOURCE = Path(__file__).parents[1] / "docs" / "source"


def test_beginner_example_gallery_page_is_in_sphinx_docs():
    page = DOCS_SOURCE / "example_gallery.rst"
    index = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")

    assert page.exists()
    assert "example_gallery" in index


def test_beginner_example_gallery_documents_all_curated_examples_and_helpers():
    text = (DOCS_SOURCE / "example_gallery.rst").read_text(encoding="utf-8")

    for card in example_gallery():
        assert card.name in text
        assert card.category in text
        assert card.question in text
        assert card.source_path in text
        assert f".. literalinclude:: ../../{card.source_path}" in text
        assert f"title: {card.title}" in text

    assert "build_spec" in text
    assert "run_model" in text
    assert "save_example_spec" in text
    assert "python -m lgca.examples.random_walk" in text
    assert text.count("Example output from ``run(steps=1)``") == len(example_gallery())
    assert "BioLGCA.ipynb" in text
    assert "Morpheus" in text


def test_example_info_api_page_has_explicit_toctree_owner():
    text = (DOCS_SOURCE / "full_api.rst").read_text(encoding="utf-8")

    assert "lgca.examples.ExampleInfo" in text
