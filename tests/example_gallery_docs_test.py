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

    assert "run_example" in text
    assert "save_example_spec" in text
    assert "BioLGCA.ipynb" in text
    assert "Morpheus" in text
