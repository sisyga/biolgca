"""The code on documentation pages runs as written."""

import re
import textwrap
from pathlib import Path

import matplotlib
import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source"


def _python_blocks(path):
    """Code of the ``.. code-block:: python`` directives of an rst page, after any option lines."""
    blocks = re.findall(r"\.\. code-block:: python\n(?:[ ]{3}:.*\n)*\n((?:(?:[ ]{3}.*)?\n)+)",
                        path.read_text(encoding="utf-8"))
    return [textwrap.dedent(block) for block in blocks]


def test_python_blocks_finds_blocks_with_options(tmp_path):
    page = tmp_path / "page.rst"
    page.write_text(".. code-block:: python\n   :caption: x\n\n   a = 1\n\n.. code-block:: python\n\n   b = 2\n",
                    encoding="utf-8")

    assert _python_blocks(page) == ["a = 1\n\n", "b = 2\n"]


@pytest.mark.parametrize("page, blocks", [("how_to/custom_interactions.rst", 7), ("how_to/research_models.rst", 5),
                                        ("how_to/fields.rst", 6), ("how_to/exploring.rst", 4)])
def test_page_code_runs(page, blocks):
    matplotlib.use("Agg")
    namespace = {"print": lambda *args, **kwargs: None}
    code = _python_blocks(DOCS / page)
    # every directive is run (none skipped by the pattern), and adding a block means updating this test
    assert len(code) == (DOCS / page).read_text(encoding="utf-8").count(".. code-block:: python") == blocks
    for block in code:
        exec(compile(block, str(DOCS / page), "exec"), namespace)  # noqa: S102

    assert namespace["result"].lgca.cell_density.sum() > 0
