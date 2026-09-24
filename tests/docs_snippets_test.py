"""The code on documentation pages runs as written."""

import re
import textwrap
from pathlib import Path

import matplotlib
import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source"


def _python_blocks(path):
    """Code of the ``.. code-block:: python`` directives of an rst page."""
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:(?:[ ]{3}.*)?\n)+)", path.read_text(encoding="utf-8"))
    return [textwrap.dedent(block) for block in blocks]


@pytest.fixture
def clean_registries():
    from lgca.pipeline import _REORIENTATION_TERMS, _TERM_ALIASES
    from lgca.plugins import default_registry

    plugins, aliases = dict(default_registry._plugins), dict(default_registry._aliases)
    terms, term_aliases = dict(_REORIENTATION_TERMS), dict(_TERM_ALIASES)
    yield
    default_registry._plugins, default_registry._aliases = plugins, aliases
    _REORIENTATION_TERMS.clear(), _REORIENTATION_TERMS.update(terms)
    _TERM_ALIASES.clear(), _TERM_ALIASES.update(term_aliases)


@pytest.mark.parametrize("page", ["how_to/custom_interactions.rst"])
def test_page_code_runs(page, clean_registries):
    matplotlib.use("Agg")
    namespace = {"print": lambda *args, **kwargs: None}
    blocks = _python_blocks(DOCS / page)
    assert len(blocks) >= 5
    for block in blocks:
        exec(compile(block, str(DOCS / page), "exec"), namespace)  # noqa: S102

    assert namespace["result"].lgca.nodes.sum() > 0
