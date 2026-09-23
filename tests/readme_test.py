"""The README code runs as written and its sweep shows the documented effect."""

import re
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture
def readme_namespace():
    matplotlib.use("Agg")
    printed = []
    namespace = {"print": printed.append}
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)
    from lgca.plugins import default_registry

    snapshot = dict(default_registry._plugins), dict(default_registry._aliases)
    try:
        for block in blocks:
            exec(compile(textwrap.dedent(block), str(README), "exec"), namespace)  # noqa: S102
            plt.close("all")
        yield namespace, printed
    finally:
        default_registry._plugins, default_registry._aliases = snapshot


def test_readme_code_runs_and_crowding_death_reduces_population(readme_namespace):
    _, printed = readme_namespace
    populations = [int(re.search(r": (\d+) cells", line).group(1)) for line in printed]
    assert len(populations) == 3
    assert populations[0] > populations[1] > populations[2]
