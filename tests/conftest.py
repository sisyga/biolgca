import os
import pkgutil
import sys
from importlib import import_module

import pytest

# Add the project root directory to sys.path
# os.path.dirname(__file__) gives the directory of conftest.py (i.e., tests/)
# os.path.join(..., '..') goes one level up to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import lgca.zoo
from lgca.fields import _REACTIONS
from lgca.mutations import _EFFECTS
from lgca.pipeline import _REORIENTATION_TERMS, _TERM_ALIASES
from lgca.plugins import default_registry
from lgca.switching import _CUES

# the library's own rules, all registered before the first test (the zoo registers its rules when imported):
# imported inside a test, they would be removed after it with the rules the test registered
for _module in pkgutil.iter_modules(lgca.zoo.__path__, "lgca.zoo."):
    import_module(_module.name)

# the legacy interaction functions, registered as "legacy.<family>.<name>" for comparisons
import tests.legacy  # noqa: F401

_REGISTRIES = (default_registry._plugins, default_registry._aliases, _REORIENTATION_TERMS, _TERM_ALIASES,
               _REACTIONS, _EFFECTS, _CUES)


@pytest.fixture(autouse=True)
def _isolated():
    """Remove what a test registers (rules, terms, reactions, effects, cues) and the pyplot figures it
    leaves open, so that tests do not depend on the order they run in (CI runs them in random order)."""
    saved = [dict(registry) for registry in _REGISTRIES]
    yield
    for registry, entries in zip(_REGISTRIES, saved):
        registry.clear()
        registry.update(entries)
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
