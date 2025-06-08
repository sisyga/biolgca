import sys
import os

# Add the project root directory to sys.path
# os.path.dirname(__file__) gives the directory of conftest.py (i.e., tests/)
# os.path.join(..., '..') goes one level up to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib.util
import pytest

HAS_MAYAVI = importlib.util.find_spec("mayavi") is not None


def pytest_runtest_setup(item):
    if not HAS_MAYAVI and "cubic" in item.name:
        pytest.skip("mayavi not installed", allow_module_level=False)

