"""Example modules must run as scripts, both from their folder and with ``python -m``.

Every example is executed in-process by ``examples_gallery_test``; these tests
cover the script entry point shared by all of them.
"""

import subprocess
import sys
from pathlib import Path


def test_example_runs_as_a_script_from_the_examples_folder():
    completed = subprocess.run(
        [sys.executable, "random_walk.py", "--steps", "1"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1] / "lgca" / "examples",
    )

    assert completed.returncode == 0, completed.stderr
    assert "Random walk example" in completed.stdout


def test_example_runs_with_python_m():
    completed = subprocess.run(
        [sys.executable, "-m", "lgca.examples.random_walk", "--steps", "1"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert "Random walk example" in completed.stdout
