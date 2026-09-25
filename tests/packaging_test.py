import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _name(requirement):
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]


def test_pyproject_separates_user_extras_from_contributor_groups():
    pyproject = _pyproject()
    dependencies = {_name(requirement) for requirement in pyproject["project"]["dependencies"]}
    extras = pyproject["project"]["optional-dependencies"]
    groups = pyproject["dependency-groups"]

    assert {"numpy", "scipy", "tqdm", "matplotlib", "ipywidgets"} <= dependencies
    assert "jupyterlab" not in dependencies
    assert set(extras) == {"yaml", "notebooks", "plot3d"}
    assert {"jupyterlab"} <= {_name(requirement) for requirement in extras["notebooks"]}
    assert {"mayavi", "PySide6"} <= {_name(requirement) for requirement in extras["plot3d"]}
    assert any(requirement.startswith("pytest") for requirement in groups["test"])
    assert "myst-nb" in groups["docs"]
    assert "ruff" in groups["dev"]
    assert "biolgca[notebooks]" in groups["dev"]
    assert {"include-group": "test"} in groups["dev"]
    assert {"include-group": "docs"} in groups["dev"]


def test_user_dependencies_declare_the_tested_minimum_versions():
    # The CI lowest-versions job installs exactly these floors; without them pip may pick ancient releases.
    project = _pyproject()["project"]
    requirements = project["dependencies"] + [
        requirement for extra in project["optional-dependencies"].values() for requirement in extra
    ]

    assert all(">=" in requirement for requirement in requirements), requirements


def test_uv_lock_and_python_pin_are_committed():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    pinned = (ROOT / ".python-version").read_text(encoding="utf-8").strip()

    assert 'name = "biolgca"' in lock
    assert pinned.startswith("3.")
    assert f"Programming Language :: Python :: {pinned}" in _pyproject()["project"]["classifiers"]


def test_ci_workflow_tests_the_locked_environment():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "- aidevelop" in workflow
    assert "- kio/development" in workflow
    assert "pull_request:" in workflow
    assert "astral-sh/setup-uv" in workflow
    assert "uv sync --locked" in workflow
    assert "python -m pytest" in workflow


def test_ci_tests_every_advertised_python_version_and_the_minimum_dependencies():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    project = _pyproject()["project"]
    advertised = [classifier.rsplit(" :: ", 1)[1] for classifier in project["classifiers"]
                  if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", classifier)]
    matrix = re.search(r"python-version: \[(.*)\]", workflow).group(1)

    assert [version.strip(' "') for version in matrix.split(",")] == advertised
    assert project["requires-python"] == f">={advertised[0]}"
    assert "--resolution lowest-direct" in workflow
    assert f'python-version: "{advertised[0]}"' in workflow


def test_readme_has_ci_badge():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "actions/workflows/ci.yml/badge.svg" in readme
