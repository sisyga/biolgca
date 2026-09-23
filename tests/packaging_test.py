from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_separates_user_extras_from_contributor_groups():
    pyproject = _pyproject()
    dependencies = set(pyproject["project"]["dependencies"])
    extras = pyproject["project"]["optional-dependencies"]
    groups = pyproject["dependency-groups"]

    assert {"numpy", "scipy", "tqdm", "matplotlib", "jupyterlab"} <= dependencies
    assert set(extras) == {"yaml", "plot3d"}
    assert "mayavi" in extras["plot3d"]
    assert any(requirement.startswith("pytest") for requirement in groups["test"])
    assert "myst-nb" in groups["docs"]
    assert "ruff" in groups["dev"]
    assert {"include-group": "test"} in groups["dev"]
    assert {"include-group": "docs"} in groups["dev"]


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


def test_project_and_ci_advertise_python_313_support():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "Programming Language :: Python :: 3.13" in _pyproject()["project"]["classifiers"]
    assert '"3.13"' in workflow


def test_readme_has_ci_badge():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "actions/workflows/ci.yml/badge.svg" in readme
