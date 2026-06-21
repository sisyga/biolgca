from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_defines_issue_86_extras():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert {"matplotlib", "mayavi"} <= set(extras["plot"])
    assert "ruff" in extras["dev"]


def test_ci_workflow_matches_issue_87_acceptance():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "- aidevelop" in workflow
    assert "- kio/development" in workflow
    assert "pull_request:" in workflow
    assert 'python -m pip install -e ".[dev]"' in workflow
    assert "python -m pytest" in workflow


def test_project_and_ci_advertise_python_313_support():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "Programming Language :: Python :: 3.13" in pyproject["project"]["classifiers"]
    assert '"3.13"' in workflow


def test_readme_has_ci_badge():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "actions/workflows/ci.yml/badge.svg" in readme
