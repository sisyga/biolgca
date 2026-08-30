from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _dependency_name(requirement: str) -> str:
    return requirement.partition(";")[0].strip().split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].lower()


def test_normal_install_includes_notebook_runtime():
    project = _project_metadata()
    names = {_dependency_name(item) for item in project["dependencies"]}

    assert {"matplotlib", "jupyterlab"} <= names
    assert "teaching" not in project["optional-dependencies"]
    assert "notebooks" not in project["optional-dependencies"]


def test_sphinx_force_executes_notebooks():
    conf = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")

    assert "'myst_nb'" in conf or '"myst_nb"' in conf
    assert 'nb_execution_mode = "force"' in conf
    assert "nb_execution_raise_on_error = True" in conf
