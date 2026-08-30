from pathlib import Path

import nbformat

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / "docs" / "source"
TUTORIALS = DOCS_SOURCE / "tutorials"
EXPECTED_NOTEBOOKS = (
    "01_fundamentals.ipynb",
    "02_collective_movement.ipynb",
    "03_combining_interactions.ipynb",
    "04_population_dynamics.ipynb",
    "05_evolutionary_lgca.ipynb",
    "06_student_project.ipynb",
)


def _notebook(name: str):
    return nbformat.read(TUTORIALS / name, as_version=4)


def _source(notebook, cell_type: str | None = None) -> str:
    return "\n".join(
        cell.source
        for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
    )


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
    conf = (DOCS_SOURCE / "conf.py").read_text(encoding="utf-8")

    assert "'myst_nb'" in conf or '"myst_nb"' in conf
    assert 'nb_execution_mode = "force"' in conf
    assert "nb_execution_raise_on_error = True" in conf


def test_primary_documentation_navigation_is_task_oriented():
    index = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
    toctree = index.split(".. toctree::", 1)[1]
    expected = (
        "getting_started",
        "tutorials/index",
        "how_to/index",
        "concepts/index",
        "example_gallery",
        "reference/index",
    )

    positions = [toctree.index(name) for name in expected]
    assert positions == sorted(positions)


def test_reference_pages_have_section_owners():
    assert (DOCS_SOURCE / "how_to" / "model_specs_and_plugins.rst").exists()
    assert (DOCS_SOURCE / "concepts" / "lgca_types.rst").exists()
    assert (DOCS_SOURCE / "reference" / "full_api.rst").exists()


def test_interaction_concepts_page_covers_every_reorientation_term():
    from lgca.pipeline import list_reorientation_terms

    text = (DOCS_SOURCE / "concepts" / "interactions.rst").read_text(encoding="utf-8")
    for name in list_reorientation_terms():
        assert f"``{name}``" in text
    assert "one sampled reorientation transition" in text
    assert "sequential pipeline" in text
    assert "N(s') = N(s)" in text


def test_fundamentals_notebook_teaches_complete_first_spec():
    source = _source(_notebook("01_fundamentals.ipynb"))

    assert "ModelSpec(" in source
    assert "InteractionPipelineSpec(" in source
    assert '"classical.random_walk"' in source
    assert "boundary" in source
    assert "seed" in source
    assert "Exercise" in source


def test_collective_notebook_compares_four_mechanisms():
    source = _source(_notebook("02_collective_movement.ipynb"))

    for name in (
        "classical.random_walk",
        "classical.alignment",
        "classical.aggregation",
        "classical.nematic",
    ):
        assert name in source
    assert "polarization" in source.lower()


def test_composition_notebook_builds_one_multi_term_sampler():
    source = _source(_notebook("03_combining_interactions.ipynb"))

    assert "ReorientationSpec(" in source
    assert source.count("ReorientationTermSpec(") >= 4
    assert "one sampled" in source.lower()
    assert "sequential" in source.lower()
    assert "chemotaxis" in source
    assert "contact_guidance" in source


def test_population_notebook_teaches_atomic_conserving_switch():
    source = _source(_notebook("04_population_dynamics.ipynb"))
    code_source = _source(_notebook("04_population_dynamics.ipynb"), "code")

    assert "BirthDeathSpec(" in code_source
    assert "PhenotypeSwitchSpec(" in code_source
    assert "s -> s'" in source or "s → s′" in source
    assert "before.sum() == after.sum()" in code_source
    population_pipeline = code_source.split("population_spec = ModelSpec", 1)[1]
    assert population_pipeline.index("BirthDeathSpec(") < population_pipeline.index("PhenotypeSwitchSpec(")


def test_maintained_notebook_set_is_exact_and_clean():
    assert tuple(sorted(path.name for path in TUTORIALS.glob("*.ipynb"))) == EXPECTED_NOTEBOOKS
    for name in EXPECTED_NOTEBOOKS:
        notebook = _notebook(name)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert cell.outputs == []


def test_notebooks_show_specs_instead_of_loading_example_specs():
    for name in EXPECTED_NOTEBOOKS:
        code_source = _source(_notebook(name), "code")
        assert "ModelSpec(" in code_source
        assert "InteractionPipelineSpec(" in code_source
        assert "get_example_spec" not in code_source
        assert "lgca.examples" not in code_source


def test_evolutionary_notebook_runs_replicates():
    source = _source(_notebook("05_evolutionary_lgca.ipynb"))

    assert '"ib.birthdeath"' in source
    assert "seeds" in source
    assert "replicate" in source.lower()
    assert "scientific conclusion" in source.lower()


def test_student_project_notebook_saves_spec_and_tests_custom_invariant():
    source = _source(_notebook("06_student_project.ipynb"))

    assert "register_plugin(" in source
    assert "save_model_spec(" in source
    assert "conserv" in source.lower()
    assert "assert" in source
