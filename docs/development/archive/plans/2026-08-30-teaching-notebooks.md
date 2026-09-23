# BioLGCA Teaching Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a six-notebook, application-driven BioLGCA curriculum that visibly teaches ModelSpec interaction composition and is the primary path through the Sphinx documentation.

**Architecture:** Keep Sphinx as the single documentation system and use MyST-NB to execute clean `.ipynb` sources during the strict docs build. Every notebook constructs its ModelSpec and interaction pipeline in visible cells; existing examples inform scenarios but are not loaded as finished specifications. Add only one supporting public API, `list_reorientation_terms()`, and retain the existing example gallery as a reference catalog.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Matplotlib, JupyterLab, nbformat, Sphinx, MyST-NB, pytest, setuptools.

**Spec:** `docs/superpowers/specs/2026-08-30-teaching-notebooks-design.md`

## Global Constraints

- Normal project installation must include Matplotlib and JupyterLab; there is no `teaching` or `notebooks` extra.
- Sphinx and MyST-NB remain documentation-build dependencies.
- The maintained notebook set is exactly `01_fundamentals.ipynb` through `06_student_project.ipynb` under `docs/source/tutorials/`.
- Every maintained notebook constructs `ModelSpec`, `InteractionPipelineSpec`, and its interactions in visible cells.
- Maintained notebooks must not import `lgca.examples.*.build_spec()` or `get_example_spec()`.
- Notebook sources contain no saved outputs or execution counts and must run from a fresh kernel without network or external data.
- Default notebook runs use explicit seeds, headless-safe Matplotlib, and complete in at most three minutes total in CI.
- Do not add a GUI, widgets, Jupyter Book, hosted notebook service, or broad plugin-registry redesign.
- Phenotypic switching is taught as an atomic full-channel-state transition `s -> s'` that conserves total particle number.
- On Windows, pytest commands use a unique repository-local `--basetemp` path.

---

### Task 1: Normal notebook dependencies and executable Sphinx configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/source/conf.py`
- Create: `tests/tutorial_notebooks_test.py`

**Interfaces:**
- Consumes: existing PEP 621 project metadata and strict `docs/build.py` entry point.
- Produces: normal dependencies `matplotlib` and `jupyterlab`; docs dependency `myst-nb`; Sphinx configuration that force-executes notebooks and raises cell errors.

- [ ] **Step 1: Write failing dependency and Sphinx configuration tests**

```python
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


def _project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def test_normal_install_includes_notebook_runtime():
    project = _project_metadata()
    names = {item.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].lower()
             for item in project["dependencies"]}
    assert {"matplotlib", "jupyterlab"} <= names
    assert "teaching" not in project["optional-dependencies"]
    assert "notebooks" not in project["optional-dependencies"]


def test_sphinx_force_executes_notebooks():
    conf = (ROOT / "docs/source/conf.py").read_text(encoding="utf-8")
    assert "'myst_nb'" in conf or '"myst_nb"' in conf
    assert 'nb_execution_mode = "force"' in conf
    assert "nb_execution_raise_on_error = True" in conf
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py --basetemp=.pytest-plan-task1`

Expected: FAIL because Matplotlib/JupyterLab and MyST-NB execution settings are absent.

- [ ] **Step 3: Move notebook runtime packages into normal dependencies**

Update `pyproject.toml` so `[project].dependencies` contains:

```toml
dependencies = [
    "numpy",
    "scipy",
    "tqdm",
    "matplotlib",
    "jupyterlab",
]
```

Keep `plot2d` as an empty compatibility extra, and keep only `mayavi` in
`plot3d`, `plot`, and `plotting`. Add `myst-nb` to both `docs` and `dev`; do not
duplicate Matplotlib or JupyterLab in those extras.

- [ ] **Step 4: Enable strict notebook execution in Sphinx**

Add `myst_nb` to `extensions` and configure:

```python
nb_execution_mode = "force"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
```

- [ ] **Step 5: Run focused tests**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py --basetemp=.pytest-plan-task1-pass`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml docs/source/conf.py tests/tutorial_notebooks_test.py
git commit -m "build: install and execute teaching notebooks"
```

---

### Task 2: Public reorientation-term discovery

**Files:**
- Modify: `lgca/pipeline.py`
- Modify: `tests/interaction_pipeline_test.py`
- Modify: `tests/tutorial_notebooks_test.py`

**Interfaces:**
- Consumes: private `_REORIENTATION_TERMS: dict[str, type[_ReorientationTerm]]`.
- Produces: `lgca.pipeline.list_reorientation_terms() -> tuple[str, ...]`, sorted and stable.

- [ ] **Step 1: Write the failing public API test**

```python
from lgca.pipeline import list_reorientation_terms


def test_reorientation_term_names_are_public_and_stable():
    assert list_reorientation_terms() == (
        "aggregation",
        "alignment",
        "chemotaxis",
        "contact_guidance",
        "nematic",
        "nematic_alignment",
        "persistent_motion",
        "persistent_walk",
        "random_walk",
        "resting_bias",
        "uniform",
    )
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `conda run -n biolgca python -m pytest -q tests/interaction_pipeline_test.py::test_reorientation_term_names_are_public_and_stable --basetemp=.pytest-plan-task2`

Expected: FAIL because `list_reorientation_terms` does not exist.

- [ ] **Step 3: Add the minimal public helper**

Immediately after `_REORIENTATION_TERMS`, add:

```python
def list_reorientation_terms() -> tuple[str, ...]:
    """Return the supported reorientation-term names in deterministic order."""

    return tuple(sorted(_REORIENTATION_TERMS))
```

- [ ] **Step 4: Add a documentation-synchronization test contract**

Extend `tests/tutorial_notebooks_test.py` with:

```python
from lgca.pipeline import list_reorientation_terms


def test_interaction_concepts_page_covers_every_reorientation_term():
    text = (ROOT / "docs/source/concepts/interactions.rst").read_text(encoding="utf-8")
    for name in list_reorientation_terms():
        assert f"``{name}``" in text
    assert "one sampled reorientation transition" in text
    assert "sequential pipeline" in text
```

Leave this test failing until Task 3 creates the page.

- [ ] **Step 5: Run the API test**

Run: `conda run -n biolgca python -m pytest -q tests/interaction_pipeline_test.py::test_reorientation_term_names_are_public_and_stable --basetemp=.pytest-plan-task2-pass`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lgca/pipeline.py tests/interaction_pipeline_test.py tests/tutorial_notebooks_test.py
git commit -m "feat: expose reorientation term names"
```

---

### Task 3: Reorganize the Sphinx learner and reference paths

**Files:**
- Create: `docs/source/tutorials/index.rst`
- Create: `docs/source/how_to/index.rst`
- Create: `docs/source/concepts/index.rst`
- Create: `docs/source/concepts/interactions.rst`
- Create: `docs/source/reference/index.rst`
- Move: `docs/source/model_specs_and_plugins.rst` -> `docs/source/how_to/model_specs_and_plugins.rst`
- Move: `docs/source/observers_and_plotting.rst` -> `docs/source/how_to/observers_and_plotting.rst`
- Move: `docs/source/custom_interactions.rst` -> `docs/source/how_to/custom_interactions.rst`
- Move: `docs/source/lgca_types.rst` -> `docs/source/concepts/lgca_types.rst`
- Move: `docs/source/lattice_geometries.rst` -> `docs/source/concepts/lattice_geometries.rst`
- Move: `docs/source/interactions_summary.rst` -> `docs/source/reference/interactions_summary.rst`
- Move: `docs/source/planned_topics.rst` -> `docs/source/reference/planned_topics.rst`
- Move: `docs/source/factory_reference.rst` -> `docs/source/reference/factory_reference.rst`
- Move: `docs/source/full_api.rst` -> `docs/source/reference/full_api.rst`
- Modify: `docs/source/index.rst`
- Modify: `docs/source/getting_started.rst`
- Modify: `docs/source/example_gallery.rst`
- Modify: path-sensitive `.rst` cross-references and tests
- Modify: `tests/tutorial_notebooks_test.py`
- Modify: `tests/example_gallery_docs_test.py`

**Interfaces:**
- Consumes: current flat Sphinx source tree and its cross-references.
- Produces: primary navigation `getting_started`, `tutorials/index`, `how_to/index`, `concepts/index`, `example_gallery`, `reference/index` and a complete interaction composition table.

- [ ] **Step 1: Write failing navigation and page-placement tests**

```python
def test_primary_documentation_navigation_is_task_oriented():
    index = (ROOT / "docs/source/index.rst").read_text(encoding="utf-8")
    expected = (
        "getting_started",
        "tutorials/index",
        "how_to/index",
        "concepts/index",
        "example_gallery",
        "reference/index",
    )
    positions = [index.index(name) for name in expected]
    assert positions == sorted(positions)


def test_reference_pages_have_section_owners():
    assert (ROOT / "docs/source/how_to/model_specs_and_plugins.rst").exists()
    assert (ROOT / "docs/source/concepts/lgca_types.rst").exists()
    assert (ROOT / "docs/source/reference/full_api.rst").exists()
```

- [ ] **Step 2: Run docs-structure tests and verify failure**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py tests/example_gallery_docs_test.py --basetemp=.pytest-plan-task3`

Expected: FAIL because the section indexes and moved pages do not exist.

- [ ] **Step 3: Move pages and create section indexes**

Use `git mv` for the files listed above. Each section index contains a short
description and a toctree of its owned pages. `tutorials/index.rst` introduces
the six-lesson progression and initially contains the six notebook docnames,
which will resolve after Tasks 4-6.

- [ ] **Step 4: Write the interaction composition page**

Create `concepts/interactions.rst` with:

- the phase order `birth/death -> phenotype switching -> reorientation -> propagation`;
- the statement that multiple `ReorientationTermSpec` terms contribute to one
  energy score and one sampled full-channel-state transition;
- the warning that sequential full reorientation operators are not additive
  biases; and
- a table containing every name returned by `list_reorientation_terms()`, its
  interpretation, required field/parameters, and aliases.

The phenotype-switching section must include:

```text
For a particle-number-conserving switch, the complete channel state s is
mapped to one admissible state s'. The transition preserves N(s') = N(s);
it is not a sequence of independent channel writes.
```

- [ ] **Step 5: Rewrite the landing and getting-started paths**

Lead `index.rst` and `getting_started.rst` with `ModelSpec`, the maintained
tutorials, and `jupyter lab`. Retain `get_lgca` only as a legacy/interactive
reference link. Describe Matplotlib and JupyterLab as normal dependencies and
remove instructions for the obsolete `plot2d` dependency choice.

- [ ] **Step 6: Repair moved cross-references and literalinclude depths**

Run `rg ':doc:|literalinclude|download' docs/source` and update moved docnames
to absolute Sphinx docnames such as `/how_to/model_specs_and_plugins` and
`/reference/full_api`. Preserve the example gallery's source literalincludes.

- [ ] **Step 7: Run structure and interaction-table tests**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py tests/example_gallery_docs_test.py --basetemp=.pytest-plan-task3-pass`

Expected: only notebook-file assertions scheduled for later tasks may fail; all
dependency, navigation, moved-page, and interaction-table assertions pass.

- [ ] **Step 8: Commit**

```bash
git add docs/source tests/tutorial_notebooks_test.py tests/example_gallery_docs_test.py
git commit -m "docs: organize learner and reference paths"
```

---

### Task 4: Fundamentals and collective-movement notebooks

**Files:**
- Create: `docs/source/tutorials/01_fundamentals.ipynb`
- Create: `docs/source/tutorials/02_collective_movement.ipynb`
- Modify: `tests/tutorial_notebooks_test.py`

**Interfaces:**
- Consumes: `ModelSpec`, `SpaceSpec`, `StateSpec`, `TimeSpec`, `InteractionPipelineSpec`, `run_model`, and standard recorders.
- Produces: two clean, executable notebooks that expose their complete model construction.

- [ ] **Step 1: Add failing notebook contract tests for lessons 1 and 2**

```python
import nbformat


TUTORIALS = ROOT / "docs/source/tutorials"


def _notebook(name):
    return nbformat.read(TUTORIALS / name, as_version=4)


def _source(notebook, cell_type=None):
    return "\n".join(
        cell.source for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
    )


def test_fundamentals_notebook_teaches_complete_first_spec():
    source = _source(_notebook("01_fundamentals.ipynb"))
    assert "ModelSpec(" in source
    assert "InteractionPipelineSpec(" in source
    assert '"classical.random_walk"' in source
    assert "boundary" in source and "seed" in source
    assert "Exercise" in source


def test_collective_notebook_compares_four_mechanisms():
    source = _source(_notebook("02_collective_movement.ipynb"))
    for name in ("classical.random_walk", "classical.alignment",
                 "classical.aggregation", "classical.nematic"):
        assert name in source
    assert "polarization" in source.lower()
```

- [ ] **Step 2: Run the focused tests and verify missing-file failures**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py -k "fundamentals or collective" --basetemp=.pytest-plan-task4`

Expected: FAIL because the notebooks do not exist.

- [ ] **Step 3: Author lesson 1 with visible model construction**

Use an 18x18 square lattice, 20 steps, density 0.15, and seed 11. The central
code cell defines the full model directly:

```python
spec = ModelSpec(
    description=Description(title="Seeded random movement"),
    space=SpaceSpec(geometry="square", dims=(18, 18), boundary="periodic"),
    state=StateSpec(density=0.15, restchannels=0),
    time=TimeSpec(steps=20, seed=11),
    dynamics=InteractionPipelineSpec(
        operators=[{"name": "classical.random_walk"}],
    ),
    analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder(), PopulationRecorder()]),
)
result = run_model(spec, showprogress=False)
```

Include state/channel inspection, periodic-versus-reflecting comparison via a
visible local `make_spec(boundary, seed)` function, density plots, exact seeded
reproduction, interpretation, and exercises.

- [ ] **Step 4: Author lesson 2 with visible interaction selection**

Define `make_collective_spec(interaction, beta, seed)` in a visible cell. Its
body constructs a complete 20x20 hexagonal `ModelSpec`; use random walk,
alignment, aggregation, and nematic plugin dictionaries. Define the global
polarization observable visibly from `lgca.calc_flux(nodes)` and compare final
polarization plus final-density panels. Explain polar versus nematic order and
end with a parameter-change exercise.

- [ ] **Step 5: Strip outputs and run focused tests**

Run a small `nbformat` cleanup that sets every code cell's
`execution_count = None` and `outputs = []`, then run:

`conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py -k "fundamentals or collective" --basetemp=.pytest-plan-task4-pass`

Expected: PASS.

- [ ] **Step 6: Execute both notebooks from fresh kernels**

Run: `conda run -n biolgca python -m jupyter nbconvert --to notebook --execute docs/source/tutorials/01_fundamentals.ipynb docs/source/tutorials/02_collective_movement.ipynb --output-dir .notebook-check-task4 --ExecutePreprocessor.timeout=120`

Expected: both execute without errors. Remove `.notebook-check-task4` after inspection.

- [ ] **Step 7: Commit**

```bash
git add docs/source/tutorials/01_fundamentals.ipynb docs/source/tutorials/02_collective_movement.ipynb tests/tutorial_notebooks_test.py
git commit -m "docs: teach fundamentals and collective movement"
```

---

### Task 5: Interaction-composition and population-dynamics notebooks

**Files:**
- Create: `docs/source/tutorials/03_combining_interactions.ipynb`
- Create: `docs/source/tutorials/04_population_dynamics.ipynb`
- Modify: `tests/tutorial_notebooks_test.py`

**Interfaces:**
- Consumes: `ReorientationSpec`, `ReorientationTermSpec`, `BirthDeathSpec`, `PhenotypeSwitchSpec`, and the phase-order compiler.
- Produces: explicit multi-term and multi-phase models plus a demonstrated particle-conservation invariant.

- [ ] **Step 1: Add failing semantic contract tests**

```python
def test_composition_notebook_builds_one_multi_term_sampler():
    source = _source(_notebook("03_combining_interactions.ipynb"))
    assert "ReorientationSpec(" in source
    assert source.count("ReorientationTermSpec(") >= 4
    assert "one sampled" in source.lower()
    assert "sequential" in source.lower()
    assert "chemotaxis" in source and "contact_guidance" in source


def test_population_notebook_teaches_atomic_conserving_switch():
    source = _source(_notebook("04_population_dynamics.ipynb"))
    assert "BirthDeathSpec(" in source
    assert "PhenotypeSwitchSpec(" in source
    assert "s -> s'" in source or "s → s′" in source
    assert "before.sum() == after.sum()" in source
    assert source.index("BirthDeathSpec(") < source.index("PhenotypeSwitchSpec(")
```

- [ ] **Step 2: Run focused tests and verify missing-file failures**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py -k "composition or population" --basetemp=.pytest-plan-task5`

Expected: FAIL.

- [ ] **Step 3: Author lesson 3 around one reorientation sampler**

Use a 12x12 square lattice and visible fields. The first combined model includes:

```python
ReorientationSpec(terms=[
    ReorientationTermSpec(name="alignment", beta=alignment_beta),
    ReorientationTermSpec(
        name="chemotaxis", beta=chemotaxis_beta,
        parameters={"field": "signal"},
    ),
])
```

The second includes `persistent_walk` and `contact_guidance` in one
`ReorientationSpec`. Add a small 3x3 parameter sweep, a directional-flux
observable, and explicit prose contrasting additive term scores with sequential
operators.

- [ ] **Step 4: Author lesson 4 around phase order and complete-state switching**

Construct a two-species square model visibly with operators in this order:

```python
operators=[
    BirthDeathSpec(
        name="birth_death",
        parameters={"birth_rate": [0.03, 0.01], "death_rate": [0.005, 0.005]},
    ),
    PhenotypeSwitchSpec(
        name="phenotype_switch",
        parameters={"rates": [[0.0, 0.08], [0.03, 0.0]]},
    ),
    ReorientationSpec(
        terms=[ReorientationTermSpec(name="random_walk")],
    ),
]
```

Before the full run, build a switch-only one-step specification with
`propagation=False`, copy the complete channel state before and after, and
assert `before.sum() == after.sum()`. Plot total population and species
fractions, then show an explicit `classical.go_or_grow` specification as the
application link.

- [ ] **Step 5: Execute and test both notebooks**

Run nbconvert with a 120-second timeout as in Task 4, strip generated outputs
from source notebooks, and run the focused semantic tests.

Expected: both fresh-kernel executions and tests pass.

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/03_combining_interactions.ipynb docs/source/tutorials/04_population_dynamics.ipynb tests/tutorial_notebooks_test.py
git commit -m "docs: teach interaction composition and switching"
```

---

### Task 6: Evolutionary and reproducible-project notebooks

**Files:**
- Create: `docs/source/tutorials/05_evolutionary_lgca.ipynb`
- Create: `docs/source/tutorials/06_student_project.ipynb`
- Modify: `tests/tutorial_notebooks_test.py`

**Interfaces:**
- Consumes: identity-based `ib.birthdeath`, portable ModelSpec JSON helpers, and public plugin registration.
- Produces: a replicate-based evolutionary application and a visible end-to-end project/custom-interaction template.

- [ ] **Step 1: Add failing notebook and whole-curriculum contracts**

```python
EXPECTED_NOTEBOOKS = (
    "01_fundamentals.ipynb",
    "02_collective_movement.ipynb",
    "03_combining_interactions.ipynb",
    "04_population_dynamics.ipynb",
    "05_evolutionary_lgca.ipynb",
    "06_student_project.ipynb",
)


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
        code = _source(_notebook(name), "code")
        assert "ModelSpec(" in code
        assert "InteractionPipelineSpec(" in code
        assert "get_example_spec" not in code
        assert "lgca.examples" not in code


def test_evolutionary_notebook_runs_replicates():
    source = _source(_notebook("05_evolutionary_lgca.ipynb"))
    assert '"ib.birthdeath"' in source
    assert "seeds" in source and "replicate" in source.lower()
    assert "scientific conclusion" in source.lower()


def test_student_project_notebook_saves_spec_and_tests_custom_invariant():
    source = _source(_notebook("06_student_project.ipynb"))
    assert "register_plugin(" in source
    assert "save_model_spec(" in source
    assert "conserv" in source.lower()
    assert "assert" in source
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py -k "maintained or loading_example or evolutionary or student_project" --basetemp=.pytest-plan-task6`

Expected: FAIL because lessons 5 and 6 are absent.

- [ ] **Step 3: Author lesson 5 with explicit identity-based specifications**

Create a visible `make_evolution_spec(seed)` using a 40-node linear lattice,
two rest channels, a seeded left-edge Boolean state, 30 steps, and:

```python
InteractionPipelineSpec(operators=[{
    "name": "ib.birthdeath",
    "parameters": {"r_b": 0.2, "r_d": 0.01, "std": 0.05},
}])
```

Run at least three explicit seeds, plot population trajectories, inspect the
final heritable `r_b` property distribution, and explain why replicate
variation prevents one realization from being a scientific conclusion.

- [ ] **Step 4: Author lesson 6 as a complete visible project template**

Start from a question and measurable prediction. Construct a baseline spec,
then define a small `ReorientationOperator` subclass that cyclically rotates
velocity channels without changing total occupancy. Register it with a unique
`PluginInfo` and `register_plugin`, use its registered name in a visible
`ModelSpec`, and assert total population is unchanged for a propagation-free
step. Save `model.json` with `save_model_spec`, record `lgca.__version__` via
package metadata, and show a project directory/checklist. Clean temporary files
at the end of the notebook using `TemporaryDirectory`, not a repository path.

- [ ] **Step 5: Execute lessons 5 and 6 and run all notebook contracts**

Run nbconvert with fresh kernels and a 120-second per-notebook timeout. Strip
source outputs again, then run:

`conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py --basetemp=.pytest-plan-task6-pass`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/05_evolutionary_lgca.ipynb docs/source/tutorials/06_student_project.ipynb tests/tutorial_notebooks_test.py
git commit -m "docs: teach evolutionary and reproducible projects"
```

---

### Task 7: Archive legacy notebooks and finish learner-facing documentation

**Files:**
- Move: `BioLGCA.ipynb` -> `notebooks/legacy/BioLGCA.ipynb`
- Move: `Evolutionary LGCA.ipynb` -> `notebooks/legacy/Evolutionary LGCA.ipynb`
- Move: `SRP_Notebook.ipynb` -> `notebooks/research_projects/SRP_Notebook.ipynb`
- Create: `notebooks/README.md`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/source/getting_started.rst`
- Modify: `docs/source/example_gallery.rst`
- Remove: obsolete `docs/source/tutorial.rst`, `docs/source/examples.rst`, and `docs/source/user_guide.rst` after their links are replaced
- Modify: `tests/examples_teaching_files_test.py`
- Modify: `tests/example_gallery_docs_test.py`
- Modify: `tests/tutorial_notebooks_test.py`

**Interfaces:**
- Consumes: completed maintained curriculum and existing historical notebook files.
- Produces: an unambiguous maintained/legacy/research-project split and a README path that reaches lesson 1 first.

- [ ] **Step 1: Write failing placement and README tests**

```python
def test_legacy_and_research_notebooks_are_clearly_separated():
    assert not list(ROOT.glob("*.ipynb"))
    assert (ROOT / "notebooks/legacy/BioLGCA.ipynb").exists()
    assert (ROOT / "notebooks/legacy/Evolutionary LGCA.ipynb").exists()
    assert (ROOT / "notebooks/research_projects/SRP_Notebook.ipynb").exists()
    readme = (ROOT / "notebooks/README.md").read_text(encoding="utf-8")
    assert "Maintained tutorials" in readme
    assert "Historical notebooks" in readme
    assert "Research-project example" in readme


def test_root_readme_leads_to_the_maintained_curriculum():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/source/tutorials/01_fundamentals.ipynb" in text
    assert text.index("ModelSpec") < text.index("get_lgca")
    assert "jupyter lab" in text.lower()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `conda run -n biolgca python -m pytest -q tests/tutorial_notebooks_test.py tests/examples_teaching_files_test.py tests/example_gallery_docs_test.py --basetemp=.pytest-plan-task7`

Expected: FAIL on old locations and old README ordering.

- [ ] **Step 3: Move historical files without rewriting them**

Use `git mv` for the three notebooks. Add `notebooks/README.md` with direct
links to maintained tutorials, a warning that legacy notebooks use older APIs
and are not executed in CI, and a description of the SRP notebook as a
project-specific historical artifact.

- [ ] **Step 4: Rewrite README and documentation entry copy**

The root README quick start installs the project normally, launches
`jupyter lab`, links lesson 1, and shows a short explicit ModelSpec. The legacy
factory example moves to a compatibility/reference section. `docs/README.md`
becomes only a strict docs-build maintainer guide. Update gallery provenance
text to point at the relocated historical notebooks without presenting them as
the learner path.

- [ ] **Step 5: Remove superseded duplicate pages and repair tests/links**

Delete the three obsolete root documentation pages only after their unique
content has a maintained destination. Update tests that assert legacy notebook
source names so they continue to verify provenance without requiring the files
at repository root.

- [ ] **Step 6: Run focused docs and notebook tests**

Run the command from Step 2 again.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add README.md docs notebooks tests
git commit -m "docs: make maintained notebooks the learner path"
```

---

### Task 8: End-to-end execution, packaging, and installation verification

**Files:**
- Verify: `.github/workflows/ci.yml`
- Verify: `tests/packaging_test.py`
- Verify: `docs/source/tutorials/*.ipynb`
- Verify: `docs/source/**/*.rst`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: evidence that tests, six fresh-kernel notebooks, strict docs, source/wheel builds, and an installed wheel all work.

- [ ] **Step 1: Run every maintained notebook with an aggregate timer**

Run all six through nbconvert in a new repository-local output directory with a
120-second per-notebook timeout. Record total wall time and verify it is below
three minutes. Remove the execution-output directory afterward; source
notebooks must remain output-free.

- [ ] **Step 2: Run the strict Sphinx build**

Run: `conda run -n biolgca python docs/build.py`

Expected: PASS with no warnings; MyST-NB executes all six notebooks.

- [ ] **Step 3: Run the complete pytest suite**

Run: `conda run -n biolgca python -m pytest -q --basetemp=.pytest-teaching-final`

Expected: all tests pass.

- [ ] **Step 4: Build source and wheel artifacts**

Run: `conda run -n biolgca python -m build`

Expected: one sdist and one wheel are created successfully and contain package
code while documentation notebooks remain repository documentation rather than
runtime package data.

- [ ] **Step 5: Smoke-test a clean wheel installation**

Create a repository-local virtual environment, install only the built wheel,
then run outside the source tree:

```python
import importlib.metadata
import jupyterlab
import matplotlib
from lgca.model import ModelSpec
from lgca.pipeline import list_reorientation_terms

assert importlib.metadata.version("biolgca") == "0.1.0"
assert "chemotaxis" in list_reorientation_terms()
assert ModelSpec().time.steps == 100
```

Also run `biolgca examples export random_walk model.json`, `biolgca validate
model.json`, and a one-step `biolgca run`. Remove the temporary environment and
smoke directory after success.

- [ ] **Step 6: Confirm CI covers notebooks and normal dependencies**

The existing docs job should install `.[docs]` and call `python docs/build.py`;
because the strict build executes notebooks, no duplicate notebook CI job is
needed. Update `.github/workflows/ci.yml` only if the actual build shows the
existing job does not exercise this path. The installed-package job's wheel
install proves normal JupyterLab/Matplotlib dependency resolution.

- [ ] **Step 7: Audit the goal against authoritative state**

Check each spec success criterion against files and command outputs. Run:

```bash
git status --short
git diff aidevelop...HEAD --check
git diff aidevelop...HEAD --stat
```

Verify there are no generated docs, notebook outputs, temporary environments,
or unrelated changes.

- [ ] **Step 8: Commit any verification-driven fixes**

```bash
git add .github/workflows/ci.yml tests/packaging_test.py tests/tutorial_notebooks_test.py docs/source pyproject.toml
git commit -m "test: verify teaching notebook release gates"
```

If no fixes were required, do not create an empty commit.
