# BioLGCA Teaching Notebooks and Documentation Design

## Purpose

BioLGCA should be easy for students to install, explore, and use for small
scientific projects. The package already contains a broad set of tested model
examples, but its learner-facing material does not provide a maintained path
from a first simulation to a reproducible project:

- the two general legacy notebooks use the old factory workflow, are long and
  depend on saved notebook state;
- the source-code example gallery is comprehensive but is a reference catalog,
  not a guided course;
- the documentation mixes tutorials, how-to material, conceptual explanation,
  and API reference;
- combining interactions is technically supported but the distinction between
  combined reorientation biases and sequential biological processes is not
  taught clearly; and
- notebook execution is not verified in CI.

This work will replace the legacy learner path with a small, executable
curriculum at the same Python and scientific level as `BioLGCA.ipynb` and
`Evolutionary LGCA.ipynb`, while expanding their applications, analysis, and
reproducibility guidance.

## Goals

1. Make a maintained sequence of six application-driven notebooks the primary
   student learning path.
2. Teach the public `ModelSpec` workflow consistently while preserving legacy
   factory documentation as reference material.
3. Make interaction composition understandable and safe to experiment with.
4. Connect simulations to quantitative analysis and biological interpretation,
   rather than presenting plots alone.
5. Teach reproducible project practice: explicit seeds, saved model
   specifications, replicates, package-version capture, and separated results.
6. Render the notebooks in the existing Sphinx documentation and execute them
   from a clean kernel in CI.
7. Keep maintenance realistic for a one-person scientific software project by
   reusing tested example builders and avoiding a second documentation system.
8. Make a normal BioLGCA installation sufficient to open and run the teaching
   notebooks, without requiring a separate teaching extra.

## Non-goals

- A graphical model-building GUI or notebook widget framework.
- A migration from Sphinx to Jupyter Book.
- One notebook for every interaction or every curated example.
- A general redesign of the interaction/plugin architecture.
- Publishing a package release or provisioning a hosted Jupyter service.
- Rewriting all reference documentation or all plots in this milestone.

## Audience and teaching level

The target reader has the same background assumed by the two general legacy
notebooks: they can read ordinary Python and NumPy code, use Jupyter, and follow
mathematical descriptions of LGCA models. The notebooks will explain BioLGCA,
the biological model, and the experimental workflow, but will not teach Python
syntax or package-management fundamentals.

Each notebook must be useful in two modes:

- a guided lesson that runs from top to bottom; and
- a starting point that a student can copy and modify for a project.

## Learner journey

The primary path will be:

1. install BioLGCA and launch JupyterLab;
2. run a first seeded simulation;
3. change parameters and initial conditions;
4. compare movement mechanisms quantitatively;
5. combine directional cues and sequential processes correctly;
6. add population dynamics and phenotypic switching;
7. run evolutionary models with stochastic replicates; and
8. turn a biological question into a saved, reproducible project.

The root README and documentation landing page will lead with this path and the
`ModelSpec` API. The legacy `get_lgca` factory remains documented for existing
users, but will no longer be the first workflow shown to new students.

## Notebook curriculum

The maintained notebooks will live in `docs/source/tutorials/` and follow the
same internal pattern: learning objectives, biological context, model setup,
baseline run, controlled experiment, quantitative interpretation, exercises,
and reproducibility checkpoint.

### 1. LGCA fundamentals and random movement

- Introduce lattice sites, velocity/rest channels, channel states, propagation,
  interaction, boundary conditions, and volume exclusion.
- Build and run a small random-walk model with `ModelSpec`.
- Inspect model state before and after one step.
- Compare at least two boundary conditions or densities.
- Demonstrate an explicit seed and exact reproduction of a short trajectory.

### 2. Collective movement

- Compare random walk, alignment, aggregation, and nematic alignment.
- Explain the biological interpretation and assumptions of each mechanism.
- Use a suitable observable, such as polarization or a clustering measure, so
  the comparison is not based only on visual inspection.
- Reuse existing curated example builders where their model definitions match
  the lesson.

### 3. Combining directional cues

- Construct a `ReorientationSpec` containing multiple
  `ReorientationTermSpec` terms.
- Study alignment plus chemotaxis and persistent motion plus contact guidance.
- Use a small parameter sweep to show reinforcement or competition between
  cues.
- Explicitly distinguish two meanings of "combining interactions":
  multiple energy/bias terms participate in one sampled reorientation
  transition, while operators in different pipeline phases represent
  sequential biological processes.
- Explain why placing two full reorientation operators sequentially is not the
  same as adding their biases.

### 4. Population dynamics and phenotypic switching

- Combine birth/death, movement, and phenotypic switching in phase order.
- Introduce go-and-grow and go-or-grow applications.
- Plot total population and phenotype fractions together with spatial output.
- State and demonstrate the conservation rule: for a particle-number-conserving
  interaction such as phenotypic switching, the operator samples a transition
  of the complete channel state, `s -> s'`, while conserving particle number.
  It must not independently edit channel occupancies in a way that creates or
  removes particles.
- Connect this rule to the established non-identity, volume-exclusion random
  walk, alignment, and chemotaxis interactions.

### 5. Evolutionary LGCA

- Replace the maintained teaching role of `Evolutionary LGCA.ipynb`.
- Introduce heritable traits, mutation/variation, selection, and spatial
  expansion using the current public API.
- Run multiple seeded replicates and display between-run variability.
- Separate a simulation result from a scientific conclusion and identify which
  parameter ranges or replicates would be needed for a real study.

### 6. Developing a reproducible student project

- Start with a biological question and a measurable prediction.
- Compose existing terms/operators before implementing a custom interaction.
- Add one small custom interaction and test its invariant.
- Save a portable `ModelSpec`, seed, BioLGCA version, parameters, and derived
  results outside the notebook's transient state.
- Provide a concise project checklist and suggested folder layout.

## Interaction discoverability

The composition notebook cannot rely on a private registry as its vocabulary.
This milestone will add one small public discovery helper that returns the
supported reorientation-term names in a stable, deterministic order. It will
not add a new registry abstraction or CLI command.

The interaction documentation will include a synchronized table covering:

- term/operator name;
- pipeline phase;
- biological interpretation;
- required parameters or fields;
- applicable model families or constraints; and
- whether it is intended to combine as a term in one reorientation sampler or
  as a sequential pipeline operator.

A test will ensure every publicly reported reorientation term is represented in
the documentation, preventing the notebook vocabulary from silently drifting.

## Documentation architecture

The existing Sphinx project remains the only documentation build. MyST-NB will
be added to render `.ipynb` files directly.

The user-facing navigation will become:

1. Getting started
2. Tutorials
3. How-to guides
4. Concepts
5. Example gallery
6. API and model reference

The target source layout is:

```text
docs/source/
|-- getting_started.rst
|-- tutorials/
|   |-- index.rst
|   |-- 01_fundamentals.ipynb
|   |-- 02_collective_movement.ipynb
|   |-- 03_combining_interactions.ipynb
|   |-- 04_population_dynamics.ipynb
|   |-- 05_evolutionary_lgca.ipynb
|   `-- 06_student_project.ipynb
|-- how_to/
|-- concepts/
|-- example_gallery.rst
`-- reference/
```

Existing pages will be moved into the appropriate section without unrelated
rewrites. Cross-references, toctrees, and path-sensitive documentation tests
will be updated with the moves. `docs/README.md` will stop duplicating the root
README and will become a short maintainer guide for building the documentation.

The large example gallery remains a reference/recipe catalog. It will point to
the tutorial sequence but will not be converted into sixteen notebooks.

## Source-of-truth and code boundaries

- Existing `lgca.examples.*.build_spec()` functions remain the source for
  curated model definitions when a notebook uses the same scenario.
- A notebook may build a small specification inline when seeing the complete
  specification is itself part of the lesson.
- Notebook cells will not copy library implementations.
- Reusable analysis code will move into a tested Python function only when it is
  used by more than one notebook and represents a generally useful observable.
- Notebook-only narrative helpers will remain in the notebook; no framework for
  authoring lessons will be introduced.

## Notebook execution and output policy

All source notebooks will be committed without saved execution counts or cell
outputs. The documentation build will execute them from top to bottom and
render the resulting figures. This avoids large, stale notebook diffs while the
built documentation still shows complete results.

Each notebook must:

- execute in a fresh kernel with no prior state;
- use deterministic seeds except where a section explicitly compares seeds;
- use only local package data;
- use static Matplotlib output suitable for headless CI;
- avoid `%matplotlib notebook`, widgets, and hidden setup state;
- close or reuse figures to prevent memory growth; and
- keep its default workload small enough that all six notebooks complete in a
  practical documentation build, with a target of at most three minutes total
  in the project CI environment.

Longer experiments will be described as optional exercises and will not be part
of the default executed path.

## Dependencies and installation

The normal project dependencies in `pyproject.toml` will include the packages a
student needs to run the curriculum:

- Matplotlib for the maintained two-dimensional plots; and
- JupyterLab for opening and running the notebooks.

There will be no `teaching` or `notebooks` optional dependency group. Existing
plotting extras will be adjusted so they do not duplicate dependencies that are
now installed normally; specialized three-dimensional plotting dependencies may
remain optional if they are not used by the curriculum.

Sphinx, MyST-NB, and other documentation-build tooling remain in the `docs`
optional dependency group. They are required to build the website, not to open
and run the teaching notebooks. This distinction keeps the normal installation
aligned with the user's request without making every student install the
documentation toolchain.

The getting-started instructions will use a normal repository installation and
then launch JupyterLab. Publishing to PyPI is outside this milestone, so the
documentation must not claim that `pip install biolgca` works until a release is
actually published.

## Legacy material

- Move `BioLGCA.ipynb` and `Evolutionary LGCA.ipynb` to
  `notebooks/legacy/` after their maintained teaching content is covered.
- Move `SRP_Notebook.ipynb` to `notebooks/research_projects/` because it is a
  specific research-project artifact rather than a general tutorial.
- Add a short `notebooks/README.md` explaining which notebooks are maintained,
  historical, or project-specific.
- Preserve the historical notebook files rather than rewriting their saved
  results. They will not execute in CI or appear in the primary documentation
  path.

## Validation and failure behavior

The implementation will add or update tests for:

- the exact maintained notebook set and tutorial toctree;
- clean notebook sources with no saved outputs or execution counts;
- public reorientation-term discovery and documentation synchronization;
- normal package metadata containing the notebook runtime dependencies;
- legacy notebook placement and status documentation; and
- existing example-gallery and documentation path expectations.

The strict Sphinx build will execute all maintained notebooks. A cell exception,
missing dependency, missing referenced page, or Sphinx warning must fail the
build. No notebook will catch unexpected exceptions merely to make CI pass.

Final verification will include:

1. `conda run -n biolgca python -m pytest -q` with a repository-local isolated
   base temporary directory on Windows;
2. the strict Sphinx documentation build with notebook execution;
3. source and wheel builds;
4. installation of the built wheel into a clean environment followed by an
   import, CLI, Matplotlib, and notebook-runtime smoke test; and
5. a check that the worktree contains only changes belonging to this goal.

## Success criteria

The goal is complete when:

- a student following the root README can reach and run the first maintained
  notebook without discovering a legacy API first;
- all six notebooks execute in order from clean kernels and are rendered in the
  documentation;
- the composition notebook teaches and demonstrates both additive
  reorientation terms and sequential pipeline phases correctly;
- the phenotypic-switching lesson states and demonstrates full-state,
  particle-number-conserving transitions;
- every notebook contains a biological application, quantitative
  interpretation, and an exercise or investigation prompt;
- a normal package installation includes JupyterLab and Matplotlib;
- the historical notebooks remain accessible but cannot be mistaken for the
  maintained learner path; and
- the full tests, strict documentation build, package build, and clean-wheel
  smoke checks pass.

