# AGENTS.md – working on biolgca

Guidelines and context for developers and coding agents working on `biolgca`.
Decisions recorded here are settled; propose changes to them, do not work
around them.

## Project overview

`biolgca` is a Python library for lattice-gas cellular automata (LGCA) in
biology. Models are classical (cells are counted) or identity-based (cells
carry labels and traits), with or without volume exclusion, with one or
several species, on 1D, square, hexagonal, cubic and 3D Moore lattices.
Fields (oxygen, signals, matrix) evolve with the cells through the `pde`
operator.

Two APIs coexist:

- **`ModelSpec` (current):** a model is a declarative spec (space, state,
  time, interaction pipeline, observers) that can be saved as JSON/YAML and
  run from Python, swept over parameters (`lgca.study.sweep`), explored live
  in a notebook (`lgca.explore`) or run with the `biolgca` command. New
  features go here.
- **`get_lgca` (quick start):** builds an LGCA object from the name of a
  standard model (the interaction names of earlier versions); the first
  tutorials and the quick starts use it. `legacy_names.py` translates every
  name into a stack of rules and compiles it into a pipeline, so both APIs
  run the same code. It gets no new names; new rules go to `ModelSpec`.

The model zoo (`lgca.zoo`) reproduces published models with the current
rules, one module and one notebook per paper.

## Environment and commands

The environment is managed with uv; `uv sync` creates `.venv/` from the
committed `uv.lock`. Run everything from the repository root:

```bash
uv sync                                    # install locked dependencies
uv run pytest -q tests/x_test.py           # one test file while you work
uv run pytest -q                           # full suite: about 2900 tests, 4 minutes
uv run ruff check <files>                  # lint the files you changed
uv run python docs/build.py --no-notebooks # strict docs build, notebooks not run (under a minute)
uv run python docs/build.py                # full docs build: runs every tutorial and zoo
                                           # notebook (about 15 minutes), as CI does
uv run jupyter nbconvert --to notebook --execute --output-dir <tmp> <notebook>  # one notebook
```

Which checks when: the test file of your change while you work; the full
suite and ruff before every commit, and the docs build without notebooks if
the commit changes docstrings or docs; the notebooks you changed, or whose
code you changed, run once by hand. The full docs build, with every notebook,
runs before pushing (and in CI).

After changing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`. CI (`.github/workflows/ci.yml`) runs the tests on Linux with
Python 3.11 to 3.14, on Windows and macOS, with the minimum dependency
versions and once in random order; executes the tutorial and zoo notebooks;
smoke-tests the installed wheel and the CLI; and builds the docs. pyamg has no
wheels for Python 3.14 yet: code paths without it are tested by
monkeypatching `lgca.fields._pyamg`.

Practical notes:

- **Testing a copy of the package** (e.g. to check that a new test fails on
  the old code): `tests/conftest.py` puts the repository root first on
  `sys.path`, so `PYTHONPATH` does not redirect pytest. Run pytest from the
  root of the copy, or a copy of `tests/` with `python -P`. Scripts outside
  pytest follow `PYTHONPATH`.
- **Heavy jobs:** the maintainer's Windows machine ran out of memory with
  several benchmarks, test runs and docs builds at once. Run at most two or
  three such jobs at a time, and benchmarks with single-threaded BLAS
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`).
- `outputs/` is git-ignored: put probe scripts, benchmark results and
  scratch files there (or in a temporary directory), never in the package.

## Where things are

`lgca/`, the package:

- Model API: `model.py` (`ModelSpec`, `build_model`, `run_model`,
  `CompiledModel` with `step`, `run` and `reconfigure`, model files),
  `pipeline.py` (`InteractionPipelineSpec`, `ReorientationSpec`, the
  Boltzmann sampler, native operators), `simulation.py` (observers and
  recorders, `SimulationRunner`), `initializers.py`, `cli.py`, `schemas/`
  (JSON schema of model files), `study.py` (`vary`, `sweep`), `explorer.py`
  (`lgca.explore`), `transaction.py` (a step is applied as a whole or not at
  all), `provenance.py` (versions, source hashes, input hashes).
- Interaction registry: `operator_base.py` (`PluginInfo`, operator base
  classes), `operator_registry.py`, `plugins.py` (registration, parameter
  validation and descriptions, built-in plugin catalogue).
- Writing rules: `lattice_state.py` (`LatticeState`, the interior of a model
  with a species axis and per-cell operations; `ContractWarning`),
  `cells.py` (the cell table and trait buffers of identity-based models,
  `state.cells`), `rules.py` (the `@interaction`, `@reorientation_term` and
  `@stack` decorators), `builtin_rules.py` (built-in rules and reorientation
  terms written with them), `mutations.py` (events that change cell traits),
  `switching.py` (switching probabilities that respond to cues),
  `fields.py` (the `pde` operator and `@reaction`; Newton's method for
  saturating uptake, `FieldSolverError`), `research_models.py` (published
  models as stacks of rules, under the legacy names), `legacy_names.py`,
  `testing.py` (`check_interaction`, `strict_contracts`).
- Model classes: `base.py`, `nove_base.py`, `ib_base.py`, `nove_ib_base.py`,
  `multispecies_base.py`; geometries in `lgca_1d.py`, `lgca_square.py`,
  `lgca_hex.py`, `lgca_cubic.py`, `lgca_3dmoore.py` and `ms_*.py`.
- Plotting: `plots.py`, `plot_data.py`, `square_plotting.py`, `plotting.py`,
  `mayavi_style.py` (optional 3D).
- `examples/`: curated runnable models (`lgca.examples.run_example`);
  `zoo/`: the model zoo (one module per paper, `_card.py` for the cards).

Elsewhere:

- `tests/`: pytest files named `*_test.py`; pytest collects only `tests/`.
  `tests/legacy/` keeps the interaction functions of earlier versions,
  frozen, as a reference for comparison tests; the package never imports it.
- `docs/source/`: Sphinx user docs (`tutorials/` and `zoo/` notebooks run in
  the full docs build). `docs/development/`: maintainer notes, not built:
  `roadmap.md` (the open work and the decisions that hold),
  `architecture.md`, and `archive/` (finished plans, specs, reviews and
  reports, kept for reference).
- `benchmarks/`: profiling and benchmark scripts.
- `notebooks/legacy/`: historical notebooks, not maintained.

## Design rules

These follow from decisions of the maintainer; the reasons and measurements
are in `docs/development/archive/reviews/2026-09-30-architecture-proposals.md`.

- **Interaction kinds and their contract.** From narrow to wide: `field`
  changes fields only (`state.set_field`); `reorientation` also moves cells
  between the channels of their node, keeping the cells of each species and,
  in identity-based models, their traits; `phenotype_switch` also changes
  species or traits, keeping the cells at every node; `birth_death` may do
  everything. A rule that moves cells and sets traits is a phenotype switch.
  The operations check the kind before they change anything: a violation
  warns once per rule and model with `ContractWarning` and runs as the wider
  kind; the test suite, `check_interaction` and `strict_contracts()` make it
  an error. A stack must be at least as wide as its operators. Operators run
  in the order they are listed, then propagation.
- **Cells change only through operations.** `cells.label`, `index` and
  `channel` are read-only; reorientation terms, reactions and stack builders
  get states they can read but not change.
- **New rules** are functions of a `LatticeState` registered with
  `@interaction`, `@reorientation_term` or `@reaction`; built-in ones go in
  `builtin_rules.py`. Use the state's operations, which act per cell, and only
  `state.rng` for random numbers. Class-based operators in `pipeline.py` are
  for cases the decorators cannot express.
- **A step is all or nothing.** A step that raises is rolled back
  (`transaction.py`): keep a rule's state on the lattice (attributes, nodes,
  fields, traits written with `trait[index] = values`), not in Python objects
  or through `TraitArray.values`, which are not rolled back.
- **Models own their configuration.** `build_model` builds from its own copy
  of the spec; operator objects in a spec are templates, copied for every
  model; rules and terms are code and copy as themselves. Change a running
  model only with `CompiledModel.reconfigure`.
- **Validation:** where going on gives correct results, warn with a category
  of its own (strict in tests) rather than raise; raise where the result
  would be wrong; put checks that cost noticeable time behind a switch.
  Error messages say what to do.
- **Provenance:** runs record versions, rule source hashes, parameters and
  input hashes; the library compares only hashes it wrote itself and warns
  on a mismatch. Read inputs once per run or sweep.
- **Species axis:** rules see channel states as `dims + (n_species, K)`, also
  for one species. Public arrays (`lgca.nodes`, recordings, model files) of
  single-species models keep `dims + (K,)`.
- **Capacity** is a crowding scale used in rates such as
  `1 - density / capacity`; it comes from `StateSpec.capacity`, not from
  operator parameters. The hard limit is volume exclusion (one cell per
  channel and species). With volume exclusion a capacity is optional
  (`state.has_capacity`); `birth_death` with `crowding=False` treats it as a
  hard limit.
- **Messages:** warn with `lgca._warnings.warn_user` (points at the user's
  code); report progress with the `logging` logger `"lgca"`, never `print`
  (the CLI prints its output directory).

## Code style

- PEP 8, lines up to about 110 characters; `ruff check` passes for new or
  rewritten modules (the legacy code is not yet clean: add no findings to
  it). Do not run `ruff --fix` on legacy modules: it reorders imports, and
  the order of the registering imports in `lgca/__init__.py` matters.
- NumPy-style docstrings for all public APIs, with examples where helpful.
- CamelCase classes, snake_case functions and methods, descriptive names,
  type hints where they help.
- Match the surrounding code: comment density, naming and idiom. Comments
  say why, not what.

## Tests and documentation

- Every change comes with tests; a fix with a test that fails without it.
  Parametrize across geometries and model families where behaviour should be
  the same. Tests use fixed seeds; statistical tests compare with tolerances
  derived from standard errors and fail for a wrong model.
- `ContractWarning` is an error in every test (`filterwarnings` in
  `pyproject.toml`); test a warning with `pytest.warns`.
- Tests do not depend on their order: `tests/conftest.py` removes the rules,
  terms and reactions a test registers and closes its pyplot figures, and CI
  runs the suite once in random order. After changing global state (a
  registry, a module-level cache), run
  `uv run --with pytest-randomly==5.0.0 pytest -q -p randomly`; it prints the
  seed, and `--randomly-seed=<seed>` repeats the order.
- Code in the README and in the how-to pages `custom_interactions`,
  `research_models`, `fields` and `exploring` is executed by
  `tests/readme_test.py` and `tests/docs_snippets_test.py`; keep it runnable
  (a new code block in these pages changes the block count in the test).
  Regenerate the README images with
  `uv run python docs/images/readme/make_readme_images.py` when their code or
  the plotted model changes.
- Keep examples in docstrings, docs, tutorials and zoo notebooks in sync with
  the API. Notebooks must run top to bottom without saved outputs.
- Record user-facing changes in `CHANGELOG.md` under "Unreleased" (what
  changed for the user, and what was wrong before). Keep
  `docs/development/roadmap.md` current: delete items that are done (the
  CHANGELOG and git keep the record), add new open work and decisions. A
  longer plan, spec or review gets a file of its own while it is worked on and
  moves to `docs/development/archive/` when it is done; do not add dated
  status notes to the roadmap.

## Git

- Development happens on `aidevelop`; `master` is what Read the Docs builds
  until the new version is released. Do not push, force-push, rewrite
  history or touch `master` unless asked.
- **Commit each finished, tested unit of work** on `aidevelop`: one fix or
  feature, with its tests and docs, after the checks above. Do not let
  unrelated work pile up uncommitted. Stage files by name, not `git add -A`.
  Before pushing, run the full docs build.
- Commit messages: `fix: ...`, `feat: ...`, `docs: ...`, `chore: ...` with a
  one-line summary, and a body that says what changed and why. PR titles:
  `[Fix|Feat|Docs] <one-line summary>`; PR bodies have a "Testing Done"
  section and reference the issues addressed.
- **Parallel work in worktrees:** several agents may work at once, each in
  its own git worktree (`.claude/worktrees/`, untracked). Give each disjoint
  files; a worktree may start from `master`, so reset its branch to
  `aidevelop` first. Shared files (`CHANGELOG.md`, the roadmap) are edited
  by whoever merges: delegated work reports its entries instead. Review a branch's diff before merging it, and merge with
  `--no-ff`.
- The stash is shared between all worktrees: do not use `git stash` while
  other worktrees are active; use a work-in-progress commit or a copy, and
  compare with `git show HEAD:<file>`.
