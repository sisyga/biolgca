# AGENTS.md – Quick Start for Developers

Guidelines and context for developers and coding agents working on `biolgca`.

## Project overview

`biolgca` is a Python library for lattice-gas cellular automata (LGCA) in
biology. Models are classical (cells are counted) or identity-based (cells
carry labels and traits), with or without volume exclusion, with one or
several species, on 1D, square, hexagonal, cubic and 3D Moore lattices.

Two APIs coexist:

- **`ModelSpec` (current):** a model is a declarative spec (space, state,
  time, interaction pipeline, observers) that can be saved as JSON/YAML and
  run from Python or the `biolgca` CLI. New features go here.
- **`get_lgca` (quick start):** builds an LGCA object from the name of a
  standard model (the interaction names of earlier versions); the first
  tutorials and the quick starts use it. `legacy_names.py` translates every
  name into a stack of rules and compiles it into a pipeline, so both APIs
  run the same code. It gets no new names; new rules go to `ModelSpec`.

## Environment and commands

The environment is managed with uv; `uv sync` creates `.venv/` from the
committed `uv.lock`. Run everything from the repository root:

```bash
uv sync                         # install locked dependencies
uv run pytest -q                # test suite (about 25 s)
uv run python docs/build.py     # strict docs build, executes all tutorials
uv run ruff check <files>       # lint the files you changed
```

After changing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`. CI (`.github/workflows/ci.yml`) runs the tests on Python 3.11 to
3.14 and with the minimum dependency versions, executes the tutorials, smoke
tests the installed wheel and CLI, and builds the docs.

## Where things are

`lgca/`, the package:

- Model API: `model.py` (`ModelSpec`, `build_model`, `run_model`, model files),
  `pipeline.py` (`InteractionPipelineSpec`, `ReorientationSpec`, the Boltzmann
  sampler, native operators), `simulation.py` (observers and recorders),
  `initializers.py`, `cli.py`, `schemas/` (JSON schema of model files).
- Interaction registry: `operator_base.py` (`PluginInfo`, operator base
  classes), `operator_registry.py`, `plugins.py` (registration, parameter
  validation and descriptions, built-in plugin catalogue).
- Writing rules: `lattice_state.py` (`LatticeState`, the interior of a
  model with a species axis and per-cell operations), `cells.py` (the cell
  table and trait buffers of identity-based models, `state.cells`), `rules.py`
  (the `@interaction`, `@reorientation_term` and `@stack` decorators),
  `builtin_rules.py` (built-in rules and reorientation terms written with
  them, including go-or-grow, `birth_death` and the switches),
  `mutations.py` (events that change cell traits: mutations of daughters and
  `trait_switch`), `switching.py` (switching probabilities that respond to
  cues such as the density), `fields.py` (the `pde` operator: fields that
  diffuse, decay and are secreted or taken up by cells), `research_models.py`
  (published models as stacks of the rules, under the legacy names),
  `legacy_names.py` (the `get_lgca` interaction names, and the deprecated
  prefixed names of model files, as stacks of rules), `testing.py`
  (`check_interaction`).
- Model classes: `base.py`, `nove_base.py`, `ib_base.py`, `nove_ib_base.py`,
  `multispecies_base.py`; geometries in `lgca_1d.py`, `lgca_square.py`,
  `lgca_hex.py`, `lgca_cubic.py`, `lgca_3dmoore.py` and `ms_*.py`.
- Plotting: `plots.py`, `plot_data.py`, `square_plotting.py`, `plotting.py`,
  `mayavi_style.py` (optional 3D).
- `examples/`: curated runnable models (`lgca.examples.run_example`).

Elsewhere:

- `tests/`: pytest files named `*_test.py`. `tests/legacy/` keeps the
  interaction functions of earlier versions, frozen, as a reference for
  comparison tests (`legacy_lgca(...)`, or `legacy.<family>.<name>` in a
  pipeline); the package never imports it.
- `docs/source/`: Sphinx user docs; the tutorials in `docs/source/tutorials/`
  run during every docs build. `docs/development/`: maintainer notes, not
  built: `usability_roadmap.md` (the current plan), `architecture.md`.
- `benchmarks/`: profiling and benchmark scripts.
- `notebooks/legacy/`: historical notebooks, not maintained.

## Conventions

- **Interactions** are of three kinds: `birth_death` changes the number of
  cells, `phenotype_switch` moves cells between species (classical models)
  or changes cell parameters (identity-based models), and `reorientation`
  rearranges a node's cells over its channels, keeping the cells per species.
  A fourth kind, `field`, changes no cells but a field (`pde`).
  Operators run in the order they are listed, then propagation.
- **New rules** are functions of a `LatticeState` registered with
  `@interaction` or `@reorientation_term`; built-in ones go in
  `builtin_rules.py`. Use the state's operations, which act per cell, and only
  `state.rng` for random numbers. Class-based operators in `pipeline.py` are
  for cases the decorators cannot express.
- **Species axis:** rules see channel states as `dims + (n_species, K)`, also
  for one species. Public arrays (`lgca.nodes`, recordings, model files) of
  single-species models keep `dims + (K,)`.
- **Capacity** is a crowding scale used in rates such as
  `1 - density / capacity`; it comes from `StateSpec.capacity`, not from
  operator parameters. The hard limit is volume exclusion (one cell per
  channel and species). With volume exclusion a capacity is optional
  (`state.has_capacity`), a soft limit in addition to the channels;
  `birth_death` with `crowding=False` treats it as a hard limit.
- **Messages:** warn with `lgca._warnings.warn_user` (points at the user's
  code); report progress with the `logging` logger `"lgca"`, never `print`.
- **Reproducibility:** tests use fixed seeds. Statistical tests compare with
  tolerances derived from standard errors and should fail for a wrong model.

## Code style

- PEP 8, lines up to about 110 characters; `ruff check` should pass for
  new or rewritten modules (the legacy code is not yet clean).
- NumPy-style docstrings for all public APIs, with examples where helpful.
- CamelCase classes, snake_case functions and methods, descriptive names,
  type hints where they help.
- Match the surrounding code: comment density, naming and idiom.

## Tests and documentation

- Run `uv run pytest -q` before proposing changes; parametrize tests across
  geometries and model families where behaviour should be the same.
- Code in the README and in `docs/source/how_to/custom_interactions.rst` is
  executed by `tests/readme_test.py` and `tests/docs_snippets_test.py`; keep it
  runnable. Regenerate the README images with
  `uv run python docs/images/readme/make_readme_images.py` when their code or
  the plotted model changes.
- Keep examples in docstrings, docs and tutorials in sync with the API. The
  tutorials must run top to bottom without saved outputs.
- Record user-facing changes in `CHANGELOG.md` under "Unreleased". When a
  roadmap item is done or a design decision changes, add a dated status note
  to `docs/development/usability_roadmap.md`.

## Contributing

- Commit or push only when asked. Development happens on `aidevelop`; the
  default branch is `master`, which Read the Docs builds until the new
  version is finished.
- PR title format: `[Fix|Feat|Docs] <one-line summary>`.
- PR body: include a "Testing Done" section and reference the issues
  addressed, if any.
