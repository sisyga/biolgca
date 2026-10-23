# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

### Fixed

- Errors raised in your own rules, reorientation terms, switch cues and
  operator constructors keep their type and traceback. Before, a `KeyError`
  became "unknown operator" or a `ValueError` naming only the key, and a
  `TypeError` inside a cue was reported as a parameter error.
- `check_interaction` checked no model for an identity-based
  `phenotype_switch` without `n_species` and reported "0 models checked, all
  passed"; it now checks them, and a check left with no model raises
  `ValueError`.
- `state.cells[name]` is a read-only copy; writing into it
  (`cells["r_b"][mask] = x`) changed nothing without a warning. Use
  `cells.set_trait`; code that edited the copy in place before passing it to
  `set_trait` takes `cells["r_b"].copy()`.
- `biolgca run/validate/sweep --plugins MODULE` finds a module next to the
  model file or in the working directory without `PYTHONPATH`; a module that
  is not found says where it was looked for.
- A reorientation term of your own put in `operators` no longer suggests the
  one-cue dict form, which then failed with "unknown operator"; it says to put
  the term in `ReorientationSpec(terms=[...])`.
- Square and hexagonal lattices given a `dims` tuple with more or fewer than
  two entries (e.g. `dims=(10, 10, 4)`) raise a `ValueError` that names
  `restchannels=` and `geometry='cubic'`. Before, the extra entries were
  dropped silently.
- Initial `nodes` of the wrong rank raise an error that names the expected
  shape, instead of "too many values to unpack". One species may be given as
  `dims + (1, K)`.
- `tumour_spheroid` computed hexagonal positions with the wrong rows shifted,
  so some lattice neighbours were 1.73 node spacings apart (initial ball,
  unstirred layer, radial rings).
- `plot_density` of multi-species models scaled colours to the channels of one
  species, so a full node looked like a half-full one; the default scale is
  now `n_species * K`. Density plots and animations of models without volume
  exclusion ignored `vmax`, so an animation took its colour scale from its
  first frame.
- `LGCA_Square.calc_velocity_correlation` and `calc_vorticity` counted each
  occupied channel of a model without volume exclusion as one cell.
- `get_prop` of identity-based models without volume exclusion ignored the
  property table passed in, and `calc_prop_mean()` without arguments raised
  `AttributeError`.
- Provenance of an uninstalled source checkout records the version "unknown"
  instead of a hard-coded "2.0.0".

### Changed

- `kappa` and `theta` of `go_or_rest` are the density cue: the relative
  density is the cells at the node over `state.capacity`, as for
  `{"name": "density"}` and the `resting` term. With volume exclusion and
  several species a full node now has density 1 (it had `n_species`), and a
  `StateSpec.capacity` set with volume exclusion counts (it was ignored).
  `go_or_grow.switch`, whose migrating and resting cells share the channels
  of a node, divides by `StateSpec.capacity` if the model sets one, else by
  K. Models of one species without `StateSpec.capacity` and models without
  volume exclusion are unchanged, as are the zoo, the examples and the
  tutorials.
- Initial `nodes` of volume-exclusion models with values other than 0 and 1
  raise a `ValueError`. Before, they were cast to booleans with a warning, so
  2 became one cell.
- The `biolgca` command shows the traceback of errors raised in user code, the
  notes ("step N was rolled back") of lgca's own errors, and `--version`; it
  warns when the `metadata.json` of an archived run cannot be read;
  `biolgca examples` lists the examples.
- Clearer messages: an unknown operator name suggests the closest registered
  name or the module to import; an unknown plugin parameter lists the
  parameters the rule takes; `vary` suggests the closest short name; NaN or
  out-of-range `switch_phenotype` rates, a trait name given as a probability
  (write `{"max": "p_rest"}`), a cell mask computed before `kill`/`divide`,
  and a reorientation term that keeps the species axis say what to do.
- `ModelRunResult` and `StateSpec` print compactly (arrays by shape and dtype).
- The warning about the rest channel of identity-based models without volume
  exclusion says which setting it changes and how to silence it.
- `lgca.base_extensions` is deprecated; import the classes from
  `lgca.ib_base`, `lgca.nove_base` and `lgca.nove_ib_base`.
- Deprecated `ParameterSpec.shape`, `dependencies` and `validator`, which
  nothing in lgca uses.

### Added

- `check_interaction` checks reorientation terms (and catches terms that draw
  from `numpy.random`), takes `fields=` for rules that read `state.field`,
  keeps the rule's first error as the `__cause__` of `InteractionCheckError`,
  and names the file and line of the rule.
- `replace=True` on `@interaction`, `@stack` and `@reorientation_term`, as the
  name-clash message says.
- `lgca.model.UncheckedChangeWarning` for a `reconfigure` change that could
  not be tried on a copy; `explorer.last_error` keeps the last error shown in
  the explorer's status line.
- CONTRIBUTING.md, CODE_OF_CONDUCT.md and GitHub issue templates; README and
  the docs home page state who BioLGCA is for and how it relates to other
  cell-based modelling tools; the README has a short project history.

### Docs

- Getting started explains ghost nodes and `lgca.nonborder`; the custom
  interaction guide gives the return shape of every reorientation coupling,
  the channel order per geometry, the module-qualified names of rules defined
  in a module, and that `kill`/`divide` shift cell positions; a new "Initial
  conditions" section covers the `region` and `from_npz` initializers and the
  `nodes` shapes and label conventions of every family; `directed_motion` and
  `steric_repulsion` are in the table of reorientation terms; installation
  from GitHub without a clone; `StateSpec.capacity` is described as a crowding
  scale; Observer hooks have docstrings.

### Tests

- New end-to-end tests of what users new to the package do
  (`tests/user_workflows_test.py`): writing an adhesion term, an
  oxygen-dependent go-or-grow switch in a module run from the command line,
  and a trait-evolution rule, checked against analytic expectations.
- Tests that repeated the function under test, checked only that code runs,
  or asserted private attributes now check results against independent
  expectations (conservation of cells, every recorded frame, every animation
  frame, standard-error tolerances, symmetric neighbourhoods); duplicates and
  tests of frozen legacy helpers were removed.

## 2.0.0 (2026-10-15)

Version 2.0 rebuilds BioLGCA around model specifications (`ModelSpec`) and
rules written as functions of the lattice state. Scripts written for 1.0 with
`get_lgca` and `timeevo` still run, but their interactions now run the new
rules: the dynamics agree in distribution, seeded runs give other numbers.
Results published with 1.0 are reproduced with the archived version they
cite; 1.0 is kept on the `v1` branch. Every change of 2.0.0, item by item, is
listed in
[docs/development/archive/releases/2.0.0.md](docs/development/archive/releases/2.0.0.md).

### Upgrading from 1.0

- Python 3.11 or newer. Minimum versions of the dependencies are declared
  (numpy 1.24, SciPy 1.12, matplotlib 3.9); pandas, pyamg (not on Python
  3.14), threadpoolctl and ipywidgets are new dependencies, and Matplotlib is
  part of the normal installation. JupyterLab is the `notebooks` extra, Mayavi
  with PySide6 the `plot3d` extra. The `plot2d`, `plot` and `plotting` extras
  and the `requirements*.txt` files are gone; the test, docs and dev tools are
  uv dependency groups.
- The legacy interaction modules are removed: `lgca.interactions`,
  `lgca.nove_interactions`, `lgca.ib_interactions`,
  `lgca.nove_ib_interactions`, `lgca.ms_interactions` and
  `lgca.identity_kernels`. `tanh_switch` is in `lgca.builtin_rules`,
  `mutation_matrix_from_trait_bins` in `lgca.legacy_names`; the old functions
  are kept unchanged in `tests/legacy` of the repository. The `wetting`
  interaction is gone; the model of Ilina et al. (2020) it served is the zoo
  entry `jamming`.
- The interaction names of `get_lgca` (`"alignment"`, `"go_or_grow"`, ...)
  run stacks of the new rules (`lgca.legacy_names`) with the old parameters
  and defaults. Where the rules differ, the details differ: growth is
  logistic, with birth and death decided at once, and identity-based
  go-or-grow switches before cells die. Multi-species models accept every
  interaction whose rules work with several species. In model files, the
  prefixed names (`"classical.alignment"`) are deprecated aliases that name
  the rules to use instead.
- Results change at the same parameters where rules were rescaled or
  corrected. `lgca.gradient` is the physical gradient (before, twice it, and
  three times on hexagonal lattices), so chemotaxis and aggregation need
  twice (hexagonal: three times) the earlier `beta`; chemotaxis takes its
  gradient with the ghost nodes and defaults to `beta=2` (was 5). The Moore
  lattice's `channel_weight` read the neighbour behind each channel instead
  of the one ahead (`nove_ib.evo_steric` changes). In a phenotype switch only
  the switching cell moves, to a random channel of its new species, and with
  volume exclusion it fails if that channel is occupied.
- The library no longer prints: messages are logged under the `lgca` logger
  at INFO level, and problems are warnings that point at your line. 1D and 2D
  plots open a new figure; pass `ax=` to draw into an existing one.

### New

- Model specifications: `ModelSpec` with `SpaceSpec`, `StateSpec`,
  `TimeSpec` and `InteractionPipelineSpec`, saved as JSON or YAML model files
  (large arrays in an array file whose hashes the model file records), with
  validation, migration of older files and provenance (versions, parameters,
  source hashes). The `biolgca` command validates, runs and sweeps model
  files and exports the curated examples. A run without a seed records the
  seed it drew; a step that fails is rolled back.
- Rules of your own: `@lgca.interaction` turns a function of the
  `LatticeState` into a rule whose cell operations keep the conservation law
  of its kind (`lgca.ContractWarning`); `@lgca.stack`,
  `@lgca.reorientation_term`, `@lgca.switch_cue`, `@lgca.mutation_effect`
  and `@lgca.reaction` add stacks, movement cues, cues of switching
  probabilities, mutation effects and field reactions.
  `lgca.testing.check_interaction` runs a rule on small models of every
  geometry and family and reports what it breaks.
- Rules without family prefix that work in every model family:
  `random_walk`, the movement cues (`polar_alignment`, `nematic_alignment`,
  `chemotaxis`, `aggregation`, `contact_guidance`, `persistent_walk`,
  `directed_motion`, `steric_repulsion`, `resting`), `birth_death`,
  `go_or_rest` with `go_or_grow.growth`, `phenotype_switch` and
  `trait_switch`. Each can act on some species or channels only.
- Switching probabilities that respond to cues (the density, fields, their
  gradients, the flux, traits, cues of your own) in tanh, Boltzmann or Hill
  form, wherever a rule takes a rate or probability.
- Identity-based models: `state.cells` with traits per cell and the
  operations `kill`, `divide`, `set_trait` and `move`; mutations as events
  with effects drawn from any NumPy distribution; families and lineages.
  Without volume exclusion the cells live in a table between rules, which
  makes go-or-grow about ten times faster.
- Fields that change during a run: the `pde` operator (diffusion, decay,
  production, secretion and uptake by cells, advection, reactions; implicit,
  explicit and steady solvers), `FieldRecorder` and field animations.
- Studies: `lgca.study.vary` and `sweep` (a pandas table over parameters and
  seeds, run in parallel), `CompiledModel.reconfigure` for changes during a
  run, and `lgca.explore`, a live view with controls in Jupyter, also of 3D
  lattices.
- The model zoo, `lgca.zoo`: eight models with notebooks that reproduce
  their results (the Allee effect of go-or-grow, the evolution of phenotypic
  plasticity, jamming in invasion, evolution at an invasion front, excitable
  media, clonal evolution of the go-or-grow switch, contact inhibition and
  the mode of tumour evolution, mutational meltdown). The research models of
  1.0 are stacks of the new rules in `lgca.research_models`, 4 to 18 times
  faster.
- Output and plots: `result.data` (and `lgca.data`) by name; observers for
  snapshots, movies and CSV files; `ax=` on 1D and 2D plots; animations that
  play in notebooks and save as GIF without ffmpeg; 3D movies and plots in
  one publication style.
- Eight tutorials (with Colab badges), how-to guides, concept pages and an
  example gallery with the `tumour_spheroid` example. CI runs on Linux,
  Windows and macOS against a uv lock file.

### Fixed (since 1.0)

- `get_lgca(ve=False)` without further arguments failed on every geometry;
  `get_lgca(ib=True, interaction="go_or_grow")` raised `IndexError` on an
  empty lattice; `get_lgca` with `n_species > 1` accepted interactions that
  crashed at the first step. Models without volume exclusion raised
  `TypeError` at the first step when given an initial state, and those with
  a capacity above the channel count and no rest channel started strongly
  polarized.
- Identity-based models without volume exclusion, whose labels start at 0,
  counted cell 0 as a family of its own; family counts and Muller plots
  change. `lgca.maxlabel` is a Python integer; with NumPy 1.x,
  `maxlabel + 1` was a float that cannot index trait arrays.
- Plots: property maps of identity-based models (2D: occupied nodes hidden;
  3D: uninitialised memory at empty nodes), plots of multi-species and
  identity-based models, ticks of hexagonal plots and discrete colour bars,
  3D isosurface levels and configuration animations.
