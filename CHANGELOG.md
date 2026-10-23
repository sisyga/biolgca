# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

Nothing yet.

## 2.0.0 (2026-10-23)

Version 2.0 rebuilds BioLGCA around model specifications (`ModelSpec`) and
rules written as functions of the lattice state. Scripts written for 1.0 with
`get_lgca` and `timeevo` still run, but their interactions now run the new
rules: the dynamics agree in distribution, but seeded runs give different
numbers.
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
  twice (hexagonal: three times) the earlier `beta`. Chemotaxis takes its
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
