# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

Fixes for the findings of the code review of 2.0.0 (November 2026).

### Changed

- Models without volume exclusion have the rest channels they are given on
  every lattice. In 1D and on the square and hexagonal lattices,
  `restchannels=3` gave one rest channel and a capacity raised by two, and
  `nodes` with several rest channels were refused; now every rest channel is
  a channel of the state, as on the 3D lattices. Runs with `restchannels > 1`
  on these lattices have more channels than before (the same default
  capacity). Identity-based models without volume exclusion keep their one
  rest channel.
- `nodes` of models without volume exclusion are `int64` however the model
  was filled (`lgca.base.COUNT_DTYPE`). Nodes given by the caller were
  `uint64`, random ones `int64`; differences of unsigned counts wrap around.
  A channel holds at most 2**63 - 1 cells.
- The recordings on a model (`lgca.nodes_t`, `lgca.dens_t`, ...) are those of
  its last run. A run that did not record the nodes left the `nodes_t` of the
  run before next to its own `dens_t`. Keep the arrays of an earlier run in
  variables of your own.
- `dims` is read in one way by every lattice class and by `ModelSpec`: one
  integer of any numeric type (also a NumPy integer or `6.0`) for every axis,
  or one length per axis in any sequence. Without volume exclusion, a NumPy
  integer gave a 50 x 50 lattice with a warning. 1D lattices no longer ignore
  further entries of `dims`, and 3D lattices without volume exclusion default
  to 10 x 10 x 10 like the others (was 50 x 50 x 50).
- With `nodes`, the capacity of a model without volume exclusion defaults to
  the number of channels of the nodes; on 3D lattices it was that of the
  `restchannels` argument, below the channels of nodes with more rest
  channels. `restchannels` beyond those of the nodes still add to it.
- `ChannelDensityRecorder` records every model without volume exclusion
  (models with one species and no identities had no `channel_pop`).
- `get_lgca` reads `nodes` given as nested lists.
- The capacity of a model defaults to its number of channels, whatever the
  number of species. With volume exclusion and several species it was the
  number of species times the channels, so a density cue or a crowding
  threshold now divides by `K` instead of `n_species * K` unless
  `capacity` is given; such a node holds up to `n_species * K` cells, so the
  relative density can exceed 1. Set `StateSpec(capacity=...)` to keep the
  old denominator.
- `nb_sum` returns floats on every lattice. Square and 3D lattices kept the
  type of the values: sums of booleans stayed booleans and small unsigned
  integers wrapped around.
- A parameter that belongs to another legacy interaction warns
  ("parameter `r_b` defined but unused") instead of being stored in silence;
  unknown parameters are refused as before.
- The class of the 3D Moore lattice with volume exclusion is `LGCA_Moore`,
  like `IBLGCA_Moore` and `NoVE_LGCA_Moore`; `LGCA_3dMoore` stays as an
  alias.
- `plot_prop_spatial` of 1D identity-based models takes the same arguments
  with and without volume exclusion (`cbar`, `cbarlabel`, `figkwargs=None`).
- `lgca.__version__` gives the installed version, and `from lgca import *`
  imports the public names only (`lgca.__all__`).
- The documentation calls a model classical only if it has volume exclusion
  and no identities (a boolean state); models that count cells without volume
  exclusion are "without identities".
- CI checks the package for unused imports and undefined names
  (`ruff check lgca --select F`); the unused imports are removed.
- Documentation: the installation page names PyPI and the front page the
  Zenodo DOI; the table of `get_lgca` interaction names per model family is
  read from the package when the docs are built
  (`lgca.legacy_names.legacy_names_table`); ghost nodes come in `r_int`
  layers; the `inflow` boundary is documented as existing on square and
  hexagonal lattices only; `plot_prop_timecourse` documents its band as the
  standard error of the mean, which it is; the boundary aliases `fixed`,
  `no_flux` and `noflux` are listed; leftover placeholder notes are removed.
- The package is written "biolgca" throughout the README, the documentation
  and the messages (BIO-LGCA remains the name of the modelling class).

### Fixed

- Identity-based models with volume exclusion gave daughters the labels of
  living cells after the `region` initializer, and `random_reset` kept the
  traits of the cells it replaced: the labels are counted again, and
  `random_reset` sets the traits and families of a new model (a model built
  from a `ModelSpec` with traits asks for a new model instead).
- `set_r_int` on an identity-based model without volume exclusion kept the
  lookup tables of the old lattice and failed at the next step.
- `lgca.capacity` and `print(lgca)` of an identity-based model without volume
  exclusion reported the number of channels while growth used the default
  capacity of the interaction (8); `set_interaction(capacity=...)` was
  ignored, and a capacity given at construction was replaced by the default
  of the next interaction.
- `calc_family_generations` raised `RecursionError` for lineages of about a
  thousand generations; it and `propagate_ancestor_to_descendants` rebuilt
  their tables on every call. Families without ancestry
  (`track_inheritance=True`) had no `maxfamily`, so `calc_family_pop_alive`
  and `recordfampop` failed.
- `calc_mean_alignment` and `OrderParameterRecorder` raised a `ValueError`
  for models without volume exclusion and several species; the flux of a node
  is that of the cells of all its species.
- 3D models with several species refused `nodes` with several rest channels,
  and identity-based 3D models without volume exclusion ended in an
  `AttributeError` for `nodes` given as cell counts.
- Errors that named neither the value nor what to do: integers and the
  density of a `ModelSpec` (`restchannels=None`, `steps=inf`), sections of
  another type (a list of operators as `dynamics`), operators and observers
  that are not objects or entries, `OrderParameterRecorder` and
  `ChannelDensityRecorder` on models with volume exclusion, the family
  analyses (`calc_family_generations`, `muller_plot`, ...) of models without
  families, and `n_species` and `capacity` of `get_lgca`.
- `lgca.interactions` and the error for an unknown interaction listed names
  the model cannot use: those of the other number of dimensions and, for
  several species, those written for the other volume exclusion.
- `calc_normalized_entropy` of a model without volume exclusion on one node
  divided by zero (`nan` and a warning); it returns 1.
- Identity-based models without volume exclusion had empty ghost nodes until
  the first step, so interactions of the first step saw no neighbours across
  a periodic boundary.

## 2.0.0 (2026-11-04)

Version 2.0 rebuilds biolgca around model specifications (`ModelSpec`) and
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
