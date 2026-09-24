# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

### Added

- `@lgca.interaction` turns a function `rule(state, r_d=0.1)` of a
  `LatticeState` into a registered interaction: parameters, defaults and
  descriptions come from the signature and docstring, calling
  `rule(r_d=0.2)` gives the entry for `InteractionPipelineSpec(operators=[...])`,
  and the name works in model files. The rule declares its kind, the families
  it supports (`"classical"`, `"nove"`) and optionally geometries, a number of
  species and momentum conservation; models it was not written for are
  rejected when they are built, and the conservation laws are checked after
  every call.
- `lgca.testing.check_interaction(rule, parameters)` runs an interaction on
  small seeded models of every supported geometry and family, with one and two
  species and periodic and reflecting boundaries, and reports broken
  conservation laws, invalid states, ghost-node changes that reach the
  lattice and unseeded randomness. `expected_growth=` compares the measured
  growth per step with the expected one.
- Go-or-grow as a two-species model: `go_or_grow.switch` (migrating and
  resting cells switch species depending on density), `go_or_grow.growth`
  (death, and division of resting cells into rest channels, with separate
  death rates if wanted) and `species_random_walk`. With
  `capacity="legacy"` (default) they reproduce `classical.go_or_grow` and
  `nove.go_or_grow` in distribution; the go-or-grow example uses them.
- `LatticeState` operations accept channel sets per species, e.g.
  `channels={0: "velocity", 1: "rest"}`.
- `lgca.LatticeState`: the interior of a classical model with a species axis
  (`dims + (n_species, K)`, also for one species) and operations with a
  defined meaning per cell: `remove_cells`, `divide_cells`, `add_cells`,
  `switch_phenotype` and `shuffle_cells`. Each works the same with and
  without volume exclusion (without it, cells in one channel die, divide and
  switch one by one), places cells only in free channels, and draws
  from the model's random generator. `commit()` checks the conservation law
  of the interaction's kind and writes the state back.
- The model and pipeline specifications (`SpaceSpec`, `StateSpec`,
  `TimeSpec`, `ModelSpec`, `InteractionPipelineSpec`, `ReorientationSpec`,
  `ReorientationTermSpec`, ...) document every field, including defaults and
  units: `density` is the mean number of cells per node, and each
  reorientation term states the score it adds.
- Every parameter of the built-in interactions has a plain-language
  description, and `describe_plugin(name)` prints a readable summary of an
  interaction: purpose, phase, model families, what it conserves and its
  parameters with defaults.
- 1D and 2D plots accept `ax=` to draw into a given axes, e.g. one panel of
  `plt.subplots`. In figures with constrained layout, colour bars stay inside
  their panel.
- Animations of recorded 2D runs accept `save_path=` and `save_kwargs=`, like
  the 3D ones, and play in Jupyter notebooks when they are the last line of a
  cell. GIF files are written with Pillow, so they need no ffmpeg. Tutorials 1
  and 2 show animations.
- `get_lgca(interaction=my_function, my_rate=0.1)` uses a function of the LGCA
  object as the interaction step; the remaining keyword arguments are stored
  in `lgca.interaction_params`. Previously this raised `AttributeError`.
- `CITATION.cff` with the BIO-LGCA method paper (Deutsch et al. 2021) and the
  heterogeneity paper (Syga et al. 2026).
- The README runs as written: `tests/readme_test.py` executes its code, and
  `docs/images/readme/make_readme_images.py` renders its figures from it.
- Versioned, shareable `ModelSpec` simulations with canonical JSON, optional
  YAML authoring, validation, migration hooks, runtime provenance, and CLI
  commands for validation, inspection, examples, and runs.
- A typed interaction-plugin registry and composable execution pipeline for
  adding and inspecting reusable interactions.
- Observer-based simulation output, plotting snapshots and movies, scheduled
  callbacks, curated runnable examples, and repeatable performance benchmarks.
- Movies of 3D simulations: `animate_density`, `animate_flux` and
  `animate_config` on cubic and Moore lattices accept `save_path=` (e.g.
  `density.mp4` with ffmpeg or `density.gif` with Pillow) and `save_kwargs=`
  with the options of Matplotlib's `Animation.save`. Frames are rendered
  offscreen, so no window opens.
- `PlotSnapshotObserver` and `AnimationObserver` support 3D models. Snapshots
  are saved as images and movies are written offscreen without blocking the
  run; the new plot kind `density_cubes` selects the voxel density plot.
  Observers now reject a plot kind the model does not provide (e.g. `flow` on
  a 3D lattice) before the simulation starts.
- Six maintained, executable teaching notebooks covering LGCA fundamentals,
  collective movement, interaction composition, population dynamics,
  evolutionary LGCA, and reproducible student projects.
- Focused regression coverage for propagation, boundary conditions,
  initialization, time evolution, interactions, plotting contracts, model
  persistence, the CLI, and installed-package smoke tests.

### Changed

- Interactions run in the order they are listed in
  `InteractionPipelineSpec.operators`; the fixed order birth/death, phenotype
  switch, reorientation is no longer enforced. `allow_custom_order` is
  deprecated and ignored, and model files no longer contain it.
- A phenotype switch moves cells between species, so a classical model with
  one species rejects it with an explanation. `classical.go_or_rest` and
  `nove.go_or_rest`, which move cells of one species between velocity and
  rest channels, are now reorientations.
- The `birth_death` interaction of multispecies models processes all nodes at
  once instead of looping over them in Python, which makes it about 45x faster
  (100 steps on 50 x 50 nodes: 0.17 s instead of 7.7 s). Runs remain
  reproducible from their seed, but seeded trajectories differ from earlier
  versions.
- Standalone 1D and 2D plots use Matplotlib's constrained layout with colour
  bars as inset axes, so axis labels, colour bars and their labels stay inside
  the figure; animations keep the layout of their first frame. Default figure
  heights follow the lattice's aspect ratio with room for labels (wide
  lattices got figures about one inch tall). 1D history plots label their time
  axis "Time step k", or "Recorded time step k" when not every step was
  recorded.
- A model run without `time.seed` draws a seed and records it in
  `result.spec.time.seed`, `result.metadata["seed"]` (with
  `metadata["seed_drawn"] = True`) and, for command-line runs, in
  `model.resolved.json`. Previously the seed was not recorded and such runs
  could not be repeated.
- Curated examples show their effect at their own settings: alignment and
  nematic alignment run at density 0.5 with beta 3 and order visibly,
  chemotaxis uses a steeper signal with reflecting walls, go-or-grow runs for
  100 instead of 15 steps so that its Allee effect (a small colony shrinks) is
  visible, and `build_spec(kappa=-4.0)` gives the invading contrast case, and
  the identity-based tumour starts from a seed, grows and mutates its
  switching steepness.
  `tests/examples_effect_test.py` checks each effect. Examples no longer
  modify `sys.path`, and the gallery lost its output listings of one-step runs.
- 1D and 2D plots open a new figure instead of drawing into the current one,
  so consecutive plot calls in a script no longer overlay each other. An empty
  current figure (e.g. from `plt.figure(figsize=...)`) is still used. To place
  a plot in an existing figure, pass `ax=`.
- `ipywidgets` is installed with BioLGCA, so progress bars render in notebooks
  instead of warning that IProgress is missing.
- The library no longer prints. Messages about default interaction parameters
  are logged at INFO level under the `lgca` logger (enable them with
  `logging.basicConfig(level=logging.INFO)`), and problems such as too few rest
  channels for go-or-grow are raised as `UserWarning`. All warnings,
  including deprecations, point at the user's line that caused them instead of
  a line inside BioLGCA.
- Registering a plugin name again from the module that registered it replaces
  the entry, so notebook cells that define plugins can be rerun. Replacing a
  plugin of another module, such as a built-in, requires
  `register_plugin(..., replace=True)`.
- The README was rewritten around runnable examples with figures.
- `profiling.py` moved to `benchmarks/profiling.py`. Maintainer notes (plans,
  milestone reports, implementation ownership) moved from the user guide and
  `benchmarks/` to `docs/development/`, which also holds the usability
  roadmap. The project's documentation URL now points to readthedocs.
- BioLGCA requires Python 3.11 or newer and supports Python 3.14; Python 3.10
  is no longer supported.
- Runtime and optional dependencies declare the oldest versions the test suite
  and tutorial notebooks pass with: numpy 1.24, scipy 1.9.2, matplotlib 3.7,
  tqdm 4.64.1, JupyterLab 4.0, PyYAML 6, Mayavi 4.9 and PySide6 6.4. Previously
  no minimum was declared, so pip could combine BioLGCA with releases that fail
  at runtime. numpy 1.24 is needed to reject ragged node arrays, matplotlib 3.7
  for inline plots in current Jupyter, and Mayavi 4.9 is the first release with
  binary wheels (older versions no longer build against current VTK).
- The development environment is managed with uv. A committed `uv.lock` and
  `.python-version` pin the complete environment; `uv sync` installs it and CI
  tests against the lock file. Test, documentation and lint tools moved from
  the `test`, `docs` and `dev` extras to PEP 735 dependency groups of the same
  names. The redundant `requirements*.txt` files and the `plot2d`, `plot` and
  `plotting` extras were removed; use the `plot3d` extra for Mayavi.
- Identity-based birth and birth-death interactions (with and without volume
  exclusion) draw all daughter birth rates of a time step in one vectorized
  truncated-normal call, which makes them about 10x faster. Runs remain fully
  reproducible from their seed, but seeded trajectories differ from those of
  earlier versions. Legacy functions and ModelSpec operators share the same
  kernels in `lgca.identity_kernels`.
- `lgca.gradient` returns the physical gradient in lattice units on every
  geometry, matching the composed chemotaxis term. Previously 1D, square,
  cubic and Moore lattices returned twice and hexagonal lattices three times
  the gradient. Aggregation, the default 2D and 3D chemotaxis fields,
  `go_or_grow_kappa_chemo` and `calc_vorticity` inherit the new scale: at the
  same `beta`, the effective sensitivity is half the previous value (a third
  on hexagonal lattices). Multiply earlier `beta` values by 2 (hex: 3) to
  reproduce previous results. The normalized 1D default chemotaxis field is
  unchanged.
- `DensityRecorder` stores densities as signed integers by default: `int16`
  with volume exclusion and `int32` (widened to `int64` on demand) without,
  instead of `float64`. This cuts recording memory by 4x or 2x. Pass
  `dtype=float` to keep floating-point output.
- Three-dimensional plots of cubic and Moore lattices share one publication
  style (`lgca.mayavi_style`): white background, a thin domain box with sparse
  ticks, a fixed oblique camera with little perspective distortion,
  anti-aliasing, correctly blended translucent isosurfaces and compact colour
  bars whose integer bins are labelled at their centres. Node `i` is drawn at
  the centre of the cell `[i, i + 1]`, so glyphs, isosurfaces and the domain box
  line up. Flux arrows are centred on their node and scaled so that the
  largest flux is 0.9 lattice units long; configuration arrows have a fixed
  length and, when a channel can hold several particles, are coloured by its
  population; sphere volumes are proportional to particle numbers. In-scene
  titles were removed; animations show the recorded time in a corner label.
  All model families on cubic and Moore lattices use one implementation.
  Animations return the Mayavi `Animator` and accept `show=False`;
  `plot_density` accepts `smooth=` for display-only Gaussian smoothing, and all
  3D plots accept `size=` and `view=`.
- The `plot3d` extra installs PySide6, which Mayavi needs to open windows and
  run animations.
- Matplotlib and JupyterLab are part of the normal installation so students can
  open the maintained notebooks and plot results without selecting extras.
  Three-dimensional Mayavi rendering remains optional.
- Documentation builds now start from clean generated sources and treat Sphinx
  warnings and notebook execution failures as errors.
- Maintained notebooks construct `ModelSpec` and interaction pipelines in
  visible cells. Historical factory-API notebooks and the research-project
  example now live in clearly labelled archive directories.

### Fixed

- `get_lgca(ve=False)` without further arguments failed on every geometry:
  the default interaction (density-dependent alignment) needs zero rest
  channels, but the default was one. Without an interaction and
  `restchannels`, the factory now uses zero rest channels.
- The README figures were ignored by git (`*.png`) and missing from the
  repository; they are now committed, and a test checks that every README
  image is tracked.
- The registry described `phenotype_switch` as channel-preserving. It lets
  cells of a multispecies LGCA change species; the number of cells at a node is
  conserved, but a switch redistributes the node's cells over its channels.
- Hexagonal lattice plots placed y-axis ticks between rows and labelled them
  with truncated row numbers (e.g. 49 instead of 50); ticks are now on whole
  rows at round intervals.
- Discrete colour bars of 1D density plots had ticks on the bin edges
  (-0.5, 0.5, ...) instead of on the integer particle numbers.
- `get_lgca(ib=True, interaction="go_or_grow")` raised `IndexError` when the
  initial lattice contained no cells.
- Particle-number-conserving interactions, including phenotypic switching,
  transition the complete channel state from `s` to `s'` while preserving the
  number of particles, consistent with legacy random-walk, alignment, and
  chemotaxis interactions for volume-excluding models.
- Density plotting and animation contracts across volume-excluding and
  non-volume-excluding model families.
- Stale generated autosummary pages causing strict documentation failures in
  reused working directories.
- NoVE lattices with `capacity` above the channel count and no rest channel
  started with the surplus particles in one velocity channel, i.e. strongly
  polarized. They now start isotropic.
- NoVE interactions raised `TypeError` on the first step when the initial
  state was given explicitly instead of drawn at random.
- `get_lgca` with `n_species > 1` accepted single-species interactions that
  crashed on the first step; it now rejects them at construction.
- Legacy chemotaxis defaulted to `beta=5`, the ModelSpec plugin to `beta=2`;
  both now use 2.
- Plotting: 1D `plot_density` with its default colour bar, flux, flow and
  configuration plots of identity-based and multi-species models (labels were
  plotted instead of counts or the species axis was rejected), square live
  density animations, stale NoVE identity-based property plots,
  `plot_prop_2dhist` without seaborn, and `list_families_alive` for crowded
  nodes.
- 3D plotting: identity-based property maps without volume exclusion drew
  uninitialized memory (values around 1e74) for empty nodes; they now show the
  mean property of occupied nodes only. Isosurface levels were clamped to the
  data range of the first animation frame. Configuration animations without
  rest channels divided by zero, NoVE animations ignored the recorded sample
  times, and multi-species configuration plots ignored the species axis.
  Identity-based live 3D animations were not implemented and now run.
- 2D identity-based property maps hid the occupied nodes: square lattices
  showed an empty plot and hexagonal lattices showed only the empty nodes.

### Removed

- The unused `lgca.interactions.disarrange` helper.
- The legacy `wetting` interaction and its `classical.wetting` ModelSpec
  port. The model is planned as an advanced tutorial with a new
  implementation; see the planned documentation topics.

### Compatibility

- The `get_lgca(...)` factory and legacy `timeevo(...)` workflow remain
  supported for interactive use.
- This development branch still reports package version `0.1.0`; a release
  version will be chosen as a separate release decision.
