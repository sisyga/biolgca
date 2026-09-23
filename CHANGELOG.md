# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

### Added

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
