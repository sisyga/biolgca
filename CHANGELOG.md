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
- `DensityRecorder` stores densities as signed integers by default: `int16`
  with volume exclusion and `int32` (widened to `int64` on demand) without,
  instead of `float64`. This cuts recording memory by 4x or 2x. Pass
  `dtype=float` to keep floating-point output.
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
