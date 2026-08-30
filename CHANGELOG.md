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

### Compatibility

- The `get_lgca(...)` factory and legacy `timeevo(...)` workflow remain
  supported for interactive use.
- This development branch still reports package version `0.1.0`; a release
  version will be chosen as a separate release decision.
