# Roadmap

Status: 2026-10-01. The open work and the decisions that hold. What was done,
and why, is in the archived plans and reviews ([archive/](archive/README.md));
the history of every note is in git.

Keep this file short: when an item is done, delete it here and record the
user-facing change in `CHANGELOG.md`; put a long design or review in its own
file and archive it when its work is done.

## Where the project stands

The usability plan of 23 September (phases 0 to 4) and the architecture plan of
the 30 September review are done, apart from the items below:

- Rules are decorated functions of a `LatticeState`, the same for all model
  families; `get_lgca` names are stacks of these rules.
- Models are `ModelSpec`s: saved as JSON/YAML, swept (`lgca.study.sweep`, the
  `biolgca` CLI), explored live (`lgca.explore`), changed while running
  (`CompiledModel.reconfigure`), with provenance in their metadata.
- Rules keep the contract of their kind (`ContractWarning`), steps are applied
  as a whole or not at all, failed runs keep what they recorded.
- Fields (`pde`): explicit, implicit and steady solvers; saturating uptake and
  reactions by Newton's method.
- The model zoo has eight entries; tutorials 1 to 8; CI on Linux, Windows and
  macOS.

## Open

### Release and distribution (when the user decides to publish)

- PyPI release with a trusted-publishing workflow, versions from git tags.
- Zenodo integration for a DOI per release.
- At the merge to `master`: in the tutorial notebooks, replace `aidevelop` by
  `master` in the Colab badges and the install cell; after a PyPI release,
  install `biolgca` from PyPI there.
- Check `lgca.explore` by hand in Colab and VS Code (so far only in a Jupyter
  kernel through nbclient).

### Features for later

- Theory next to simulation (`lgca.theory`): mean-field equations and linear
  stability (polar and nematic alignment, aggregation, chemotaxis), compared
  with simulations; also for cell–field models.
- Course pack: one-semester exercises for MSc students (physics, applied
  mathematics, computational modelling), a 45-minute exercise every second
  week, solutions public in the repository, figures and animations.
- An advanced example: multispecies tumour growth without volume exclusion on
  hexagonal and 3D Moore lattices, nodes coloured by the local mean class
  property.
- Fields: systems of fields solved together, heterogeneous or field-dependent
  diffusion, fields on a finer grid than the lattice, fixed values at interior
  nodes (vessels), a helper for physical units.

### Optional engineering (when profiling or users ask)

- Field solver, direct backend: a chord finishing step and SuperLU's
  `MMD_AT_PLUS_A` ordering (measured: halves the cost of boundary-dominated
  steady fields; 1.45 to 3.4x faster factorizations).
- Field solver: `reaction(derivative=..., monotone=...)` so that reactions
  known to be monotone get full Newton steps (self-enhancing reactions now
  converge at Picard's rate: a logistic step from 0.1 takes 19 of the 20
  allowed iterations); skip the error estimate where the floor is active
  everywhere; the BDF/Radau Jacobian of the explicit solver still uses a
  reaction's loss rate, not its derivative.
- Contracts of class-based operators: fingerprints before and after each
  operator under `strict_contracts()` (20 to 26 % of a step), once third-party
  operators exist.
- Read-only arrays and frozen containers in `model.spec`.
- Checkpoints: `CompiledModel.checkpoint()`/`restore()` on the step
  transaction, an NPZ checkpoint of identity-based states, the Explorer's
  trial step as a rolled-back step instead of a deep copy.

### Known small issues

- Traits named only in a mutation's probability (a trait-valued `max`,
  `kappa` or `theta`) are not checked before the first block writes.
- `vary` on a spec that holds a single-cue operator object fails: it treats
  every object with `.terms` as a `ReorientationSpec`.
- `lgca.explore` cannot run a `from_npz` model (no `resource_base`); the
  message "Relative initializer resources require resource_base" also appears
  for an absolute path with `trusted_paths=True`.

## Decisions that hold

- `get_lgca` is the quick start and runs the standard models (tutorials 1 and
  2); `ModelSpec` follows from tutorial 3. `get_lgca` gets no new names.
- No deprecation release for the legacy interactions: published papers cite
  Zenodo snapshots of the old `master`.
- pandas is a dependency (`sweep`); JupyterLab is the `notebooks` extra.
- Preferred citation: Syga et al. 2026 (EPJ ST); the 2021 paper describes
  BIO-LGCA.
- The species axis is internal; public shapes stay backward compatible. The
  kind that changes species or traits is `phenotype_switch`; operators run in
  the order they are listed; Boltzmann sampling is the default for
  reorientation, not a requirement; species are not tied to channels.
- 3D models are explored with Matplotlib, not Mayavi.
- The 30 September decisions (contracts and enforcement, rollback, templates,
  Newton, provenance) are summarized in `AGENTS.md` under "Design rules"; the
  measurements behind them are in
  [the archived proposals](archive/reviews/2026-09-30-architecture-proposals.md).
