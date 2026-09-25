# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

### Added

- `birth_death(channels=...)`: only the cells in a set of channels die and
  divide, and their daughters go to that set, e.g. `channels="rest"` for
  growth by resting cells; with volume exclusion the logistic factor counts
  the cells of the set.
- `go_or_rest` for several species, with `kappa` and `theta` per species.
- Every part of the dynamics can act on some species only: `species` for
  `birth_death`, `go_or_rest`, the single-cue operators and
  `ReorientationSpec(parameters={"species": ...})` (the other species keep
  their cells and channels), and `sensed_species` for the cues computed from
  the cells (whose cells they sense), e.g. species 1 aligning with its own
  kind while species 0 follows a signal. `LatticeState.sensing(species)`
  gives rules the state of some species only.
- `trait_switch`: cells of identity-based models change their traits at any
  time by events written as mutations (probability, effects, bounds), e.g.
  switching on an alignment strength; the new effect operation `"set"`
  replaces a trait by a value, and `"when"` limits an event to cells with
  some values of a trait (so two or more states switch at their own rates).
- Switching probabilities that respond to the surroundings (`lgca.switching`):
  the rates of `phenotype_switch` and the probabilities of `trait_switch`
  events and mutations can be `{"max": p, "cues": [...]}`, i.e.
  `p (1 + tanh(Σ kappa (cue - theta))) / 2` as in go-or-grow, with the cues
  `density`, `field`, `gradient` and `flux` (for chosen species), traits as
  cues or as per-cell sensitivities, and cues of your own
  (`lgca.switch_cue`).
- The Boltzmann form of switching probabilities: `{"rate": r, "cues":
  [{"name": "density", "beta": 3.0}, ...]}` is the weight
  `w = r exp(Σ beta cue)` of a switch against staying (weight 1). An event
  of `trait_switch` or of a mutation happens with probability `w / (1 + w)`;
  a row of `phenotype_switch` rates chooses among its switches and staying,
  `w_ab / (1 + Σ w_ab')`, without the bound of 1 on the row. `beta` may name
  a trait in identity-based models. For two states it is the tanh form with
  `beta = 2 kappa` and `rate = exp(-2 kappa theta)`.
- `directed_motion`, a cue by which cells move along a given vector field
  (`{"name": "directed_motion", "parameters": {"beta": 2, "field": "flow"}}`).
- The research models of earlier versions as stacks of the generic rules, in
  `lgca.research_models` and under the legacy names without family prefix:
  `go_or_grow_kappa`, `go_or_grow_kappa_chemo`, `go_or_grow_glioblastoma`,
  `evo_steric`, `birthdeath_cancerdfe`, `go_and_grow_mutations`,
  `birthdeath_discrete` and `excitable_medium`. They work in identity-based
  models with and without volume exclusion, match the legacy interactions in
  distribution (`tests/research_models_test.py`) and run 4 to 18 times
  faster (`benchmarks/research_models.py`; the excitable medium 2 times). The
  how-to page "Research models from generic rules" shows how they are built.
- `lgca.stack`: a decorator that turns a function returning a list of
  operators into one operator with its own parameters, e.g. a published
  model. Parameters named in `traits=` give the initial cell traits.
- Mutations as events (`lgca.mutations`): `"mutation": {"probability": p,
  "traits": {"r_b": effect, ...}}`, or a list of such kinds of mutation.
  An effect is a draw from any NumPy distribution (`{"distribution":
  "exponential", "scale": 0.01}`), a fixed value, or a function of your own
  registered with `lgca.mutation_effect`; it adds, subtracts or multiplies,
  within optional bounds (clipped or redrawn). With `new_family=True`,
  mutated daughters found new families (`Cells.found_families`).
- `birth_death` works in every model family and takes rates per cell in
  identity-based models (`"birth_rate": "r_b"` names a trait), `mutation`
  and `new_family`, and a `mutation_matrix` for the species of daughters in
  classical models with several species. It matches the legacy
  `classical.birthdeath` (in the mean), `ib.birthdeath`,
  `nove_ib.birthdeath` and `multispecies.birthdeath` in distribution
  (`tests/birth_death_test.py`). `go_or_grow.growth` takes the same
  `mutation` and `new_family` options.
- Reorientations restricted to a channel set: `ReorientationSpec(parameters=
  {"channels": "velocity"})`, also for the single-cue operators, moves only
  the cells in these channels and only among them, in every family.
- `steric_repulsion`, a term by which cells avoid crowded neighbours, and
  `LatticeState.neighbor_values` (a field at the neighbour each channel
  points to). `go_or_rest` and `go_or_grow.switch` take
  `density="neighbourhood"`: the switch senses the mean density of the node
  and its neighbours.
- Names without family prefixes for movement: `random_walk` (cells move to
  random channels, optionally within a channel set or for some species) and
  every built-in cue as an operator of its own (`polar_alignment`,
  `nematic_alignment`/`nematic`, `persistent_walk`/`persistent_motion`,
  `aggregation`, `chemotaxis`, `contact_guidance`, `resting_bias`), e.g.
  `{"name": "chemotaxis", "parameters": {"beta": 2.0, "field": "signal"}}`.
  They work in every model family. `polar_alignment` takes `include_center`
  and `normalize` (density-independent alignment); with them it is
  `nove.dd_alignment` and `nove.di_alignment` seed for seed.
- Identity-based models without volume exclusion hold their cells in a table
  between rules: boundary conditions and propagation move it with lookup
  tables derived from the geometry's own transport code, and `lgca.nodes`
  builds the lists of labels only when it is read. A go-or-grow step on a
  100 x 100 lattice takes 13 ms (legacy `nove_ib.go_or_grow` 125 ms); code
  that reads or edits `lgca.nodes` keeps working. `NodeRecorder` stores the
  cells of such models as compact tables in `lgca.cells_t` (label, node and
  channel of every cell per recorded time); `lgca.nodes_t` builds the lists
  from them when it is first read.
- Reorientation terms scaled by a cell trait:
  `ReorientationTermSpec("polar_alignment", beta=1.0, trait="alignment")`
  gives every cell of an identity-based model its own strength (also
  `term(beta=..., trait=...)` for terms written with
  `@lgca.reorientation_term`). Without volume exclusion each cell draws its
  channel from its own weights; with it, the labelled node states follow the
  joint Boltzmann distribution, sampled by a Metropolis chain per node, run
  for all nodes at once, with `ReorientationSpec(parameters={"sweeps": 10})`
  proposals per channel (51 ms per step on a 100 x 100 hex lattice).
- Rules for individual cells of identity-based models: `state.cells` holds
  one entry per living cell (label, node, channel) with its traits
  (`cells["kappa"]`) and the operations `kill`, `divide` (daughters inherit
  all traits; `new_family=True` starts a family per daughter), `set_trait`,
  `move` and `pick`, which act on all cells at once instead of looping over
  nodes. `remove_cells`, `divide_cells` and `shuffle_cells` work in
  identity-based models too. `StateSpec(traits={"kappa": 4.0})` gives the
  initial cells their traits, one value for all or one per cell.
  `check_interaction(..., traits=...)` checks rules that read traits.
- `go_or_rest` and `go_or_grow.growth` run in identity-based models, and
  their parameters take a trait name for values per cell, e.g.
  `go_or_rest(kappa="kappa", theta=0.6)`; `go_or_grow.growth(mutation=
  {"kappa": 0.2}, new_family=True)` mutates daughters and tracks lineages.
  One step matches `ib.go_or_grow` and `nove_ib.go_or_grow` in distribution;
  a step on a 100 x 100 lattice takes 23 ms with volume exclusion (legacy
  243 ms) and 89 ms without (legacy 226 ms).
- `ReorientationSpec` works in every model family. Without volume exclusion,
  every cell chooses its channel independently with `P(i) ∝ exp(Σ beta w_i)`
  (a multinomial draw per node and species), where `w_i` is the score of one
  cell in channel `i`; without terms this is `nove.random_walk`, and
  `polar_alignment` is `nove.dd_alignment`. Identity-based models, with and
  without volume exclusion, update their cell numbers the same way as the
  classical models and then place each node's cells on the occupied channels
  at random. Terms written with `@lgca.reorientation_term` work unchanged in
  all of them. With volume exclusion, seeded results are unchanged.
- Rules written with `@lgca.interaction` can declare the identity-based
  families `"ib"` and `"nove_ib"`. `LatticeState` reads their cell numbers
  (labels > 0, or the length of each channel's list), and `shuffle_cells`
  moves the labelled cells within a channel set, so cells outside it keep
  label and channel; the other operations and assigning `state.counts` are
  refused for now. `channel_random_walk` supports these families, and
  `check_interaction` checks them, including that every node keeps its cells.
- `@lgca.reorientation_term(coupling=...)` defines a new cue for
  `ReorientationSpec` as a function that returns a field from the lattice
  state: a vector per node (`"flux"`, cells move along it), a tensor
  (`"nematic"`), a number (`"rest"`) or channel weights (`"channels"`).
  Calling the term with `beta=` and its parameters gives the
  `ReorientationTermSpec`. The built-in terms are defined the same way, with
  unchanged results.
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
- Go-or-grow from separate rules: `go_or_rest` (cells move between velocity
  and rest channels depending on density), `go_or_grow.growth` (death, and
  division of resting cells into rest channels, with separate death rates if
  wanted) and `channel_random_walk` (a random walk within a set of channels
  and species). With `when_full="legacy"` (default) they reproduce
  `classical.go_or_grow` and `nove.go_or_grow` in distribution; the example,
  tutorial 4 and the README use them. In the two-species form, migrating and
  resting cells are species and `go_or_grow.switch` is the switch.
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

- The interaction names of `get_lgca` run stacks of the rules without family
  prefix (`lgca.legacy_names`): `get_lgca(interaction="alignment", beta=2)`
  runs `polar_alignment`, `interaction="go_or_grow"` runs `go_or_rest`,
  `go_or_grow.growth` and `random_walk`, and so on for every name and family,
  with the legacy parameters and defaults. The dynamics agree with the legacy
  functions in distribution (`tests/legacy_names_test.py`), but seeded runs
  give other numbers than before, and details differ where the rules differ:
  growth is logistic with birth and death decided at once, identity-based
  go-or-grow switches before cells die, and the default concentration of
  `chemotaxis` takes its gradient with the model's ghost nodes. The research
  models run 4 to 18 times faster, identity-based growth 11 to 16 times.
  `lgca.interaction`, `lgca.interaction_params`, `lgca.interactions` and
  functions passed as the interaction keep working.
- Multi-species models built with `get_lgca` accept every interaction name
  whose rules work with several species (e.g. `alignment`, `birthdeath` with
  volume exclusion), not only the few the legacy classes had.
- The prefixed names of model files (`"classical.alignment"`,
  `"nove_ib.go_or_grow"`, ...) are deprecated aliases of these stacks: they
  warn (`FutureWarning`) and name the rules to use instead, and
  `list_plugins()` lists them only with `deprecated=True`. The examples,
  tutorials and documentation use the rules without prefix.
- Identity-based growth rules combine freely in a pipeline; the restriction
  to the `ib.*` growth operators is gone.
- Models refuse node counts whose sum over a node does not fit a signed
  64-bit integer when they are built.
- `phenotype_switch` is a rule on the lattice state, 18 to 28 times faster
  (100 x 100 hex nodes: 16 ms per step with volume exclusion, 9 ms without,
  instead of 463 and 164 ms; since the target channels below, 7 ms with
  volume exclusion). A switching cell goes to a random channel of its new
  species, and the other cells stay in their channels; before, a switch at a
  node redistributed all its cells over the channels. With volume exclusion
  the switch fails if that channel is occupied, so a cell switching into a
  species with n cells in C channels succeeds with probability 1 - n/C
  (before: whenever any channel was free). Cells switching into the same
  species at a node pick distinct channels, and occupancy is judged at the
  start of the step. `go_or_grow.switch(when_full="reject")` follows these
  rules.

- Faster rules: the lattice state reads and writes the interior as a view
  and keeps counts as `int8` with volume exclusion (`state.counts` has that
  type then), cells are counted with a matrix product, identity-based cells
  are ranked with one sort of the selected cells, and `random_walk` draws
  the state of a node from the enumerated states with as many cells (0.8 ms
  per step on 100 x 100 hex nodes, legacy `classical.random_walk` 1.0 ms;
  on 20 x 20 x 20 Moore nodes 1.8 ms against 4.1 ms). The states are
  enumerated within the limits of the Boltzmann sampler's enumeration (a
  million states, 64 MiB), per number of cells where all states together are
  too many. `birth_death` with a random walk
  takes 1.4 ms per step where `classical.birthdeath` took 1.6 ms, and
  identity-based growth is 11 to 14 times faster than `ib.birthdeath` and
  `nove_ib.birthdeath`. Seeded runs of `random_walk` differ from before.
- The Moore lattice's `channel_weight` gave every channel the value of the
  neighbour behind it instead of the one ahead; legacy models on the Moore
  lattice that used it (`nove_ib.evo_steric`) change.
- `nematic_alignment` and `contact_guidance` score a channel with traceless
  tensors, c cᵀ - |c|² I / d: resting scores 0, moving along the axis above
  and across it below it. Before, resting tied with moving across the axis,
  so cells rested less than in the legacy `classical.nematic`, which the
  terms now match in 2D. Seeded results of models with rest channels change.

- `channel_random_walk` is now called `random_walk`; the name `random_walk`
  used to be an alias of `classical.random_walk`, which it replaces in every
  family (same distribution, different random numbers).

- The option of `go_or_rest`, `go_or_grow.switch` and `go_or_grow.growth`
  for full channels is now called `when_full` ("legacy" or "reject"; it was
  `capacity`, which is the crowding scale everywhere else).

- The `chemotaxis` term uses the same gradient as `aggregation` and
  `LatticeState.gradient`: centred differences with ghost nodes, which for a
  named field repeat its edge values. Inside the lattice nothing changes for
  linear signals; at the edge, and on the hexagonal lattice for curved
  signals, seeded runs differ from before.
- `birth_death` is one rule for every model family (classical and
  identity-based, with and without volume exclusion). Birth and death are
  decided at the same time, on the state at the start of the step, as in
  the legacy `birthdeath` rules; before, cells died first. Growth is
  logistic: with volume exclusion a daughter goes to a random channel of its
  species and survives if it was empty, as in the legacy `classical.birth`
  and `ib.birthdeath`; a capacity (`StateSpec.capacity`, always set without
  volume exclusion) scales divisions by `1 - n / capacity`. With volume
  exclusion the capacity is optional, a soft limit in addition to the
  channels: species with volume exclusion now compete for space only if the
  model sets one. `crowding=False` gives the former single-species
  behaviour: cells divide with `birth_rate`, and the capacity is a hard
  limit. The operator no longer takes `capacity`; set `StateSpec.capacity`.
  The alias `birthdeath_native` is gone. Seeded runs differ from before.
- The README go-or-grow example uses the go-or-grow rules.
- The custom interaction guide, the concepts page on interactions,
  tutorials 4 and 6 and the README use the decorators: rules are written as
  functions of the lattice state and tested with `check_interaction`.
  Tutorial 4 builds go-or-grow from `go_or_rest`, growth and a random walk.
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

- In identity-based models without volume exclusion, labels start at 0, and
  `init_families` put cell 0 into family 0, the root of the family tree, so
  it and its descendants counted as an extra family. Cell 0 now belongs to
  family 1 like the other initial cells (homogeneous), or founds family 1
  (heterogeneous). Family counts and Muller plots of such runs change.

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

- The legacy interaction modules `lgca.interactions`,
  `lgca.nove_interactions`, `lgca.ib_interactions`,
  `lgca.nove_ib_interactions`, `lgca.ms_interactions` and
  `lgca.identity_kernels`, the class-based `Native*` operators that ported
  them to model files, `lgca.classical_operators`, `LegacyInteractionOperator`
  and `lgca.plugins.resolve_operator_capacity`. `tanh_switch` is in
  `lgca.builtin_rules`, `mutation_matrix_from_trait_bins` in
  `lgca.legacy_names`. The legacy functions are kept, unchanged, in
  `tests/legacy` of the repository as a reference for tests.
- The unused `lgca.interactions.disarrange` helper.
- The legacy `wetting` interaction and its `classical.wetting` ModelSpec
  port. The model is planned as an advanced tutorial with a new
  implementation; see the planned documentation topics.

### Compatibility

- The `get_lgca(...)` factory and legacy `timeevo(...)` workflow remain
  supported for interactive use; their interactions now run the rules (see
  "Changed"). Published results of earlier versions are reproduced with the
  archived versions of BioLGCA they cite.
- This development branch still reports package version `0.1.0`; a release
  version will be chosen as a separate release decision.
