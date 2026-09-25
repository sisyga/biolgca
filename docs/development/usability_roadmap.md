# Usability roadmap

Status: draft, 2026-09-23. Written after a usability review of the `aidevelop`
branch from the point of view of a Master's or early PhD student who wants to
implement a new interaction and study it, and of a teacher who uses BioLGCA in
a course. The goal is a tool that is easier to learn and extend than Morpheus
for lattice-gas models, and that offers things Morpheus cannot: rules written
in plain Python, analysis in the same notebook, zero-install use in the browser
and mean-field theory next to simulation.

The document has three parts: the issues found (with IDs used throughout), the
phased plan with API sketches and acceptance criteria, and open decisions.

## Reference point: Morpheus

Morpheus (https://gitlab.com/morpheus.lab/morpheus) is a multiscale modelling
environment from TU Dresden. Its strengths for students and teaching:

- a GUI editor for a single declarative model language (MorpheusML), with
  context-sensitive in-app documentation and a model graph view;
- rules written as mathematical expressions, without compiling;
- an example menu with built-in and contributed models, and a public model
  repository with persistent identifiers;
- parameter sweeps and batch runs from the GUI, plus FitMultiCell for
  parameter estimation;
- live visualisation during a run;
- installers for Linux, macOS and Windows, free online courses, and Python
  bindings.

Its weak spots, which are BioLGCA's openings:

- rules that expressions cannot express need a C++ plugin and a rebuild;
- analysis happens in separate tools;
- it simulates Cellular Potts models, which have no simple mean-field
  description, and it does not offer lattice-gas models;
- it needs a local installation.

## Part 1: Issues

Severity: **H** blocks or misleads a typical student, **M** costs hours,
**L** is cosmetic or rare.

### A. First contact

| ID | Sev | Issue |
| --- | --- | --- |
| A1 | H | Hosted docs (readthedocs, github.io) show the old `get_lgca`-only documentation. `pyproject.toml` links to github.io (a README render). `docs/source/index.rst` still installs with pip. |
| A2 | H | Not on PyPI: no `pip install biolgca`, so no Colab or Binder. Installing git and uv on 30 course laptops is the largest hurdle for teaching. |
| A3 | M | Internal development material in user-facing places: `how_to/model_specs_and_plugins.rst` sections "Numerical implementation ownership", the duplication inventory and memory-budget internals; `docs/superpowers/`; milestone reports in `benchmarks/`; `profiling.py` and a demo script in the repository root. |
| A4 | M | Two front doors. `get_lgca` needs 3 lines; a `ModelSpec` random walk needs 5 imports from 3 modules and about 20 lines, yet the docs present `ModelSpec` as the way to learn. |
| A5 | L | No `CITATION.cff` and no software DOI. |

### B. Building a model

| ID | Sev | Issue |
| --- | --- | --- |
| B1 | H | Spec dataclasses have one-line docstrings. `help(StateSpec)` does not say that `density` is particles per node (not channel occupancy), which geometry names exist, or what `parameters` vs `capacity` mean. |
| B2 | M | Plugin names encode the backend family (`classical.`, `nove.`, `ib.`, `nove_ib.`, `multispecies.`), plus unprefixed names (`birth_death`, `phenotype_switch`) and aliases. A student must learn the family taxonomy before choosing a mechanism. |
| B3 | H | Mechanism coverage across families is sparse. Composed `ReorientationSpec` works only for classical volume exclusion. Chemotaxis, alignment and contact guidance are unavailable for NoVE and identity-based models; NoVE has no plain birth-death plugin; identity-based models only random-walk. "Evolution plus chemotaxis" cannot be built without writing an operator. Root cause: each mechanism is implemented per family (40 plugins; `pipeline.py` is 2800 lines) instead of once. |
| B4 | M | `classical.birth` and `classical.birthdeath` contain a random walk, contradicting the documented phase order (growth, switching, reorientation, propagation). Combined with a reorientation operator, cells reorient twice. |
| B5 | M | Curated examples are tuned for test speed, not science: alignment, nematic alignment and chemotaxis show almost no effect at their settings, and `go_or_grow` runs only 15 steps, too short to see its Allee effect. Examples use a `sys.path` workaround (`ensure_project_root_on_path`). |
| B6 | M | Parameter meaning, sign and scale (κ sign, θ, β after the gradient unification) are not shown where a user picks them; `describe_plugin` shows validator strings. |
| B7 | H | No support for studying a model: changing one setting needs nested `dataclasses.replace`; no replicate or parallel runner; results are attributes set on `result.lgca` (`n_t`, `dens_t`) instead of a table. |
| B9 | M | A run without `time.seed` draws an unrecorded seed: metadata and `model.resolved.json` store `seed: null`, so the run cannot be repeated. Draw a seed, use it and record it. |
| B8 | M | Model files: numeric fields are stored inline in JSON with `__tuple__` wrappers (a 50×50 field is 88 KB). The CLI cannot load a custom plugin module, so models with custom rules cannot be rerun with `biolgca run`. |

### C. Writing a new interaction

| ID | Sev | Issue |
| --- | --- | --- |
| C1 | H | Boilerplate and internal vocabulary: `PluginInfo` with `operator_kind` strings and `backend_families`, a factory, `register_plugin`. `PluginInfo` exposes migration bookkeeping (`port_status`, `test_status`, `legacy_source`) that docs and tutorial 6 fill in. Declared `ParameterSpec` defaults are not applied, so every factory repeats them. |
| C2 | M | Registering the same name again raises, so re-running a notebook cell fails. Tutorial 6 works around it with `try: describe_plugin(...) except KeyError`. |
| C3 | H | New directional biases cannot join a composed `ReorientationSpec`: the term table `_REORIENTATION_TERMS` is private and term parameters are restricted to `field`. The natural BIO-LGCA extension, a score for each candidate channel state, is closed; a student must write a whole operator including the sampler. `operator_base.ReorientationTerm` is exported but unused. |
| C4 | H | Operators work on raw state with traps: `lgca.nodes` includes ghost nodes; `lgca.nodes[lgca.nonborder]` is a copy, so in-place edits are silently lost; seeding a node needs the `r_int` offset; the node dtype differs per family (bool, uint labels, int counts, object lists, extra species axis). |
| C5 | M | No test helper checks an interaction for conservation, capacity, dtype, ghost nodes and seed reproducibility, although tutorial 6 asks students to test. |
| C6 | L | `get_lgca(interaction=my_function)` fails with `AttributeError: 'function' object has no attribute 'replace'`. Assigning `lgca.interaction = f` works but is undocumented. |

### D. Running and seeing results

| ID | Sev | Issue |
| --- | --- | --- |
| D1 | M | About 120 `print` calls in the library ("sensitivity set to beta = 2.0", "Random walk interaction is used.") fill notebooks and sweep logs; real warnings are printed, not raised. |
| D2 | M | 2D plot methods draw into `plt.gcf()`: in a script, `plot_density()` followed by `plot_flux()` overlays both into one figure (reproduced on square, hex, 1D). |
| D3 | L | Hex y-axis ticks read 49 instead of a round number; long colour-bar labels are clipped. |
| D4 | H (teaching) | 2D animations cannot be saved (`save_path` exists only in 3D) and show only their first frame in a notebook with the inline backend. No tutorial shows moving cells. |
| D5 | L | Two implementations of most interactions (legacy functions and `Native*` operators with parity tests): students who read the source to learn find two versions. |

## Part 2: Plan

Each phase lists work items with the issues they close, an API sketch where
the design matters, and acceptance criteria. Phases 0 and 1 remove most of the
friction for a new student; phase 2 is the large refactor that makes phases 3
and 4 cheaper.

### Phase 0: Quick fixes (about one week)

Independent, low-risk items, in order.

**0.1 Documentation front door** (A1, A5)
- Point `Documentation` in `pyproject.toml` to readthedocs; switch readthedocs
  to build `aidevelop` until it is merged, then `master`. Use the uv install
  text in `docs/source/index.rst`.
- Add `CITATION.cff` (software entry plus the two papers) and a citing section
  on the docs landing page. Later: Zenodo GitHub integration for a DOI per
  release.
- Accept: the hosted docs show the tutorials; GitHub shows "Cite this
  repository".
- Status (2026-09-23): done in the repository (URL, install text,
  `CITATION.cff`, citing section). Readthedocs keeps building `master` until
  this work is merged there. Still open: Zenodo.

**0.2 Separate internal notes from user docs** (A3)
- Move `docs/superpowers/` and the milestone reports from `benchmarks/` to
  `docs/development/`; move `profiling.py` to `benchmarks/`.
- Move the internal sections of `how_to/model_specs_and_plugins.rst`
  (implementation ownership, duplication inventory, kernel details) to
  `docs/development/architecture.md`; keep one short user-facing paragraph on
  recording-memory limits.
- Remove `multispecies_nove_tumor_growth_demo.py`; it returns as a clean
  advanced example (4.4) once phases 0–2 are done.
- Accept: the user docs contain no references to internal migration state.
- Status (2026-09-23): done.

**0.3 Plugin registration that works in notebooks** (C2, C6)
- `register_plugin(info, factory, replace=False)`; registering an identical
  name from the same module (a re-executed cell) replaces silently, a clash
  with a different module raises with a hint to use `replace=True`.
- `get_lgca(interaction=callable)` sets the callable as the interaction and
  passes the remaining keyword arguments as `interaction_params`.
- Simplify tutorial 6 and the custom interaction guide accordingly.
- Status (2026-09-23): done. Tutorial 6 registers without the try/except;
  function interactions are documented in the factory reference.

**0.4 Quiet library output** (D1)
- Replace default-parameter messages with `logging.getLogger("lgca")` at INFO
  level and real warnings ("system will die out", 1D nematic) with
  `warnings.warn`. Keep `print_interactions` and `print_nodes`.
- Accept: `get_lgca(...)` and `timeevo` print nothing by default;
  `logging.basicConfig(level="INFO")` shows the chosen defaults.
- Status (2026-09-23): done. Warnings use `lgca.base.warn_user`, which points
  at the first caller outside the package. All library warnings, including
  deprecations, now use it (`lgca/_warnings.py`).

**0.5 Plotting that behaves in scripts and notebooks** (D2, D3, D4)
- `setup_figure` opens a new figure unless `figindex` or a new `ax=` argument
  is given.
- 2D `animate_*` accept `save_path=` and `save_kwargs=` like the 3D ones, and
  return an object that displays as an HTML5 video in Jupyter
  (`_repr_html_` via `to_jshtml`).
- Fix hex tick positions and colour-bar label clipping.
- Add an animation to tutorials 1 and 2.
- Status (2026-09-23): done. `lgca.plots.lattice_axes`, `colorbar_axes` and
  `make_animation`/`LatticeAnimation` hold the shared behaviour. Tutorial 2
  now draws its hexagonal density panels with `plot_density(ax=...)` instead
  of `imshow`. Also fixed: 1D colour-bar ticks on bin edges. `ipywidgets` is a
  dependency so tqdm progress bars work in notebooks. D3 confirmed and fixed:
  on wide lattices and in 1D flux plots, colour bars and labels were cut off.
  Standalone plots now use constrained layout with inset colour bars and
  aspect-aware default sizes; `tests/plot_figure_test.py` checks that labels
  stay inside the figure.

**0.6 Documented specs** (B1, B6)
- NumPy docstrings for `SpaceSpec`, `StateSpec`, `TimeSpec`, `AnalysisSpec`,
  `ModelSpec`, `InteractionPipelineSpec`, `ReorientationSpec`,
  `ReorientationTermSpec`, including units and conventions (`density` per
  node, β scale, κ sign).
- Plugin parameter descriptions in plain language; `describe_plugin` prints a
  readable card.
- Status (2026-09-23): done. Descriptions of built-in parameters live in a
  glossary in `lgca/plugins.py` (shared meanings plus per-plugin overrides);
  a test requires one for every built-in parameter.

**0.7 Example audit** (B5)
- Each example must show its effect at its default settings (e.g. go-or-grow
  grows) and produce one gallery figure. Keep fast variants for tests through
  a `steps` argument, not by shortening the example itself. Remove
  `ensure_project_root_on_path`.
- Status (2026-09-23): done. At their own settings, alignment, nematic
  alignment and chemotaxis now show their effect (they were too dilute or too
  weak); go-or-grow runs 100 steps so its Allee effect is visible and takes
  `build_spec(kappa=...)` for the invading contrast case; the
  identity tumour is seeded, grows and mutates kappa, and uses
  `state.capacity` instead of the deprecated parameter. `tests/examples_effect_test.py`
  checks each promised effect. The gallery lost its stale "run(steps=1)"
  output blocks.
- Performance note: `NativeBirthDeathOperator` applied multispecies birth and
  death node by node. Done: it is vectorized over sites (45x faster) and the
  example is back to 50 x 50.

**0.8 Record the seed of every run** (B9)
- If `time.seed` is missing, draw one from `numpy.random.SeedSequence`, use it,
  and write it to the run metadata and `model.resolved.json`, so every run can
  be repeated.
- Status (2026-09-23): done for `ModelSpec` runs and the CLI. `get_lgca(seed=None)`
  still creates an unrecorded generator; it will inherit the behaviour when
  `get_lgca` builds a `ModelSpec` internally (phase 2.4).

### Phase 1: A simple way to add interactions (three to four weeks)

Four conventions make interactions easy to write and hard to get wrong:

1. **Every model has a species axis internally.** Inside the library and in
   the state that interactions see, channel states have the shape
   `dims + (n_species, K)`; a single-species model has `n_species = 1`. Code
   written for one species then works unchanged for several. Public shapes do
   not change: model files, `lgca.nodes`, recorded histories and
   `measurements.npz` of single-species models keep `dims + (K,)`; the
   conversion happens at the boundary.
2. **Each kind of interaction has one meaning.**
   - *Birth/death* changes the number of cells.
   - *Phenotype switch* changes what a cell is. In classical models, a
     phenotype is a species, and a phenotype switch moves cells along the
     species axis (for example from "migrating" to "resting" in go-or-grow).
     In identity-based models, it changes a cell's parameters (Phase 2.2).
     Single-species classical models have no phenotype switch.
   - *Reorientation* rearranges the cells of a node over its channels and
     conserves the number of cells of each species at the node. The
     conservation law defines the kind, not the sampling method: Boltzmann
     sampling of combined scores is the default and the way to combine
     directional cues, but deterministic rules such as the HPP collision rule
     (two cells meeting head-on turn by 90°) are reorientations too.
3. **The order of interactions is part of the model.** The pipeline applies
   interactions in the order they are listed, followed by propagation. The
   fixed order birth/death, switch, reorientation is dropped.
4. **Interactions use operations with a defined meaning, not raw arrays.**
   Without volume exclusion a channel holds several cells, so a rule that
   kills whole channels has the right mean but the wrong fluctuations (for 5
   cells per channel and a death probability of 0.5: variance 6.25 instead
   of 1.25), and no consistency check catches it. Operations such as "every
   cell dies with probability p" are implemented and tested once per model
   family.

Where cells may sit is up to each interaction. The library does not restrict
species to channels; an interaction that needs it, such as go-or-grow keeping
resting cells in rest channels, guarantees it itself.

**1.1 Free order and interaction kinds** (B2, B4)
- Operators run in list order; drop the order check and deprecate
  `allow_custom_order` (accepted and ignored for one release, with a warning).
  Propagation always comes last. The run metadata already records the
  resolved schedule.
- Documentation: the order is a modelling decision; two reorientation
  operators in a row are two random decisions, while cues that should compete
  belong as terms of one `ReorientationSpec`.
- Keep the restriction on combining several identity-based growth operators
  (it is about shared daughter-trait bookkeeping, not order).
- The kind `phenotype_switch` keeps its name. Compilation rejects a phenotype
  switch in a single-species classical model with a message that explains
  the species axis. `classical.go_or_rest` and `nove.go_or_rest`, which move
  cells between velocity and rest channels of one species, become
  reorientations.
- The concepts page defines the three kinds, the species axis and the
  meaning of "phenotype" in classical and identity-based models.
- Status (2026-09-23): done except the concepts page, which follows with 1.7.
  The how-to on model specs explains the kinds and the order; tutorials 3
  and 4 no longer describe a fixed order.

**1.2 Go-or-grow from separate rules** (planned as a two-species model; see status)

Migrating cells (species 0, in velocity channels) and resting cells (species
1, in rest channels) are separate species. The model is the pipeline

1. `go_or_grow_switch(kappa, theta)`: phenotype switch with density-dependent
   rates. A migrating cell becomes resting with probability
   `(1 + tanh(kappa * (rho - theta))) / 2` and a resting cell becomes
   migrating with the complementary probability, where `rho` is the number of
   cells at the node divided by its capacity. Switched cells are placed in
   free rest or velocity channels respectively, so each species stays in its
   channels.
2. `birth_death`: both species die with their own death rate (default equal);
   resting cells divide into free rest channels.
3. `random_walk` for the migrating species, among the velocity channels.

This order is the order of the current (legacy) rule. The formulation is more
flexible: migrating and resting cells can have different death rates or
motility, recorders and plots separate them, and the switch acts per cell
from its current state.

- The switch offers two ways to handle full target channels:
  `capacity="legacy"` (renamed `when_full` on 2026-09-24, see 2.2) reproduces the current rule (the number of switching
  cells is binomial over the cells that fit, and both directions are computed
  from the counts before the step); `capacity="reject"` lets each cell try and
  rejects a switch into full channels.
- Validation before switching the example over: compare distributions with
  the legacy rule (resting fraction versus density, population growth, the
  Allee effect of the go-or-grow example). Seeded trajectories will differ
  because random numbers are drawn in a different order.
- `classical.go_or_grow` and `get_lgca(interaction="go_or_grow")` keep the
  legacy single-species implementation until Phase 2.4; the example and
  tutorial 4 move to the two-species model.
- Phase 1 covers classical models with and without volume exclusion.
  Identity-based go-or-grow, where every cell has its own `kappa` and
  `theta`, follows with per-cell operations in Phase 2.
- Status (2026-09-24): done in `lgca/builtin_rules.py`. Decided afterwards:
  the default formulation is single-species, which is easier to explain:
  `go_or_rest` (a reorientation between velocity and rest channels),
  `go_or_grow.growth` (resting cells are the cells in rest channels) and
  `channel_random_walk` over the velocity channels. The two-species form
  (`go_or_grow.switch`, the same growth rule, the walk restricted to species
  0) stays available. Both accept `capacity="legacy"` or `"reject"`; without
  volume exclusion both modes are the legacy rule. Validation: one step on
  3600 nodes over the whole density range matches the legacy rule per
  density level for both forms, with and without volume exclusion
  (`tests/go_or_grow_test.py`, which also detects `"reject"` as a different
  model), and 300 seeded runs of the two-species form agreed with the legacy
  rule within one standard error at t = 10, 30 and 60. The example, tutorial
  4 and the README use the single-species form; the example still shows the
  Allee effect (12 cells shrink to 2; with kappa = -4 they grow to 1827).

**1.3 Lattice view with operations** (C4)

A `LatticeState` passed to user code hides ghost nodes, dtypes and the
differences between model families:

```python
# read-only views, interior nodes only, species axis always present
state.counts              # cells per channel, shape dims + (n_species, K)
state.density             # cells per node, shape dims
state.species_density     # cells per node and species, shape dims + (n_species,)
state.flux                # sum of cell velocities per node, shape dims + (d,)
state.neighbor_sum(a)     # sum of a over the interaction neighbourhood
state.gradient(f)         # centred gradient in lattice units; f is an array or a field name
state.field("signal")     # named field from StateSpec.fields
state.c, state.K, state.velocitychannels, state.restchannels, state.capacity
state.rng, state.step, state.geometry, state.dims, state.n_species

# operations; probabilities broadcast against dims or dims + (n_species,)
state.remove_cells(p)                          # every cell dies independently with probability p
state.divide_cells(p, channels="rest")         # every cell divides with probability p into a free channel
state.add_cells(n, channels="rest")            # n new cells per node and species, into free channels
state.switch_phenotype(rates, channels=...)    # rates[a][b]: probability that a cell of species a becomes b;
                                               # switched cells go to free channels of the given set
state.shuffle_cells("velocity", species=0)     # random walk of one species within a channel set
state.counts = new                             # expert access: replace the whole state (checked)
```

- `channels=` selects where new or switched cells go: `"rest"`,
  `"velocity"`, `"same"` (keep the channel if it is free for the new species)
  or explicit channel indices. This is how an interaction keeps species in
  their channels.
- Each operation is implemented with volume exclusion (at most one cell per
  channel and species) and without it (binomial and multinomial sampling per
  cell). The only hard limit is volume exclusion; `capacity` is the crowding
  scale that rules use (e.g. `1 - density / capacity`), in every family.
- Writes are checked: at most one cell per channel and species with volume
  exclusion, non-negative integers, no NaN, and the
  conservation law of the kind (reorientation keeps cells per node and
  species; a phenotype switch keeps cells per node).
- Only `state.rng` is available for randomness, so runs stay reproducible.
- Status (2026-09-24): done in `lgca/lattice_state.py` (`lgca.LatticeState`),
  tested on every geometry with and without volume exclusion and with one and
  two species (`tests/lattice_state_test.py`). Decisions made on the way:
  - Capacity is a soft limit in every family: the crowding scale that rules
    use, not enforced by the operations. With volume exclusion the hard limit
    is one cell per channel and species (default capacity `n_species * K`);
    without it there is none, as in the existing NoVE rules and initial
    states.
  - With volume exclusion, a phenotype switch needs a channel that was free
    for the new species before the switch; competing cells are chosen at
    random, and the rest keep their species (as in legacy go-or-grow).
    `channels="same"` keeps each cell in its channel.
  - `divide_cells(channels="same")` (daughter in the mother's channel) exists
    only without volume exclusion.
  - `neighbor_sum` wraps around periodic boundaries and sees zeros beyond
    reflecting and absorbing walls, as the model's own `nb_sum` does after
    its boundary conditions. `gradient` takes centred differences with the
    model's own `gradient` method on the padded lattice: arrays get the same
    ghost values as in `neighbor_sum`, named fields keep the ghost values the
    model stores for them (edge values for `StateSpec.fields`).
  - Done (2026-09-24): with several species `birth_death` runs on the
    operations and `capacity` is a soft limit (divisions with probability
    `birth_rate * (1 - density / capacity)`, no capacity by default); with
    one species it stays a hard limit by design.
  - `commit()` checks the conservation law of the kind and writes the
    interior; the pipeline integration comes with the decorator (1.4).

**1.4 `@interaction` decorator** (C1)

```python
from lgca import interaction

@interaction(kind="birth_death", families=("classical",))
def crowding_death(state, r_d=0.1):
    """Each cell dies with probability r_d * density / K."""
    state.remove_cells(r_d * state.density / state.K)

@interaction(kind="reorientation", families=("classical",), geometries=("square",))
def hpp_collision(state):
    """Two cells meeting head-on leave at right angles (HPP rule)."""
    ...  # deterministic rearrangement of state.counts; conservation is checked

spec = ModelSpec(..., dynamics=InteractionPipelineSpec(
    operators=[crowding_death(r_d=0.2), {"name": "classical.random_walk"}]))
```

- The decorator builds `PluginInfo` from the signature: parameters from
  keyword defaults (no default means required), description from the
  docstring, name from `module.function` unless given. Defaults are applied by
  the framework, not repeated by hand.
- Calling the decorated function with parameters returns an operator entry
  for `operators=[...]`; the name also works in JSON.
- `families=` (and optionally `geometries=`) is explicit. A model of another
  family is rejected at compilation with a clear message instead of being
  assumed to work. Identity-based families are not supported in Phase 1.
- `conserves=("momentum",)` declares further conservation laws, which the
  test helper checks (HPP conserves momentum).
- Drop `port_status`, `test_status` and `legacy_source` from the public
  `PluginInfo` (keep them internally until phase 2 removes the legacy code).
- The class-based operator API stays for advanced cases (setup caches,
  custom validation, dependencies).
- Status (2026-09-24): done in `lgca/rules.py` (`from lgca import
  interaction`). Additions: `n_species=` for rules that need a number of
  species (checked when the model is built); calling a rule with a
  `LatticeState` applies it directly; parameter descriptions come from a
  numpydoc `Parameters` section. Declared momentum conservation is checked
  after every call, like the conservation law of the kind. The schedule no
  longer shows `port_status` and `legacy_source`; the fields stay in
  `PluginInfo` with defaults and are documented as internal.

**1.5 Public reorientation terms** (C3)

Most BIO-LGCA biases couple a field to the candidate state in a few ways,
which the existing terms already implement. Expose them for the Boltzmann
sampler:

```python
from lgca import reorientation_term

@reorientation_term(coupling="flux")        # score = g(r) · J(s')
def drift(state, direction=(1.0, 0.0)):
    return np.broadcast_to(direction, state.dims + (len(direction),))

ReorientationTermSpec(name="drift", beta=2.0, parameters={"direction": [0, 1]})
```

Couplings: `"flux"` (vector field · flux of the candidate), `"nematic"`
(tensor field : nematic tensor), `"rest"` (scalar · rest occupancy) and
`"channels"` (per-channel weights). An advanced `score(features, state)` form
stays available. Rewrite the built-in terms on the same public API. Like the
combined sampler, terms work for classical models with volume exclusion (one
or several species) in Phase 1; the sampler without volume exclusion follows
in 2.1. Rules that are not Boltzmann samplers use `@interaction(kind=
"reorientation")` instead.
- Status (2026-09-24): done. `lgca.reorientation_term` in `lgca/rules.py`;
  a term is a function of a `LatticeState` (the state before the
  reorientation) that returns a field, and the coupling turns it into a
  score. Fields that broadcast are accepted (one vector for the whole
  lattice). All built-in terms are now defined this way in
  `lgca/builtin_rules.py`, and the private term classes are gone; seeded
  runs are unchanged (the batched-sampling test compares states and random
  number streams) and so is the run time. Terms list the fields they read
  as inputs in the schedule. The advanced `score(features, state)` form was
  not added: every built-in term fits one of the four couplings.
  Decided (2026-09-24): `chemotaxis` uses `state.gradient(field)` like
  `aggregation`: centred differences with ghost nodes, which for a named
  field repeat its edge values unless set otherwise. Linear ramps keep their
  exact gradient inside the lattice on every geometry; across the edge the
  slope is halved. Seeded chemotaxis runs differ from before.

**1.6 Interaction test helper** (C5)

```python
from lgca.testing import check_interaction

report = check_interaction(crowding_death, parameters={"r_d": 0.2})
```

Runs the interaction on small seeded lattices of every supported geometry
(1D, square, hex, cubic, Moore) and declared family, with one and two species,
and checks: ghost nodes untouched after boundary application, volume
exclusion and dtype respected, conservation law of the kind and declared extra laws
(e.g. momentum), same seed gives same result, no NaN. With
`expected_rates=...` it also compares measured death, birth or switch
frequencies with the declared ones. Raises with a readable report; usable as a
one-line pytest. The operations of 1.3 are tested this way across the full
geometry-by-family matrix; HPP serves as the test case for a deterministic
reorientation.
- Status (2026-09-24): done in `lgca/testing.py`. It checks every declared
  geometry and family with one and two species (two and three for a
  phenotype switch, or the rule's `n_species`) and periodic and reflecting
  boundaries, also for registered built-in names. Changes to ghost nodes
  count as errors only when they reach the lattice through the boundary
  conditions (the legacy `classical.birth` writes ghost nodes that periodic
  boundaries overwrite). Instead of `expected_rates`, `expected_growth`
  compares the relative change of cells per species in one step on a large
  lattice; switch rates are not measured yet. `prepare=` arranges the random
  initial states for rules that expect a layout (go-or-grow).

**1.7 Documentation**
- Rewrite `how_to/custom_interactions.rst` around 1.3–1.6: a growth rule, a
  movement bias, a deterministic reorientation (HPP), a test.
- Tutorial 4 builds go-or-grow as a two-species model; tutorial 6 uses the
  decorator; add exercises that write a term.
- Concepts page: the three kinds of interaction, the species axis, free order.
- Status (2026-09-24): done. The how-to shows a growth rule, a phenotype
  switch, a movement bias (term), HPP and `check_interaction`; its code runs
  in `tests/docs_snippets_test.py`. The concepts page on interactions
  explains the three kinds, species and order, and lists the terms with
  their couplings. Tutorial 4 ends with go-or-grow built from `go_or_rest`
  (kappa = 4 and -4, moving and resting cells plotted separately); tutorial 6 writes
  its rule with the decorator and runs `check_interaction`; tutorials 3 and
  6 have exercises that write a term. The README section "Write your own
  interaction" uses the decorator, and its go-or-grow block the go-or-grow
  rules.

**Supported in Phase 1**

| | Birth/death | Phenotype switch | Reorientation |
|---|---|---|---|
| Classical, one species, VE and NoVE | decorator and operations | none by design | decorator (any conserving rule); Boltzmann terms for VE |
| Classical, several species, VE and NoVE | decorator and operations | decorator and operations | decorator (any conserving rule); Boltzmann terms for VE |
| Identity-based, VE and NoVE | built-in operators only | built-in operators only | built-in operators only |

Accept: a student writes, registers, tests and sweeps a new death rule, a new
bias or a deterministic collision rule in under 30 lines without reading
library source; the same rule runs with one or several species and with or
without volume exclusion; the two-species go-or-grow model matches the legacy
rule in distribution, including its Allee effect; re-running every notebook
cell works.

### Phase 2: One mechanism for all model families (one to two months)

**2.1 Common representation** (B3, D5)

All classical families store channel counts as `dims + (n_species, K)`
(`n_species = 1` for one species), so every mechanism has one implementation.
Model files and results keep accepting and returning single-species arrays
without the species axis. Reorientation:

- with volume exclusion: the existing Boltzmann sampler, which enumerates
  admissible channel states;
- without volume exclusion: each cell samples its channel independently,
  P(i) ∝ exp(β·score_i) (multinomial per node; cost O(K), not O(2^K));
- identity-based families: when movement does not depend on cell traits, apply
  the count update and reassign the existing labels to channels at random
  (cells are exchangeable). Trait-dependent movement, such as identity-based
  go-or-grow where every cell has its own `kappa` and `theta`, needs a
  per-cell decision: each cell rests or moves with its own probability, and
  capacity conflicts are resolved in a random order.
- Status (2026-09-24): the sampler part is done. `ReorientationSpec` runs in
  every family: every term gives a weight per channel (all four couplings are
  linear in the channel occupation), so one set of term fields serves both
  samplers. Without volume exclusion each cell draws its channel from the
  softmax of the summed weights (one multinomial per node and species); it
  reproduces `nove.random_walk` and `nove.dd_alignment` seed for seed on 1D,
  square, cubic and Moore lattices, and on hex in distribution only, since
  the neighbour sums of `LatticeState` round differently in the last bit.
  Identity-based models (VE and NoVE) run the classical sampler on their
  cell numbers, with the same random numbers, and then place the node's labels
  on the occupied channels in random order; `LatticeState` reads their cell
  numbers but refuses to change them until 2.2. With volume exclusion the
  per-channel weights give bit-identical seeded runs and are not slower
  (`benchmarks/composed_reorientation.py`; differences were within the noise). Tests:
  `tests/reorientation_sampler_test.py`.
  Decided and done (2026-09-24): decorated rules may declare the families
  `"ib"` and `"nove_ib"`. `LatticeState` keeps the labels; `shuffle_cells`
  draws the cell numbers as in the classical model (same random numbers) and
  places the labelled cells of the shuffled channel set on them in random
  order, so cells outside the set keep label and channel (e.g. resting cells
  under `channel_random_walk(channels="velocity")`). Assigning `state.counts`
  is refused for identity-based models (reading works), because an array of
  cell numbers does not say which cell went where; the other operations
  follow in 2.2. Still open for 2.1: trait-dependent movement such as
  identity-based go-or-grow.

**2.2 Cell-level operations for identity models**

`state.cells` exposes per-cell properties (arrays keyed by trait) with
`divide(parents, **traits)`, `kill(cells)` and `set_trait(cells, name,
values)`, built on `lgca.identity_kernels`. Mutation rules become small
functions. `phenotype_switch` for identity-based models changes cell
parameters with `set_trait`. The operations of 1.3 (`remove_cells`,
`divide_cells`) get identity-based implementations that pick random cells and
let daughters inherit traits.

Design (agreed 2026-09-24, after profiling). The legacy identity kernels loop
over nodes and channels in Python: on a 100 x 100 square lattice a step takes
90 to 240 ms (e.g. `nove_ib.birthdeath` 150 ms, `ib.go_or_grow` 243 ms),
while the same work on flat per-cell arrays takes about 1 ms.

- **Cell table.** One entry per living cell: label, node, channel. `state.cells`
  exposes it; `cells.node` indexes lattice fields (`state.density[cells.node]`)
  and `cells["kappa"]` gives a trait per cell. Operations (`kill`, `divide`,
  `set_trait`, moving cells between channel sets) and `remove_cells`,
  `divide_cells`, `shuffle_cells` are array operations on the table. With
  volume exclusion, cells that compete for free channels are ranked in random
  order within their node by one sort of all cells (the legacy rule: a random
  subset of the cells that fit tries), without a loop over nodes.
- **Traits.** `lgca.props` holds each trait as a NumPy buffer with spare
  capacity (doubling when full), indexed by label, with list-like `append`,
  `extend` and indexing. Lists cost 10 ms per step to read for 30k living
  cells after 930k births (every vectorized read converts the whole list);
  the buffer costs 0.04 ms to read and 0.01 ms to append. Labels are never
  reused; rows of dead cells stay for lineages.
- **Families.** Daughters inherit the parent's family; `new_family=True`
  starts a new one (lineage tracking is optional because it costs time).
  Muller plots and the family functions must keep working.
- **Initial traits.** `StateSpec(traits={"kappa": 4.0, "theta": 0.6})`: one
  value for all initial cells or an iterable with one value per initial cell.
- **Rules with per-cell parameters.** A rule parameter takes a number (the
  same for all cells) or the name of a trait (per cell), chosen per
  parameter, e.g. `go_or_rest(kappa="kappa", theta=0.6)`. Identity-based
  go-or-grow uses the classical order (switch, growth, walk); the legacy
  order of `ib.go_or_grow` (death first) stays in the legacy rule only.
- **Stage B, NoVE storage.** Without volume exclusion, converting the table
  back to lists costs 40 ms (reading them 4 ms). The NoVE identity model
  stores the table and builds `lgca.nodes` only when it is read. Propagation
  uses a slot lookup table obtained by passing slot IDs through the classical
  NoVE `apply_boundaries()` and `propagation()` (interior and ghost sources
  separately, split further until no two IDs meet); the prototype matched the
  list propagation on 5 geometries x 3 boundaries. Recorders store the table;
  a test checks that a pipeline of new rules never builds `lgca.nodes`.
- **Trait-dependent reorientation** (e.g. an alignment strength per cell):
  with volume exclusion the labelled node state has the joint weight
  `P(σ) ∝ exp(Σ_a β_a w_σ(a))`. Assigning cells one after another in random
  order gives a different distribution (two identical cells, weights 1, 1, 3:
  0.100 instead of 1/7 for the pair of velocity channels), so it is not used.
  Decided: a Metropolis sampler that proposes swapping the contents of two
  channels on all nodes at once (prototype: 11 ms per 20 proposals per node
  for 10k hex nodes; about 40 proposals per node reach the exact
  distribution for 4 cells on hex). It starts from a random arrangement, so
  stopping early weakens the cue instead of adding persistence; the number
  of proposals is a parameter that scales with the number of channels. It is
  used only when a term depends on cell traits; trait-independent movement
  keeps the exact classical sampler with random labels. Tests compare it with
  full enumeration on small nodes and with the classical sampler for equal
  traits. Exact alternatives were rejected: enumeration (up to 5040 states on
  hex, 466 ms per step) and dynamic programming over channel subsets (2^K
  states, 117 ms on hex, 4096 subsets with 6 rest channels, impossible on
  Moore). Weights do not depend on the new positions of other cells at the
  same node (terms stay linear in the occupation); within-node interactions
  would need an energy of the whole node state and are not planned now.
  Without volume exclusion each cell samples its own channel, exactly.

- Status (2026-09-24): stage A done. `lgca/cells.py` holds `TraitArray` and
  `Cells`; `LatticeState` builds the cell table for identity-based models
  (VE from the label array, NoVE from the lists) and writes it back on
  commit, checking that reorientations and phenotype switches keep every
  node's cells. `StateSpec.traits`, `check_interaction(traits=...)`, and
  `go_or_rest`, `go_or_grow.growth`, `channel_random_walk` in the families
  `ib` and `nove_ib` (parameters take a number or a trait name). Decorated
  growth rules may be combined with the legacy identity growth operators.
  Validation (`tests/identity_go_or_grow_test.py`): one step with a kappa per
  cell matches `ib.go_or_grow` and `nove_ib.go_or_grow` per density level
  (r_d = 0, where the orders agree); on nodes without full channels every
  cell rests with its own probability; mutated daughters have the right
  spread; new families are recorded and plotted. Speed (100 x 100 square,
  per step): 23 ms with VE (legacy 243 ms), 89 ms without (legacy 226 ms),
  where converting the table to lists and counting list lengths dominate;
  stage B removes this. Found on the way: `init_families` put NoVE cell 0
  into the root family 0 (fixed), and a string-valued `capacity` parameter
  clashed with `StateSpec.capacity` (fixed; the parameter name of the
  go-or-grow mode is still open). Still open: the Metropolis sampler for
  trait-dependent reorientation, and stage B.
- Status (2026-09-24): trait-dependent reorientation done.
  `ReorientationTermSpec(trait=...)` scales a term per cell. NoVE: exact
  per-cell draw. VE: Metropolis per node, all nodes at once, from a random
  arrangement; proposals pick a cell uniformly and swap its channel's
  contents with another channel (symmetric, the cell number is fixed), with
  O(1) bookkeeping per accepted swap; `parameters={"sweeps": 10}` gives
  `10 * K` proposals per node. Measured from a random start with a strong
  field (occupied-first proposals, `sweeps * K` per node): 10 sweeps reach
  a long reference chain within sampling noise for K = 5 (square, 4 cells),
  12 (hex with 6 rest channels, 10 cells) and 27 (Moore, 15 cells); 5 do
  not. (An earlier estimate of 20 sweeps counted proposals per cell.)
  Decided (2026-09-24): the chain length scales with the total number of
  channels, default 10 sweeps. Tests
  (`tests/trait_reorientation_test.py`): the labelled distribution matches
  enumeration for cells of strengths 0, 1, 2, 4 on 3600 hex nodes; equal
  strengths reproduce the classical sampler; the NoVE draw is per cell.
  Time per step: 51 ms (hex, 10k nodes), 36 ms (square), 106 ms (Moore, 8k
  nodes, K = 26).
- Status (2026-09-24): stage B done. `NoVE_IBLGCA_base` holds either the
  lists (`nodes`) or a table of labels and padded slots, whichever was
  written last; reading `nodes` builds the lists, `_cell_table()` builds the
  table. `LatticeState` reads and writes the table (cells in flight in ghost
  slots are kept aside), `update_dynamic_fields` counts from it, and
  `apply_boundaries` and `propagation` (the geometry methods are wrapped in
  `__init_subclass__` and `set_bc`) use lookup tables from
  `lgca.cells.slot_maps`, built once per model (0.4 s for Moore with
  reflecting walls, otherwise below 0.05 s). Periodic, reflecting and
  absorbing boundaries; inflow keeps the lists. The table moves cells exactly
  like the list code on 5 geometries x 3 boundaries
  (`tests/cell_store_test.py`), and a pipeline of rules builds no lists.
  Speed (100 x 100 square, per step, NoVE): go-or-grow rules 13 ms (legacy
  125 ms), `ReorientationSpec` 10 ms.
- Decided and done (2026-09-24): `NodeRecorder` stores cell tables for these
  models (`lgca.cells_t`, a `CellHistory` of label, node and channel per
  recorded time); `nodes_t` is built from them when first read (0.34 s for
  21 times on 100 x 100). A step with recording takes 19 ms instead of 52.
  The go-or-grow option for full channels is renamed `when_full` (it was
  `capacity`, which elsewhere is the crowding scale; `theta` is relative to
  that crowding scale).

**2.3 Names without family prefixes** (B2)

`random_walk`, `alignment`, `chemotaxis`, `birth_death`, `go_or_grow`, ...
resolve to the implementation for the family chosen by `StateSpec`; an
unsupported combination raises a clear error. Old prefixed names remain as
deprecated aliases for one release.

- Status (2026-09-24): movement done. `random_walk` (the former
  `channel_random_walk`) and every built-in cue registered as an operator of
  its own (`lgca.rules.register_single_cue`: a ReorientationSpec with one
  term) work in every family, so no per-family dispatch is needed: one
  implementation per name. `polar_alignment` got `include_center` and
  `normalize`. Parity (`tests/single_cue_test.py`): one step matches
  `classical.alignment`, `persistent_walk`, `aggregation`, `chemotaxis`
  (given the same gradient), `nematic` and `contact_guidance` in
  distribution (score mean and spread per density level); the NoVE
  alignments match seed for seed. Growth (birth/death, go-or-grow variants)
  is next.
- Status (2026-09-24): `nematic_alignment` and `contact_guidance` score with
  traceless tensors c cᵀ - |c|² I / d. They equal the legacy tensors c cᵀ - I/2
  in 2D, so the parity test now includes a rest channel; in 3D the legacy
  tensor is not traceless (a perpendicular cell scored -1/4 against resting),
  and on Moore the velocities have unequal lengths. Averaged over directions,
  moving now scores like resting on every geometry (tested).
- Status (2026-09-24): `birth_death` is one decorated rule for all four
  families (the class-based native operator is deleted). Death first, then
  division with `birth_rate * (1 - n / capacity)`. With volume exclusion
  and one species the factor comes from the legacy mechanism (a daughter
  picks a random channel, distinct per node, and survives if it is empty;
  sampled as a hypergeometric count), which gives exactly r_b n (K - n) / K.
  A first version that scaled the rate and then also needed a free channel
  counted crowding twice (0.284 instead of 0.32 births at n = 4 of K = 5);
  `test_volume_exclusion_gives_logistic_growth` catches it. Several species
  keep the Phase 1 competition through total density / `StateSpec.capacity`
  (default n_species * K). `crowding=False` is the former single-species
  behaviour (hard capacity; several species share the room by a
  hypergeometric draw). Rates can be trait names; `mutation` takes a
  standard deviation, `{"std", "bounds"}` (truncated normal) or `{"step",
  "probability"}`; `new_family` takes a probability; `mutation_matrix` sets
  the daughters' species. Parity in distribution with `classical.birth`
  (mean only: legacy fills each empty channel with r_b n / K),
  `ib.birthdeath` (legacy `ib.birth` draws target channels with replacement,
  so colliding daughters are lost), `nove_ib.birth` and `multispecies.birth`
  with a mutation matrix. Legacy rules that let dying cells divide in the
  same step differ by O(r_b r_d).
- Status (2026-09-24, user decisions): birth and death are simultaneous
  (decided on the state at the start of the step, as in the legacy
  `birthdeath` rules), because the expected change is then exactly
  r_b n (1 - n / capacity) - r_d n; death first is two stacked operators.
  With volume exclusion and several species, exclusion acts only within a
  species (classical LGCA); `StateSpec.capacity`, if set, adds a soft limit
  on all cells (`LatticeState.has_capacity`). Parity now includes deaths
  (`classical.birthdeath` in the mean; `ib.birthdeath`, `nove_ib.birthdeath`,
  `multispecies.birthdeath` in distribution). All legacy research rules get
  ported as stacks of generic rules under their legacy names, which needs
  custom mutation-effect distributions.
- Status (2026-09-24): research models ported. `lgca.stack` makes one
  operator from a list of operators (StackOperator builds them when the model
  is built, with the model's LatticeState). `lgca.research_models` registers
  `go_or_grow_kappa`, `go_or_grow_kappa_chemo`, `go_or_grow_glioblastoma`,
  `evo_steric`, `birthdeath_cancerdfe`, `go_and_grow_mutations`,
  `birthdeath_discrete` (stacks, identity-based families) and
  `excitable_medium` (a rule, classical). Mutations are events
  (`lgca.mutations`): probability, per-trait effects from any NumPy
  distribution, fixed values or registered functions (`@mutation_effect`),
  add/subtract/multiply, bounds clipped or redrawn, lists of kinds;
  `new_family=True` means mutated daughters found families (decided with the
  user; the per-daughter probability is gone). Generic additions:
  `channels` for reorientations, `steric_repulsion`, `neighbor_values`,
  `go_or_rest(density="neighbourhood")`, `Cells.found_families`.
  Parity in distribution away from the edges (`tests/research_models_test.py`;
  in a pipeline the legacy operators see empty ghost nodes). Differences:
  family traits become cell traits; density cues see the density after
  growth; legacy `go_or_grow_kappa_chemo` normalized the neighbourhood sum by
  v instead of v + 1 (kappa = 0 in its test); `birthdeath_discrete` clips at
  a_max; Moore `channel_weight` pointed backwards (fixed).
  Speed (`benchmarks/research_models.py`, 50 warm-up steps, every
  measurement in a fresh process): research stacks 3.6-17x faster,
  identity-based growth 11-14x, the excitable medium 2x, classical growth
  with a random walk 1.1x (after `int8` counts with volume exclusion: int64
  temporaries of 560 kB made glibc map fresh pages every step). Multispecies
  growth without volume exclusion is 0.8x or 1.3x depending on whether glibc
  trims the heap (bimodal across identical seeded runs; with
  MALLOC_TRIM_THRESHOLD_ raised it is steadily 1.3x). int32 counts there
  made it worse (mixed-type conversions). Optimizations on the way: interior
  as slice views, channel sums as matrix products, table-drawn occupancy
  states, distinct-target births as one random subset per node, one random
  number per channel for birth and death, ranking only selected cells,
  argsort(node + random) instead of lexsort, cheap identity conservation
  check.
- Status (2026-09-24): uniform placement (`lattice_state.random_occupancy`)
  draws one enumerated state per node: from one table of all states when
  they fit the limits of `get_permutations` (10^6 states, 64 MiB: up to 20
  channels), else from the cached states of each number of cells, else
  (e.g. half-full Moore nodes) by ranking random keys. The earlier cutoff of
  12 channels was not measured; the table is 5-9x faster than ranking at
  every size tested (7-20 channels, 10^4 nodes), and the Moore random walk
  is 2.3x faster than legacy. The `occupations` cache is shared with the
  Boltzmann sampler of channel subsets; the full-set sampler keeps
  `get_permutations`, whose row order its seeded results depend on.

**2.4 Remove duplicates** (B4, D5)
- Separate the random walk from `classical.birth*`.
- After parity tests pass on the new implementations, delete the legacy
  interaction functions and `Native*` duplicates, then the parity tests.
  `get_lgca` builds a `ModelSpec` internally.

Accept: the coverage matrix (mechanism × family) is full for movement terms,
birth/death and switching; `pipeline.py` shrinks substantially; the test suite
has no parity tests.

- Decided (2026-09-24, user): prefixed names stay as aliases that warn
  (removed in the next release); the legacy interaction functions are not
  deleted but frozen in `tests/legacy`, so the comparison tests keep a
  reference (this replaces "the test suite has no parity tests"); functions
  passed to `get_lgca(interaction=...)` keep running on the model object;
  legacy details that the rules do differently (e.g. `classical.birth`,
  classical go-or-grow's birth cap) follow the rules and are documented.
- Status (2026-09-24): done. `lgca/legacy_names.py` translates every
  `get_lgca` name of every family into a stack of rules with the legacy
  parameters and defaults (growth with a random walk, so `birth*` no longer
  moves cells itself); the prefixed model-file names are registered as
  deprecated aliases (`PluginInfo.deprecated`: `FutureWarning`, hidden from
  `list_plugins`). `LGCA_base.set_interaction` is one method for all
  families: it derives a `ModelSpec` from the model object and compiles the
  stack, so `get_lgca` and model files run the same code; `lgca.interaction`,
  `interaction_params` and `interactions` keep working. Multi-species models
  now accept every name whose rules take several species. Generic additions
  for the translations: `birth_death(channels=...)`, `go_or_rest` with a
  kappa and theta per species (multispecies go-or-grow), `directed_motion`
  (a given gradient for legacy chemotaxis). Deleted from the package: the
  five legacy interaction modules and `identity_kernels` (moved to
  `tests/legacy` with their `set_interaction` code, registered there as
  `legacy.<family>.<name>`), all `Native*` duplicates, the legacy plugin
  registrations and `classical_operators`: `pipeline.py` 2951 -> 972 lines,
  `plugins.py` 1822 -> 350, 5900 lines removed in all. Comparison tests use
  the frozen functions (several had compared with the `Native*` ports, and a
  few had silently compared a translation with itself);
  `tests/legacy_names_test.py` compares every name of every family with its
  legacy function after one step (changes of cells, resting cells and flux
  per node, 100 x 100 nodes; a 15 % change of a rate shows at 3-5 sigma).
  Coverage: movement and growth rules work in every family, species switching
  in classical models with several species. Open: identity-based models have
  no generic rule that switches a discrete cell trait at given rates (the
  counterpart of `phenotype_switch`). Speed against the frozen legacy
  functions (`benchmarks/research_models.py`): research models 3.6-18x,
  identity-based growth 11-16x, classical growth 1.1x, multispecies growth
  0.8-1.4x depending on the allocator (unchanged).
- Decided and done (2026-09-24, user): with several species every part of
  the dynamics can act on chosen species (`species` on growth, go-or-rest,
  reorientations and single cues; a single cue is a `ReorientationSpec` with
  one term, and its other species now keep their channels), and cues take
  `sensed_species` (whose cells they read; default all), via
  `LatticeState.sensing`. The open switching gap is closed by reusing
  mutations: `trait_switch` applies mutation events to all living cells
  (identity-based, with and without volume exclusion), with a new operation
  `"set"`; two states with their own rates are one event with a `choice`
  draw. Rates that depend on the surroundings or on the current value are
  not covered yet.
- Status (2026-09-24): switching responds to the surroundings
  (`lgca.switching`, user request). A probability is a number or
  `{"max", "cues"}` = `max (1 + tanh(Σ kappa_k (c_k - theta_k))) / 2`, the
  go-or-grow switch generalised to several cues (density, field, gradient,
  flux, cell traits; `sensed_species`; `@switch_cue`), in the rates of
  `phenotype_switch` and in the events of `trait_switch` and mutations, which
  also take `"when"` (the event applies to cells with some trait values, so
  any rate matrix between discrete states). `phenotype_switch` became a rule
  on `LatticeState.switch_phenotype` (the class-based operator looped over
  nodes: 463 -> 16 ms per step with volume exclusion, 164 -> 9 without, 100 x
  100 hex); switched cells take a free channel of their new species instead
  of all cells of the node being redistributed. Tests:
  `tests/switching_test.py` (probabilities by density and field level against
  the formula, per-cell sensitivities, when-conditions, custom cues).
- Decided and done (2026-09-25, user): species switches with volume
  exclusion aim at a random channel of the new species (in `channels`,
  default all) and fail if it is occupied, so one switcher succeeds with
  probability 1 - n/C; switchers into the same species pick distinct
  channels (at most C of them, chosen at random); occupancy is judged at the
  start of the step. `channels="same"` is unchanged. Before, a switch
  succeeded whenever any channel was free. `LatticeState._switch_ve` also got
  faster (100 x 100 hex, two species: 15.6 -> 7.3 ms, `"same"` 11.5 -> 7.6
  ms). Tests: success rate per fill level on every geometry (Moore with 27
  channels uses the non-enumerated placement), more switchers than channels,
  vacated channels.
- Decided and done (2026-09-25, user): switching probabilities in the
  Boltzmann form, `{"rate": r, "cues": [{"name", "beta", ...}]}`, the weight
  `w = r exp(Σ beta_k c_k)` against staying: `w / (1 + w)` for a single
  event (`trait_switch`, mutations), `w_ab / (1 + Σ w_ab')` within a
  `phenotype_switch` row, with no row-sum bound (a row is all Boltzmann or
  0; mixing forms in a row is refused). Cues take `beta` only (a threshold is
  a factor of the rate); `beta` may name a trait. For two states it equals
  the tanh form (`beta = 2 kappa`, `rate = exp(-2 kappa theta)`), documented
  for teaching in the concepts page. Computed in log space, so large weights
  do not overflow. Tests: row choice between two switches by density level
  and a trait-valued `beta` in `trait_switch` against the formula (a row
  normalisation without staying fails them), the tanh equivalence,
  overflow, validation.
- Done (2026-09-25, delegated to Claude; confirmed by the user the same day: matching legacy
  `go_or_rest` is not required, and no exact sampler is needed): go-or-rest as a
  Boltzmann reorientation is the new term `resting` (also a single cue),
  not a replacement of the rule `go_or_rest`. A lone cell rests with a
  switching probability `p` (`lgca.switching`, default the go-or-grow
  switch), via the rest score `log(p/(1-p)) + log(v/r)`. Without volume
  exclusion this is exactly `go_or_rest` followed by a velocity random walk.
  With volume exclusion it is not close to `go_or_rest`
  (`when_full="legacy"`): on the square lattice with one rest channel
  (kappa 5, theta 0.75) the rest channel is occupied with 0.03/0.18/0.62 at
  2/3/4 cells under `go_or_rest` (probability p whatever the number of
  cells) and 0.08/0.57/0.96 under `resting` (Fisher's noncentral
  hypergeometric law with odds `p/(1-p) · v/r`). So the legacy names and the
  go-or-grow examples keep `go_or_rest`; the name `go_or_rest` was not
  turned into a `ReorientationSpec` (a term of that name would also collide
  with the rule as a single cue). Per-cell kappa and theta: reorientation
  terms may return `CellWeights` (identity-based); `_cell_scores` sums
  per-cell scores and `_metropolis` works on a `(cells, K)` score matrix
  (trait-scaled chemotaxis unchanged in speed, 17 ms per step on 100 x 100
  hex). Speed per step (100 x 100 hex, density 0.4): `resting` 2.1 ms vs
  `go_or_rest` + `random_walk` 5.2 ms (classical), 3.2 vs 4.0 (NoVE), 2.3 vs
  3.2 (NoVE identity-based with trait kappa), but 16.9 vs 3.8 for
  identity-based with volume exclusion and trait kappa (Metropolis); an
  exact sampler for per-cell rest scores is possible. `go_or_grow.switch`
  stays as it is: the two-species legacy form, whose `"reject"` mode now
  follows the `phenotype_switch` target-channel rule. Tests:
  `tests/resting_test.py` (NoVE per density level against p; VE against the
  noncentral hypergeometric law on lin/square/hex with one and two rest
  channels; the Boltzmann form and beta = 0; trait kappa in both
  identity-based families; two cells with their own odds at a node against
  the exact Boltzmann weights).

### Phase 3: Studying a model (two to three weeks)

**3.1 Varying specs** (B7)

```python
from lgca.study import vary, sweep

variant = vary(spec, {"time.steps": 200, "dynamics.operators[0].terms[1].beta": 5})
```

**3.2 Sweeps and replicates** (B7)

```python
table = sweep(
    spec,
    grid={"dynamics.operators[0].terms[1].beta": [0, 5, 10, 20]},
    seeds=range(10),
    measure={"x_flux": mean_x_flux, "population": lambda r: r.data["n"][-1]},
    n_jobs=4,
)
table.groupby("beta").x_flux.agg(["mean", "std"])
```

- Returns a pandas DataFrame (adds pandas as a dependency) with one row per
  run: parameters, seed, measurements, BioLGCA version.
- Runs in parallel with `concurrent.futures`; custom interactions must be
  importable from a module in workers (document this).
- `biolgca sweep model.json --vary beta=0,5,10 --seeds 0:10 --output runs/`.

**3.3 Results as data** (B7)
- `ModelRunResult.data` mapping observer outputs by name (`result.data["n"]`,
  `result.data["density"]`), with times; keep the `lgca.n_t` attributes for
  compatibility.

- Done (2026-09-25): `ModelRunResult.data`, a `RunData` mapping (in
  `lgca.simulation`, with the table `RECORDED`): built-in recorders under
  descriptive names (`population` with alias `n`, `density`, `nodes`,
  `channel_population`, `moving`, `resting`, `family_population`, order
  parameters) and `ScalarTimeSeriesRecorder` metrics; `data.steps(name)`;
  a missing name lists what was recorded. References are taken at the end
  of the run (lazy lists of labels for NoVE identity-based histories), so a
  later run does not change them. The CLI writes `measurements.npz` from the
  same table (keys unchanged). Tutorials, README and examples migrated.
  User decisions (2026-09-25): pandas becomes a dependency for `sweep`,
  which takes the number of parallel workers as a parameter and offers a
  long table (one row per run and time); `get_lgca` stays the quick start
  and the way to run the included standard models, `ModelSpec` follows.

**3.4 Model files** (B8)
- Arrays above a size threshold go to an NPZ sidecar next to the JSON
  automatically; tuples become lists in the schema.
- `biolgca run model.json --plugins my_project.interactions` imports trusted
  plugin modules named on the command line (never from the model file).

Accept: tutorial 3's sweep is a single `sweep(...)` call with error bars; a
model with a custom rule reruns from the CLI.

### Phase 4: Beyond Morpheus (ongoing)

**4.1 Zero install** (A2)
- Release to PyPI with a trusted-publishing workflow; versions from git tags.
- "Open in Colab" badge on every tutorial (first cell installs BioLGCA when it
  is missing).
- A JupyterLite site on GitHub Pages running the tutorials in the browser
  (NumPy, SciPy and Matplotlib run in Pyodide). Morpheus cannot offer this.

**4.2 Live exploration**
- `lgca.explore(spec)` with ipywidgets: sliders for β, density and rates,
  play/pause, flux or density view (optional `widgets` extra).

**4.3 Theory next to simulation**
- `lgca.theory`: mean-field equations and linear stability (dispersion
  relation) for classical Boltzmann rules; predict the pattern wavelength or
  the onset of order, then compare with the simulation. Tutorial 7.
  No Cellular Potts tool offers this, and it is where LGCA are strongest.

**4.4 Course pack and model zoo**
- Instructor notes, exercise solutions in a separate folder excluded from the
  built docs, slide-ready figures.
- A model zoo of published LGCA models, each with ModelSpec, notebook and
  paper link: adhesion-driven invasion (Ilina et al. 2020, planned),
  go-or-grow (Böttger et al. 2015), genotypic and phenotypic heterogeneity
  (Syga et al. 2026).
- An advanced example of multispecies tumour growth without volume exclusion
  on hexagonal and 3D Moore lattices, colouring nodes by the local mean class
  property (replaces the removed `multispecies_nove_tumor_growth_demo.py`).

**4.5 Platform coverage**
- CI on Windows and macOS in addition to Linux.

## Part 3: Open decisions

1. **Front door.** Keep `get_lgca` as the documented quick start (as the README
   now does) and `ModelSpec` for studies, or add a short `ModelSpec`
   constructor (for example `lgca.model(geometry="hex", dims=..., interactions=[...])`)
   and retire `get_lgca` from the docs.
2. **pandas** as a hard dependency for `sweep`, or return a list of records
   with an optional DataFrame conversion.
3. **Legacy removal timing.** Phase 2.4 deletes the legacy interaction
   functions. Is there external code (theses, papers) that imports them
   directly and needs a deprecation release first? Decided (2026-09-24): no
   deprecation release. The current `master` is archived and published
   papers cite Zenodo snapshots, so this release replaces the legacy
   interactions directly.
4. **Preferred citation.** `CITATION.cff` currently lists the software with the
   two papers as references; decide whether one paper should be the preferred
   citation.

Decided (2026-09-23): the species axis is internal only and public shapes stay
backward compatible; the kind that changes species or cell parameters keeps
the name `phenotype_switch`; interactions run in the order they are listed;
Boltzmann sampling is the default for reorientation, not a requirement; the
library does not restrict species to channels.
