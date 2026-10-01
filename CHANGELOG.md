# Changelog

This file records notable user-facing changes. Changes remain under
`Unreleased` until a release version and date are selected.

## Unreleased

### Added

- An advanced example, `lgca.examples.tumour_spheroid`, with a notebook in
  the example gallery: a tumour spheroid of proliferating, quiescent and
  necrotic cells (three species without volume exclusion) in the oxygen they
  consume, on a hexagonal lattice (a cross-section) and on the 3D Moore
  lattice. It follows Dormann & Deutsch (2002) in outline, with measured
  parameters: oxygen diffusion and consumption, Michaelis–Menten uptake,
  thresholds of quiescence and necrosis, doubling time, cell volume and an
  unstirred layer of medium. Its 3D spheroid doubles in 24 h, its centre
  runs out of oxygen at about 0.53 mm across, and after a week oxygen
  reaches 180 µm into it, close to the measured 188–220 µm.
- `lgca.plot_data.mean_species_property(lgca, values)`: the mean over the
  cells of every node of a property of their species (one value per
  species), the counterpart of `mean_trait` for models with several species;
  also for recordings of `DensityRecorder`. Draw it with
  `lgca.plot_scalarfield`.
- Provenance: `model.metadata["provenance"]`, and so `metadata.json` of
  `biolgca run`, records the versions of Python and the packages, the
  platform, the operators with their parameters including defaults, the
  SHA-256 hashes of the source code of every rule, term and reaction the
  model runs, and of the files it read; `reconfigure` keeps it current. The
  command line adds the hash of the model file and the git commit of BioLGCA
  when it runs from a checkout. `sweep` records the same in
  `table.attrs["provenance"]`, and `biolgca sweep` in `sweep.json`.
- Hash pins where the library writes the file: a model file records a hash
  of every array in its array file and warns when it is loaded after the
  array file changed (e.g. a `model.yaml` saved next to `model.json` wrote its
  arrays to the shared `model.arrays.npz`: 200 cells became 400 without
  notice); the archive of `biolgca run` records the hashes of its files in
  `metadata.json`, and `biolgca run` and `validate` warn when an archived model
  changed after the run that wrote it.
- Field solver failures raise `lgca.fields.FieldSolverError`, a
  `RuntimeError` with a `kind` (`"no_steady_state"`, `"singular"`,
  `"nonlinear"`, `"linear"`, `"non_finite"`, `"explicit"`); the operator's
  statistics count them by kind (`"failure_kinds"`).
- `CompiledModel.reconfigure(changes)` changes the dynamics of a running
  model from its next step on, with the paths of `lgca.study.vary` (e.g.
  `{"birth_rate": 0.3}` after a burn-in). The change is tried on a copy of the
  model first and applied only if that works; `metadata["reconfigurations"]`
  records the step and the old and new values, and `model.initial_spec`
  keeps the spec the model was built from, so the run can be repeated.
  `lgca.explore` applies its sliders this way.
- Model zoo (`lgca.zoo`, roadmap 4.4): published LGCA models written with
  the current rules, each a module with a card (reference, question, result
  reproduced, fidelity), the paper's parameters with their spec paths,
  `build_spec(full=False, ...)` and measurements, and a notebook that
  reproduces a result of the paper. `lgca.zoo.catalogue()`, `load(name)`,
  `parameter_table(name)`. First entry: the emerging Allee effect of
  go-or-grow (Böttger et al. 2015): extinction frequency against the
  initial density, bimodal fates near the threshold, and why the lattice's
  threshold (0.25) lies far below the mean-field one (0.42). Second entry:
  the evolution of phenotypic plasticity (Syga et al. 2024): the three
  evolutionary regimes of the go-or-grow switch in 2D (S1–S3 Figs) and in
  1D at the paper's size (Fig 3 A–F). Third entry: jamming transitions in
  invasion (Ilina et al. 2020): the invasion modes of Fig 5d, a coarse phase
  diagram (Fig 5e) and, beyond the paper, invasion from a spheroid; its
  energy terms, matrix degradation and influx are written with the public
  decorators in the entry's module. Further entries: evolution at an
  invasion front (Syga et al. 2026, the evolutionary LGCA: the front
  accelerates as the fastest-dividing cells gather at it) and discrete
  excitable media (Syga et al. 2019: spiral waves in the LGCA and in its
  mean field, Barkley's model, and their break-up by fluctuations). A new
  model: clonal evolution of the go-or-grow switch (the research model
  `go_or_grow_glioblastoma`), in which drivers change the birth rate and the
  switch; repulsive clones take over and the tumour grows faster than with
  a fixed switch. And contact inhibition and the mode of tumour evolution
  (the research model `evo_steric`, after Noble et al. 2022): glands whose
  capacity stops division, and driver mutations; the more cells stop moving in a full
  tissue, the more the tumour's evolution changes from one sweep after
  another to branching and finally stalls, measured with Noble's indices
  (drivers per cell, clonal diversity). And drivers, passengers and
  mutational meltdown (Syga 2023, PhD thesis, after McFarland et al. 2014):
  the tug-of-war around the critical population size N*, invasion as an
  escape, and a coarse probability-of-cancer diagram; a control without
  mutations shows that small immobile tumours die out by chance, and only
  tumours near N* by their mutational load. `ZooEntry` cards of new models
  have no paper.
- Rules of the kind `"field"` change fields but no cells, with
  `LatticeState.set_field(name, values)` (also available to other rules),
  e.g. a matrix degraded by the cells.
- `lgca.explore` shows the mean trait of the cells at each node
  (`view="mean kappa"`) in identity-based models, and takes a trait as a
  measure (its mean over the cells).
- `lgca.explore(spec, controls)` (roadmap item 4.2) runs a model live in a
  Jupyter notebook (JupyterLab, Colab, VS Code): play, pause, step and
  reset buttons, the steps per frame, sliders or dropdowns for the
  parameters you name (paths and short names as in `lgca.study.vary`), a
  choice of view (density, per species, flux, or a field; a kymograph of
  the last steps on 1D lattices) and a time series of the population, the
  mean of a field or your own measures. A change of a parameter of the
  dynamics applies to the running lattice, which keeps its cells; a change
  of the lattice, state or seed builds the model again. Every change is
  tried first on a copy for one step, so an invalid value is explained and
  the control goes back. `explorer.spec` is the model with the current
  values. Frames are drawn with Matplotlib into an ipywidgets image, about
  ten per second; no new dependencies. How-to "Exploring a model live";
  tutorial 8 explores the onset of aggregation with it.
- Tutorials 7, oxygen-limited growth (a colony that takes up oxygen
  supplied from the edges grows as a proliferating rim around a hypoxic
  core; an identity-based colony in which division and consumption evolve
  together), and 8, aggregation (cells that secrete a chemokine and follow
  it: the onset of aggregation, and how decay stops coarsening). A how-to,
  "Fields and multiscale models", collects the options of the `pde`
  operator.
- Fields that change during a run (first part of roadmap item 4.6): the
  pipeline operator `pde` (`lgca.fields.PDESpec`, a new operator kind
  `"field"`) updates a field of `StateSpec.fields` by one step of
  `dc/dt = D Δc + P − L c`, in its listed place among the cell operators.
  Terms: diffusion, decay, production (a number or a map) and cells that
  secrete (`{"production": r}`) or take up the field (`{"uptake": r}`,
  saturating with `"saturation": K, "n": 1`), selected by `species` and
  `channels`; in identity-based models the rate may name a cell trait.
  Boundaries: periodic, no flux (the default on non-periodic lattices), a
  fixed value, or one condition per side on 1D, square and cubic lattices.
  Solvers: `"implicit"` (backward Euler with SciPy's sparse solvers, the
  default), `"explicit"` (`scipy.integrate.solve_ivp`, RK45 by default) and
  `"steady"`: the field at equilibrium with the current cells at every step
  and when the model is built, for fields much faster than the cells
  (oxygen, growth factors). The steady solver uses conjugate gradients with
  an algebraic multigrid preconditioner (pyamg) that is kept while the cells
  change little; it costs about one go-or-grow step on 100² to 400²
  lattices (`benchmarks/fields.py`).
  The Laplacian uses the cells' neighbourhood on every lattice
  (`lgca.fields.laplacian`). Chemotaxis and the `field` and `gradient` cues
  read the updated field.
- A number in `StateSpec.fields` is a uniform initial value.
- `lgca.study.vary` and `sweep` change the parameters of a `PDESpec`
  (e.g. `"decay"` or `"dynamics.operators[pde].parameters.cells[0].uptake"`).
- The Hill form of switching probabilities, `{"max": 0.1, "hill": [{"name":
  "field", "field": "oxygen", "K": 0.2, "n": 1}]}`: `max` times a product
  of saturating responses `cⁿ / (Kⁿ + cⁿ)` to cues (decreasing for negative
  `n`), wherever the tanh and Boltzmann forms are accepted; `K` and `n` may
  name traits. In identity-based models `max` may name a trait too, e.g. a
  maximal division rate that evolves while division responds to a field.
- `go_or_rest` takes `probability=`, any switching probability, so cells
  can rest in response to a field or any other cue; `kappa` and `theta`
  remain the short form of the density switch.
- `birth_rate` and `death_rate` of `birth_death` accept switching
  probabilities (tanh, Boltzmann and Hill forms, also one per species), so
  cells can divide and die in response to fields, the density and other
  cues. Numbers and trait names mean what they meant.
- The model graph (`describe_model_graph`) draws the fields that the cues
  of an operator's probabilities read.
- Advection in the `pde` operator: `advection=[vx, vy]` or the name of a
  field of velocities adds `−∇·(v c)`, upwinded along the channels of every
  lattice (conserves the total, keeps `c ≥ 0`, adds a numerical diffusion of
  about `|v|/2`). With advection the iterative solvers use BiCGSTAB, and the
  steady solver pyamg's AIR multigrid, which stays at a few iterations per
  step also when advection dominates.
- Reactions of your own in the `pde` operator: `@lgca.reaction` (also
  `lgca.fields.reaction`) registers `function(state, c, **parameters)`
  returning `(production, loss_rate)`, used as `reactions=[{"name": ...,
  parameter: value}]`. Terms that depend on the field are iterated to
  convergence in the implicit and steady solvers; fields that react with
  each other are updated in the order of their operators.
- New dependencies: `pyamg>=5.1` (not on Python 3.14, which it has no
  wheels for yet; the steady solver then uses SciPy alone) and
  `threadpoolctl>=3.0` (field solvers run with one BLAS thread, 3 to 9
  times faster). The SciPy minimum rose from 1.9.2 to 1.11.
- `FieldRecorder(["oxygen"])` records fields: `result.data["oxygen"]`, and
  `field_oxygen`/`field_oxygen_steps` in the `measurements.npz` of
  `biolgca run` (one pair per recorded field).
- `animate_scalarfield` animates a recorded field on square and hexagonal
  lattices (also `lgca.plotting.animate(lgca, "scalarfield", data=...)`);
  `plot_scalarfield` of 1D models draws a field history as a kymograph.
- "Open in Colab" badges on the tutorials; in Colab their first cell
  installs BioLGCA.
- CI runs the tests and the command line on Windows and macOS as well.
- The preferred citation is Syga et al. 2026 (`CITATION.cff`, README,
  docs); the 2021 BIO-LGCA paper remains listed.
- A process sweep started from a script without `if __name__ ==
  "__main__":` fails with an explanation instead of a multiprocessing
  `EOFError`.
- `lgca.study`: `vary(spec, {"time.steps": 200, "kappa": 4})` returns a
  copy of a model with values changed by path (`dynamics.operators[0].kappa`,
  entries by name such as `operators[go_or_rest]`, or a short name that
  occurs once in the model). `sweep(spec, grid=..., seeds=..., measure=...)`
  runs every combination and returns a pandas DataFrame with one row per
  run, or with `long=True` one row per run and recorded step; measures are
  functions of the result or names of recordings (whose recorders are added).
  `n_jobs` runs several at the same time, in worker processes (default) or
  threads (`backend="threads"`); `plugins` names modules the workers import.
- `biolgca sweep model.json --vary kappa=-4,0,4 --seeds 0:10 --measure
  population --long --n-jobs 4 --output runs/` writes `table.csv` and
  `sweep.json`. `biolgca run`, `validate` and `sweep` take `--plugins
  MODULE` to import trusted modules with your own rules.
- pandas is a dependency (`pandas>=2.1`).
- Model files keep large arrays in an array file: `save_model_spec(spec,
  "model.json")` writes arrays with more than 100 elements (nodes, fields)
  to `model.arrays.npz` next to it, and `load_model_spec` reads them from
  there (`max_inline_array=None` keeps them inline). Tuples are written as
  lists; files with `{"__tuple__": ...}` still load.
- Tutorial 3 sweeps two cue strengths over ten seeds with one `sweep` call
  and plots error bars.
- `lgca.data` for models built with `get_lgca`: the recordings of the last
  `timeevo` by name, as `result.data` of a model run.
- The quick starts (docs home, getting started) and tutorials 1 and 2 use
  `get_lgca` and its standard models; tutorial 3 introduces `ModelSpec`,
  showing that a standard model written as a specification runs the same
  trajectory, and the later tutorials use it.
- `result.data`: the data recorded in a model run by name, e.g.
  `result.data["population"]` (alias `"n"`), `result.data["density"]` or
  the metrics of a `ScalarTimeSeriesRecorder`, with
  `result.data.steps(name)`; `list(result.data)` shows what was recorded.
  The attributes `lgca.n_t`, `lgca.dens_t`, ... remain. The tutorials,
  README and examples use `result.data`.
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
- `resting`, go-or-rest as a term of the Boltzmann reorientation: a cell
  alone at its node rests with a switching probability (default the
  go-or-grow switch of the density, kappa 5 and theta 0.75; any response to
  cues, also in the Boltzmann form), via the rest score
  `log(p / (1 - p)) + log(v / r)`. It combines with other cues in one
  decision. Without volume exclusion every cell rests with `p`, as after
  `go_or_rest` and a velocity random walk; with volume exclusion the node's
  cells choose together, and crowded nodes fill their rest channels more
  often than with `go_or_rest` (documented in the concepts page).
  `go_or_rest` is unchanged.
- Reorientation terms can give every cell its own weights in identity-based
  models (`lgca.rules.CellWeights`), e.g. `resting` with `kappa` and `theta`
  as traits. The Metropolis sampler works on per-cell scores.
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
- `lgca.explore` draws 3D lattices: in perspective (the nodes with cells,
  where a field is high, or the flux as arrows), turned with a slider, or as
  a plane that a slider moves through the lattice (`slice="z"` or
  `slice=("z", 3)`). It uses Matplotlib, so it needs no Mayavi.
- 3D plots: cubic and Moore models animate fields (`animate_scalarfield`,
  `AnimationObserver(kind="scalarfield", field=...)`) and the density as
  cubes (`animate_density_cubes`, `kind="density_cubes"`), and show the
  channel configuration live (`live_animate_config`), as 2D models do.

### Changed

- Models without volume exclusion draw the random numbers of switches
  between species, `go_or_rest`, `birth_death`, killing and dividing cells
  and the spreading of cells over channels only for the nodes and channels
  that hold cells: the same results for the same seed, as NumPy draws
  nothing for empty ones, but these draws take a fifth of the time on a
  sparse lattice (a 3D Moore lattice of 32³ nodes with three species and a
  small tumour; a step takes about a quarter less).
- `pde` operators that follow each other update their fields
  simultaneously: each reads the other fields as they were before the first
  of them, like the equations of a Morpheus `System`, so their order no
  longer matters; steady fields among them are also solved this way when the
  model is built. The new `InteractionPipelineSpec.field_updates`
  (`dynamics.field_updates` in model files; `vary` and `reconfigure` change
  it) chooses between `"simultaneous"`, the default, and `"sequential"`, in
  which each reads the fields as the operators before it left them, as
  before. Both are first order in time. Results change only where such
  operators read each other's fields (in a reaction, or as production or
  advection); no model of the library, the zoo or the docs does. Before, a
  field that a reaction copied from another got the other's new value or
  its old one depending on which operator was listed first.
  `metadata["schedule"]` joins fields updated together with `&`, and
  `metadata["field_updates"]` records the choice.
- An operator object of a registered interaction, e.g. from
  `create_plugin`, counts as its mapping `{"name": ..., "parameters": ...}`:
  `InteractionPipelineSpec` holds the mapping, which `vary`, `sweep`,
  `lgca.explore` and `reconfigure` can change, also by short names, and which
  a model file saves as before. Before, `vary` could not change such an
  object, not even by its full path, and short names skipped it. It stays an
  object, a template as before, if it holds more than its name and
  parameters give (attributes set on it, or what it stored while it ran in a
  model), if another operator object of the list refers to it, or if its
  name is deprecated; `vary` then explains this. Operator objects otherwise
  remain for class-based operators that are not registered.
- Files that a model made in Python reads, e.g. the state of a `from_npz`
  initializer, are found as `np.load` finds them: relative to the working
  directory, or absolute (`build_model`, `run_model`, `sweep` and
  `lgca.explore` without `resource_base`). They needed `resource_base`, and an
  absolute path also `trusted_paths=True`. With `resource_base`, the directory
  of a model file as `biolgca` passes it, they must be relative and stay
  inside it, as before. A file that is missing says where relative paths were
  looked for.
- Rules keep the contract of their kind. An operation that a rule's kind
  does not allow (e.g. `cells.set_trait` in a reorientation, `kill` or
  `divide` in a phenotype switch, `move` or `shuffle_cells` in a `field`
  rule), or assigned `state.counts` that break the kind's conservation law,
  warns with the new `lgca.ContractWarning`, once per rule and model, and the
  step goes on as the wider kind would apply it. The model still reports the
  declared kind (e.g. in its metadata, that the rule keeps the cells), so
  declare the kind the rule needs. Before, broken conservation laws raised `ValueError` when
  the state was committed, after traits, labels and families had been
  written, and trait writes were not checked at all: a reorientation could
  set traits, or divide and kill the daughter, which left orphan labels,
  trait rows and families. `lgca.testing.strict_contracts()` and
  `check_interaction` turn the warning into an error, raised before the
  operation changes anything; the test suite does so for every test. A
  reorientation may not set traits: a rule that moves cells and sets traits
  (e.g. a memory of the heading) is a phenotype switch. A stack narrower than
  one of its operators warns when the model is built. Rules of kind
  `reorientation` and `phenotype_switch` in identity-based models became 1.1
  to 1.7 times faster: the operations check the kind, instead of a recount of
  the cells at every commit.
- `cells.label`, `cells.index` and `cells.channel` are read-only; cells change
  through their operations. Writing them directly could lose cells, move them
  to other nodes or duplicate labels, and no check noticed.
- Reorientation terms, the reactions and cell terms of `pde`, and stack
  builders get a state that they can read but not change: its operations
  raise `TypeError`.
- Fields: saturating (Hill) uptake in the implicit and steady solvers is
  solved by Newton's method instead of Picard iteration. It converges where
  Picard iteration cycled or failed (steep uptake with `n >= 2`, strong uptake
  in implicit steps, steady fields started at zero), usually in one to three
  linear solves per step, and 1.8 to 7 times cheaper per step in the
  benchmarks (tutorial 7's model at 200² nodes, 3D spheroids). It stops
  when the estimated error of the field is at most `rtol` relative to its
  largest value, also where a fixed boundary value supplies most of the
  field; there Picard iteration on the AMG and CG backends had stopped at
  errors of up to 1e-4. Seeded runs with saturating uptake change within the
  tolerance (tutorial 7: 6,186 cells after 200 steps instead of 6,211).
  `solver_options={"nonlinear": "picard"}` restores the earlier iteration.
  `metadata["fields"][name]["nonlinear"]` records which iteration a run used.
- Fields: reactions (`@reaction`) are solved by Newton's method too, with
  their derivative taken by a finite difference; a reaction's production and
  loss rate at a node should depend on the field at that node only. Where a
  reaction enhances itself (production rising with the field, or loss
  falling: logistic growth below its capacity, autocatalysis, switches), it is
  linearized as in Picard iteration, so the field reaches the stable
  equilibrium its dynamics reach, at Picard iteration's rate. Saturating
  uptake together with reactions, and saturating uptake written as a
  reaction, now converge where Picard iteration did not, and a field that a
  reaction takes to 0 converges on the iterative backends.
  `metadata["fields"][name]["floored_iterations"]` counts the iterations that
  used Picard's linearization. Seeded runs with reactions change within
  `rtol`; a steady logistic field started at exactly 0 now stays 0 (an
  equilibrium) instead of raising.
- Fields: a steady field that only the cells remove (no decay, no fixed
  boundary value) now raises when production reaches the total saturable
  uptake, where no steady state exists, instead of failing to converge or
  publishing a meaningless field. It starts from the uniform level at which
  production and uptake balance.
- A step is applied as a whole or not at all. If it raises, or is
  interrupted, the model is put back in the state before the step (cells in
  their order, traits, labels, families, fields, the random stream), the
  exception says so in a note, and the model can step again: retried steps
  give the same run as steps that never failed. Before, the operators before
  the failing one stayed applied, so `step()` again applied them a second
  time, and the Explorer's Step and Play went on from that state. The
  checkpoint costs one copy of `nodes` and the fields and no measurable time;
  `model.rollback = False` saves the memory, and a model whose step failed
  then refuses further steps (`RuntimeError`) until it is rebuilt, e.g. with
  `build_model(model.spec)`. Models made by `get_lgca` with a function as
  interaction are rolled back too. Not rolled back: what operators keep
  themselves (e.g. the statistics and multigrid hierarchies of field solvers)
  and writes through `TraitArray.values`.
- A run that fails keeps what it recorded up to the last completed step: the
  recordings (`lgca.n_t` and the other arrays, field recordings, the cell
  histories of identity-based models) no longer end in empty rows for the
  steps that did not run, the observers write their files (e.g. the time
  series CSV, movies), the error says where the run stopped, and
  `model.metadata["runtime"]` has the `failed_step`. Observers of your own can
  implement `truncate(lgca, step)`.
- `sweep(..., errors="record")` (`biolgca sweep --errors record`) goes on when
  a run fails: the run gets a row with its values, its seed and the error in a
  column `error`, and a warning counts the failed runs. By default
  (`errors="raise"`) a failed run stops the sweep, now without running the
  other runs first: thread and process pools ran every run of the sweep before
  they raised the error.
- `lgca.explore` says when a failed step was rolled back, and draws the steps
  of a frame that were applied before it.

- Every model owns its configuration. `build_model` (and so `run_model`,
  `sweep`, `lgca.explore` and the command line) builds from its own copy of
  the spec and keeps it as `model.spec` / `result.spec`: changes the caller
  makes afterwards to the dicts, lists and arrays of the spec (operator
  parameters, also nested ones such as switch rates, initial nodes, traits,
  the initializer, the operator list) reach neither the running model nor
  `model.spec`, and the running model does not change `model.spec` (a lattice
  of label lists gets lists of its own). Operator objects in
  `dynamics.operators` (e.g. from `create_plugin`, or your own
  `InteractionOperator` subclasses), also those a stack returns, are
  templates: every model runs its own copy, `model.pipeline.operators[i]`,
  and the object in the spec never runs. Before, all models built from one
  spec ran the same object, which stores what it learns about its model (the
  capacity, a stack's operators, a field's matrices): building a second model
  changed the first one's dynamics (a capacity-2 model grew to 1,593 cells
  instead of 70 after a capacity-50 variant was built from the same object),
  runs of a sweep in threads were labelled with a capacity they did not run
  with, and an Explorer change that was rejected still changed the running
  model. The operator objects of one list (the model's, or a stack's) are
  copied together, so operators that refer to each other still do, and
  everything an operator holds is copied with it, also a list or dict handed
  to its constructor and the object of a bound method: code that reads an
  operator object after a run, or a list the operator appends to, must read
  `model.pipeline.operators[i]` instead, or hand the operator an observer of
  `spec.analysis`. Observers, rules, reorientation-term definitions,
  `PluginInfo` and functions are shared, not copied. An operator or another
  part of the spec that cannot be copied (a lock, an open file, a wrapper
  whose `__getattr__` recurses) is rejected when the model is built, and by
  `sweep` before the first run, also in a value of the grid, with a message
  naming it; an operator then needs a `__deepcopy__` method. A `pde`
  operator that has already run can serve as a template: its copies start
  with fresh statistics and leave out its matrices, which the new model
  computes. The copy costs well under a millisecond for most models (0.04 ms
  for a 50 x 50 model, 0.3 ms for 200 x 200 identity-based nodes given as a
  label array); nodes given as lists of labels (identity-based without
  volume exclusion) are copied for `model.spec` and for the lattice, about
  20 ms each for 200 x 200 nodes (the build takes about a sixth longer). The
  copy takes memory as long as the model exists: initial nodes, traits and
  fields given as arrays are held by `model.spec` and by the lattice, and
  label lists nearly double the memory of a model after its first step
  (67 instead of 38 MB for 300 x 300 nodes). `lgca.explore` keeps one copy
  for itself and its model.
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
  and tutorial notebooks pass with: numpy 1.24, scipy 1.9.2, matplotlib 3.9,
  tqdm 4.64.1, JupyterLab 4.0, PyYAML 6, Mayavi 4.9 and PySide6 6.4. Previously
  no minimum was declared, so pip could combine BioLGCA with releases that fail
  at runtime. numpy 1.24 is needed to reject ragged node arrays, matplotlib 3.9
  for plots in notebooks with current IPython (older versions fail at the first
  plot), and Mayavi 4.9 is the first release with
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
- Matplotlib is part of the normal installation, so results can be plotted
  without selecting extras. JupyterLab is in the `notebooks` extra: `uv sync`
  installs it with the development tools, and with pip use
  `pip install "biolgca[notebooks]"`. Installs without it (Colab, clusters,
  editors with their own notebook support, packages depending on BioLGCA)
  stay small: 35 packages instead of about 105. Three-dimensional Mayavi
  rendering remains optional.
- Documentation builds now start from clean generated sources and treat Sphinx
  warnings and notebook execution failures as errors.
- Maintained notebooks construct `ModelSpec` and interaction pipelines in
  visible cells. Historical factory-API notebooks and the research-project
  example now live in clearly labelled archive directories.

### Fixed

- Mutation and `trait_switch` events check the traits that their probability
  reads (a trait as `max`, as a cue's `kappa` or `theta`, or a `trait` cue)
  before the first event changes a trait, as they check the traits they
  change. Before, a missing one raised a `KeyError` only after the earlier
  events had written their traits (the step was then rolled back).
- `biolgca validate` makes the checks that `biolgca run` makes before it
  writes: two recorders that would write the same file (e.g. two
  `ScalarTimeSeriesRecorder`s without `output_path`, both `time_series.csv`)
  and a `FieldRecorder` of a field the model does not have. `validate`
  accepted both, and `run` rejected the second only after it had written
  `model.resolved.json`.
- Fields keep the values of their boundary condition beyond the lattice edge
  (the ghost nodes that `state.gradient` reads at the edge nodes) where a rule
  writes them with `set_field`. They took the values at the edge until the
  `pde` ran again, so rules after it saw a wrong gradient at the edge nodes
  with periodic and fixed-value boundaries (in a test, −4 became 0.5), and a
  field without `pde` had this wrong gradient on a periodic lattice from the
  start. A field without `pde` now wraps around where the lattice does, and
  takes the values at the edge beyond walls as before. A ramp on a periodic
  lattice falls back where the edges meet, which cells there now sense:
  tutorial 3 and the composed example of the model-spec how-to put their
  signal ramps on lattices with walls.
- `lgca.explore` runs models that start from a file (`from_npz`): it takes
  `resource_base` and `trusted_paths` like `build_model`, and builds the model
  again from the same file when a control changes the lattice or the state.
  It raised "Relative initializer resources require resource_base".
- A sweep reads its input files (the arrays of a `from_npz` initializer) once
  before the first run: a file edited during a sweep changed the runs that
  started later (seeds 0 and 1 from 300 cells, 2 and 3 from 100). `biolgca
  run` and `biolgca sweep` copy their inputs before the first run and run from
  the copies, so the archive holds what ran (it was copied afterwards).
- Interrupting (Ctrl-C) the Boltzmann reorientation of an identity-based
  model with volume exclusion could replace every label by 1: the sampler
  wrote the occupied channels and then the labels. It now writes the lattice
  once, also in multispecies models whose reorientation moves some species
  only.
- A reorientation term that draws random numbers moved the model's random
  stream when the model was built.
- Model files keep the `sensed_species` of reorientation terms, all parameters
  of single-cue operators made with `create_plugin` (beta, `sensed_species`,
  trait and the cue's own parameters), NumPy numbers in the spec (e.g. a seed
  or capacity from `np.arange`) and a term's trait given as a NumPy string.
  Species-specific channel sets (`channels={0: ...}`) keep working after a JSON
  round trip, which turns their keys into strings. A reorientation term placed
  directly in `dynamics.operators` ran with beta 1 and was saved without its
  settings; building and saving reject it with a pointer to
  `ReorientationSpec(terms=[...])`.
  `time.seed` must be a non-negative integer.
- Cell numbers: assigned counts are checked for node totals beyond the signed
  int64 of the samplers before conversion (float counts by their exact totals)
  and are copied, not shared with the caller; `add_cells` and
  `divide_cells(channels="same")` refuse results beyond it. Cell selectors
  reject float positions, out-of-range or wrapping indices and 2-D position
  arrays such as `np.argwhere` output (integral floats and 2-D arrays used to be
  accepted); `shuffle_cells` and `random_walk` reject negative, wrapping,
  boolean and float species.
- Non-finite runtime values no longer silently change events: rates and cues
  read from traits, traits that scale reorientation terms, mutation effects and
  mutation results that overflow raise an error that names the trait, and
  mutation conditions with reversed or NaN ranges raise. A mutation event checks
  all its traits and computes all its effects before writing any, so a rejected
  event changes no trait. A failed `divide(new_family=True)` no longer leaves
  trait arrays of different lengths, which made every later division fail.
- Rules of kind `field` in identity-based models must keep every cell's label,
  node and channel: a division followed by a kill replaced a cell unnoticed.
  `FamilyPopulationRecorder` follows families founded in a rule body, and on a
  model without cell identities it raises a clear error.
- Fields: rules read their own `set_field` updates, also through `gradient`;
  steady fields inside (nested) stacks are at equilibrium from the first step;
  production maps must be finite and non-negative; a non-finite solver result is
  rejected and leaves the field unchanged; a declared field is always copied to
  the lattice. Saturating uptake and reactions iterate until both the change and
  the residual are within `rtol`. A solve still within 1000 times this tolerance
  after `max_iterations` warns and uses the last iterate; a worse one raises and
  says whether to raise `max_iterations` or to use `solver="explicit"`. Before,
  every unconverged iterate was published after one warning, however far off.
  Hill uptake no longer overflows at extreme concentrations. The `"auto"`
  backend factors a matrix that does not depend on the cells once on lattices
  of up to 10⁶, 10⁵ and 6,000 nodes in 1D, 2D and 3D (steady fields in 3D, and
  without pyamg in any dimension, up to 20,000) and iterates above; before, it
  factored every such matrix, however large, so seeded runs with such a field
  on larger lattices change within the solver tolerance. Without pyamg, a
  steady field whose matrix the cells change (uptake) on a 3D lattice is solved
  with conjugate gradients; before, lattices of up to 20,000 nodes factored that
  matrix again in every solve (1.8 s per solve at 27³ nodes, against about
  15 ms).
  Multigrid hierarchies are built with a fixed seed, so that runs with a steady
  field repeat exactly. `metadata["fields"]` names the backend used and counts
  failed updates. BDF and Radau get the Jacobian of saturating uptake and
  reactions (hundreds to thousands of times fewer evaluations on stiff uptake).
  Integer solver options are stored as integers, and invalid ones (infinite,
  NaN, strings, booleans) raise an error naming the option. A steady solution
  that overflows is reported as such, not as having no unique solution.
- Sweeps reject grid keys that set the same parameter, or a part of another
  (aliases, named or negative operator selectors, `time` with `time.steps`),
  empty axes and grids, and seeds that are not non-negative integers. Run
  folders are bounded in length and distinct also on case-insensitive file
  systems. Long tables keep a run that recorded nothing. Runs in threads no
  longer share the operator objects of the model or the grid (see "Every
  model owns its configuration" under Changed). Operator objects of
  registered rules, stacks and reorientation terms pickle by reference, like
  functions, so sweeps in worker processes (the default with `n_jobs` > 1)
  take them; before, such a model had to run in threads.
  Short parameter names (`vary`, `sweep`, `explore`) no longer fail on a model
  that holds a single-cue operator object.
  `biolgca sweep` rejects a repeated `--vary`, archives every varied NPZ input
  (reading all of them first, so that a sweep into its own folder keeps its
  inputs) and records in `sweep.json` which file each copy was made from; the
  model's own file is not needed when every run replaces it.
  `mutational_meltdown.scan` accepts generators and rejects invalid seeds.
- Recorders: metric names cannot be `step` or an alias of `result.data`.
  Recorders that would write the same output name are rejected before the first
  step, and by `biolgca validate`, `biolgca run` and `sweep` before anything is
  written, unless they record the same quantity (a `PopulationRecorder` and a
  population CSV may use different schedules). Subclassed recorders and scalar
  schedules without samples appear in `result.data`, and sweeps measure the
  metrics of an existing `ScalarTimeSeriesRecorder` instead of adding a
  recorder. `ScalarTimeSeriesRecorder(metrics={})` raises.
- `lgca.explore` offers only scalar fields as views and measures, starts with
  vector fields such as advection velocities, works under windowed matplotlib
  backends such as TkAgg (its figure never opens a window or stays in pyplot),
  rejects controls that set the same place or a part of another, accepts a
  control for a whole parameter mapping, and warns when it cannot try a change
  on a copy of the model.
- Zoo: clone-trait summaries include the first cell in models without volume
  exclusion; radial profiles handle extinction and a colony at the centre.
- Documentation: the decorator API pages get distinct file names (strict builds
  on Windows), and references to `lgca.interaction`, `lgca.stack` and
  `lgca.reorientation_term` link to them.
- 3D isosurfaces (`plot_density`, `plot_scalarfield` and their animations
  on cubic and Moore lattices) are drawn at the levels of `contours=`, which
  stay fixed during an animation. They were drawn at five levels spread over
  the data of the first frame, and an animation that started from an empty
  lattice or a uniform field showed no surfaces at all. 3D scalar fields hide
  NaN nodes as they hide masked ones.
- Built wheels include the model zoo; the installed-package CI smoke test
  imports its catalogue and runs an entry.
- Model files preserve the shape of inline arrays, including empty arrays
  and identity-based channel lists, when loaded from JSON or YAML.
- Sweeps reject measure names that overwrite parameter, seed or step
  columns. Explorer controls read omitted parameter defaults consistently
  from dataclass and dictionary operator specifications.
- The steady field solver accepts fixed boundaries with value zero, and
  advection responds to successive in-place changes of a velocity field.
- Trait-based reorientation works when restricted to a single channel.
  Oversized candidate tables are rejected before enumeration, including
  restricted channel subsets on Moore lattices.
- Hexagonal scalar-field plots render all sites when given an unmasked
  masked array or a scalar boolean mask.
- Zoo core/rim summaries handle extinction, and excitable-medium return
  times are NaN when no later frame exists. The documented Barkley equation
  now agrees with the implemented scaling.
- Recording no longer changes a run of an identity-based model without
  volume exclusion. `FamilyPopulationRecorder`, `PerTypeRecorder`,
  `OrderParameterRecorder` and the flux views of `lgca.explore` read
  `lgca.nodes`, which builds the label lists from the cell table; the table
  rebuilt from the lists orders the cells differently, so the cells got
  other random numbers in later steps and a recorded run differed from an
  unrecorded one with the same seed (in distribution the dynamics were
  right). They now read the cell table.
- Reading `lgca.nodes` no longer changes the rest of a run of an
  identity-based model without volume exclusion, whoever reads it: plotting
  and animation observers, the estimate of the recording size at the start
  of `run()` (so a run continued after `step()` calls differed from one
  uninterrupted run), or a notebook. A cell table rebuilt from label lists
  that were read but not changed keeps the order of its cells.
- Reorientation terms and field reactions see `StateSpec.capacity` as
  rules do. With volume exclusion they used the number of channels, so a
  density cue such as that of `resting` or a reaction written with
  `state.density / state.capacity` responded to another crowding scale than
  the rules of the same model. `LatticeState(model.lgca)` uses it too, and
  the run metadata records it as `capacity` (it recorded the number of
  channels of volume-exclusion models).
- `sweep` rejects a seed in the grid (`"time.seed"` or `"seed"`, also
  `biolgca sweep --vary seed=...`) and asks for `seeds=` (`--seeds`). The
  seed of the grid was replaced by the seeds of the sweep, so rows labelled
  with different seeds were identical runs.
- Models that read a file, e.g. the state of a `from_npz` initializer, can
  be swept: `sweep(..., resource_base=, trusted_paths=)` passes the
  directory to every run (also in worker processes), and `biolgca sweep`
  uses the directory of the model file and takes `--trusted-paths`. It
  copies the file to `resources/` in the output directory, as `biolgca run`
  does, so the model in `sweep.json` can be run again.
- Runs of a sweep no longer write to the same files. Plot snapshots and
  movies of every run went to the observer's path, so each run overwrote
  the previous one's (in parallel, at the same time). A sweep now keeps no
  files by default: observers that only draw or write files do not run.
  `sweep(..., keep_files=True)` (`biolgca sweep --keep-files`) keeps them,
  in a folder per run named after its values and seed, e.g.
  `snapshots/kappa=2_seed=1/`. Plotting observers draw one at a time, so
  that runs in threads do not draw into each other's figures.
- The model of the zoo entry `jamming` can be saved as a model file:
  `build_spec` returns the model alone, and the notebook records the
  observables (`single_cells`, `velocity_correlation`) with a recorder of
  its own. `build_spec` takes `steps` (default 250, of which
  `jamming.TRANSIENT = 50` are the transient) instead of `transient` and
  `steps`. Runs no longer share a CSV file in the temporary directory. A
  test saves every entry and runs it in a new Python process; loading a
  model that names the rules of an entry that was not imported says which
  module to import (`--plugins lgca.zoo.jamming`).
- `ScalarTimeSeriesRecorder` writes a CSV file only to an `output_path`
  you give; the values are in `result.data` under the metrics' names. It
  wrote `time_series.csv` to the working directory by default. `biolgca
  run` still writes `time_series.csv` to its output directory.
- The zoo index no longer calls the capacity of `evolution_modes` hard.
- `PlotSnapshotObserver` and `AnimationObserver` draw a field by its name:
  `kind="scalarfield", field="oxygen"`. Without `field=` they failed only
  when the first frame was drawn (the animation after the whole run).
  `PlotSnapshotObserver` on a 1D model fails when the run starts and says
  to plot the recorded run as a kymograph; it failed at the first
  snapshot. The documentation lists the kinds each observer draws.
- The largest cell label of identity-based models (`lgca.maxlabel`) is a
  Python integer. It was a NumPy integer, and with NumPy 1.x `maxlabel + 1`
  became a float that cannot index the trait arrays.
- CSV files of `ScalarTimeSeriesRecorder` and `CSVSnapshotObserver` are
  written as UTF-8 on every system (Windows used its local encoding).
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
