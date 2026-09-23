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
| B5 | M | Curated examples are tuned for test speed, not science. `go_or_grow` (κ=4, θ=0.75, 15 steps) shrinks from 12 to 4 cells in 150 steps and cannot answer its stated question. Examples use a `sys.path` workaround (`ensure_project_root_on_path`). |
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
  at the first caller outside the package. Remaining `warnings.warn` calls
  without a useful stack level can move to it when touched.

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
  dependency so tqdm progress bars work in notebooks. Still to check:
  clipping of long colour-bar labels at the figure edge (D3).

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

**0.8 Record the seed of every run** (B9)
- If `time.seed` is missing, draw one from `numpy.random.SeedSequence`, use it,
  and write it to the run metadata and `model.resolved.json`, so every run can
  be repeated.

**0.7 Example audit** (B5)
- Each example must show its effect at its default settings (e.g. go-or-grow
  grows) and produce one gallery figure. Keep fast variants for tests through
  a `steps` argument, not by shortening the example itself. Remove
  `ensure_project_root_on_path`.

### Phase 1: A simple way to add interactions (two to three weeks)

**1.1 Lattice view for operators** (C4)

A `LatticeState` object passed to user code hides ghost nodes and dtypes:

```python
state.counts            # interior channel counts, int array (dims..., K)
state.counts = new      # writes interior, applies boundaries, updates density
state.density           # particles per node
state.neighbor_sum(a)   # sum of a over the interaction neighbourhood
state.gradient(f)       # physical gradient in lattice units
state.field("signal")   # named field from StateSpec.fields
state.c, state.K, state.velocitychannels, state.restchannels
state.rng, state.step, state.geometry, state.dims
```

Species axis for multispecies models; identity-based families get
`state.cells` in phase 2.

**1.2 `@interaction` decorator** (C1)

```python
from lgca import interaction

@interaction(kind="birth_death")
def crowding_death(state, r_d=0.1):
    """Each cell dies with probability r_d * density / K."""
    p = r_d * state.density[..., None] / state.K
    state.counts = state.counts * (state.rng.random(state.counts.shape) >= p)

spec = ModelSpec(..., dynamics=InteractionPipelineSpec(
    operators=[crowding_death(r_d=0.2), {"name": "classical.random_walk"}]))
```

- The decorator builds `PluginInfo` from the signature: parameters from
  keyword defaults (no default means required), description from the
  docstring, name from `module.function` unless given.
- Calling the decorated function with parameters returns an operator entry
  for `operators=[...]`; the name also works in JSON.
- `families=` defaults to all families the lattice view supports; the
  conservation law is optional metadata.
- Drop `port_status`, `test_status` and `legacy_source` from the public
  `PluginInfo` (keep them internally until phase 2 removes the legacy code).
- `ParameterSpec` defaults are applied by the framework; factories stop
  duplicating them.

**1.3 Public reorientation terms** (C3)

Most BIO-LGCA biases couple a field to the candidate state in one of three
ways, which the existing terms already implement. Expose them:

```python
from lgca import reorientation_term

@reorientation_term(coupling="flux")        # score = g(r) · J(s')
def drift(state, direction=(1.0, 0.0)):
    return np.broadcast_to(direction, state.dims + (len(direction),))

ReorientationTermSpec(name="drift", beta=2.0, parameters={"direction": [0, 1]})
```

Couplings: `"flux"` (vector field · flux of the candidate), `"nematic"`
(tensor field : nematic tensor), `"rest"` (scalar · rest occupancy), and
`"channels"` (per-channel weights). An advanced `score(features, state)` form
stays available. Rewrite the built-in terms on the same public API.

**1.4 Interaction test helper** (C5)

```python
from lgca.testing import check_interaction

report = check_interaction(crowding_death, parameters={"r_d": 0.2},
                           geometries=("lin", "square", "hex"))
```

Checks on small seeded lattices: ghost nodes untouched after boundary
application, dtype and capacity respected, declared conservation holds, same
seed gives same result, no NaN. Raises with a readable report; usable as a
one-line pytest.

**1.5 Documentation**
- Rewrite `how_to/custom_interactions.rst` around 1.2–1.4: a growth rule, a
  movement bias, a test.
- Tutorial 6 uses the decorator; add exercises that write a term.

Accept: a student writes, registers, tests and sweeps a new death rule or a new
bias in under 30 lines without reading library source; re-running every
notebook cell works.

### Phase 2: One mechanism for all model families (one to two months)

**2.1 Common count representation** (B3, D5)

All families expose channel counts `(dims..., [species,] K)`. Mechanisms act
on counts:

- reorientation for VE enumerates admissible channel states (existing
  Boltzmann sampler);
- reorientation for NoVE samples each cell independently over channels,
  P(i) ∝ exp(β·score_i) (multinomial per node; cost O(K), not O(2^K));
- identity-based families apply the count update and then reassign the existing
  labels to channels uniformly at random (cells are exchangeable for
  movement), keeping properties attached to labels.

**2.2 Cell-level operators for identity models**

`state.cells` exposes per-cell properties (arrays keyed by trait) with
`divide(parents, **traits)` and `kill(cells)`, built on
`lgca.identity_kernels`. Mutation rules become small functions.

**2.3 Names without family prefixes** (B2)

`random_walk`, `alignment`, `chemotaxis`, `birth_death`, `go_or_grow`, ...
resolve to the implementation for the family chosen by `StateSpec`; an
unsupported combination raises a clear error. Old prefixed names remain as
deprecated aliases for one release.

**2.4 Remove duplicates** (B4, D5)
- Separate the random walk from `classical.birth*`.
- After parity tests pass on the new implementations, delete the legacy
  interaction functions and `Native*` duplicates, then the parity tests.
  `get_lgca` builds a `ModelSpec` internally.

Accept: the coverage matrix (mechanism × family) is full for movement terms,
birth/death and switching; `pipeline.py` shrinks substantially; the test suite
has no parity tests.

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
   directly and needs a deprecation release first?
4. **Preferred citation.** `CITATION.cff` currently lists the software with the
   two papers as references; decide whether one paper should be the preferred
   citation.
