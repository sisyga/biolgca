# `aidevelop` Review and Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement the user-selected milestone task by task. Use `superpowers:test-driven-development` for each bug fix or feature, `superpowers:using-git-worktrees` for isolated implementation, and `superpowers:verification-before-completion` before claiming a task complete. Stop at each milestone boundary for review; do not automatically execute later milestones.

**Goal:** Make the `aidevelop` model/pipeline work scientifically safe, straightforward to run and share, maintainable by one person, and measurably efficient without prematurely adding a GUI or compiled backend.

**Architecture:** Keep the existing `get_lgca(...)` API as the interactive/compatibility layer. Make a strict, versioned, pure-data `ModelSpec` the reproducible layer, compiled through one validated registry and one simulation runner. Keep NumPy and Matplotlib as the default numerical and plotting backends; optimize measured hotspots before considering optional JIT or a different 3-D renderer.

**Tech stack:** Python 3.10-3.13, NumPy, SciPy, pytest, stdlib `argparse`, JSON as canonical interchange, optional PyYAML for authoring, `jsonschema` for development-time schema tests only, Matplotlib for 1-D/2-D, Mayavi isolated as the current optional 3-D backend.

**Spec:** This document is both the review record and the implementation specification.

**Review date:** 2026-08-21

**Review range:** `origin/master...aidevelop`, merge base `bace041376dba6665c8d743edba06ec138fcaad5`, reviewed head `303d8f381b6b39040292d1bc887e6324dd17b38a`.

---

## Verdict

`aidevelop` is directionally strong but should not be merged as the new reproducible workflow yet. There is no P0 repository-wide failure and the existing core suite is healthy, but several P1 defects can change scientific results, report false provenance, lose particles, or make supported model combinations fail.

The branch already contains most of the right foundations: a model specification, JSON/YAML persistence, a plugin registry, a phased interaction pipeline, observers, runtime metadata, curated examples, and extensive legacy-equivalence tests. The priority is to make those foundations strict and internally consistent, not to add another front end.

Decisions:

- **Add shareable templates:** yes. Use a strict, versioned JSON wire format and optional YAML authoring. Do not add XML as another source of truth.
- **Add a GUI now:** no. First add schema validation, a small CLI, curated templates, and initial-condition presets. Reconsider a schema-generated notebook/browser form only after these are stable and users still struggle.
- **Move to Numba/Cython/C++ now:** no. First fix unbounded allocations, use existing vectorized kernels, remove Python site loops, and establish repeatable benchmarks. Numba is a later optional experiment for a remaining fixed-dtype hotspot. Cython/C++ are not justified for a one-person package today.
- **Replace Matplotlib:** no. Fix data selection, lifecycle, and artist construction. Keep Mayavi optional and isolated; evaluate PyVista only if 3-D becomes a maintained priority.
- **Target 100% coverage:** no. Protect scientific invariants and every discovered regression. Track coverage for the new orchestration modules, but do not impose a broad repository-wide percentage gate initially.

---

## Review evidence

### Branch and GitHub snapshot

- The branch changes 67 files with approximately 12,508 insertions and 388 deletions relative to the merge base.
- `origin/master` is 7 commits ahead of the merge base; `aidevelop` is 15 commits ahead. Integrate `origin/master` before opening a new merge PR.
- There are **0 open pull requests**.
- The latest `aidevelop` GitHub Actions run is green: [run 27903736806](https://github.com/sisyga/biolgca/actions/runs/27903736806).
- There are **12 open issues**, triaged later in this document.
- Historical PR [#94](https://github.com/sisyga/biolgca/pull/94) merged the earlier cleanup into `master`; the current 15-commit branch delta is subsequent work.
- `.github/workflows/ci.yml` runs on pushes to `master`, but pull-request CI currently excludes PRs targeting `master`.

### Verification performed

The controlled full-suite command was:

```powershell
conda run -n biolgca python -m pytest -q --basetemp .codex-pytest-basetemp-01a023bd -p no:cacheprovider
```

Result:

```text
527 passed, 1 skipped, 8 warnings in 53.52s
```

The warnings are Matplotlib pending deprecations from `lgca/square_plotting.py:76`. The skipped test is `tests/nove_test.py:574`, described as unstable with missing Numba even though Numba is not a project dependency. `coverage`/`pytest-cov` is not installed in the environment, so no trustworthy line-coverage percentage is claimed.

A supplemental 90-case smoke matrix across six model families, five geometries, and periodic/reflecting/absorbing boundaries passed basic propagation mass and recording-shape invariants in under four seconds. This is cheap enough to check in after its assertions are tightened.

### Strengths worth preserving

- The concise exploratory workflow remains available and documented.
- Existing propagation, boundary, initialization, and long-run identity tests cover a broad geometry/backend matrix.
- Native interaction implementations have extensive deterministic comparisons to legacy behavior.
- Schema version metadata, safe YAML loading, NumPy round-tripping, a model graph, conservation metadata, and runtime provenance are valuable foundations.
- Curated executable examples are useful onboarding and teaching assets.
- Pipeline phases and named observers are a better extension boundary than adding more flags to `timeevo()`.

---

## Findings ordered by importance

Priority meanings:

- **P1:** fix before merging `aidevelop` as the recommended workflow.
- **P2:** fix in the first hardening/usability release after the P1 gate.
- **P3:** measure or prototype later; do not block the merge.

### P1: scientific correctness and reproducibility

#### F1. Phenotype switching is not implemented as a particle-conserving state transition

`NativePhenotypeSwitchOperator._switch_channel()` in `lgca/pipeline.py:2165-2175` processes Boolean species sequentially. When two particles propose the same target species in one channel, the fallback can point to a slot already occupied by the first particle, collapsing two particles into one despite the declared mass-conservation law.

Confirmed deterministic probe:

```text
before: 2
after:  1
channel: [False, True]
```

This is not merely a collision-handling defect. For a particle-number-conserving interaction, the elementary update must be a transition of the complete local channel state,

```text
s -> s', with N(s') = N(s),
```

where `s'` is sampled from the admissible state space for the model's volume-exclusion rules. Conservation should follow from how candidate output states are constructed, not from repairing sequential particle writes afterward.

The legacy interactions demonstrate both established patterns:

- NoVE random walk and alignment in `lgca/nove_interactions.py:13-111` compute the full node density and sample a complete replacement channel vector from a multinomial distribution with exactly that particle count.
- VE alignment and chemotaxis in `lgca/interactions.py:128-167` and `lgca/interactions.py:221-261` select complete channel configurations from occupancy permutations with the same particle number.

Required fix: define phenotype switching as an atomic `s -> s'` kernel over the complete multispecies node/channel state. The transition-rate matrix determines phenotype probabilities, while the candidate-state construction enforces particle number and volume exclusion. Document explicitly whether channel identities are preserved or resampled; either choice must produce one valid complete state. Add deterministic collision-heavy and small-state conservation regressions before changing the algorithm.

#### F2. Composed operators can read stale dynamic fields

`CompiledPipeline.execute_step()` in `lgca/pipeline.py:110-138` refreshes `cell_density` and related fields only after all operators. A birth/death operator can mutate `nodes`, then a phenotype/reorientation operator reads the pre-birth density.

Confirmed seed-0 birth -> go-or-rest probe:

```text
particles after birth: 2
cached density: 1
stale result: [0, 1, 0, 1, 0]
refreshed result: [0, 1, 0, 0, 1]
```

Required fix: establish and test an explicit field-refresh contract between operators. The simple correct first implementation is to refresh after each node-mutating operator; optimize redundant refreshes only after operator dependency/output metadata is authoritative.

#### F3. Moore nematic dynamics fail on the lazy permutation path

For `K > 15`, `lgca/base.py:1093-1107` creates `_si_cache` but does not create `si` or provide `get_si_permutations()`. Both legacy nematic code and `lgca/pipeline.py:1839` still index `lgca.si[n_particles]`.

A sparse one-particle Moore `classical.nematic` run fails with:

```text
AttributeError: 'LGCA_3dMoore' object has no attribute 'si'
```

The same defect appears for a square lattice with enough rest channels to make `K > 15`. Add a lazy tensor-permutation accessor and use it from every caller.

This path also has an OOM risk: `math.comb(26, 13) == 10,400,600`; current code can materialize hundreds of MiB of Boolean permutations and multi-GiB float copies. Cache limits must be byte-based, and combinatorial enumeration needs a documented safety threshold or a non-enumerating specialized sampler.

#### F4. Model and plugin typos silently select different science

`model_spec_from_dict()` in `lgca/model.py:159-203` uses defaults without rejecting unknown keys. `_validate_spec()` at `lgca/model.py:683-689` checks only three invariants. `validate_plugin_parameters()` in `lgca/plugins.py:349-398` validates known keys but ignores extras; probability checks accept `NaN` because both range comparisons are false.

Confirmed examples:

- `geomtry: square` loads and silently runs the default `hex` geometry.
- `birth_raet` is accepted and the operator uses the default birth rate.
- a `NaN` probability compiles successfully.

Required fix: strict unknown-key, type, shape, finite-number, and range validation with full configuration paths and close-match suggestions. Audit built-ins for currently accepted but undeclared parameters before switching rejection on.

#### F5. `StateSpec` can override canonical sections and falsify metadata

`_build_lgca()` in `lgca/model.py:692-713` creates canonical keyword arguments and then overlays `state.parameters`. `_attach_fields()` at `lgca/model.py:740-750` performs arbitrary `setattr()` calls.

Confirmed:

- a spec declaring periodic boundary, one rest channel, and seed 1 ran with reflecting boundary, three rest channels, and seed 999 while metadata still reported the declared values;
- a field named `nodes` replaced the simulator state and left `nodes` inconsistent with cached density during compilation.

Required fix: reject reserved keys in `state.parameters`; reject fields colliding with simulator attributes; normalize one canonical configuration before construction; generate provenance from that normalized configuration.

Capacity also has two sources of truth: the state/backend can use one capacity while a native interaction independently defaults to another. Capacity should live in model state. During migration, an operator-level `capacity` may be accepted only when equal to the normalized state value.

#### F6. Sparse recorder schedules return plausible false zeros and save no memory

The runner skips callbacks for unobserved steps, but `DensityRecorder`, `NodeRecorder`, `PopulationRecorder`, and related recorders allocate `timesteps + 1` rows and index by absolute step (`lgca/simulation.py:101-181`). Unobserved rows remain zero and are indistinguishable from genuine zero measurements.

Confirmed:

```text
Schedule(every=2), four steps -> [4, 0, 4, 0, 4]
```

A 10,000-step 256 x 256 density record still allocates about 4.9 GiB even with `every=100`.

Required fix: store compact samples plus an explicit step vector such as `dens_steps`. Preserve the current dense shape for the default every-step schedule, and document the sparse result contract.

#### F7. Family recording fails for native mutating interactions

`FamilyPopulationRecorder` in `lgca/simulation.py:209-254` detects family mutation by comparing the legacy `lgca.interaction` function to a hardcoded list. `ModelSpec` construction sets the underlying interaction to `only_propagation`, so native mutating operators are misclassified and a fixed-width array is allocated.

A deterministic one-step native glioblastoma mutation run raised:

```text
ValueError: could not broadcast input array from shape (11,) into shape (2,)
```

Required fix: declare a `mutates_families` capability in operator metadata and let the recorder inspect the compiled pipeline through the runner/context. Cover every native family-mutating plugin.

#### F8. Multispecies density plotting is broken

`species_density` has shape `dims + (n_species,)` (`lgca/multispecies_base.py:111-113`). The plotting paths forward that array to code that expects exactly two dimensions (`lgca/plots.py:619`). Direct `plot_density()`, `lgca.plotting.plot()`, and `AnimationObserver(kind="density")` all fail for a square multispecies model with `ValueError: too many values to unpack`.

Required fix: default density plots to aggregate `cell_density` and offer an explicit `species=<index>`/`aggregate=` selector. Test static, facade, and observer paths with the Agg backend.

#### F9. Pull requests targeting `master` do not run CI

`.github/workflows/ci.yml:9-12` limits `pull_request.branches` to `aidevelop` and `kio/development`. The eventual merge PR to `master` could therefore bypass PR checks.

Required fix: include `master` before opening the PR and keep the existing Python 3.10-3.13, wheel-install, and documentation jobs.

### P2: portability, lifecycle, and maintainability

#### F10. An advertised example cannot be saved

`model_spec_to_dict()` supports a hardcoded subset of runtime operator classes (`lgca/model.py:383-405`). The bundled `custom_rest_or_align` example embeds `LegacyInteractionOperator`, so `model_spec_to_json(get_example_spec("custom_rest_or_align"))` raises `TypeError`.

Persisted specs must contain pure data and registered names, not arbitrary runtime objects. Every example advertised as portable must round-trip and run. Python-only examples must be explicitly labeled nonportable with an actionable error.

#### F11. Registry mutations are not atomic or collision-safe

`lgca/plugins.py:286-314` checks canonical names only against canonical names and aliases only against aliases, and mutates the registry before completing validation. Aliases can shadow canonical plugins and failed registration can leave partial state.

Required fix: validate the complete canonical-plus-alias namespace first, then commit the mutation atomically. Add both cross-namespace collision directions and rollback tests.

#### F12. Registry metadata is not yet a complete extension contract

Some operators consume parameters not declared by their metadata, and reorientation terms resolve through a private dictionary rather than the public registry. Meanwhile `lgca/plugins.py` and `lgca/pipeline.py` are 2,013 and 2,755 lines respectively, with defaults and interaction knowledge duplicated across legacy and native paths.

Required direction: structured parameter constraints; explicit term registration; one definition per built-in containing metadata, validation, factory, and numerical implementation; legacy APIs become adapters. Do this one interaction family at a time behind parity tests, not as a big-bang rewrite.

#### F13. The two runners duplicate lifecycle behavior

`SimulationRunner` in `lgca/simulation.py` and `_PipelineRunner` in `lgca/model.py` separately implement setup, notification, stepping, finalization, and progress behavior. This contributed to capability and scheduling inconsistencies.

Required direction: one runner with a pluggable step function/context. Keep the old `timeevo()` methods as compatibility adapters.

#### F14. Observer state and figures can leak across runs

- Reusing `ScalarTimeSeriesRecorder` or `CSVSnapshotObserver` does not clear prior records/paths.
- `PlotSnapshotObserver` retains full figure results even after closing them; a 100-snapshot probe retained 100 figures and about 47.5 MiB.
- `AnimationObserver` keeps a frame list and a second contiguous history; a 100-frame 256 x 256 probe retained about 149 MiB with a much higher peak.
- Per-step operator timing dictionaries grow without bound; a 50,000-step propagation-only run retained about 11.9 MiB, implying roughly 238 MiB per million recorded phases.

Required fix: reset observer state in `setup()`, make result retention opt-in, release frames after conversion or stream saved movies, and aggregate timings online by default. Keep detailed timing traces explicitly opt-in and bounded.

#### F15. Plot observer contracts do not match Mayavi

`AnimationObserver` expects an object with `.save` and `._fig`, while cubic Mayavi animation methods call `mlab.show()` and return `None`. Snapshot cleanup always calls Matplotlib close.

Required fix: either define a small renderer lifecycle protocol (`save`, `close`, `animation`) or reject unsupported observer/backend combinations early with an actionable message. Do not pretend the generic observer works for 3-D until it does.

#### F16. YAML support is advertised but its dependency is not installable as documented

The loader recommends docs/dev extras for PyYAML, but neither includes it. Add a dedicated `yaml = ["PyYAML>=6"]` extra and correct the error/docs, or declare JSON-only. This plan chooses the dedicated optional extra.

### P2/P3: performance findings

Preliminary microbenchmarks show where optimization effort should go:

| Kernel | Vectorized/specialized | Generic Python pipeline | Slowdown |
|---|---:|---:|---:|
| Birth/death, 64 x 64 | 0.59 ms | 51.35 ms | 87.6x |
| Birth/death, 128 x 128 | 2.16 ms | 205.24 ms | 95.0x |
| Random reorientation, 32 x 32 | 0.09 ms | 13.27 ms | 154x |
| Random reorientation, 64 x 64 | 0.27 ms | 56.60 ms | 209x |

The compositional loops are in `lgca/pipeline.py:409-463`, `lgca/pipeline.py:2152-2185`, and `lgca/pipeline.py:2689-2755`. These measurements must be checked into a repeatable harness before serving as release claims, but the profile already supports a clear ordering:

1. reuse/vectorize specialized NumPy kernels;
2. stop unbounded metadata and observer allocations;
3. guard combinatorial permutation generation;
4. fast-path identity/diagonal multispecies mutation matrices;
5. replace the object/list IB-NoVE representation with a packed numeric representation if it remains a user-relevant bottleneck;
6. only then prototype optional Numba for a stable fixed-dtype loop.

Dense propagation itself is not the first target: measured classical 1-D/square/hex/cubic propagation was about 9.7-14.3 ns per site, with Moore around 50.9 ns per site. Object-list IB-NoVE propagation was roughly two orders of magnitude slower; Numba will not rescue Python objects.

Do not add Cython or C++ in this plan. Their build, ABI, debugging, packaging, and contributor cost across Python 3.10-3.13 outweighs the evidence currently available.

### P2/P3: plotting assessment

Matplotlib is not the main problem. Current code creates Python artists per lattice site (`RegularPolygon`, arrows, circles, text) and retains histories/figures. Keep Matplotlib and improve how it is used:

- square scalar/density fields: `imshow` or `pcolormesh`;
- hex fields: a reused `PolyCollection`;
- configurations/flux: `Quiver` and collections rather than one artist per cell;
- rendering functions receive already selected, plot-ready arrays;
- numerical extraction/normalization is testable without a display;
- Agg smoke tests cover figure creation and saving.

Keep `plot2d` as the recommended extra. Do not make typical users install Mayavi through `.[plot]`; document `.[plot2d]`, and keep 3-D explicit. Evaluate PyVista only after the renderer contract exists and only if Mayavi maintenance or Python support is a demonstrated blocker.

---

## Usability architecture

### The three supported workflows

1. **Exploration/backward compatibility**

   ```python
   from lgca import get_lgca

   lgca = get_lgca(...)
   lgca.timeevo(...)
   ```

   Keep this concise and stable.

2. **Reproducible/shareable simulation**

   ```powershell
   biolgca validate model.json
   biolgca run model.json --output runs/example-001
   ```

   The run directory contains the resolved versioned spec, runtime metadata, and observer outputs. Relative paths resolve against the spec or run directory, not the caller's accidental working directory.

3. **Python extension work**

   Users register a named interaction or initializer, then reference that stable name from a model spec. A JSON/YAML file never executes an arbitrary import path by itself.

   A standalone `biolgca run` process can resolve only built-ins and registrations shipped with BioLGCA. Until an explicit trusted discovery mechanism is designed, external plugins require a Python launcher that imports and registers the plugin before loading/running the model. Custom observers remain Python-only in schema v1.

### JSON/YAML instead of MorpheusML XML

Borrow the useful MorpheusML properties, not its syntax:

- versioned declarative files;
- clear separation of description, space, state, time, dynamics, and analysis;
- schema validation before execution;
- named operators with documented parameters and constraints;
- shareable templates and reproducible output manifests.

JSON is the canonical wire representation because it is already implemented, unambiguous, easy to validate, and available in the standard library. YAML is an optional human-authoring syntax that must normalize to the exact same data model. XML would create a third parser, schema, documentation path, and round-trip burden without improving BioLGCA's current capabilities. If real Morpheus interoperability is requested later, implement an explicit, tested subset importer/exporter rather than making XML a second canonical format.

Keep model configuration separate from simulation state/checkpoints. Large node arrays, trajectories, and restart data belong in NPZ initially (and possibly Zarr/HDF5 only after a concrete scale requirement), referenced by path and optional checksum from the model/run manifest.

### GUI decision

Do not build a bespoke GUI in this roadmap. The schema and registry are not strict enough yet, and duplicating validation in a desktop UI would make correctness worse.

After strict schema + CLI + templates have shipped, test the workflow with real users. If a visual editor is still valuable, prototype a generated Jupyter widget or small local web form from the JSON Schema and registry metadata. It must emit the same validated model file and invoke the same runner; it must not introduce another internal representation. Live parameter editing from issue #19 is a separate runtime-control feature and should not drive the first UI.

### Initial-condition presets

Issue #20 is high-value usability work. Add named pure-data initializers after schema hardening:

- the existing `StateSpec.density` for random-density initialization;
- one parameterized `region` initializer with center/left/corner placement, radius/extent, and density;
- `from_npz` for large explicit states.

Each initializer validates dimensional compatibility and uses the model RNG. Do not embed large arrays in ordinary JSON/YAML templates.

---

## Test coverage judgment

For a one-person scientific package, behavioral confidence is more valuable than a high global line-coverage number. The current suite is broad, but the new failure modes are at compositions and invalid-input boundaries.

| Area | Current assessment | Required addition |
|---|---|---|
| Propagation | Strong across 1-D, square, hex, cubic, and Moore; classical, identity, NoVE, and multispecies variants | Check in the cheap geometry x family invariant matrix; include one exact channel-mapping assertion per geometry |
| Boundary conditions | Strong legacy periodic/reflecting/absorbing coverage | Extend the compact invariant matrix to native ModelSpec construction and multispecies crossing cases |
| Initialization | Good random and provided-node coverage | Strict invalid-shape/type tests; named initializer tests; deterministic seeds |
| Time evolution | Good basic runner/recording coverage | Cross-operator field freshness, sparse schedules, observer reuse, zero-step behavior, family mutation |
| Interactions | Extensive square-periodic native-vs-legacy parity | Collision-heavy multispecies switching, high-channel geometry smoke, phase-boundary composition |
| Persistence | Good happy-path JSON/NumPy round trips | Unknown keys/types, non-finite values, every portable example round-trips and runs, schema version rejection/migration |
| Registry/modularity | Many built-ins registered and described | Alias/canonical collisions, atomic rollback, complete parameter inventories, custom plugin recipe |
| Plotting | Basic single-species square observer tests | Multispecies aggregate/species selection, all public plot paths, figure lifecycle, headless save smoke, 3-D rejection/contract |
| Performance | Ad hoc profiler only | Repeatable benchmark scenarios and JSON baseline output; no brittle wall-clock CI gate initially |

Coverage policy:

- Add `pytest-cov` to the development extra and record the initial report.
- Do not fail CI on a repository-wide percentage initially.
- Every bug in this review gets a deterministic regression test.
- Every new public branch or validation rule gets focused tests.
- Track changed-module coverage for `model.py`, `pipeline.py`, `plugins.py`, and `simulation.py`; set a floor only after measuring the baseline and excluding backend-specific display code.
- Replace the skipped monolithic NoVE characteristic test with small seeded per-interaction invariants; remove the stale Numba wording.

---

## Open issue and PR triage

There are no open PRs. Do not open a new PR until Tasks 0-7 are green.

| Issue | Recommendation | Roadmap placement |
|---|---|---|
| [#28 Absolute and relative colour scaling](https://github.com/sisyga/biolgca/issues/28) | Keep, but fold into one plotting-consistency milestone with #27/#26/#11/#5 | Task 12 |
| [#27 Density colourbar label striding](https://github.com/sisyga/biolgca/issues/27) | Keep or close as a child of the consolidated plotting issue | Task 12 |
| [#26 Plot slicing](https://github.com/sisyga/biolgca/issues/26) | Audit current support, define remaining geometry/backend matrix, then narrow | Task 12 |
| [#23 Callback in `timeevo()`](https://github.com/sisyga/biolgca/issues/23) | Observer API largely supersedes it; add/document a minimal function observer if needed, then close | Task 6/13 |
| [#20 More initial conditions](https://github.com/sisyga/biolgca/issues/20) | Keep and prioritize after strict schema; implement named initializers | Task 9 |
| [#19 Edit parameters during live simulation](https://github.com/sisyga/biolgca/issues/19) | Defer; separate runtime pause/resume/control from GUI editing | Future, after Task 13 |
| [#18 Plot custom live quantity](https://github.com/sisyga/biolgca/issues/18) | `ScalarTimeSeriesRecorder` covers recording, not a live second window; narrow the issue to the unmet live UI | Future UI |
| [#11 Legend for flux plot](https://github.com/sisyga/biolgca/issues/11) | Include in consolidated plotting consistency work | Task 12 |
| [#10 Ideas](https://github.com/sisyga/biolgca/issues/10) | Split remaining concrete work into issues and close the umbrella checklist | Task 13 |
| [#5 Flow plots](https://github.com/sisyga/biolgca/issues/5) | Audit existing geometry/backend coverage, then close completed portions and retain exact gaps | Task 12 |
| [#3 Inheritance utilities](https://github.com/sisyga/biolgca/issues/3) | Most checklist items are complete; split remaining diversity/initialization/stopping items and close umbrella | Task 13/future |
| [#2 Parameter scans](https://github.com/sisyga/biolgca/issues/2) | Keep, but implement only after stable specs/output manifests; reuse the CLI runner rather than a second simulator | Future after Task 9 |

Create focused issues for F1-F9 before or with implementation so the correctness work is traceable. Avoid adding review comments to unrelated old issues.

---

## Ordered implementation plan

**Execution scope:** use one Goal Mode run per selected milestone. The default first run is Milestone A (Tasks 0-7). Milestone B (Tasks 8-9) is a second run after reviewing A. Tasks 10-12 are ordered follow-up candidates, not automatic scope; select and refine each from the then-current architecture/benchmarks. Task 13 closes whichever scope is chosen for release.

### Task 0: Isolate the work, integrate `master`, and restore the merge gate

**Files:**

- Modify: `.github/workflows/ci.yml`
- Verify: `pyproject.toml`

**Steps:**

1. Use `superpowers:using-git-worktrees` to create `codex/aidevelop-hardening` from `aidevelop`.
2. Fetch current refs and merge `origin/master` into that working branch; do not rewrite the shared `aidevelop` history.
3. Run the full suite before resolving any behavior changes.
4. Add `master` to `pull_request.branches` in `.github/workflows/ci.yml`.
5. Keep the Python 3.10-3.13 test matrix, installed-wheel smoke test, and docs build.
6. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q
   ```

7. Commit: `ci: run pull request checks for master`

**Done when:** the branch includes current `master`, the baseline is recorded, and PRs targeting `master` trigger every current CI job.

### Task 1: Add the compact core invariant matrix

**Files:**

- Create: `tests/core_invariants_test.py`
- Modify: `tests/nove_test.py`

**Steps:**

1. Parameterize small deterministic cases across supported model families, geometries, and periodic/reflecting/absorbing boundaries.
2. Assert exact one-step channel displacement for one representative channel per geometry.
3. Assert particle/identity conservation for propagation-only steps where the boundary should conserve it.
4. Assert absorbing loss only at the expected boundary and reflecting direction mapping for one boundary crossing.
5. Assert seeded initialization reproducibility and recorder shape/step metadata.
6. Split the skipped NoVE characteristic test into small per-interaction invariants, or delete only the unstable redundant portion after equivalent coverage exists.
7. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/core_invariants_test.py tests/nove_test.py
   ```

8. Commit: `test: add cross-geometry core invariants`

**Done when:** the checked-in matrix runs in a few seconds, the stale Numba skip is removed or accurately scoped, and failures identify the exact family/geometry/boundary combination.

### Task 2: Implement phenotype switching as a full-state transition

**Files:**

- Modify: `lgca/pipeline.py`
- Modify: `tests/interaction_pipeline_test.py`
- Modify if needed: `tests/multispecies_test.py`

**Steps:**

1. Add the deterministic two-species, same-channel regression from F1 and confirm current execution changes `N(s)` from two to one.
2. Write the local transition contract in the operator docstring: `s` is the complete multispecies node/channel state; `s'` has the same shape/dtype, satisfies the backend's exclusion constraints, and obeys `N(s') == N(s)`. State explicitly whether the phenotype-switch interaction preserves or resamples channel identities.
3. Add exhaustive small-state tests (for example, two species and two channels) covering every admissible Boolean `s`. Across representative seeds/rate matrices, assert valid output and exact particle-number conservation.
4. Add semantic tests: zero rates return `s` unchanged; a one-particle forced transition changes phenotype as specified; collision-heavy states never merge or delete particles; and empirical frequencies for a one-particle case agree with the rate matrix within a suitable statistical tolerance.
5. Use the legacy full-state patterns as the implementation reference: multinomial sampling conditioned on total density for NoVE states and fixed-occupancy candidate states/permutations for VE states. Do not repair the existing sequential `_switch_channel()` writes with an after-the-fact collision rule.
6. Replace `_switch_channel()` with a state-level sampler such as `_sample_state(state, rates, rng)` (or a vectorized equivalent) that constructs one admissible `s'` atomically. Avoid enumerating a combinatorial state space when constrained count sampling can generate an equivalent valid state.
7. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/interaction_pipeline_test.py tests/multispecies_test.py
   ```

8. Commit: `fix: model phenotype switching as a conserved state transition`

**Done when:** each lattice-site update maps one complete admissible `s` to one complete admissible `s'`, conservation is guaranteed by construction, rate/channel semantics are documented and tested, and the native-vs-legacy suite remains green.

### Task 3: Define correct field freshness between pipeline operators

**Files:**

- Modify: `lgca/pipeline.py`
- Modify: `tests/interaction_pipeline_test.py`
- Modify: `tests/model_spec_test.py`

**Steps:**

1. Add the deterministic birth -> `classical.go_or_rest` regression from F2.
2. Add one switch -> reorientation composition test that reads a field produced by the preceding phase.
3. Confirm the first test differs from an explicitly refreshed reference on current code.
4. Refresh dynamic fields after each operator that may mutate nodes. Initially treat every interaction operator as node-mutating unless it explicitly declares otherwise.
5. Apply boundaries before a downstream operator only when that operator's documented dependencies require valid ghost/boundary state; encode and test this contract rather than guessing per plugin.
6. Re-run the focused composition and parity suites.
7. Benchmark the added refresh cost on a small representative pipeline and record it; correctness takes precedence.
8. Commit: `fix: refresh dynamic fields between pipeline phases`

**Done when:** composed execution matches the explicitly refreshed reference and operator dependency semantics are documented in code.

### Task 4: Repair and bound lazy permutation/tensor generation

**Files:**

- Modify: `lgca/base.py`
- Modify: `lgca/interactions.py`
- Modify: `lgca/pipeline.py`
- Modify: `tests/interaction_pipeline_test.py`
- Modify: `tests/classical_test.py`

**Steps:**

1. Add a sparse Moore native nematic regression and the corresponding legacy regression.
2. Add a `K > 15` square/contact-guidance regression where the geometry is supported.
3. Implement `get_si_permutations(n_particles)` using `get_permutations()` and `_si_cache`; replace direct `lgca.si[...]` calls.
4. Remove repeated full candidate `.astype(float)` copies where Boolean arrays suffice for dot/einsum operations.
5. Compute `math.comb(K, n)` before enumeration. Reject configurations above a documented safe candidate/byte budget with an actionable message until a specialized sampler supports them.
6. Bound caches by estimated bytes, not number of occupancy keys.
7. Test both a safe lazy case and a rejected combinatorial case without allocating the huge matrix.
8. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/interaction_pipeline_test.py tests/classical_test.py
   ```

9. Commit: `fix: make lazy tensor permutations safe`

**Done when:** sparse Moore nematic works through both APIs and high-combination inputs fail early instead of risking OOM.

### Task 5: Make model loading strict and provenance truthful

**Files:**

- Modify: `lgca/model.py`
- Modify: `lgca/plugins.py`
- Modify: relevant capacity consumers in `lgca/pipeline.py`
- Create: `tests/model_spec_validation_test.py`
- Modify: `tests/model_spec_test.py`
- Modify: `tests/legacy_interaction_registry_test.py`

**Steps:**

1. Add failing tests for unknown top-level/section keys, misspellings, wrong scalar/sequence types, invalid geometry/boundary, fractional/bool steps, non-finite rates, and unknown plugin parameters.
2. Add failing tests proving `state.parameters` cannot override `bc`, `seed`, `nodes`, `density`, `restchannels`, `interaction`, or other canonical fields.
3. Add failing tests for field names colliding with `nodes`, `cell_density`, RNG, geometry, or methods.
4. Inventory all built-in parameters and add currently legitimate undeclared parameters to metadata.
5. Add reusable strict helpers that report full paths such as `model.state.densitty` and suggest `density` with `difflib.get_close_matches`.
6. Normalize and validate once before building. Construct the LGCA and metadata from the normalized object.
7. Make state/backend capacity canonical. During migration, reject a conflicting operator capacity and accept an equal duplicate with a deprecation warning.
8. Reject unknown plugin parameters and non-finite numerical values before operator construction.
9. Make registry registration atomic and reject alias/canonical collisions in both directions.
10. Run:

    ```powershell
    conda run -n biolgca python -m pytest -q tests/model_spec_validation_test.py tests/model_spec_test.py tests/legacy_interaction_registry_test.py
    ```

11. Commit in two reviewable units:

    ```text
    fix: validate model specifications strictly
    fix: make plugin registration atomic
    ```

**Done when:** invalid or ambiguous configurations fail before stepping and before trajectory/output allocation, normalized metadata matches actual runtime values, and every built-in has a complete parameter contract. Task 5 validates domain semantics and the current section layout but does not add or freeze new runtime-class serialization tags; Task 8 defines the stable pure-data wire tags.

### Task 6: Correct recorder schedules, capabilities, reuse, and timing

**Files:**

- Modify: `lgca/simulation.py`
- Modify: `lgca/model.py`
- Modify: `lgca/pipeline.py`
- Modify: `lgca/plugins.py`
- Modify: `lgca/plotting.py` and any analysis consumers of recorded `*_t` arrays
- Modify: `tests/simulation_test.py`
- Modify: `tests/model_spec_test.py`
- Modify: `tests/plotting_observer_test.py`

**Steps:**

1. Add failing sparse-schedule tests for node, density, population, and per-type recorders. Assert compact values plus explicit sampled steps. Also reject fractional/bool `Schedule.every` values and invalid explicit step values instead of truncating them.
2. Preserve dense `timesteps + 1` behavior for the default every-step schedule. For sparse schedules, define a paired contract such as `dens_t` + `dens_steps` (and corresponding names for every recorder) with equal leading lengths and no placeholder rows.
3. Audit every plotting/analysis consumer of `nodes_t`, `dens_t`, and related attributes. Make consumers use the paired step vector, or reject unsupported sparse input explicitly. Add compatibility tests for default-dense and sparse histories.
4. Add a deterministic native family-mutation + `FamilyPopulationRecorder` regression.
5. Add a structured `mutates_families` capability to operator metadata and inspect the compiled pipeline rather than `lgca.interaction` when available.
6. Add observer-reuse tests; clear records, paths, frames, results, and derived outputs in `setup()`.
7. Replace the default list of per-step timing dictionaries with online aggregates (`count`, total, min, max by operator/phase).
8. Add an explicit `timing_trace` option with a documented bound for users who need per-step details.
9. Add a minimal `FunctionObserver` only if issue #23's callback use case is not cleanly expressible/documented with the base `Observer`.
10. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/simulation_test.py tests/model_spec_test.py tests/plotting_observer_test.py
   ```

11. Commit: `fix: make observer outputs explicit and bounded`

**Done when:** sparse outputs cannot be mistaken for zero data, mutating family runs record successfully, reused observers start clean, and default metadata is bounded in step count.

### Task 7: Fix plotting correctness and lifecycle before optimizing rendering

**Files:**

- Modify: `lgca/plotting.py`
- Modify: `lgca/square_plotting.py`
- Modify: `lgca/multispecies_base.py` only if the selection API belongs there
- Modify: `lgca/lgca_cubic.py`
- Modify: `tests/plotting_observer_test.py`

**Steps:**

1. Add Agg regressions for multispecies direct plotting, facade plotting, and zero-step `AnimationObserver` finalization.
2. Default `kind="density"` to aggregate `cell_density`; add and validate explicit per-species selection.
3. Record sampled step indices in animations, especially for irregular explicit schedules.
4. Make `PlotSnapshotObserver` result retention opt-in. When output is saved and retention is false, close and release the figure.
5. Release frame-list references after creating an in-memory animation; when saving, stream to the writer instead of keeping two full histories where practical.
6. Add early backend capability validation for Mayavi paths. Do not call Matplotlib close on non-Matplotlib objects.
7. Replace `fig.set_tight_layout(...)` with the supported layout-engine API and make the full suite warning-clean for this deprecation.
8. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/plotting_observer_test.py
   ```

9. Commit: `fix: make plotting observers backend and species aware`

**Done when:** multispecies plots work, observer memory has an explicit retention policy, unsupported 3-D combinations fail early, and headless save tests pass.

### Task 8: Make persisted model specs pure data and schema-backed

**Files:**

- Modify: `lgca/model.py`
- Modify: `lgca/pipeline.py`
- Modify: `lgca/simulation.py`
- Modify: `lgca/examples/__init__.py`
- Modify: `lgca/examples/custom_rest_or_align.py`
- Create: `lgca/schemas/model-spec-v1.schema.json`
- Modify: `pyproject.toml`
- Modify: `tests/model_spec_test.py`
- Modify: `tests/examples_gallery_test.py`
- Modify: `docs/source/model_specs_and_plugins.rst`

**Steps:**

1. Define one stable pure-data wire shape for registered operators, built-in observers, and an optional initializer tag: `name`/`type`, `parameters`, schedule/output options, and no runtime object serialization. Schema v1 enumerates built-in observer schemas; custom observers remain Python-only.
2. Make Python dataclass/runtime objects compile from and serialize to that shape through one code path.
3. Add an explicit portable/nonportable classification. JSON must never execute arbitrary import paths.
4. Convert the curated custom example to a registered named interaction or label it Python-only with an actionable save error. Prefer registration because it ships with the package.
5. Include the optional `state.initializer = {"name": ..., "parameters": ...}` shape and resource-path semantics in schema v1 before publishing it; Task 9 supplies the first built-in initializer implementations.
6. Add a JSON Schema matching the strict parser, add `jsonschema>=4` to the development extra only, and test that allowed keys/types stay synchronized. Runtime loading continues to use the dependency-light strict parser.
7. Add round-trip-and-run tests for every portable curated example.
8. Add `yaml = ["PyYAML>=6"]` and package the schema. Correct loader errors and docs to say `pip install biolgca[yaml]`.
9. Add schema-version rejection and a tested migration hook for future versions; do not invent migrations until version 2 exists.
10. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/model_spec_test.py tests/examples_gallery_test.py
   ```

11. Commit: `feat: define portable versioned model specifications`

**Done when:** every advertised portable example round-trips through JSON, validates against v1 schema, and produces the same seeded result after reload.

### Task 9: Add a small CLI, templates, and initial-condition presets

**Files:**

- Create: `lgca/cli.py`
- Create: `lgca/initializers.py`
- Modify: `lgca/model.py`
- Modify: `lgca/plugins.py` or the extracted registry module
- Modify: `pyproject.toml`
- Create: `tests/cli_test.py`
- Create: `tests/initializers_test.py`
- Modify: `README.md`
- Modify: `docs/source/getting_started.rst`

**Steps:**

1. Add stdlib-`argparse` commands:

   ```text
   biolgca examples list
   biolgca examples export NAME PATH
   biolgca validate MODEL
   biolgca run MODEL --output RUN_DIR
   ```

2. Make `validate` parse, normalize, resolve plugins, and report all safe-to-collect validation errors without allocating a full trajectory.
3. Make `run` create an explicit run directory containing `model.resolved.json`, `metadata.json`, and observer outputs. Reject accidental overwrite unless the user supplies an explicit option.
4. Resolve relative input paths relative to the model file and output paths relative to the run directory. By default, imported resources must remain under the model directory and generated outputs under the run directory after path resolution; reject absolute/`..` escapes unless a clearly named trusted-local opt-out is supplied.
5. Keep `StateSpec.density` as the random initializer. Add one parameterized region initializer (center/left/corner placement, radius/extent, density) and one NPZ-backed initializer. Use the model RNG and validate dimensional compatibility.
6. Export two or three small canonical templates from the curated examples; do not maintain a second hand-written copy of every model.
7. Test exit codes, helpful errors, deterministic runs, output manifest contents, path containment, no-overwrite behavior, and rejection of unregistered external plugin names using pytest temp paths.
8. Run:

   ```powershell
   conda run -n biolgca python -m pytest -q tests/cli_test.py tests/initializers_test.py
   conda run -n biolgca biolgca examples list
   ```

9. Commit: `feat: add validated model CLI and initializers`

**Done when:** a new user can export, validate, run, save, and share a simulation without constructing nested Python classes or using a GUI.

Stop for milestone review here. Tasks 10-12 remain in this document so priorities and dependencies are not lost, but each requires explicit selection for a later Goal Mode run and may warrant a smaller task-specific plan.

### Task 10: Simplify the extension architecture incrementally

**Files:**

- Modify: `lgca/pipeline.py`
- Modify: `lgca/plugins.py`
- Modify: `lgca/simulation.py`
- Modify: `lgca/model.py`
- Create flat focused modules as the first family is extracted, for example:
  - `lgca/operator_base.py`
  - `lgca/operator_registry.py`
  - `lgca/classical_operators.py`
- Modify: `tests/legacy_interaction_registry_test.py`
- Modify: `tests/model_spec_test.py`
- Create: `docs/source/custom_interactions.rst`

**Steps:**

1. Freeze the tested public extension contract: `PluginInfo`, structured parameter schema, factory, operator dependencies/outputs/capabilities, and explicit registration.
2. Keep `plugins.py` as a compatibility facade while moving registry mechanics to a focused module.
3. Extract one complete classical interaction vertical slice—metadata, validation, factory, numerical kernel—into one focused module.
4. Make both legacy `get_lgca(interaction=...)` and ModelSpec resolve that single definition; remove only the duplication made obsolete by this slice.
5. Preserve native-vs-legacy parity tests before extracting the next family.
6. Add public registration for composable reorientation terms if they remain part of the supported extension API.
7. Consolidate `SimulationRunner` and `_PipelineRunner` into one lifecycle engine with a pluggable step function. Keep `timeevo()` as an adapter.
8. Document a 30-50 line custom interaction example, including registration, parameters, a conservation declaration, and one test.
9. Do not add Python entry-point discovery yet; explicit import/registration is simpler and adequate for a one-person package.
10. Run the full suite after each extracted interaction family.
11. Suggested commits:

    ```text
    refactor: centralize plugin registration
    refactor: share classical interaction definitions
    refactor: use one simulation runner
    docs: explain custom interaction registration
    ```

**Done when:** adding a built-in interaction does not require editing two giant registries and duplicated defaults, and a third-party interaction has one documented registration path.

### Task 11: Establish repeatable performance baselines and take safe wins

**Files:**

- Modify or replace: `profiling.py`
- Create: `benchmarks/README.md`
- Create: `tests/profiling_test.py` for harness smoke/format only
- Modify: `lgca/pipeline.py`
- Modify: `lgca/ms_interactions.py`
- Modify: relevant NoVE/identity representation code only after a dedicated design task

**Steps:**

1. Add reproducible scenarios for propagation-only and representative interactions across classical VE, NoVE, identity VE, identity NoVE, and multispecies; include 1-D, square/hex, cubic/Moore where meaningful.
2. Record Python/NumPy/BioLGCA version, seed, lattice size, steps, wall time, peak memory, and output-observer policy in JSON/CSV.
3. Keep benchmarks out of the default pytest timing gate; pytest only verifies the harness runs and output schema is stable.
4. Add identity/diagonal fast paths for multispecies mutation matrices to avoid `... x S x S` temporaries.
5. Replace generic per-site birth/death and known reorientation cases with grouped/vectorized NumPy paths while preserving seeded statistical/invariant tests.
6. Re-run baselines and report both speed and peak-memory deltas. Do not merge an optimization that changes invariants or merely shifts allocation elsewhere.
7. Write a separate design note before replacing object/list IB-NoVE state with packed/CSR-like numeric storage; this is a data-model migration, not a local optimization.
8. Prototype Numba only if a fixed-dtype loop still accounts for at least about 25% of representative runtime after the NumPy work. It must be optional, have a NumPy fallback, and demonstrate a material end-to-end gain.
9. Do not add Cython/C++ in this task.
10. Suggested commits:

    ```text
    perf: add reproducible simulation benchmarks
    perf: vectorize generic interaction kernels
    perf: fast path identity mutation matrices
    ```

**Done when:** optimization decisions are backed by checked-in reproducible measurements and default behavior remains dependency-light.

### Task 12: Consolidate and optimize plotting without changing libraries

**Files:**

- Modify: `lgca/plotting.py`
- Modify: `lgca/square_plotting.py`
- Modify: `lgca/plots.py`
- Modify: geometry-specific plotting modules
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create/modify: headless numerical and render smoke tests

**Steps:**

1. Separate plot-data extraction/normalization from renderer calls and test the numerical layer directly.
2. Use `imshow`/`pcolormesh` for square scalar fields and reusable collections for hex/configuration rendering.
3. Benchmark representative 20 x 20, 40 x 40, and 256 x 256 density/configuration plots before and after.
4. Centralize color normalization, absolute/relative scales, colorbar formatter/locator behavior, labels, and slicing so issues #28/#27/#26/#11 do not require copying fixes across classes.
5. Audit flow plotting support by geometry/backend and close or narrow #5 based on tests.
6. Make `plot2d` the documented default extra. Avoid the umbrella extra that forces Mayavi on 2-D users, or clearly label it.
7. Define the renderer lifecycle contract before evaluating PyVista. Keep Mayavi until a replacement has feature-parity tests and a manageable packaging story.
8. Commit plotting consistency and performance separately.

**Done when:** common plot behavior is implemented once, large 2-D plots avoid per-site artist construction, open plotting issues have exact tested disposition, and ordinary 2-D installs do not pull a 3-D stack.

### Task 13: Documentation, issue cleanup, and release gate

**Files:**

- Modify: `README.md`
- Modify: `docs/source/getting_started.rst`
- Modify: `docs/source/model_specs_and_plugins.rst`
- Modify: `docs/source/observers_and_plotting.rst`
- Modify: `docs/source/interactions_summary.rst`
- Modify: `docs/source/planned_topics.rst`
- Modify: `.github/workflows/ci.yml` only if verification gaps remain

**Steps:**

1. Make the quick exploratory path and reproducible model-file path the first two workflows in the README.
2. Document JSON as canonical, YAML as optional, the schema version, safe plugin resolution, run-directory contents, and config-vs-checkpoint separation.
3. Document sparse recorder step arrays and plotting species selection.
4. Add examples for `validate`, `run`, and `examples export` that CI executes as smoke tests.
5. Triage the 12 open issues according to the table above. Split umbrella issues before closing them; do not lose remaining requirements.
6. Build docs locally:

   ```powershell
   conda run -n biolgca python -m sphinx -W -b html docs/source docs/_build/html
   ```

7. Run the release gate:

   ```powershell
   conda run -n biolgca python -m pytest -q
   conda run -n biolgca python -m build
   ```

8. Install the wheel into a clean environment or use the existing CI wheel smoke job; run the CLI from outside the source tree.
9. Use `superpowers:requesting-code-review`, then `superpowers:finishing-a-development-branch` to decide merge/PR handling.

**Done when:** docs build without warnings, all tests and installed-wheel smoke checks pass, GitHub issues reflect current reality, and the PR to `master` has green CI.

---

## Milestones and stop points

### Milestone A: merge safety

Complete Tasks 0-7. This is the minimum code-correctness merge gate for `aidevelop` if `ModelSpec` remains explicitly experimental. If the merge advertises model files as the recommended shareable workflow, Task 8 is part of this gate too.

Success criteria:

- no known particle-loss or stale-field composition bug;
- lazy Moore nematic works and dangerous enumeration is bounded;
- invalid/misspelled specs fail before a simulation starts;
- metadata matches actual runtime configuration;
- sparse recorders are explicit and family mutation records correctly;
- multispecies plotting works;
- PRs to `master` run CI;
- full suite passes.

### Milestone B: usable reproducible workflow

Complete Tasks 8-9.

Success criteria:

- portable examples round-trip through schema-valid JSON;
- optional YAML installs and behaves identically;
- `biolgca validate/run/examples` works from an installed wheel;
- run directories contain resolved provenance;
- common initial conditions require no manual node mutation;
- a useful simulation can be configured and shared without a GUI.

### Milestone C: sustainable architecture and measured speed

Complete Tasks 10-12 selectively, based on maintained use cases.

Success criteria:

- one documented interaction extension path;
- one runner lifecycle;
- repeatable performance baselines;
- generic kernels no longer pay avoidable Python-site loops for common cases;
- plotting behavior is centralized and large 2-D plots use efficient artists;
- no compiled dependency is added without measured end-to-end justification.

### Milestone D: release/readiness cleanup

Complete Task 13 for the selected release scope and open the PR. Tasks 10-12 do not have to be included unless they were explicitly selected.

---

## Explicitly out of scope for this plan

- a bespoke desktop GUI;
- MorpheusML-compatible XML as a second canonical format;
- arbitrary code execution from model files;
- Cython or a C++ rewrite;
- a wholesale switch from Matplotlib to Plotly/Bokeh/HoloViews;
- an immediate Mayavi-to-PyVista rewrite;
- a repository-wide 100% coverage target;
- a full parameter-scan framework before run manifests and output semantics stabilize;
- embedding large trajectories or checkpoints directly in JSON/YAML.

These can be reconsidered only after the earlier milestones provide evidence that they solve a remaining user problem.
