# BioLGCA branch code review, 30 September 2026

> Archived on 2026-10-01: its findings are fixed or listed as known issues. Open items moved to the [roadmap](../../roadmap.md); kept for reference.

The review corrected 26 bounded defect groups and added 86 regression cases.
Three architectural findings remain open: incomplete state transactions, nonlinear
solver robustness, and ownership of a compiled model's configuration. Their
reproductions and repair plans are below. No changes were committed or pushed.

A second pass on the same day re-checked every fix, completed eleven of them, relaxed
F11 and F18, which rejected valid cases, and fixed further defects found along the way;
see [Follow-up review](#follow-up-review-second-pass). The architecture findings, renamed
CR-A1 to CR-A3, have revised plans in
[architecture_proposals_2026-09-30.md](2026-09-30-architecture-proposals.md), which
supersede the plans and the order of work below.

## Scope and method

Reviewed `aidevelop` at `ee3e492`, including changes since the earlier repaired
milestone `5978778f`. The focus was the current model API and its connections to
rules, identity-based cell tables, switching and mutations, PDE fields, recording,
parameter sweeps, CLI archives, Explorer, plotting, model-zoo measurements and
packaging. The legacy implementations were used as comparison coverage, rather
than as a definition of scientific correctness.

The unchanged branch passed its existing suite: **2,149 passed, 4 skipped**.
Independent checks then exercised conservation laws, analytic field solutions,
seeded model-file replay, integer limits, failed-rule state, measurement names,
generator inputs and archive replay after removal of source files. An archived
copy of the unchanged package was imported explicitly to check that the new
regressions detect defects in the original code.

Severity in this report: **P1** can silently change biological dynamics, discard
scientific data or misrepresent a run; **P2** breaks a supported workflow or has a
substantial resource cost. Severity is about the failing case, not every use of
the affected feature.

## Corrected defects

The regressions are in [review_regressions_test.py](../../../../tests/review_regressions_test.py).
The Windows documentation correction is verified by the strict Sphinx build.

| ID | Severity | Failing behavior and correction |
| --- | --- | --- |
| F01 | P1 | Model-file serialization dropped a reorientation term's `sensed_species`, changing who senses whom after JSON/YAML replay. Preserve scalar/list/array selections and add the schema property; compare seeded dynamics before and after replay. |
| F02 | P2 | A steady PDE inside a stack, including nested stacks, was not initialized at model construction. Forward field attachment through every stack; a uniform source 2 and decay 1 now give equilibrium 2 immediately. |
| F03 | P1 | Signed/float channel counts could overflow a node's signed 64-bit total, and assigned arrays could be mutated after validation through a shared buffer. Check totals over species and channels before casting, check finite integer additions and own the assigned buffer. |
| F04 | P1 | JSON turns species keys in channel dictionaries into strings; lookup silently fell back to all channels. Normalize and validate each key, including duplicate representations, before selecting channels. |
| F05 | P1 | A custom rule read the old field after calling `set_field`, including when taking its gradient. Read staged field updates and use the same padding as publication, so successive operations in a rule use its current values. |
| F06 | P1 | Cell selectors truncated float positions and wrapped large unsigned indices, potentially selecting the wrong cell. Accept integer positions or correctly shaped masks, validate bounds before conversion and preserve valid negative indexing. |
| F07 | P1 | NaN in a runtime trait or cue could silently suppress growth, death or switching. Reject non-finite per-cell rate parameters and cue inputs before those events. This checks runtime values, not only initial traits. |
| F08 | P2 | Integral float solver options passed validation and subsequently failed in `range`; numeric string tolerances passed validation without being normalized. Store validated integer and float values in solver options. |
| F09 | P1 | A named production map could contain negative or non-finite sources without the checks applied to constant production. Validate the map at construction and on every use, preserving the previous field on rejection. |
| F10 | P1 | Hill uptake overflowed or underflowed when individual powers of finite concentrations were unrepresentable. Evaluate the loss rate in log space; verify its scaling identity at concentrations around `1e100` and `1e-100`. |
| F11 | P1 | An unconverged nonlinear field iterate was published after a warning. Check the nonlinear equation's residual and raise with diagnostics before storing an inaccurate result. This is a guard; the solver limitations in CR-A2 remain. |
| F12 | P1 | Non-finite solver output could be stored in a field even for finite input values whose result overflows. Check the entire result before publication and leave the previous field intact. |
| F13 | P2 | `backend="auto"` forced sparse LU for a constant field on arbitrarily large lattices, bypassing the existing direct-solver size limit. Use the iterative backend above 20,000 nodes; verify the route and analytic solution on 20,001 nodes. |
| F14 | P1 | Equivalent sweep paths, including named/negative operator selectors and implicit `parameters`, could overwrite an axis or split the same parameter across columns. Canonicalize paths for duplicate detection and coalesce aliases across explicit points, while retaining readable named columns. Explorer uses the same duplicate detection. |
| F15 | P1 | Sweep seeds were truncated or reinterpreted from floats and booleans; empty axes could produce no runs without a useful error. Require non-negative integer seeds and nonempty axes before dispatch. |
| F16 | P2 | Array-valued sweep parameters produced file names too long for Windows. Bound folder names with a stable digest, retain distinct run folders and replace quadratic duplicate-name counting with a counter. |
| F17 | P1 | Repeated CLI `--vary` keys were silently discarded by constructing a dictionary. Reject repeated declarations before creating output files. |
| F18 | P1 | Scalar metrics could overwrite `step`, become unreachable through the `n` alias, or collide with recorded arrays/fields. Validate metric names and output ownership before advancing. Matching standard population recorders still coexist with default scalar CSV output. |
| F19 | P2 | Subclassed recorders and scalar schedules with no samples were absent from `result.data`. Recognize subclasses and expose empty arrays with paired steps. Sweeps also respect existing custom metrics instead of adding an overriding built-in recorder. |
| F20 | P2 | Explorer offered vector fields as scalar images and could fail on startup for valid models using advection. Offer scalar fields as views; verify that 1D, square and cubic models with vector fields can advance. |
| F21 | P1 | Clone-trait summaries skipped label 0 in NoVE identity-based models, where it is the first real cell. Skip the sentinel only in volume-exclusion models and handle an empty family table. |
| F22 | P1 | Mutational-meltdown scans exhausted generator axes after the first outer iteration and omitted combinations. Materialize each axis once; a 2 × 2 × 2 scan now evaluates eight distinct jobs. |
| F23 | P2 | Radial profiles failed after extinction and divided by a zero front radius for a population only at the centre. Return zero densities and undefined statistics after extinction; preserve central-node densities and traits with undefined normalized radii. |
| F24 | P2 | Sphinx generated colliding Windows filenames for `Interaction`/`interaction` and `Stack`/`stack`, causing strict documentation errors. Assign distinct decorator filenames in `autosummary_filename_map`. |
| F25 | P1 | A CLI sweep archive copied only the base NPZ initializer and retained original file references in varied inputs. Archive and rewrite every varied NPZ initializer declaration; replay the archive after deleting the originals. |
| F26 | P1 | Fixed mutation effects and callbacks could write NaN/infinity to traits; reversed or NaN condition ranges silently disabled events. Validate fixed values, each draw and condition order before the affected trait is written. |

The fixes retain the two APIs and existing model families. They do not replace
the PDE algorithm or impose a transaction around the whole simulation. In
particular, F11/F12 protect publication of the affected field, but do not undo
earlier operators in a failed step; that is CR-A1.

## Architectural findings and repair plans

### CR-A1 — P1: rule validation does not make state changes transactional

Sources: [LatticeState](../../../../lgca/lattice_state.py),
[Cells](../../../../lgca/cells.py), [pipeline execution](../../../../lgca/pipeline.py),
[CompiledModel](../../../../lgca/model.py).

**Reproduction.** Start an identity-based model with one cell, fitness 7 and
family tracking. Run this invalid reorientation rule:

```python
@interaction(kind="reorientation", families=("ib", "nove_ib"))
def invalid_division(state):
    state.cells.set_trait([0], "fitness", 9)
    state.cells.divide([0], new_family=True)
```

Conservation validation rejects the added cell. The lattice still contains its
original single cell and the model's step stays zero. However, its fitness has
become 9, `maxlabel` increased, trait storage contains a daughter, and a new
family has been appended. This occurs with and without volume exclusion.

A second example runs as a `kind="field"` rule in a NoVE identity-based model:

```python
state.cells.divide([0], channels="same", new_family=True)
state.cells.kill([0])
```

The channel counts are unchanged, so validation accepts it, but the original
cell label 0 has been replaced by daughter label 1. A field rule has changed
cell identity and lineage despite its advertised conservation contract.

**Cause and impact.** Counts and the cell-position table are staged locally, but
`set_trait`, `_inherit` and `_found_families` write live trait/lineage storage.
The field-kind check compares counts without retaining the original cell
identities. Other operators in the pipeline commit independently, and
`CompiledModel.step` increments time only after the entire pipeline succeeds.
Consequently, catching an exception and retrying can continue from a partially
advanced model at the old step number. Direct class-based operators can also
write the LGCA without the decorated-rule checks. The failed-rule examples are
verified; the broader pipeline consequence follows from its execution order.

**Plan.**

1. Give a rule an owned mutation journal or staged trait/lineage buffers in
   addition to counts and positions. Publish all affected state only after its
   validation succeeds. Include RNG state in the failure contract; avoid an
   unconditional copy of the whole lattice at every step.
2. Validate the complete contract for each kind. Field rules preserve counts,
   identities, traits and lineage; reorientation preserves the cells per node
   and species; phenotype switching preserves identities and cell numbers
   while permitting its documented species/trait changes. Put these checks at
   the common execution boundary, including class-based and stacked operators.
3. Define failed-step semantics explicitly. Either roll back the entire step,
   or mark a partially advanced model unusable until checkpoint restoration.
   An operator transaction alone does not undo an earlier successful operator.

**Acceptance.** Inject failures after trait writes, division, new-family
creation, field updates and a later pipeline operator. Compare nodes, cell
tables, traits, label counters, family graphs, fields, time and RNG state with
the pre-step snapshot across classical, identity-based, VE and NoVE models.
Test count-preserving identity replacement as a forbidden field operation.
Until the failure contract is implemented, rebuild or restore a model after a
failed step instead of treating it as unchanged.

### CR-A2 — P2: nonlinear field iteration needs a robust convergence strategy

Sources: `_solve_system`, `_solve_linear` and `_hill_rate` in
[fields.py](../../../../lgca/fields.py).

**Reproduction.** On three periodic 1D nodes with one cell per node, use a steady
field initially zero, diffusion 1, production 1, uptake 2, saturation 1 and Hill
exponent 2. Its equation is

```text
0 = Δu + 1 − 2 u² / (1 + u²).
```

The uniform non-negative equilibrium is exactly `u = 1`. Construction instead
raises "the steady problem has no unique solution". At the zero starting
iterate, the effective uptake loss is zero and the first *linearized* matrix
is a singular periodic Laplacian. Its singularity is not non-existence of the
nonlinear equilibrium.

A separate, implicit, diffusion-free example starts at `u = 2`, with uptake 4,
saturation 1 and exponent 64. For a unit time step its solution satisfies
`u + 4u⁶⁴/(1 + u⁶⁴) = 2`, with root about `0.98332`. The fixed-point iteration
oscillates and previously published a value near 2 with a residual near 4.
F11 now rejects that result and preserves the old field, but does not provide
a convergent algorithm for the case. Merely increasing the iteration limit
does not resolve a cycling fixed-point map.

**Plan.**

1. Separate existence/uniqueness checks for the physical problem from failure
   of one frozen linearization. Handle steady nonlinear uptake from zero
   without claiming that uptake cannot remove the field.
2. Add bracketed scalar solves for uncoupled monotone local equations, and a
   damped Newton or trust-region method with a sparse Jacobian for coupled
   fields. Use a positivity-preserving line search and residual-based stopping.
   Keep the inexpensive existing iteration where it is demonstrably adequate.
3. Report nonlinear convergence, linear convergence and singularity as separate
   diagnostics, including iterations and the final scaled residual.

**Acceptance.** Test the analytic uniform equilibrium from zero, steep Hill
uptake against a bracketed reference root, mass balance with sources/boundaries,
and backend agreement. Include non-periodic boundaries, species/trait-dependent
uptake and reaction terms. Check time-step convergence for evolving fields and
large-lattice memory behavior before changing the default algorithm. Existing
field tests and tutorials pass with the publication guard; these harder cases
remain limitations.

### CR-A3 — P1: a compiled model does not own an immutable configuration snapshot

Sources: `ModelSpec`, `build_model` and model-file serialization in
[model.py](../../../../lgca/model.py); parameter updates and archived runs in
[study.py](../../../../lgca/study.py), [cli.py](../../../../lgca/cli.py) and
[explorer.py](../../../../lgca/explorer.py).

**Reproduction.** Build a three-cell model with an operator mapping that refers
to a caller-owned dictionary:

```python
parameters = {"birth_rate": 0.0, "death_rate": 0.0}
# Build with {"name": "birth_death", "parameters": parameters}.
parameters["death_rate"] = 1.0
model.step()
```

The compiled operator still uses death rate zero and all three cells survive.
But `model.spec.dynamics.operators[0]["parameters"]["death_rate"]` is now 1.
Serializing the reported spec therefore describes different dynamics. A frozen
dataclass does not freeze caller-owned nested dictionaries or arrays.

**Cause and impact.** Configuration is normalized, operators receive their own
parameters, and the public spec retains nested mutable objects. These objects
are not one authoritative snapshot of what actually ran. Results, archives and
live controls depend on that snapshot for reproducibility. Runtime field values
also live on the LGCA separately from the initial declarations in
`ModelContext.fields`; the design should distinguish declared initial values
from runtime state rather than exposing both as interchangeable authorities.
The parameter divergence above is directly reproduced; the wider ownership
risks should be covered by the acceptance tests rather than assumed fixed.

**Plan.**

1. At build time, create an owned configuration snapshot, copy nested parameter
   data and own input arrays. Compile from that same snapshot and return it as
   the run's specification. Define immutable views or explicit update methods.
   Do not blindly deepcopy GUI widgets, open files or arbitrary observer state.
2. Separate reproducible configuration from live runtime fields and observers.
   Record live Explorer changes explicitly with their effective step; updates
   should replace a configuration snapshot instead of mutating earlier results.
3. Archive resources before execution and record content hashes and resolved
   seeds per run. F25 repairs omitted varied files, but copying sweep inputs
   only after the runs does not prove that the archived bytes were the ones
   consumed if an input changed during execution.

**Acceptance.** Mutate original parameter dictionaries, traits, operator lists
and input arrays after construction. The compiled dynamics and saved spec must
remain consistent. Replay from archived inputs after deleting originals;
compare seeded histories and hashes. Test that later Explorer changes do not
alter an earlier run's recorded configuration.

## Suggested order of work

Superseded by the order in the architecture proposals.

1. **CR-A1: state contracts and failure semantics.** First close count-preserving
   identity replacement, then make rule publication atomic, then define the
   pipeline/checkpoint failure boundary.
2. **CR-A3: owned configuration and provenance.** Establish the owned snapshot
   before extending Explorer updates or adding more archive formats.
3. **CR-A2: nonlinear solver.** Retain the new rejection guard while implementing
   and validating the stronger algorithm against analytic reference cases.

Each stage should be a separate, reviewable change with the acceptance probes
above. These are focused boundary repairs, not a proposal to rewrite the model
API or lattice backends.

## Verification and limits

- **Full suite: 2,235 passed, 4 skipped, 62 warnings** in 187.46 seconds after
  the final mutation guard (`uv run --no-sync python -m pytest -q tests`, with
  an isolated temporary directory and cache provider disabled).
- **86 new regression cases.** The unchanged package fails 81 and passes 5;
  the passing cases include compatibility/control cases. All 86 pass with the
  fixes and are included in the full-suite result.
- **Strict documentation build passed**, including all **8 tutorials and 8 zoo
  notebooks** executed from source after the final mutation guard (422 source
  files; 9 minutes 29 seconds of reading/execution).
- A wheel was built and installed in an isolated environment. Imported code was
  confirmed to come from its `site-packages`; zoo loading, schema resources,
  CLI export/validation/run, a two-process Windows sweep and archive replay
  after deleting original NPZ files passed. The rebuilt final wheel, including
  the mutation guard, also passed the repeated smoke check.
- Ruff passes the new tests and the changed rule, cell, CLI, field, state,
  study, Explorer, switching, mutation and zoo modules. `model.py` and
  `simulation.py` retain **16 pre-existing** Ruff findings, verified against
  the unchanged source. No new findings were added there.
- Evidence and reproducible architecture probes are kept locally under ignored
  `outputs/code-review-2026-09-30/`, including `check_architecture.py`, full
  test/doc logs and original-source comparison logs. The regression file and
  this report are the versionable evidence.

Validation used Windows, Python 3.13.12 and the locked dependencies. Other Python
versions, minimum dependencies and other operating systems were not rerun here.
Native Mayavi/Qt windows were not exercised; the supported headless plotting and
Explorer paths are covered by the suite. Full publication-size simulations and
independent reproduction of every cited paper were outside this code review.
Passing legacy comparisons and notebook execution alone do not establish those
scientific claims.

## Follow-up review (second pass)

**Method.** Each fix was re-checked against the unchanged package (imported
explicitly) and the fixed tree: the defect was reproduced on the former, the fix
confirmed on the latter, and the fix probed for regressions, over-restriction,
sibling code paths and test strength. Every issue of medium or high severity was
then reproduced again, independently, by a reviewer asked to refute it. The
confirmed issues were fixed in five groups of disjoint files, each checked by a
second reviewer, and every behavioural fix has a regression test that fails on
the tree before this pass. The architectural findings were investigated, two
competing designs were prototyped for each and compared on each other's cases;
the results are in [the architecture proposals](2026-09-30-architecture-proposals.md).

### Verdicts on F01 to F26

13 fixes were correct as they were, 11 were correct but incomplete, and two
rejected valid cases (F11, F18). All of them are completed or corrected now.

| ID | Verdict | Follow-up |
| --- | --- | --- |
| F01 | correct | The same defect class elsewhere: single-cue operators made with `create_plugin` were saved without beta, `sensed_species`, trait and cue parameters; NumPy numbers in the spec and a NumPy-string trait could not be saved. A reorientation term placed directly in `dynamics.operators` ran with beta 1 and was saved without its settings; building and saving reject it. |
| F02 | correct | |
| F03 | correct, incomplete | `add_cells` and `divide_cells(channels="same")` could still overflow a node's int64 total; float counts are now checked by exact totals, and non-finite ones raise `ValueError`. |
| F04 | correct | Tests cover duplicate and boolean keys. |
| F05 | correct | |
| F06 | correct, incomplete | 2-D position arrays are rejected with a message naming their shape, also when they select no cell; the species selector of `shuffle_cells` and `random_walk` still wrapped -1 and uint64 values. |
| F07 | correct, incomplete | A trait that scales a reorientation term was not checked (NaN sent every cell to channel 0 without volume exclusion). Messages name the trait. |
| F08 | correct, incomplete | inf, NaN, None and strings raised `OverflowError` or raw `int()` errors (a YAML `.inf` crashed the CLI). |
| F09 | correct | |
| F10 | correct | Every call became about 7x slower; the direct formula is used again, with the log form only where a power over- or underflows. |
| F11 | rejected valid cases | It rejected solves that were exact up to the linear solver's rounding (an inactive reaction on an ill-conditioned steady field), and slowly converging but accurate Michaelis-Menten solves (tutorial 7's spheroid with `solver="implicit"` and D = 5 failed at step 0); stopping on the residual alone also made direct solves less accurate (4e-4 against 1e-6). Now the residual that updating the terms adds and the relative change must both be within `rtol`; a solve within 1000x warns and uses the iterate, a worse one raises with advice from its convergence rate; a non-finite iterate raises at once. |
| F12 | correct | |
| F13 | correct, incomplete | A node count ignores the dimension and the solver: 1D and 2D fields became 2 to 12 times slower, and implicit 3D fields under 20,000 nodes still factored (10 s and 700 MB at 27³), where CG is faster per step. The limits are now per dimension (10⁶, 10⁵ and 6,000 nodes); steady fields in 3D keep their factors up to 20,000 nodes, because multigrid iterations in every step are 1.3 to 3 times slower. |
| F14 | correct, incomplete | A key inside another (`time` with `time.steps`, a whole operator with one of its parameters) still overwrote an axis, also for Explorer controls. Messages name the user's keys; invalid indices give clear errors. |
| F15 | correct, incomplete | An empty list of points still gave an empty table. An invalid `spec.time.seed` is named as such, and the model rejects it. |
| F16 | correct | Folder names that differ only in case are made distinct. |
| F17 | correct | |
| F18 | rejected valid cases | Requiring equal schedules rejected setups that worked, including `biolgca run` of a zoo model with a time-series CSV. Recorders of the same quantity may now use any schedules; the checks run before any output is written (`validate`, `run`, `sweep`). |
| F19 | correct, incomplete | Long tables dropped runs that recorded nothing, which F19 made reachable for scalar metrics. |
| F20 | correct, incomplete | A vector field as a measure failed with a misleading error. The Explorer failed under windowed backends such as TkAgg (older than this review), so the new test failed when run on its own on Windows. |
| F21 | correct | |
| F22 | correct | Seeds are validated. |
| F23 | correct | |
| F24 | correct | References to the decorators now link to their pages. |
| F25 | correct, incomplete | Numbered copies could overwrite inputs stored in `resources/` (a sweep into its own folder), so the archive replayed other inputs; all inputs are read first. `sweep.json` records where each copy came from; a missing model file that every run replaces is not needed. |
| F26 | correct, incomplete | Results that overflow to infinity were still written, and an event with two traits wrote the first before the second was rejected. |

### Further fixes

Found while checking the fixes and the architectural findings:

- Threaded sweeps shared operator objects between concurrent runs, which store
  model-dependent state such as the capacity: rows were labelled with a capacity
  they did not run with (CR-A3).
- A rule of kind `field` could replace a cell's identity, the review's second
  CR-A1 reproduction: field rules must keep every cell's label, node and channel.
- A failed `divide(new_family=True)` left trait arrays of different lengths, and
  every later division failed.
- `FamilyPopulationRecorder` failed after the step when a rule founded families
  in its body, and with an `AttributeError` on models without cell identities.
- BDF and Radau were given a Jacobian without the nonlinear terms: 40,761
  right-hand-side evaluations instead of 215 on stiff uptake.
- A field given with its border nodes was attached to the lattice by reference.
- The Explorer skipped the validation of a change silently when it could not
  copy the model; it warns now. A control for a whole parameter mapping crashed.
- An overflowing steady solution was reported as having no unique solution.
- Runs with a steady field did not repeat exactly (differences near 1e-10):
  pyamg seeds its multigrid hierarchy from NumPy's global generator. The
  hierarchy is now built with a fixed seed, leaving the global generator as it
  was.
- `vary`, `sweep` and `explore` failed on any short parameter name when the model
  held a single-cue operator object.

### Verification

- Full suite: **2,423 passed, 4 skipped** (Windows, Python 3.13, locked dependencies).
- The new and changed tests of this pass fail on the tree before it (about 160
  cases) and pass now.
- Strict documentation build: succeeded, executing all 16 notebooks (8 tutorials, 8 zoo models).
- Ruff passes on every changed module and test file, except for findings that
  predate the review: 16 in `model.py` and `simulation.py`, and 15 in
  `tests/model_io_test.py`, `tests/model_spec_test.py` and `tests/simulation_test.py`.
- Probes and prototypes are kept locally in the ignored folder
  `outputs/code-review-2026-09-30/followup-probes/`.

### Still open

Besides CR-A1 to CR-A3, these were found and left alone (low severity, or they
need a decision):

- A rule's `set_field` gives a field edge-valued ghost nodes, as documented, also
  when a `pde` owns the field; until the `pde` runs again this replaces its
  wrapped (periodic) or fixed boundary values, and gradients at the seam differ.
  Padding by the owner's boundary condition is a design decision.
- Traits named only in a mutation's probability (a trait-valued `max`, `kappa` or
  `theta`) are not checked before the first block writes.
- `biolgca validate` accepts two `ScalarTimeSeriesRecorder`s without
  `output_path`, which both write `time_series.csv`, and a `FieldRecorder` of an
  undeclared field; `run` rejects them, the latter after writing
  `model.resolved.json`.
- ~~Without pyamg (Python 3.14), a steady field with uptake on a 3D lattice of up
  to 20,000 nodes calls `spsolve` in every Picard iteration.~~ Fixed on
  2026-10-01: such fields use CG in 3D (measured 1.8 s per solve at 27³ with
  `spsolve`, about 15 ms with CG).
- `vary` on a spec that holds a single-cue operator object fails, because it
  treats every object with `.terms` as a `ReorientationSpec`.
- Order-dependent tests: `plugin_registry_test` fails after `research_models_test`
  or `study_test`, which register rules without parameter descriptions, and two
  `model_spec_test` cases fail after `zoo_test`. The suite passes in its default
  order.
