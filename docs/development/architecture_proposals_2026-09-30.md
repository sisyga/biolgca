# Architecture proposals after the 30 September 2026 review

Status: proposal, 2026-09-30; decisions, CR-A3 PR1 to PR3, CR-A1 PR0, PR1 and
PR3, and CR-A2 PR1 on 2026-10-01 (see [Decisions](#decisions-for-the-maintainer)). Follow-up to [the code review](code_review_2026-09-30.md),
whose three open architectural findings are renamed here CR-A1 to CR-A3, so that
they no longer collide with the IDs A1 to A5 of the [usability roadmap](usability_roadmap.md).

| ID | Finding | Review | Follow-up assessment | Recommended first PR |
| --- | --- | --- | --- | --- |
| CR-A1 | State transactions and rule contracts | P1 | P1 for silent contract violations of custom rules, stacks and class-based operators; P2 for leaks on the exception path (P1 in the Explorer, which continues after a failed step) | Contract checks where each operation is called (about 1 day). **Done 2026-10-01** |
| CR-A2 | Nonlinear field solver | P2 | P2; on the pre-review code P1 (Picard published fields wrong by up to about 280x) | Newton's method for saturating uptake (2 to 2.5 days) |
| CR-A3 | Configuration ownership and provenance | P1 | P1, but through a case the review missed: operator objects shared by every model built from a spec. The review's own reproduction is P2 | Owned copy of the spec, operator objects as templates (about 1 day). **Done 2026-10-01** |

The follow-up (see the review) already fixed the bounded parts that need no
design decision; they are listed under each finding. What remains needs the
decisions collected at the end.

Each proposal was checked against the code: the reviewer's reproductions were
re-run, the mechanism was traced, two competing designs were prototyped as
monkeypatches, and a third pass compared them on each other's test cases. The
probe scripts are kept locally in the ignored folder
`outputs/code-review-2026-09-30/followup-probes/`, including prototypes to start
from: `probes/synth-A1/quick_wins_wt2.diff` (CR-A1 PR0; its parts listed as fixed
below are applied already), `probes/design-A2-newton/newton_design.py` with
`probes/synth-A2/variants.py` (CR-A2 PR1) and `probes/synth-A3/qw_patch.py`
(CR-A3 PR1). They were written against the tree before the follow-up fixes.

## CR-A1: state transactions and rule contracts

### What was verified (CR-A1)

Both reproductions of the review hold. A rejected rule also advances the RNG.
Beyond them, several contract violations are not rejected at all; they commit
while the model metadata advertises conservation:

- A `field` or `reorientation` rule can change traits with `set_trait`, or divide
  and kill the daughter, which leaves an orphan label, a trait row and an empty
  family (`LatticeState._same_cells` compares labels per node only).
- Writing `cells.channel` directly (`cells.channel = zeros`) loses cells or moves
  them to another node, and `commit` accepts it.
- A stack declared `kind="field"` or `"reorientation"` can kill or grow the
  population: the declared kind of a stack is never compared with its operators.
- A class-based operator that advertises a conservation law can empty the
  lattice; only `lgca.testing.check_interaction` checks the law, offline.
- `@reorientation_term` functions, reactions and stack builders get writable
  states. Terms run already at build (`validate` calls `prepare`), so a term that
  writes traits or draws random numbers changes the initial state and the RNG.

On the exception path:

- A failing later operator leaves the earlier operators applied, the RNG advanced
  and the step counter unchanged. The Explorer catches the error and continues
  from this state, and its status line clears on the next click.
- A failed run leaves zero-filled recordings whose step arrays claim every step,
  and loses the outputs written only in `finalize` (time-series CSV, movies).
- An interrupt in the Boltzmann path of identity-based VE models can replace every
  label with 1: the path writes occupancy first and labels second (27 % of a step
  lies in this window on a 150 x 150 hex lattice).

There is no checkpoint facility; `from_npz` refuses identity-based states.

### Fixed in the follow-up (CR-A1)

- `Cells._inherit` checks every trait and the family precondition before
  extending anything; a failed `divide(new_family=True)` no longer leaves the
  model unable to divide again.
- `FamilyPopulationRecorder` follows families founded in a rule body.
- A mutation event checks all its traits and computes all its effects before
  writing any, and rejects results that overflow.
- A `field` rule in an identity-based model must keep the whole cell table (the
  review's second reproduction).

### Recommendation (CR-A1)

Use two mechanisms, not three: **check the contract where an operation is
called**, and **roll back a failed step as a whole**. Do not stage trait writes
per rule: rules read traits they wrote earlier in the same rule (mutations are
applied in order), so staging needs read-your-writes overlays, costs O(maxlabel)
per rule (labels are never reused) and still leaves partial steps when a later
operator fails.

The contract, enforced once:

| kind | counts | positions (label, node, channel) | new labels | new families | traits |
| --- | --- | --- | --- | --- | --- |
| field | equal per channel | unchanged | no | no | no |
| reorientation | equal per node and species | may move within the node | no | no | no (decided) |
| phenotype_switch | equal per node | may move | no | yes | yes |
| birth_death | any | any | yes | yes | yes |

A `LatticeState` built outside a step (`kind=None`) has no contract. Terms,
reactions and stack builders get read-only states. A stack must be at least as
wide as each of its operators (field < reorientation < phenotype_switch <
birth_death); all 43 buildable built-in stacks already are.

Decision 2 (2026-10-01): a reorientation rule may not write traits.
Reorientation rearranges cells over the channels of a node, within their species
(multispecies models) and with their traits unchanged (identity-based models).
A rule that moves cells and sets traits, e.g. a heading memory, is a
phenotype_switch rule, which may do both.

### Enforcement: error, warning or switch (measured 2026-10-01)

The maintainer proposed warning instead of raising, because some models may
need what the contract forbids, and a debug switch in case the checks are
expensive. Probes against the tree of 2026-09-30
(`followup-probes/round3/contracts/`) established:

- **Operation-level violations can be warned about safely.** For all 36
  rejected case x family combinations, a `commit` that warned and then
  committed without the check gave an end state bit-identical to the same rule
  declared `birth_death` (cells, traits, `maxlabel`, families, random stream).
  What is wrong afterwards is only the declared kind: the schedule in the
  metadata ("conserves mass"), `str(plugin)` and the models `check_interaction`
  tests. No committed case left a stale cache; nothing in the engine (order,
  caches, recorders, model files) depends on the kind beyond these checks and
  the metadata.
- **Direct writes to the cell arrays cannot be warned about.**
  `cells.channel[:] = 0` loses 132 of 245 cells under every kind except
  `field`, and `cells.label[0] = cells.label[1]` commits duplicate labels under
  `birth_death`. Only read-only arrays (an error at the write) help.
- **The entry checks cost nothing measurable.** `Cells._allow` takes 88 ns per
  call, the read-only arrays 2 us per structural operation; step ratios on
  150 x 150 models were 0.85 to 1.02 (noise). They need no switch. What is
  expensive today are the commit-time recounts (a narrow kind is 1.1 to 1.6x
  slower than the same rule declared `birth_death`), which entry checks on
  every operation would make redundant for identity-based models, and PR2's
  fingerprints (20 to 26 % of a step).
- **`birth_death` allows everything**, is the fastest kind, and changes no
  result. What no kind can express comes from missing operations: moving an
  identity-based cell to another node (today only by writing `cells.index`
  under `birth_death`, which read-only arrays would forbid) and creating a cell
  without a mother.
- **The rule API is unreleased** (not on master, not on PyPI), so no external
  model depends on the current behaviour. Relaxing an error into a warning
  later breaks no model; tightening a warning into an error does.
- **Warnings need care in a loop.** Python shows a warning once per text and
  line: today's messages contain node counts, so a violating rule gave 107
  warnings in 1,000 steps, and 1,000 when a steady `pde` is in the pipeline
  (`fields.py` enters `catch_warnings` every step, which resets the record).

Recommended enforcement, which follows the maintainer's proposal where the
measurements allow it:

1. An operation the rule's kind does not allow (`set_trait`, `found_families`,
   `divide`, a kill or a move in a narrower kind; a stack narrower than its
   operators, at build) warns once per operator with a `ContractWarning`
   (a `UserWarning` subclass; constant text naming the rule, the operation and
   the kind to declare) and is applied as the wider kind would apply it.
   `lgca.testing.check_interaction`, the test suite and CI turn the category
   into an error (`warnings.simplefilter("error", ContractWarning)`, or a
   `strict_contracts()` helper).
2. Direct writes to `cells.label`, `cells.index` and `cells.channel` raise
   (read-only arrays); a `Cells.relocate` operation can come when a model needs
   to move identity-based cells between nodes.
3. The expensive checks (PR2's fingerprints for class-based operators) are off
   by default and on under the switch, which CI enables.
4. Entry checks replace the identity-based commit-time recounts.

Rollback is a separate question from enforcement: it decides what state a model
is in after any exception in a step (a rule's own error, a solver failure,
Ctrl-C), and it matters only when the model is used afterwards. Measured on the
2026-09-30 tree (`followup-probes/round3/failed_steps_hashes/`): after a failed
step the lattice holds step n minus the failing operator and propagation, with
the RNG advanced and the step counter unchanged, and nothing marks it; calling
`step()` again applies the earlier operators a second time (76 cells against 60
in a clean run). The Explorer's Step and Play continue from that state out of
sight (198 -> 232 -> 266 cells while the label stays at step 2); so do notebooks
that keep stepping a built model, `run()` after Ctrl-C, and retry loops that
change the solver and step again. `run_model`, `sweep` and the CLI discard the
model and are not affected. Rollback (PR1) costs 0.01 to 0.65 % of a step and
one copy of `nodes`.

**PR0 [Fix] Contract checks (about 1 to 1.5 days).**

- `Cells._allow(operation, kinds)` runs first in `set_trait`, `found_families`
  and `divide` (and in kills and moves of narrower kinds): before anything is
  written or drawn, it warns as described under Enforcement, or raises under
  strict contracts. Read-only states raise `TypeError`.
- `Cells.label`, `index` and `channel` become read-only properties over
  non-writeable views; operations store new arrays through one private setter.
  This costs nothing per step (a commit-time recount measured 5 % of a step).
- `StackOperator.validate` compares the stack's kind with its operators.
- The Boltzmann paths write the lattice once.
- Build-time `prepare` of terms restores the RNG state.
- Until PR1: a model whose step raised refuses further steps until it is
  rebuilt (`CompiledModel.step`), so the Explorer and retry loops cannot continue
  from a partial step.

A patch of about 115 lines was prototyped and passed 1,588 targeted tests; the
seeded end states of seven pipelines were bit-identical.

Done on 2026-10-01, with the enforcement of decision 3: `LatticeState._allow`
(called by `Cells` too) warns with `ContractWarning` (exported as
`lgca.ContractWarning`) once per operator and widens the law the state keeps;
classical commits compare assigned counts with the law and warn the same way;
`lgca.testing.strict_contracts()`, `check_interaction` and the test suite
(`filterwarnings` in `pyproject.toml`) make it an error. The identity-based
commit-time recounts are gone (narrow kinds 1.1 to 1.7x faster, now as fast as
`birth_death`). Tests in `tests/contracts_test.py`: every operation x narrower
kind x family is stopped before any change under strict contracts, and warns
once and ends bit-identical to the rule declared `birth_death` otherwise.
Reactions get their read-only state with CR-A2 PR1, which rewrites the code
around them.

**PR1 [Feat] A step is all or nothing (about 3 days).** A private
`lgca/transaction.py`:

```python
class StepTransaction:
    """Everything a step may change, restored in place if the step raises (KeyboardInterrupt included)."""
    def begin(self): ...     # nodes (or the NoVE cell table, in its order), dict(lgca.__dict__),
                             # rng.bit_generator.state, copies of the declared fields,
                             # (TraitArray, size) per trait, lengths of family_props
    def record(self, trait, index): ...   # undo entry, called by TraitArray.__setitem__
    def rollback(self): ...
    def commit(self): ...
```

- Entry points: `CompiledPipeline.execute_step` (covers `CompiledModel.step`,
  `get_lgca` models and the Explorer), `LegacyInteraction.__call__`, and the raw
  callable branch of `LGCA_base.timestep`. Transactions live in a
  `WeakKeyDictionary`, not on the lattice, so deep copies do not copy them.
- Cost measured at 0.01 to 0.65 % of a step, against 130 to 300 % for a deep
  copy (which stays as the test oracle). One extra copy of `nodes` in memory;
  `model.rollback = False` opts out and falls back to PR0's refusal.
- A rolled-back exception gets a note (`exc.add_note`) naming the step.
- Not rolled back, documented: writes through `TraitArray.values`, legacy list
  properties, operator-internal caches (PDE statistics, AMG hierarchies) and
  Python closures. Rules keep their state on the lattice.

Done on 2026-10-01: `lgca/transaction.py` as above, entered by
`CompiledModel.step` (which covers `run`, the Explorer and `get_lgca` models
with an interaction name), `LegacyInteraction.__call__` and the function branch
of `LGCA_base.timestep`. `CompiledModel.rollback` (default True) keeps the
transaction on the model; `rollback = False` falls back to PR0's refusal.
Transactions nest (each records where its part of a trait's journal starts).
Measured on 150 x 150 hex models of the four families (birth_death and
random_walk): no difference beyond the noise (+-3 %). Tests in
`tests/rollback_test.py`: five families x four injection points (a rule after
set_trait, divide with new families and a field rule; an interrupt;
propagation; boundaries) restore the snapshot, and retried steps equal clean
ones bit for bit; an interrupted Boltzmann reorientation; `get_lgca` with a
function.

**PR2 [Feat] Backstop for class-based operators (about 2 days).** Compare
cheap marks (trait writes, `maxlabel`, `maxfamily`) before and after each
operator, including those in stacks, and raise `ContractError(ValueError)`.
Count fingerprints cost 20 to 26 % of a step, so they run only for third-party
class-based operators and under `lgca.testing.strict_contracts()`, which CI
enables for the invariant and research-model tests.

**PR3 [Fix] Failed runs and the Explorer (1.5 to 2 days).** `SimulationRunner.run`
tracks the last completed step, truncates recordings to it
(`Observer.truncate`), always calls `finalize`, and records
`metadata["runtime"]["failed_step"]`. The Explorer keeps the error visible and
says that the step was not applied. A sweep in which one run fails discards
every result today (thread and process pools even finish all runs first): give
`sweep` an error policy, e.g. a column with the error of each failed run, and
cancel pending runs when it raises.

Done on 2026-10-01: `Observer.truncate(lgca, step)` (built-in recorders through
`RECORDED`, `FieldRecorder`, the cell histories of `NodeRecorder`);
`SimulationRunner.run` truncates to the last completed step, finalizes the
observers and adds a note where the run stopped; `metadata["runtime"]` gets
`failed_step` and the last completed `end_step`. The Explorer shows the notes of
an error (e.g. that the step was rolled back) and draws the applied steps of a
frame. `sweep(errors="raise" | "record")`: "raise" cancels the runs that have
not started (pool `shutdown(cancel_futures=True)`), "record" gives failed runs a
row with an `error` column and warns with their number; `biolgca sweep
--errors record` as well. Tests in `tests/failed_runs_test.py`.

**PR4 (optional) Checkpoints.** `CompiledModel.checkpoint()` and `restore()` on
the transaction's list of state, an identity-aware NPZ checkpoint, and
`Explorer._trial` as a rolled-back step on the live model instead of a deep copy.

**Acceptance.** For each family (classical VE, NoVE, multispecies VE, identity
VE, identity NoVE) and injection point (a later operator, a rule after
`set_trait` + `divide(new_family=True)`, a field operator followed by a failure,
propagation, boundaries, a `KeyboardInterrupt` in `place_labels`), the state
after the failed step equals a deep copy taken before it: nodes or the cell table
in order, traits (values, dtype, length), `maxlabel`, `family_props`, fields with
ghost nodes, RNG state, step. Three retried steps equal three clean steps bit for
bit. A contract matrix {field, reorientation, phenotype_switch} x {`set_trait`,
`divide`, `found_families`, divide-then-kill, `move` in a field rule} raises
before any change. The two over-wide stacks raise at build; every registered
stack builds. A failure at step 3 of 6 leaves three recorded rows and the CSV of
steps 0 to 2.

Effort: 7.5 to 8.5 developer days for PR0 to PR3.

## CR-A2: nonlinear field solver

### What was verified (CR-A2)

Both reproductions hold, with independent reference solutions (u = 1 and
u = 0.98332 from `brentq`). The Picard contraction factor has the closed form
`|(n - 1) - n h(u*)| a r / (1 + a r)`, which tends to `n - 1` for strong uptake:
Picard always converges for `n = 1`, becomes arbitrarily slow for `n = 2` and
cycles or diverges for `n >= 3`. The failures are therefore broad, not limited
to `n = 64`:

- implicit Hill uptake with `n = 2` from about 16 cells' worth of uptake per
  step, with `n >= 3` from 4;
- steady fields in the geometry of tutorial 7 at `D = 100` as soon as `n = 2`;
- a steady state that does not exist (production above the total saturable
  uptake, no decay, no fixed boundary) was reported as "raise max_iterations";
- steady fields started at zero with `n > 1` are reported as having "no unique
  solution", because the first linearization is singular.

On the pre-review code these cases were published after one warning, with errors
of up to about 280x and values of 1.3e4 where no steady state exists. **Runs made
before this branch with Hill `n >= 2` and non-trivial uptake should be treated as
suspect.** Shipped tutorials, zoo models and examples use `n = 1` or linear terms
and are not affected.

### Fixed in the follow-up (CR-A2)

- F11's guard no longer rejects converged solves whose residual is dominated by
  the linear solver's rounding, and no longer aborts slowly converging but
  accurate Michaelis-Menten solves: within 1000x of the tolerance it warns and
  uses the iterate; beyond that it raises, and says whether more iterations can
  help or the iteration cycles (then `solver="explicit"` works).
- BDF and Radau get the true Jacobian of saturating uptake and reactions
  (up to 1000x fewer right-hand-side evaluations on stiff uptake).
- `_hill_rate` evaluates the direct formula and uses the log form only where a
  power over- or underflows (F10 had made every call about 7x slower).
- Multigrid hierarchies are built with a fixed seed: pyamg drew from NumPy's
  global generator, so runs with a steady field did not repeat exactly.

### Recommendation (CR-A2)

**Newton's method for saturating uptake**, with three restrictions that came out
of comparing the prototypes:

1. The Jacobian `base + s diag(loss + w h'(c))` is still an M-matrix for Hill
   uptake (`h' >= 0`), and symmetric positive definite without advection, so the
   existing CG, AMG and AIR backends and the AMG reuse stay valid. `h'` in log
   form: `n r (1 - h)` with `1 - h = exp(-logaddexp(0, n log(c/K)))` (relative
   error 1.2e-13 over c/K from 1e-30 to 1e30).
2. Projected Armijo line search (`max(c + lambda d, 0)`, lambda halved down to
   2^-10; per node when there is no diffusion). The first iteration in iterate
   form, so that problems linear in c keep today's results bit for bit; later
   iterations in correction form with an Eisenstat-Walker inner tolerance floored
   at half the outer one (equal tolerances stall at 1 to 2x rtol).
3. For steady problems with no decay and no fixed boundary: an exact existence
   check (total production below the sum of the saturable uptake rates) and a
   uniform start that balances production and uptake. **Only when every
   field-dependent term is monotone**: with a logistic reaction the balance
   step returns the unstable root 0.

Reactions stay on Picard in the first PR. Newton picks the unstable equilibrium
of autocatalytic reactions where Picard finds the stable one, and the reaction
API has no derivatives. A second PR adds a floored finite-difference derivative;
`reaction(derivative=...)` only if profiling asks for it.

Rejected: Anderson acceleration with pseudo-transient continuation. It needs six
internal modes with constants tuned on one reaction, and it failed six cases that
the restricted Newton method solves (steep implicit uptake, near-capacity steady
states, repro 2 written as a reaction), while Newton solved all 91 of its cases
with fewer iterations.

On tutorial 7's workload Newton was faster than Picard (2.2 s against 5.6 s for
200 steps, a median of 1 iteration against 8) and solved `n = 2`, where Picard
fails.

### Cost with time-scale separation (measured 2026-10-01)

The maintainer asked whether Newton is too expensive when the field is much
faster than the cells (`solver="steady"`, or `"implicit"` with fast diffusion),
where it is solved in every step. A validated Newton prototype
(`followup-probes/round3/newton/newton_patch.py`, 91 reference cases) was
benchmarked on tutorial 7's model at 100² to 400² (D = 2e4 and 100, Hill n = 1
and 2), on 3D balls of cells at 30³ to 70³, and on a 2D implicit field that
relaxes within about 5 steps; an independent run on a quiet machine confirmed
the ratios (`round3/verify/`).

- **Per linear solve, Newton costs what Picard costs**: 0.98 to 1.04x on the
  direct backend, where both factor a new matrix in every iteration (the
  cells' uptake changes its diagonal), 1.14 to 1.75x on AMG and 1.65 to 1.73x
  on CG (Picard's later solves start almost converged).
- **Per LGCA step, Newton was cheaper in every case**: 2.1 to 4.3x on AMG and
  CG (2D, 3D, steady and implicit), 3.7 to 5.2x on direct at matched accuracy.
  From the previous step's field it needs 1 to 3 solves; Picard needs 8 to 20,
  or fails (the implicit field with n = 2 and strong uptake at step 1).
- **Against linear uptake**, one solve per step and the cheapest field with
  cells, Newton costs 1.1 to 2.2x.
- **Time-scale separation does not make it expensive**: the number of solves is
  set by how far the cells move the equilibrium in one step, not by D. The
  explicit solver is what does not scale: 10³ to 10⁴ times slower than steady
  Newton at D = 2e4.
- **The field dominates the step** with any method: 82 to 92 % with today's
  Picard on AMG, 54 to 77 % with Newton on AMG.
- **Levers**: the warm start (in place; losing it costs 3.8 to 5.9x) and AMG
  reuse (in place; 2.9 to 4.5x). On the direct backend, keeping one
  factorization across steps and iterating with it (chord Newton) costs 22.5 ms
  per step at 200² and D = 2e4, against 121 ms for Newton and 107 ms for linear
  uptake, which refactors today; on AMG it does not pay. SuperLU with
  `permc_spec="MMD_AT_PLUS_A"` and symmetric mode factors 1.45x (2D) to 3.4x
  (30³) faster. Updating the field every k steps saves at most the field's
  share, at a lag error 10² to 10⁴ times the solver error; not recommended.
- **The stopping test needs care.** A residual test relative to |right| loosens
  as D grows, because a fixed boundary's inflow dominates |right|: at D = 2e4
  Newton stopped at an error of 1.1e-4, against 1.5e-6 for Picard on direct.
  Today's default (Picard on AMG or CG) delivers only 1e-5 to 1.6e-4 there,
  because the inexact inner solves pass its change test early, so Newton loses
  nothing against it; but PR1 should use a test whose scale does not grow with
  the boundary inflow. Adding Picard's change test costs up to 2.45x in mild
  cases.
- Seeded trajectories change within the tolerance (3,994 against 3,964 cells
  after 100 steps of tutorial 7 at 200²).

Found on the way and fixed on 2026-10-01: without pyamg (Python 3.14), steady
fields with uptake on 3D lattices of up to 20,000 nodes factored their matrix
in every solve (1.8 s at 27³); they now use CG (about 15 ms).

**PR1 [Fix] Newton for saturating uptake (2 to 2.5 days).** An `_HillUptake`
term object with `monotone = True` and `derivative`; `_solve_system` dispatches
to `_newton` unless `solver_options["nonlinear"] == "picard"` or a term is not
monotone; `FieldSolverError(RuntimeError)` with a `kind` (`no_steady_state`,
`nonlinear`, `linear`, `singular`, `non_finite`), counted in the statistics. The
regression test `test_nonconvergent_nonlinear_fields_fail_without_publishing_an_inaccurate_solution`
must then assert u = 0.98332 (keeping a `"picard"` copy that raises).

Done on 2026-10-01 (decision 4), by a delegated agent on a branch, reviewed and
merged: `_HillUptake` (`monotone`, `derivative`, `maximum`), `_newton` with the
projected Armijo line search (per node without diffusion and advection), the
first iteration in iterate form and later ones in correction form with
Eisenstat-Walker tolerances. **Stopping test:** an error estimate, the
simplified Newton correction `J^-1 F(c)` with the last matrix (one
back-substitution with the kept factors on the direct backend, one solve to
relative tolerance 0.1 on AMG and CG), at most `rtol * max(c)` after a full
step, or `F` at rounding level; it does not loosen with the boundary inflow.
Measured at 200² and D = 2e4 (field ms per step / max error against a tight
reference): direct n = 1 271 / 3.8e-9 against Picard 1887 / 1.5e-6, AMG 53 /
1.4e-7 against 98 / 1.6e-4; n = 2 similar. It costs about 2x the prototype's
single iteration there, which stopped at 25 to 110 times the tolerance; a chord
finishing step on the direct backend (about 15 lines, measured 0.9 to 1.4x the
prototype at errors below 1e-6) is left to PR4. `FieldSolverError` has a sixth
kind, `explicit`, for the explicit solver's failures. The existence check runs
also with `"picard"` when every term is monotone. Linear problems and the
`"picard"` path are bit-identical to before (35 snapshot cases). Tests in
`tests/fields_newton_test.py` (94). Reactions and cell terms get read-only
states (CR-A1 PR0).

**PR2 [Feat] Reactions in Newton (about 1.5 days).** **PR3 [Docs] (0.5 day)**,
including `fields_spec.md`, which describes Picard, and benchmarks for Hill
`n = 2` and `4`. **PR4 (optional) [Perf] Direct backend (about 1 day)**: the
SuperLU ordering above, and a factorization kept across steps for matrices the
cells change (chord Newton for saturating uptake, preconditioned iteration for
linear uptake).

**Acceptance.** Repro 1 from 0, 1e-12, 0.5 and 5, periodic and no-flux: u = 1 to
1e-8. Repro 2: u = 0.98332043697 to 1e-6 in at most 10 iterations. Implicit and
steady sweeps over K, n in {1, 2, 3, 4, 8, 64} and uptake against `brentq`.
Near capacity (P = 1.9 against 2) from 0: u = 19^(1/n). Above capacity: kind
`no_steady_state`. Mass balance with no-flux boundaries and advection. Backend
agreement at rtol 1e-10. Logistic from 0.1, 1.5 and 3 gives 1; bistable
autocatalysis from 0.2 gives 0. First-order convergence in the number of
substeps.

**Risk.** Accepted fields move within rtol, so seeded runs with nonlinear fields
diverge from earlier ones (tutorial 7: 6,211 against 6,186 cells after 200 steps).
Record the solver option in the provenance (CR-A3) and keep `"picard"`.

Effort: about 5 developer days. CR-A2 touches only `fields.py`, its tests and
docs, and can proceed in parallel with CR-A1 and CR-A3.

## CR-A3: configuration ownership and provenance

### What was verified (CR-A3)

The review's reproduction holds (a caller edits the parameter dictionary after
the build: `model.spec` changes, the dynamics do not). On its own this is P2.
The P1 case is different: `_compile_operator` returns an `InteractionOperator`
object given in a spec as it is, and `validate` stores model-dependent state on
it (`_capacity`, a stack's operators). So:

- Building a second model from the same spec changes the first model's
  dynamics (a capacity-2 model reached 1,593 cells instead of 70 after a
  capacity-50 variant was built).
- A sweep with `backend="threads"` produced rows labelled with a capacity they
  did not run with; the processes backend refuses such specs and recommends
  threads.
- A rejected Explorer change still changed the running model.

Also found: operators copy their parameters one level deep, so nested values
(switch rates) stay shared with the caller; `model.spec` keeps the caller's
`nodes`, traits, initializer and operator list; a live Explorer change leaves
`metadata` stale (`operator_names`, schedule, propagation) and rewrites an
earlier result's `context.spec`; the CLI sweep copies its inputs after the runs
and records no hashes, dependency versions or parameter defaults.

### Fixed in the follow-up (CR-A3)

- Threads sweeps give each run its own copy of operator objects.
- Declared fields are always copied to the lattice (a padded-shape field was
  attached by reference).
- NumPy scalars in the spec serialize; single-cue operator objects keep their
  parameters in model files.
- The Explorer warns when it cannot try a change on a copy of the model.
- Short parameter names in `vary`, `sweep` and `explore` skip operator objects,
  which they cannot change, instead of failing.
- The CLI sweep reads every varied input before writing its archive and records
  where each archived file came from.

### Recommendation (CR-A3)

An **owned copy** of the configuration at build, **operator objects as
templates**, and an explicit **`reconfigure`** for live changes, with the
provenance of the "serialization as source of truth" design. Making every build
go through the serializer was rejected as the mechanism: every future spec
feature would have to serialize before it could run, `model.spec` would change
types (dataclasses to mappings, aliases renamed), and its rule for unregistered
objects depended on garbage collection (a lattice and its model refer to each
other).

**PR1 [Fix] Each model owns its configuration and operators (about 1 day).**

```python
def _owned_spec(spec: ModelSpec) -> ModelSpec:
    """A copy of ``spec`` owned by the model: later changes of the caller's dicts, lists and arrays reach
    neither the running model nor ``model.spec``. Observers record the run and stay the caller's objects."""
    memo = {} if spec.analysis is None else {id(spec.analysis): spec.analysis}
    return deepcopy(spec, memo)


def _template_copy(operator):
    """The model's own copy of an operator object given in a spec (the object is a template)."""
    memo = {id(value): value for value in (operator.__dict__.get("rule"), operator.info) if value is not None}
    return deepcopy(operator, memo)  # the rule and its PluginInfo are shared descriptions
```

`build_model` compiles from `_owned_spec(spec)` and returns it as `model.spec`;
`_compile_operator` returns `_template_copy(spec)` (with a clear error when an
object cannot be copied, and a `PDEOperator.__deepcopy__` that drops solver
caches, since SuperLU objects cannot be copied). The running operator is
`model.pipeline.operators[i]`. Four existing tests read the caller's object
after the build and must switch to it. Build overhead measured at 0.1 to 1 ms,
and about 0.8 microseconds per channel for identity-based states given as label
lists.

**Done on 2026-10-01** (decision 1: templates, "to be safe; cheap as it is done
only once per model"), implemented, reviewed through two lenses and corrected:

- `build_model` compiles from `_owned_spec(spec)`, one `deepcopy` of the whole
  spec, and `_compile_operator` returns `_template_copy(operator, memo)`. The
  operator objects of one list (the pipeline's, or a stack's) are copied with
  one memo, so operators that refer to each other still do; the review found
  that per-object copies cut such links (a driver operator drove a copy of the
  `pde` that ran, field total 0 against 82). Observers of `spec.analysis` are in
  the memo, so an operator may still report to them.
- Rules, reorientation terms and `PluginInfo` copy as themselves
  (`__deepcopy__` returns `self`) instead of the proposed memo, which would
  have shared only the top level; rules and terms also pickle by reference, so
  processes sweeps take operator objects of registered rules.
- A `pde` operator copied as a template leaves out the matrices its `setup`
  computes (a context variable set only during template copies); a plain
  `deepcopy` of a running model keeps them. `setup` starts fresh statistics.
- Anything that cannot be copied raises a `ValueError` naming the operator or
  the path (e.g. `model.state.parameters.lock`); `sweep` copies the model and
  every non-scalar grid value before the first run. `study._own_operators` is
  gone.
- The Explorer keeps one copy of the spec for itself and its model; a live
  change compiles a new pipeline on a new context and swaps it in only if it
  succeeds, so a rejected change leaves the running model as it was.
- Cost: build time within noise except for label lists (identity-based without
  volume exclusion), about 1.16x at 200² (the lattice also gets lists of its
  own). Memory: `model.spec` holds a second copy of initial nodes, traits and
  fields given as arrays (label lists: 67 instead of 38 MB at 300² after the
  first step). Documented in `build_model` and the CHANGELOG.
- Behaviour change, documented: a list or dict handed to an operator's
  constructor, and the object of a bound method, are copied with the operator;
  code that reads them after a run reads `model.pipeline.operators[i]`.
- `vary` still shares the parts it does not change with its input (documented;
  copying there would copy every sweep run twice).

**PR2 [Feat] `CompiledModel.reconfigure(changes)` (about 1 day).** Validates on a
copy of the lattice, then swaps in a new context, pipeline, spec and metadata
(never changing the shared ones), and appends `{from_step, changes: {path: {old,
new}}}` to `metadata["reconfigurations"]`; `initial_spec` keeps the built spec.
The Explorer calls it instead of assigning private members. Replay:
`build_model(initial_spec)`, n steps, `reconfigure(record)`, m steps equals the
live run.

Done on 2026-10-01: paths go through `resolve_path` and must lie in
`dynamics` (space, state and time raise, pointing at a new build); the new
values are copied; `_try_step` (moved from the Explorer) runs one step of the
new pipeline on a deep copy; then a new context with a copy of the metadata
(operator names, schedule, capacities refreshed by `_pipeline_metadata`, shared
with `build_model`) is swapped in with the pipeline and spec. `from_step` is the
first step with the new values. Tests in `tests/reconfigure_test.py` (replay in
three families, earlier results untouched, five rejected changes leave the
model and its random stream as they were).

**PR3 [Feat] Read once; provenance (1.5 to 2 days).** The CLI copies and
hashes its inputs before any run, and runs from the copies; `sweep` reads each
input once. `metadata.json`, `sweep.json` and `sweep()`'s table attributes get
a provenance block: Python, NumPy, SciPy and pandas versions, platform, package
version (and VCS commit in the CLI), plugins and a hash of each rule's source,
effective operator parameters with defaults, input hashes (per run when the
inputs vary) and the reconfigurations. Hash pins live only where the library
writes the file itself: the reference to a model file's array file
(`<stem>.arrays.npz`, which older versions ignore) and `model.resolved.json`
in a CLI archive; a mismatch warns and names both hashes. No pin that users
type (decision 5, below).

What hashes are for (measured 2026-10-01, `round3/failed_steps_hashes/`). A
hash does nothing unless something compares it. Reading each input once, not
hashing, is what prevents a file edited during a sweep from changing half its
runs (seeds 0 and 1 started from 300 cells, 2 and 3 from 100, and the archive
kept the edited file), and a rebuild from `model.spec` after the file changed.
Hashes serve two uses:

- **The library compares hashes it wrote.** This catches the most common case
  without any effort from the user: model files share array files
  (`model.json` and `model.yaml` share `model.arrays.npz`; a copy
  `model_v2.json` still points to it), so saving one model silently changes
  another (200 to 400 cells in the probe), and an array of the same shape
  swapped in is checked only for shape and dtype. It also catches an edited
  CLI archive (population 300 against 200 with identical `metadata.json`).
- **A person compares two `metadata.json`** when a re-run disagrees with a
  published result: rare but decisive, and only useful with the code identity
  (rule source, versions, commit), since changed rule code is probably the most
  frequent drift and is recorded nowhere today.

A `sha256` parameter of `from_npz` typed by users is friction, and today's
version rejects it ("unknown parameters"); `from_npz` is also a niche feature
(no identity-based states, in no example or tutorial). Side findings, open:
`lgca.explore` cannot run a `from_npz` model (no `resource_base`), and the
"Relative initializer resources require resource_base" message also appears for
an absolute path with `trusted_paths=True`.

Done on 2026-10-01 (decision 5): `lgca/provenance.py`; `build_model` records
`metadata["provenance"]` (versions, platform, operators with effective
parameters, source hashes of rules, terms and reactions, input hashes, which the
`from_npz` initializer returns for the bytes it read) and `reconfigure` refreshes
the operator part. `sweep` reads each `from_npz` file once before the runs and
gives every run its bytes (a context variable in the initializer, passed to
worker processes per job); `table.attrs["provenance"]`. The CLI stages its input
copies in a temporary directory before the first run, runs from them, and moves
them into the output only on success (a failed build or sweep leaves no output
directory); `metadata.json` and `sweep.json` get the provenance with the model
file's hash and the BioLGCA git commit (`git rev-parse` of the package, with a
changed flag). Pins: array references in model files carry `sha256` of the
array's values (not of the compressed file), checked on load with a warning;
`metadata.json` records `archive` hashes, which `biolgca run` and `validate`
compare when they load a model next to it. The acceptance's "an edited archive
fails" became a warning, as decision 5 says. Tests in `tests/provenance_test.py`.

**PR4 (optional)** read-only arrays and frozen containers in `model.spec`.

**Acceptance.** Caller edits after the build (parameters, nested rates, nodes,
traits, initializer, appending to the operator list) change neither the
dynamics nor `model_spec_to_dict(model.spec)`. A capacity-2 model steps
identically whether or not a capacity-50 variant is built from the same operator
object in between, in all four families. A threads sweep equals the serial one.
`reconfigure` refreshes the metadata, leaves earlier results alone and is atomic.
An input rewritten during a CLI sweep changes no row; an edited archive fails
with a message naming the hash.

Effort: 3.5 to 4 days for PR1 to PR3.

## Order of work

1. ~~**CR-A3 PR1**~~: done on 2026-10-01.
2. ~~**CR-A1 PR0**~~: done on 2026-10-01.
3. ~~**CR-A1 PR1 and PR3**~~: done on 2026-10-01.
4. ~~**CR-A3 PR2 and PR3**~~: done on 2026-10-01.
5. ~~**CR-A2 PR1**~~ (done on 2026-10-01); **PR2 and PR3** next: reactions in
   Newton, docs and benchmarks. PR4 (direct backend, chord step) when profiling
   asks for it.

## Decisions for the maintainer

Taken on 2026-10-01:

1. **Operator objects in a spec are templates**, copied for every model ("to be
   safe; cheap as it is done only once per model"). Implemented (CR-A3 PR1).
2. **A reorientation rule may not write traits**: reorientation changes channels
   within a species (multispecies) or with fixed traits (identity-based).

Also taken on 2026-10-01, as recommended after the measurements:

3. **Enforcement of the contracts and rollback** (see CR-A1, Enforcement). The
   maintainer proposed warnings with a switch for expensive checks. Decided:
   operation-level violations warn once per operator (`ContractWarning`) and are
   applied as the wider kind would, which the probes showed to be exact; strict
   mode, which CI and `check_interaction` use, makes them errors; direct writes
   to the cell arrays raise (a warning cannot make them correct); the entry
   checks need no switch (no measurable cost), the fingerprints of class-based
   operators run only under it. Separately, rollback by default (0.01 to 0.65 %
   of a step), with `model.rollback = False` to save memory, and until then
   refusing further steps after a failed one.
4. **Newton by default** for saturating uptake, with `"picard"` to restore the
   old iteration: the measurements show no cost reason against it, also with
   time-scale separation (see CR-A2, Cost). Seeded runs with nonlinear fields
   change within the tolerance.
5. **Input hashes**: read inputs once; hashes in `metadata.json`, `sweep.json`
   and the sweep table, with the code identity; pins only in references the
   library writes (array files of model files, CLI archives), warning on a
   mismatch; no `sha256` parameter for users (see CR-A3, PR3).
