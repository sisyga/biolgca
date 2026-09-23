# Reopened milestone 1 validation

Target: `aidevelop`; starting revision `5b9efe7` (the merged tree of #128).
This report covers the second review's 22 new findings and reopened #113/#118.
The original report remains historical evidence, not proof of these new contracts.

## Supported feature combinations (#150)

Assertions live in the regression tests below; this matrix reuses them rather
than duplicating tests. Seeds are fixed. Scientific expectations are independent
of legacy implementation parity unless a row explicitly tests RNG compatibility.

| Combination | Independent expectation | Regression tests |
|---|---|---|
| Explicit, generated and NPZ unsigned counts with random walk and both alignment samplers | Counts remain nonnegative integers, total mass is conserved; overflow is rejected before a draw | `tests/nove_sampler_test.py` |
| Channel reversal followed by NoVE alignment, with/without propagation | Hand-computed periodic edge state is `[0, 10]`, not the stale pre-reversal direction | `tests/nove_sampler_test.py` |
| Identity birth and property-consuming growth in both orders | Every living ID indexes every required property; inherited marker traits remain valid; unsupported growth lifecycles fail at compilation | `tests/identity_composition_test.py` |
| Compiled run, legacy time evolution and direct runner continuation with dynamic families | One live member per newly mutated family, family totals equal physical population, IDs remain unique; split/uninterrupted states and RNG match | `tests/compiled_continuation_test.py` |
| NoVE alignment measurement with periodic, reflecting and absorbing boundaries | State and RNG are unchanged, including an injected failure; uniform square director alignment is 1 periodic and 0.75 at walls | `tests/measurement_state_test.py` |
| Direct/facade animations, selected channels and sparse/nonuniform times | Displayed density comes from selected node channels and titles use their paired sample times, even when density recording has a different schedule | `tests/plotting_observer_test.py`, `tests/animation_contract_test.py` |
| Phenotype transition conservation metadata and numeric flux | One particle preserves mass but can reverse momentum; metadata advertises this correctly | `tests/scientific_transition_matrix_test.py` |

## Acceptance coverage

| Issue | Implemented contract and validation |
|---|---|
| #113, #148 | Shared history/time resolver used by direct and facade config, density, flux and flow animations across square/hex backends. Dense, sparse, nonuniform and explicit-time tests inspect each renderer's title artist. Cubic Mayavi adapters are tested with a fake rendering module; actual Mayavi rendering is not claimed. |
| #118 | Attached compiled context is used by direct runners and legacy continuation; dynamic family histories are checked against physical populations after each run. |
| #129 | Observational alignment computation never replaces simulation nodes; wall normalization and exceptional paths are covered. |
| #130 | Native NoVE sampling validates and converts totals safely from all supported count sources, including unsigned arrays. |
| #131 | Alignment declares boundary-node and density dependencies; sequential operator regression checks edge refresh. |
| #132 | Shared daughter inheritance extends missing traits after operator-owned mutations; supported VE growth pairs and legacy callers are tested, unsafe mixed growth is rejected. |
| #133 | Global polarization sums species before vector norm; opposing, aligned, empty and unequal populations use hand-derived values. |
| #134 | Chemotactic gradients use physical lattice coordinates, including staggered hex rows; physical x/y/z ramps check every site and channel score. |
| #135 | Named scalar/director fields are prepared from current values once per application; between-step and preceding-writer changes affect actual trajectories. |
| #136 | Direct and ModelSpec NoVE identity input rejects duplicate, negative, noninteger and unindexable IDs; zero and sparse IDs remain valid. |
| #137 | Selected-channel density uses node counts and node sample times, including rest channels and animation observers. |
| #138 | Operator capacity resolves explicit value, canonical state, then fallback; all eight affected registrations cover absent/matching/conflicting inputs and runtime metadata. |
| #139 | Plain JSON/YAML nested node arrays normalize before construction; handwritten specs and CLI cover VE, counts and sparse IDs, with field-specific malformed-input errors. |
| #140 | Output preflight reserves archive artifacts and all scheduled observer paths before build/write; sentinels, duplicate paths, ancestor conflicts and moved-archive replay are tested. |
| #141 | Odd-row periodic hex transport fails before mutation through direct, legacy and compiled paths; static plotting stays valid and even-row diagonal seams are reciprocal. |
| #142 | Extra NoVE Poisson draws have physical state shape, independent of capacity; tracked RNG calls prove bounded allocation at capacity one million. Poisson additivity preserves the distribution; seeded initialization above channel capacity intentionally changes. |
| #143 | Occupancy-group candidate caching and bounded batch scoring remove per-site Python sampling; timing and distribution evidence is below. |
| #144 | Phenotype transition metadata no longer claims momentum conservation; a seeded counterexample checks measured mass and flux. |
| #145 | Empty/out-of-horizon animation schedules fail during setup, before dynamics; initial-only and final-only schedules remain valid. |
| #146 | Each result snapshots cumulative start/end steps and declares local sample-time origin; two sparse serialized runs reconstruct cumulative sample times. |
| #147 | `identity_kernels.inherit_missing_properties` owns missing daughter traits after native/legacy owned mutation. Canonical normalization owns state aliases; `plugins.resolve_operator_capacity` owns explicit/canonical/default precedence. Both call paths are documented in the model-spec guide and covered above. |
| #149 | README composed polar alignment uses `polar_alignment`; the deprecated nematic alias is explained explicitly. README Python blocks execute and polar/nematic score regressions remain active. |
| #150 | The cross-feature matrix above names independent expectations and reuses individual fix regressions. |

## Composed sampler profile and repeated benchmark (#143)

`composed_reorientation.py` builds fresh models for three repeats, each with five
propagated steps and seed 143. Initial occupancy is one or two of four channels.
Dedicated and composed aggregation both use beta 2; combined adds nematic beta 1.
`composed-before.json` is the scalar implementation at `c591dc8`;
`composed-after.json` is the batched implementation. Raw repeats and profiles are
retained in those files. Times below are median seconds for **all five steps**.

| Size | Initial occupancy | Dedicated before/after | Composed before/after | Combined before/after |
|---|---:|---:|---:|---:|
| 64 x 64 | 1 | 0.00578 / 0.00641 | 0.53928 / 0.00668 | 0.66175 / 0.00753 |
| 64 x 64 | 2 | 0.00610 / 0.00612 | 0.67515 / 0.00708 | 0.80256 / 0.00810 |
| 128 x 128 | 1 | 0.01916 / 0.01955 | 2.16092 / 0.02082 | 2.63429 / 0.02416 |
| 128 x 128 | 2 | 0.02092 / 0.02126 | 2.69443 / 0.02335 | 3.19171 / 0.02804 |

These Windows measurements diagnose this implementation and machine, not a
portable speed guarantee. The 64 x 64 combined one-step profile went from
127,700 calls (0.217 s, 4,096 per-site sampler calls) to 796 calls (0.002 s,
one batched sampler call). Candidate features are reused per occupied count;
score arrays and cached feature arrays each have a separate 32 MiB guard.

`composed_batch_test.py` checks seeded scalar/batched trajectories and RNG states
for all built-in terms, square/hex and one/two species under ordinary and tiny
batch limits. It also checks the independent one-particle resting-bias weights
`(1, 1, 3)` on 10,000 sites against probabilities `(0.2, 0.2, 0.6)` within six
binomial standard errors, local mass, feature reuse and allocation guards.
Existing once-per-step preparation tests remain in `interaction_pipeline_test.py`.
Draw order is preserved; matrix arithmetic may differ in final floating-point
bits across numerical libraries, so universal bitwise compatibility is not claimed.

## Final validation

- Windows `biolgca` Conda environment: **1,379 passed, 1 skipped**, 53 warnings,
  using `python -m pytest -q --basetemp=.pytest-tmp-reopened-final`.
  The skip requires Windows symlink privileges. The first full run identified
  two old odd-row hex transport fixtures; they now use supported even-row grids
  without changing their species/count assertions.
- `python docs/build.py`: clean Sphinx build with warnings treated as errors
  succeeded, forcing all six maintained tutorial notebooks through fresh kernels.
- `git diff --check 5b9efe7`: clean.
- Remote Python 3.10-3.13, installed wheel/CLI and documentation CI results are
  recorded in PR #151 and final issue comments before milestone closure.
