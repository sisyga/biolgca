# Architecture notes

Internal notes for maintainers, moved out of the user guide
(`docs/source/how_to/model_specs_and_plugins.rst`). They describe which module
owns which numerical update while legacy interaction functions and native
pipeline operators coexist (see phase 2 of `usability_roadmap.md`).

## Numerical implementation ownership

`lgca.identity_kernels.inherit_missing_properties` owns completion of a
volume-exclusion daughter's property row. The native and legacy VE growth
operators first append their explicitly mutated traits, then call this helper
to inherit every other cell property from the parent, without additional RNG
draws. Thus composed growth and downstream property consumers see complete rows.
Family membership is inherited unless the growth rule explicitly creates a new
family. Family-level mutation rules remain owned by the corresponding operator.

Multiple identity growth operators are supported for `ib.birth`,
`ib.birthdeath`, `ib.birthdeath_discrete`, `ib.go_or_grow` and
`ib.go_and_grow_mutations`. Other identity growth combinations are rejected
during compilation because their daughter-property lifecycles are not shared;
use one growth operator on those backends.

Capacity precedence for native NoVE identity operators is owned by
`lgca.plugins.resolve_operator_capacity`: an explicit operator value wins,
otherwise `state.capacity` supplies the value, otherwise the operator's
documented default applies. Factories leave an omitted capacity absent;
`validate_plugin_parameters` rejects a genuinely conflicting explicit override
before the operator resolves its capacity. `model._normalize_and_validate_spec` owns
the deprecated state-parameter alias, and the constructor receives the normalized
state capacity. Runtime metadata records the resulting active operator capacity.

Ordinary and multispecies NoVE initialization draw the excess rest contribution
as one Poisson variable with the summed mean. Initialization uses one represented
channel array plus one spatial (and, if present, species) array; memory does not
grow with carrying capacity. Poisson additivity preserves the distribution, but
for capacities above the channel count the random draw order and therefore exact
trajectories for historical seeds change. Repeated runs of the new implementation
with the same seed remain reproducible.

`lgca.identity_kernels` holds the shared identity kernels: `apply_identity_birth`,
`apply_identity_birthdeath` and `apply_nove_identity_birth`. The legacy
functions in `ib_interactions` and `nove_ib_interactions` and the corresponding
native operators call them. The kernels own birth attempts, daughter
ID/property updates and final shuffling; adapters own setup and parameter
sourcing. Mutated daughter traits are drawn in one vectorized call to
`sample_truncated_normal` per time step.

Remaining duplication:

* Identity VE go-or-grow and mutation rules occur in
  `ib_interactions` and native classes in `pipeline`.
* NoVE/identity-NoVE alignment, birth/death and switching occur in their legacy
  interaction modules and native pipeline classes.
* Classical reorientation has legacy score code and dedicated pipeline operators;
  composed spatial terms now prepare fields once per application.
* `classical_operators` owns the previously extracted classical random walk.

`operator_base` owns lifecycle contracts and metadata types;
`operator_registry` owns name/alias resolution; `plugins` re-exports these
public types and owns registration/parameter contracts. For subsequent slices,
put numerical updates in the relevant focused kernel module, keep legacy and
native adapters small, and retain independent scientific invariants alongside
seeded parity tests. Do not move unrelated kernels as part of a correctness fix.

## Reorientation batching and memory

Dedicated vector and tensor reorientation samplers process candidate
scores in batches with a conservative 32 MiB temporary budget. This preserves
site order and RNG draws; it does not change the transition model.

Composed Boltzmann reorientation also batches sites by occupancy and caches
candidate flux, nematic scores and rest occupancy once per group. Score batches
and cached candidate features each have a 32 MiB budget; these are temporary
array limits, not a total process-memory limit. Candidate enumeration has its
own size guard. The sampler retains one categorical uniform draw per nonempty
site/species in spatial order, including fully occupied sites. Scalar-reference
regressions preserve seeded trajectories for the tested mixed terms and species.
Floating-point matrix evaluation can differ in its final bits across numerical
libraries, so cross-platform bitwise trajectories are not promised. Boltzmann
transition weights are unchanged. See `benchmarks/composed_reorientation.py`
and the reopened milestone validation report for repeated multi-step timings.
