# Milestone 1 scientific and workflow validation

Target: `aidevelop`; starting revision `5bf6ea3`.

The existing deterministic/parity tests remain in place. The additions below
assert independent scientific expectations rather than agreement with a second
implementation. Fixed seeds are used throughout. Statistical checks use six
standard errors; their probability models and sample counts are in the tests.
Hypothesis is not added: exhaustive small state spaces and hand-derived fixtures
cover these boundaries without another dependency.

| Issues | Independent oracle | Test location |
|---|---|---|
| #103, #100 | All 64 three-species/two-channel states, six relabelings: forbidden transitions, isolated species, mass, dtype and capacity | `scientific_transition_matrix_test.py` |
| #104, #111, #117, #126 | Hand-computed neighbor scores across geometries, empty species equivalence, bounded spatial call counts, scoped trajectories, opposite-heading polar/nematic distinction | `interaction_pipeline_test.py` |
| #105 | Empty/partial/full-site conservation with and without propagation | `interaction_pipeline_test.py` |
| #106, #125 | Unique physical IDs across five geometries and daughter property indexing | `core_invariants_test.py` |
| #107, #120 | Poisson density independent of capacity; shared direct/ModelSpec/NPZ count validation | `initializers_test.py` |
| #108 | Symmetric shared-capacity competition under species permutations | `interaction_pipeline_test.py` |
| #109, #119, #127 | Counts independent of IDs; pre-evolution recorder collision/budget rejection; analytical allocation estimates | `simulation_test.py` |
| #110, #114, #115 | Outside sentinel preservation, subprocess measurement persistence, moved archive replay and original-output protection | `cli_test.py` |
| #112 | Exact clockwise permutation, unchanged rest channels and rotated flux | Executed tutorial 6 |
| #113 | Dense/sparse/nonuniform animation titles, 1D sample labels, property and family time arrays | `plotting_observer_test.py` |
| #116, #118, #121, #122, #123 | Hand-authored dimensions, split/uninterrupted RNG parity, live compiled stepping, metric identity, capacity provenance and invalid term contracts | `model_spec_test.py` |
| #124 | Authoritative public type identity and extension registry compatibility | `legacy_interaction_registry_test.py` |
| #127 | Bounded candidate batches preserve seeded trajectories; composed/recorded benchmark | `model_spec_test.py`, `profiling_test.py`, `profiling.py` |

The real filesystem-symlink escape regression is conditional on Windows symlink
privileges. A separate controlled resolved-target regression always verifies
the same confinement check and unchanged outside sentinel.

Recording estimates cover fixed buffers; Python containers, list-valued states,
dynamic family growth, simulation state and plotting allocations are additional.
Benchmarks report diagnostic machine-specific timings, not performance gates.

## Final verification

- Windows, Conda `biolgca`, Python 3.13.5 / NumPy 2.4.6: full suite at
  `267250f` passed **973 tests**, with **1 symlink-privilege skip** and 30 warnings.
- The six exhaustive matrix cases added in `3999426` separately passed; no
  production code changed between these two test runs.
- Clean `python docs/build.py` passed with Sphinx warnings treated as errors;
  all six maintained notebooks executed in clean kernels. In this uninstalled
  worktree, `PYTHONPATH` was explicitly set to the repository root.
- Composed nematic/aggregation with density recording every two steps, 32x32,
  20 steps, seed 7, one diagnostic repeat: **0.864 s**, **758906 bytes** peak
  Python-tracked memory. This is not a bound on total process memory.
- `git diff --check` passed. GitHub PR #128 targets `aidevelop` and runs the
  existing Python 3.10-3.13, installed-wheel/CLI and documentation checks.
