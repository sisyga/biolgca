# Architecture notes

Internal notes for maintainers, moved out of the user guide
(`docs/source/how_to/model_specs_and_plugins.rst`). They describe which module
owns which numerical update (see phase 2 of `usability_roadmap.md`).

## Rules on the lattice state

`lgca.lattice_state.LatticeState` is the layer between rules and model
classes. It copies the interior channel states of a classical model into an
integer array with a species axis (`dims + (n_species, K)`), offers the
per-cell operations (`remove_cells`, `divide_cells`, `add_cells`,
`switch_phenotype`, `shuffle_cells`), neighbour sums and gradients that pad
by the boundary conditions, and writes the result back with `commit()`, which
checks the conservation law of the kind. Each operation is implemented once
with volume exclusion (one cell per channel and species) and once without.

`lgca.rules` turns functions of a `LatticeState` into registered operators
(`@interaction`, via `FunctionInteractionOperator`) and into reorientation
terms (`@reorientation_term`, a field plus a coupling, evaluated by
`pipeline._FieldTerm`). `lgca.builtin_rules` defines the built-in rules and all
built-in reorientation terms this way. `lgca.testing.check_interaction` runs
a rule on every declared geometry and family.

## Cells of identity-based models

`lgca.cells.Cells` is the table of living cells (label, flat node index,
channel) that `LatticeState.cells` builds for identity-based models; its
operations (`kill`, `divide`, `move`, `pick`, `set_trait`) are array
operations, and competition for free channels with volume exclusion is
resolved by ranking the cells of each node in random order (one `lexsort` of
all cells). `LatticeState` recounts the cells per channel after every change
and writes the labels back on `commit()`. Traits are `lgca.cells.TraitArray`
buffers in `lgca.props`, one row per label, converted from lists on first use;
they keep list-like `append` and `extend` for code that treats them as lists.
`Cells.divide` appends a complete row for every trait (and a family,
optionally a new one), so growth rules combine freely.

Without volume exclusion the model itself can hold the table
(`NoVE_IBLGCA_base._cell_table()`, labels and padded slots `node * K +
channel`, ghost nodes included for cells in flight beyond reflecting walls).
The `nodes` property builds the object array of label lists only when it is
read and then makes the lists the state again; writing `nodes` does the
same. `update_dynamic_fields`, `apply_boundaries` and `propagation` act on the
table when it is the state. The lookup tables (`lgca.cells.slot_maps`) come
from passing slot IDs through the classical NoVE model's own boundary and
propagation code, so a new geometry needs no table code of its own.

## The get_lgca front door

`lgca.legacy_names` holds one stack per interaction name of earlier versions
and family (classical, NoVE, identity-based with and without volume
exclusion, several species), with the legacy parameters and defaults. The
stacks are registered under the prefixed model-file names
(`classical.alignment`, ...), marked deprecated (`PluginInfo.deprecated`):
`create_plugin` warns, `list_plugins` hides them. `LGCA_base.set_interaction`
looks the name up for the model's family, derives a `ModelSpec` from the model
object (geometry, dims, boundary, channels, family, the capacity) and compiles
the stack into the pipeline that `timestep()` runs, as for a model built from a
spec. `lgca.interaction` applies the stack once without propagation, and
`lgca.interaction_params` holds the parameters with their defaults. A function
passed as the interaction runs on the model object instead, without a pipeline.

Capacity comes from `StateSpec.capacity`; a `capacity` parameter of a legacy
name must equal it (`validate_plugin_parameters` rejects a conflict). For
`get_lgca`, the capacity keyword or the legacy default of the interaction
(8, or 512 for `steric_evolution`) becomes the capacity of the derived spec.

Ordinary and multispecies NoVE initialization draw the excess rest contribution
as one Poisson variable with the summed mean. Initialization uses one represented
channel array plus one spatial (and, if present, species) array; memory does not
grow with carrying capacity. Poisson additivity preserves the distribution, but
for capacities above the channel count the random draw order and therefore exact
trajectories for historical seeds change. Repeated runs of the new implementation
with the same seed remain reproducible.

The legacy interaction functions live in `tests/legacy`, with the
`set_interaction` code that set them up, only as a reference for comparison
tests (`tests/legacy_names_test.py` compares every name with its translation).

`operator_base` owns lifecycle contracts and metadata types;
`operator_registry` owns name/alias resolution; `plugins` re-exports these
public types and owns registration/parameter contracts.

## Reorientation batching and memory

Composed Boltzmann reorientation turns every term into one weight per channel
(`_FieldTerm.weights`; all couplings are linear in the channel occupation) and
adds them up once per step. With volume exclusion it batches sites by
occupancy, converts the candidate states of each occupancy once, and scores a
batch with one matrix product; the candidate matrix and the score batches each
have a 32 MiB budget, which are temporary array limits, not a total
process-memory limit. The sampler retains one categorical uniform draw per
nonempty site/species in spatial order, including fully occupied sites.
Without volume exclusion it draws one multinomial per node and species from the
softmax of the weights, which reproduced the legacy `nove.random_walk` and
`nove.dd_alignment` bit for bit (on hex only in distribution, since the
neighbour sums round differently). Identity-based models reuse both samplers
for their cell numbers and then place the node's labels on the occupied
channels in random order (one extra uniform draw per channel or cell).
Scalar-reference regressions preserve seeded trajectories for the tested mixed
terms and species.
Floating-point matrix evaluation can differ in its final bits across numerical
libraries, so cross-platform bitwise trajectories are not promised. Boltzmann
transition weights are unchanged. See `benchmarks/composed_reorientation.py`
and the reopened milestone validation report for repeated multi-step timings.
