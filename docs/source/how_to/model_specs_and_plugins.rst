Model specs and interaction plugins
===================================

The traditional :func:`lgca.get_lgca` API is still supported. New code can also
describe a complete simulation with :class:`lgca.model.ModelSpec`, which keeps
model setup, interaction dynamics, logging and plotting separate. The design is
inspired by Morpheus-style model declarations: dynamics are named plugins with
metadata, validation and an explicit execution order, while propagation remains
the deterministic lattice movement phase.

Pipeline phases
---------------

One timestep is compiled as:

1. particle-number changing operators, such as birth, death and division
2. phenotype/species switching operators
3. reorientation operators, including Boltzmann samplers over channel states
4. deterministic propagation

The default compiler enforces this order. Pass
``InteractionPipelineSpec(allow_custom_order=True)`` only for specialised
experiments where the order itself is part of the model.

Minimal ModelSpec
-----------------

.. code-block:: python

   from lgca.model import (
       AnalysisSpec,
       Description,
       ModelSpec,
       SpaceSpec,
       StateSpec,
       TimeSpec,
       run_model,
   )
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, NodeRecorder

   spec = ModelSpec(
       description=Description(title="native random-walk model"),
       space=SpaceSpec(geometry="hex", dims=(20, 20), boundary="reflecting"),
       state=StateSpec(density=0.2, restchannels=0),
       time=TimeSpec(steps=100, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.random_walk"}],
       ),
       analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   print(result.metadata["schedule"])
   result.lgca.plot_density()

The result exposes the final LGCA object, the compiled pipeline, metadata and
any observer outputs. For example, ``NodeRecorder`` writes ``lgca.nodes_t`` and
``DensityRecorder`` writes ``lgca.dens_t``.

Portable model files
--------------------

JSON is the canonical portable ModelSpec v1 representation. YAML is an
optional authoring syntax for the same data model; install it with
``pip install biolgca[yaml]``. Both formats contain only data:

* registered interactions use ``name`` and ``parameters``;
* composed reorientation uses the stable ``type: reorientation`` form;
* analysis contains only the documented built-in observer types;
* an optional ``state.initializer`` contains a registered initializer name and
  parameters.

Every portable file carries the top-level ``schema_version``. The current
value is ``1``. Unknown future versions are rejected instead of being guessed
at. Legacy versionless data is normalized to version 1 before validation.
Save and archive canonical JSON for reproducibility; YAML is a convenience
syntax and resolves to the same ModelSpec.

Model files never contain Python import paths and never execute arbitrary
modules. A third-party interaction may be referenced after a trusted Python
launcher imports and registers it, but standalone model loading resolves only
plugins shipped with BioLGCA. Custom Python observers remain Python-only and
raise an actionable portability error if a caller tries to save them.

.. code-block:: python

   from lgca.model import load_model_spec, run_model, save_model_spec

   save_model_spec(spec, "model.json")
   loaded = load_model_spec("model.json")
   result = run_model(loaded, showprogress=False)

The packaged ``model-spec-v1.schema.json`` is intended for editors and
development-time validation. Runtime loading uses BioLGCA's dependency-light
strict parser, so ``jsonschema`` is needed only by contributors and tooling.

Configuration, outputs and checkpoints
--------------------------------------

A ModelSpec is configuration, not a serialized simulator or trajectory. Keep
it small enough to review and version-control. Numeric initial channel states
can live in a companion NPZ file referenced by the ``from_npz`` initializer;
resource paths are resolved relative to the model file and cannot escape that
directory unless trusted-path handling is explicitly enabled.

The command-line runner writes ``model.resolved.json`` and ``metadata.json``
plus configured observer outputs into an explicit run directory. The resolved
model records the configuration that was actually compiled, while metadata
records runtime provenance and the resolved operator schedule. These files do
not constitute a restart checkpoint. In particular, identity-based particle
properties are not yet supported by ``from_npz`` and must not be reconstructed
from labels alone.

Composed dynamics
-----------------

Composed pipelines make the conservation laws visible. The following example
uses a multispecies volume-exclusion model with particle turnover, phenotype
switching and a reorientation sampler that combines nematic alignment and
chemotaxis.

.. code-block:: python

   import numpy as np

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import (
       InteractionPipelineSpec,
       PhenotypeSwitchSpec,
       ReorientationSpec,
       ReorientationTermSpec,
   )
   from lgca.simulation import DensityRecorder

   signal = np.linspace(0.0, 1.0, 30)[:, None] + np.zeros((30, 30))

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30), boundary="periodic"),
       state=StateSpec(
           density=0.25,
           restchannels=1,
           n_species=2,
           fields={"signal": signal},
       ),
       time=TimeSpec(steps=50, seed=3),
       dynamics=InteractionPipelineSpec(
           operators=[
               {
                   "name": "birth_death",
                   "parameters": {
                       "birth_rate": [0.01, 0.005],
                       "death_rate": [0.002, 0.002],
                   },
               },
               PhenotypeSwitchSpec(
                   name="phenotype_switch",
                   parameters={"rates": [[0.0, 0.02], [0.01, 0.0]]},
               ),
               ReorientationSpec(
                   terms=[
                       ReorientationTermSpec(name="nematic_alignment", beta=1.0),
                       ReorientationTermSpec(
                           name="chemotaxis",
                           beta=0.4,
                           parameters={"field": "signal"},
                       ),
                   ],
               ),
           ],
       ),
       analysis=AnalysisSpec(observers=[DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)

Identity-based family dynamics
------------------------------

Identity-based models can use native plugins for family and mutation dynamics.
For example, the glioblastoma go-or-grow interaction tracks family-level
proliferation rates and switching sensitivities:

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, NodeRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(40, 40), boundary="periodic"),
       state=StateSpec(
           density=0.6,
           restchannels=1,
           volume_exclusion=False,
           identity_based=True,
           parameters={"capacity": 8},
       ),
       time=TimeSpec(steps=100, seed=5),
       dynamics=InteractionPipelineSpec(
           operators=[
               {
                   "name": "nove_ib.go_or_grow_glioblastoma",
                   "parameters": {
                       "r_b": 0.2,
                       "r_d": 0.01,
                       "r_m": 0.001,
                       "fitness_increase": 1.1,
                       "kappa": 5.0,
                       "theta": 0.5,
                   },
               }
           ],
       ),
       analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   family_rates = result.lgca.family_props["r_b"]

Plugin registry
---------------

The registry provides the machine-readable interaction catalogue. All built-in
registered interactions are native operators and have unit-test coverage against
the previous interaction semantics.

.. code-block:: python

   from lgca.plugins import describe_plugin, interaction_coverage_table, list_plugins

   for plugin in list_plugins(kind="interaction"):
       print(plugin.name, plugin.operator_kind, plugin.port_status)

   info = describe_plugin("nove_ib.go_or_grow_kappa_chemo")
   print(info.parameters)

   rows = interaction_coverage_table()

Use plugin metadata when building UIs, validation reports, model provenance
tables or migration audits.

See :doc:`custom_interactions` for the supported extension contract, complete
registration example and conservation guidance. Portable model files resolve
registered names only; importing third-party Python remains the responsibility
of a trusted launcher.

CLI measurement files
---------------------

CLI runs persist numeric recorder outputs in ``measurements.npz`` next to
``metadata.json``. Array names match the Python attributes: ``nodes_t``,
``dens_t``, ``n_t``, ``channel_pop_t``, ``velcells_t``, ``restcells_t``,
``fam_pop_t``, and the four order-parameter arrays. Each includes its paired
``*_steps`` array (order parameters share ``order_parameter_steps``).
CSV snapshot and scalar observers continue to write their declared CSV files.
Identity-based NoVE node histories contain Python lists and are rejected before
CLI execution; use channel-density or density recording for portable numeric data.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   with np.load("run/measurements.npz", allow_pickle=False) as data:
       plt.plot(data["n_steps"], data["n_t"])
   plt.xlabel("Simulation step")
   plt.ylabel("Population")

Sparse 1D history images label actual times on sample rows; distances between rows
represent samples, not elapsed time. Property/family histories and the
``lgca.plotting.animate(..., steps=...)`` facade use paired simulation times.
Explicit history data defaults to dense times when ``steps`` is omitted.

The CLI archives a portable ``model.resolved.json`` with relative observer paths.
NPZ initializer inputs are copied to ``resources/initial_state.npz`` and the
archived declaration points there. Move the whole run directory together, then
validate or rerun its model into a new output directory without ``--trusted-paths``.

Objects returned by ``build_model`` and ``run_model`` retain their compiled
dynamics. Their ``lgca.timestep()``, ``lgca.timeevo(...)`` and live animations
advance that same pipeline, preserving RNG state and cumulative operator time.
``compiled.step()`` advances one step. Each recording run still starts its
observer schedule at local step zero and replaces that run's history arrays.

``state.capacity`` supplies the carrying capacity for NoVE models and the shared
birth limit for native VE ``birth_death``. An operator capacity that conflicts
with it is rejected. VE channel occupancy remains limited by ``K`` per species.
Metadata records ``channel_capacity`` separately from per-operator
``growth_capacities``; with one native birth/death operator, ``capacity`` reports
its active configured limit.

Numerical implementation ownership
----------------------------------

``lgca.identity_kernels.inherit_missing_properties`` owns completion of a
volume-exclusion daughter's property row. The native and legacy VE growth
operators first append their explicitly mutated traits, then call this helper
to inherit every other cell property from the parent, without additional RNG
draws. Thus composed growth and downstream property consumers see complete rows.
Family membership is inherited unless the growth rule explicitly creates a new
family. Family-level mutation rules remain owned by the corresponding operator.

Multiple identity growth operators are supported for ``ib.birth``,
``ib.birthdeath``, ``ib.birthdeath_discrete``, ``ib.go_or_grow`` and
``ib.go_and_grow_mutations``. Other identity growth combinations are rejected
during compilation because their daughter-property lifecycles are not shared;
use one growth operator on those backends.

Capacity precedence for native NoVE identity operators is owned by
``lgca.plugins.resolve_operator_capacity``: an explicit operator value wins,
otherwise ``state.capacity`` supplies the value, otherwise the operator's
documented default applies. Factories leave an omitted capacity absent;
``validate_plugin_parameters`` rejects a genuinely conflicting explicit override
before the operator resolves its capacity. ``model._normalize_and_validate_spec`` owns
the deprecated state-parameter alias, and the constructor receives the normalized
state capacity. Runtime metadata records the resulting active operator capacity.

Ordinary and multispecies NoVE initialization draw the excess rest contribution
as one Poisson variable with the summed mean. Initialization uses one represented
channel array plus one spatial (and, if present, species) array; memory does not
grow with carrying capacity. Poisson additivity preserves the distribution, but
for capacities above the channel count the random draw order and therefore exact
trajectories for historical seeds change. Repeated runs of the new implementation
with the same seed remain reproducible.

The first shared identity kernel is ``lgca.identity_kernels.apply_identity_birth``.
Both ``ib_interactions.birth`` and ``NativeIdentityBirthOperator`` call it. The
kernel owns birth attempts, daughter ID/property updates, and final shuffling;
adapters own setup and parameter sourcing. RNG draw order remains unchanged.
The existing ``ib_interactions.trunc_gauss`` helper remains the mutation sampler.

The remaining duplication inventory is intentionally incremental:

* Identity VE birth/death, go-or-grow, and mutation rules occur in
  ``ib_interactions`` and native classes in ``pipeline``.
* NoVE/identity-NoVE alignment, birth/death and switching occur in their legacy
  interaction modules and native pipeline classes.
* Classical reorientation has legacy score code and dedicated pipeline operators;
  composed spatial terms now prepare fields once per application.
* ``classical_operators`` owns the previously extracted classical random walk.

``operator_base`` owns lifecycle contracts and metadata types;
``operator_registry`` owns name/alias resolution; ``plugins`` re-exports these
public types and owns registration/parameter contracts. For subsequent slices,
put numerical updates in the relevant focused kernel module, keep legacy and
native adapters small, and retain independent scientific invariants alongside
seeded parity tests. Do not move unrelated kernels as part of a correctness fix.

Recording and temporary-memory budgets
--------------------------------------

``lgca.simulation.estimate_recording_bytes(lgca, timesteps, observers)`` reports
fixed numeric buffers and sample-time arrays before allocation. Runners reject
estimates above 512 MiB before observer setup. Reduce the horizon, use
``Schedule(every=...)`` or explicit sample steps, select a smaller density dtype,
or stream snapshots with ``CSVSnapshotObserver``. No samples are silently dropped.
For a deliberate larger allocation, use
``compiled.run(max_recording_bytes=your_budget)`` or the same keyword on
``SimulationRunner``; ``None`` disables this guard.

The estimate is a lower bound: Python sample-index maps, object/list payloads,
dynamically growing family histories, model state and renderer buffers cost extra.
Dedicated vector, tensor and wetting reorientation samplers process candidate
scores in batches with a conservative 32 MiB temporary budget. This preserves
site order and RNG draws; it does not change the transition model.
