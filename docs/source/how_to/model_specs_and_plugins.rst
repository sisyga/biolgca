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
