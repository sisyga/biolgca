.. _interaction_chapter:

Built-in interactions
=====================

Interactions define the local update applied before propagation. Built-in
interactions are selected with the ``interaction`` keyword when constructing a
simulator with :func:`lgca.get_lgca`. The valid names depend on the selected
model family.

New model specifications use the same interaction library through canonical
plugin names. Plugins expose their phase, parameters, backend family and
conservation contract, so pipelines can separate particle-number changes,
phenotype/species switching, reorientation and deterministic propagation. All
built-in registered interactions are ported as native operators and are tested
against the previous interaction semantics.

Supported names by family
-------------------------

.. list-table::
   :header-rows: 1

   * - Model family
     - Interaction names
   * - Classical LGCA with volume exclusion
     - ``go_and_grow``, ``go_or_grow``, ``alignment``, ``aggregation``,
       ``random_walk``, ``persistent_motion``, ``birthdeath``,
       ``excitable_medium`` (2D/3D), ``nematic`` (2D/3D), ``chemotaxis``
       (2D/3D), ``contact_guidance`` (2D), ``only_propagation``
   * - Identity-based LGCA with volume exclusion
     - ``go_or_grow``, ``go_and_grow``, ``random_walk``, ``birth``,
       ``birthdeath``, ``birthdeath_discrete``, ``go_and_grow_mutations``
       (most geometries), ``only_propagation``
   * - Classical LGCA without volume exclusion
     - ``dd_alignment``, ``di_alignment``, ``go_or_grow``, ``go_or_rest``
   * - Identity-based LGCA without volume exclusion
     - ``go_or_grow``, ``go_or_grow_kappa``, ``go_or_grow_glioblastoma``,
       ``birth``, ``birthdeath``, ``birthdeath_cancerdfe``,
       ``random_walk``, ``randomwalk``, ``steric_evolution``
   * - Multi-species LGCA
     - Classical interaction names plus ``excitable_medium_ms`` for
       volume-exclusion multi-species models

Use ``lgca.print_interactions()`` on an initialized simulator to inspect the
exact list for that class.

Legacy names and plugin names
-----------------------------

The legacy factory API remains the shortest way to run an existing rule:

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(geometry="hex", interaction="alignment", seed=1)
   lgca.timeevo(timesteps=50, showprogress=False)

The :class:`lgca.model.ModelSpec` pipeline uses canonical plugin names. Most
plugins are namespaced by backend family, for example
``"classical.alignment"``, ``"ib.birthdeath_discrete"``,
``"multispecies.go_or_grow"`` or
``"nove_ib.go_or_grow_glioblastoma"``. Shared native operators such as
``"birth_death"`` and ``"phenotype_switch"`` are backend-aware and validate the
model context during setup.

.. code-block:: python

   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(20, 20), boundary="reflecting"),
       state=StateSpec(density=0.2, restchannels=0),
       time=TimeSpec(steps=50, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.alignment", "parameters": {"beta": 2.0}}],
       ),
   )

   result = run_model(spec, showprogress=False)

Plugin introspection
--------------------

Use the registry to discover available operators and to build validation,
provenance or migration reports:

.. code-block:: python

   from lgca.plugins import describe_plugin, interaction_coverage_table, list_plugins

   for plugin in list_plugins(kind="interaction"):
       print(plugin.name, plugin.operator_kind, plugin.backend_families)

   info = describe_plugin("classical.go_or_grow")
   print(info.parameters)
   print(info.conservation_law.describe())

   coverage = interaction_coverage_table()

The ``operator_kind`` field maps plugins to the pipeline phases documented in
:doc:`model_specs_and_plugins`. This is the preferred interface for tools that
need to reason about conservation laws or whether a rule changes particle
number, phenotype identity or channel occupancy.

Classical LGCA
--------------

.. automodule:: lgca.interactions
   :members:
   :noindex:

Identity-based LGCA
-------------------

.. automodule:: lgca.ib_interactions
   :members:
   :noindex:

LGCA without volume exclusion
-----------------------------

.. automodule:: lgca.nove_interactions
   :members:
   :noindex:

Identity-based LGCA without volume exclusion
--------------------------------------------

.. automodule:: lgca.nove_ib_interactions
   :members:
   :noindex:

Multi-species LGCA
------------------

.. automodule:: lgca.ms_interactions
   :members:
   :noindex:

Custom interactions
-------------------

Direct interaction callables remain supported on concrete LGCA objects. A
callable receives the LGCA instance, reads parameters from
``lgca.interaction_params`` and mutates ``lgca.nodes`` or related state before
propagation.

For reusable rules, prefer adding an :class:`lgca.plugins.InteractionOperator`
subclass and registering it with :func:`lgca.plugins.register_plugin`. Plugin
operators can validate the model context, advertise dependencies and outputs,
declare conservation laws, and participate in :class:`lgca.model.ModelSpec`
pipeline compilation. The existing native operators in :mod:`lgca.pipeline`
are the most complete templates for new birth/death, phenotype-switch and
reorientation rules.
