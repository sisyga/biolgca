.. _interaction_chapter:

Built-in interactions
=====================

Interactions define the local update applied before propagation. Built-in
interactions are selected with the ``interaction`` keyword when constructing a
simulator with :func:`lgca.get_lgca`. The valid names depend on the selected
model family.

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

Placeholder. Interaction callables receive the LGCA instance and mutate
``lgca.nodes`` or related state. Detailed guidance for custom interaction
registration, parameter validation and dynamic-field updates is still planned.
