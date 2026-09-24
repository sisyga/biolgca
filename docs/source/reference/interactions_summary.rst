.. _interaction_chapter:

Built-in interactions
=====================

Interactions are the local update of a time step, applied before
propagation. They are rules of the lattice state with a name, parameters and
a kind: ``reorientation`` rearranges the cells of a node over its channels,
``birth_death`` changes the number of cells, and ``phenotype_switch`` moves
cells between species (see :doc:`/concepts/interactions`). A
:class:`~lgca.model.ModelSpec` lists them by name, in the order they apply:

.. code-block:: python

   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(20, 20), boundary="reflecting"),
       state=StateSpec(density=0.2, restchannels=1),
       time=TimeSpec(steps=50, seed=1),
       dynamics=InteractionPipelineSpec(operators=[
           {"name": "birth_death", "parameters": {"birth_rate": 0.1, "death_rate": 0.02}},
           {"name": "polar_alignment", "parameters": {"beta": 2.0}},
       ]),
   )
   result = run_model(spec, showprogress=False)

The rules work in every model family (classical or identity-based, with or
without volume exclusion, one or several species) unless stated otherwise;
a rule that does not support a model says so when the model is built. With
several species, ``species`` chooses the species a rule acts on and
``sensed_species`` the species a cue senses (see
:doc:`/concepts/interactions`, "Species").

The rules
---------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Name
     - What cells do
   * - ``random_walk``
     - Move to random channels of their node, within a set of channels.
   * - ``polar_alignment``
     - Move in the direction of the cells at the neighbouring nodes.
   * - ``nematic_alignment``
     - Move along the axis of the cells at the neighbouring nodes.
   * - ``persistent_walk``
     - Keep the direction they had at the node.
   * - ``aggregation``
     - Move up the gradient of the cell density.
   * - ``chemotaxis``
     - Move up the gradient of a field (``StateSpec.fields``).
   * - ``directed_motion``
     - Move along a given vector field, e.g. a flow.
   * - ``contact_guidance``
     - Move along the axis of a director field, e.g. fibres.
   * - ``steric_repulsion``
     - Avoid crowded neighbouring nodes.
   * - ``resting_bias``
     - Prefer rest channels.
   * - ``go_or_rest``
     - Start resting on crowded nodes and moving on sparse ones.
   * - ``birth_death``
     - Die and divide; crowded nodes have less room for daughters.
   * - ``go_or_grow.growth``
     - Die, and divide only while resting.
   * - ``go_or_grow.switch``
     - The two-species form of ``go_or_rest``: migrating and resting cells
       are species.
   * - ``phenotype_switch``
     - Switch species at given rates, which may respond to cues such as the
       density or a field (several species).
   * - ``trait_switch``
     - Change their traits by events, as daughters do by mutations, at rates
       that may respond to cues (identity-based models).
   * - ``excitable_medium``
     - Barkley kinetics of resting inhibitors and moving activators
       (classical models).

The cues from ``polar_alignment`` to ``resting_bias`` are terms of the
Boltzmann reorientation; several combine into one decision in a
:class:`~lgca.pipeline.ReorientationSpec`. :mod:`lgca.builtin_rules` holds
their definitions, and :doc:`/how_to/research_models` the published models
built from them (``go_or_grow_kappa``, ``evo_steric``, ...).

Interaction names of get_lgca
-----------------------------

``get_lgca(interaction=name, **parameters)`` accepts the interaction names of
earlier versions of BioLGCA, with their parameters and defaults. Each runs a
stack of the rules above (:mod:`lgca.legacy_names`), so seeded runs give other
numbers than earlier versions, while the dynamics agree in distribution
(``tests/legacy_names_test.py``):

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(geometry="hex", interaction="alignment", beta=2.0, seed=1)
   lgca.timeevo(timesteps=50, showprogress=False)

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Runs
   * - ``random_walk`` (``diffusion``)
     - ``random_walk``
   * - ``alignment``, ``dd_alignment``
     - ``polar_alignment``
   * - ``di_alignment``
     - ``polar_alignment`` with ``normalize=True``
   * - ``persistent_motion``
     - ``persistent_walk``
   * - ``aggregation``, ``excitable_medium``
     - the rule of the same name
   * - ``nematic``, ``contact_guidance``
     - ``nematic_alignment``, ``contact_guidance`` (a ``director`` field;
       default: the first axis)
   * - ``chemotaxis``
     - ``chemotaxis`` up a concentration peaked in the middle of the
       lattice, or ``directed_motion`` along a given ``gradient``
   * - ``go_or_rest``
     - ``go_or_rest``, then ``random_walk`` over the velocity channels
   * - ``go_or_grow``
     - ``go_or_rest``, ``go_or_grow.growth``, ``random_walk`` over the
       velocity channels; identity-based models give every cell its own
       ``kappa`` and ``theta``, which daughters inherit with a normal change
       (``kappa_std``, ``theta_std``); with several species, every species has
       its own ``kappa`` and daughters change species (``kappa_std`` or
       ``mutation_matrix``)
   * - ``birth``, ``birthdeath``, ``go_and_grow``
     - ``birth_death``, then ``random_walk`` (``resting_bias`` with
       ``gamma``); identity-based models give every cell its own birth rate,
       which daughters inherit with a normal change (``std``, truncated to
       ``[0, a_max]``); with several species, daughters change species
       (``std`` or ``mutation_matrix``)
   * - ``birthdeath_discrete``, ``go_and_grow_mutations``,
       ``birthdeath_cancerdfe``, ``go_or_grow_kappa``,
       ``go_or_grow_glioblastoma``, ``steric_evolution``
     - the research model of the same name (``evo_steric`` for
       ``steric_evolution``)
   * - ``excitable_medium_ms``
     - ``excitable_medium`` with two species
   * - ``only_propagation``
     - nothing: cells only move along their channels

The defaults depend on the family, as before: without an ``interaction``,
models without volume exclusion align (``dd_alignment``), the others walk at
random. ``lgca.print_interactions()`` lists the names of a model, and
``lgca.interaction_params`` holds the parameters with their defaults.
Multi-species models accept every name whose rules work with several species.

Model files of earlier versions name these stacks with the family as a
prefix, e.g. ``"classical.alignment"`` or ``"nove_ib.go_or_grow"``. The
prefixed names still work but are deprecated: they warn and name the rules to
use instead.

Plugin introspection
--------------------

Every name a model file can use is a registered plugin with its kind,
families, parameters and description:

.. code-block:: python

   from lgca.plugins import describe_plugin, interaction_coverage_table, list_plugins

   for plugin in list_plugins(kind="interaction"):
       print(plugin.name, plugin.operator_kind, plugin.backend_families)

   print(describe_plugin("go_or_rest"))
   coverage = interaction_coverage_table()

``list_plugins(deprecated=True)`` includes the deprecated prefixed names.

Custom interactions
-------------------

A function can serve as the interaction of a ``get_lgca`` model:
``get_lgca(interaction=my_function, my_rate=0.1)``. It receives the LGCA
instance, reads its parameters from ``lgca.interaction_params`` and changes
``lgca.nodes`` before propagation. See :doc:`factory_reference` for an
example. Such a function runs as it is, without the checks of the rules, and
a model with it cannot be saved as a model file.

For reusable rules, write a function of the lattice state and decorate it
with :func:`lgca.interaction`, or with :func:`lgca.reorientation_term` for a
cue of the Boltzmann reorientation. The rule gets a name for model files,
works with and without volume exclusion and for any number of species, and is
checked against the conservation law of its kind;
:func:`lgca.testing.check_interaction` tests it on every lattice and family it
supports. :doc:`/how_to/custom_interactions` shows examples, and
:mod:`lgca.builtin_rules` contains the built-in rules written this way.
