Research models from generic rules
==================================

Earlier versions of BioLGCA had one interaction function per published
model, such as ``nove_ib.go_or_grow_glioblastoma``. Each of them combined a
few steps: a phenotype switch, growth with mutations, movement. These models
are now written as *stacks* of the generic rules, in the module
:mod:`lgca.research_models`, under the names of the legacy interactions
without the family prefix. They work in identity-based models with and
without volume exclusion, run 4 to 18 times faster than the legacy code
(`Speed`_), and show how to build a published model from the rules.
``get_lgca(interaction="go_or_grow_glioblastoma", ...)`` runs them as well, as
do model files with the prefixed names (see :ref:`interaction_chapter`).

Using a research model
----------------------

A research model is one operator with its own parameters. Parameters that
name a cell trait, such as ``r_b`` and ``kappa`` below, give the initial
value of that trait unless ``StateSpec.traits`` sets it:

.. code-block:: python

   import numpy as np

   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(30, 30), boundary="reflecting"),
       state=StateSpec(density=1, restchannels=1, volume_exclusion=False, capacity=8,
                       identity_based=True),
       time=TimeSpec(steps=40, seed=1),
       dynamics=InteractionPipelineSpec(operators=[
           {"name": "go_or_grow_glioblastoma", "parameters": {"r_b": 0.3, "r_m": 0.01}},
       ]),
   )
   result = run_model(spec, showprogress=False)
   clones = len(np.unique(np.asarray(result.lgca.props["family"])))

The models and the rules they stack:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Stack
   * - ``go_or_grow_kappa``
     - ``go_or_rest`` sensing the neighbourhood density with a switch
       steepness ``kappa`` per cell, ``birth_death`` (death),
       ``go_or_grow.growth`` whose daughters inherit ``kappa`` with a
       normal change, ``random_walk`` over the velocity channels.
   * - ``go_or_grow_kappa_chemo``
     - as ``go_or_grow_kappa``, with moving cells following ``aggregation``
       instead of a random walk.
   * - ``go_or_grow_glioblastoma``
     - as ``go_or_grow_kappa``, with driver mutations that found clones with
       a higher birth rate and a new ``kappa``.
   * - ``evo_steric``
     - ``birth_death`` with driver mutations, then a reorientation with
       ``steric_repulsion`` and ``resting_bias``.
   * - ``birthdeath_cancerdfe``
     - ``birth_death`` with two kinds of mutation (drivers and passengers
       with exponential effects), then ``resting_bias``.
   * - ``go_and_grow_mutations``
     - ``birth_death`` (death), ``birth_death`` (division; mutated daughters
       found families, drivers raise the birth rate), ``random_walk``.
   * - ``birthdeath_discrete``
     - as ``go_and_grow_mutations``, with the birth rate mutating up or down
       by a fixed step.
   * - ``excitable_medium``
     - a rule of its own for classical models (Barkley kinetics of resting
       inhibitors and moving activators).

Writing a stack
---------------

A stack is a function of the lattice state that returns operator entries,
decorated with :func:`lgca.stack`. This is the glioblastoma model:

.. literalinclude:: ../../../lgca/research_models.py
   :pyobject: go_or_grow_glioblastoma

The function runs once, when the model is built; ``state`` gives access to
e.g. ``state.capacity``. Your own stacks work the same way:

.. code-block:: python

   from lgca import stack

   @stack(kind="birth_death", families=("ib", "nove_ib"), traits="r_b")
   def grow_and_walk(state, r_b=0.3, r_d=0.1, std=0.02):
       """Cells with their own, mutating birth rate grow logistically and walk at random."""
       return [
           {"name": "birth_death", "parameters": {
               "birth_rate": "r_b", "death_rate": r_d,
               "mutation": {"r_b": {"distribution": "normal", "scale": std, "bounds": [0, 1]}}}},
           {"name": "random_walk"},
       ]

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30)),
       state=StateSpec(density=0.2, restchannels=1, identity_based=True),
       time=TimeSpec(steps=30, seed=2),
       dynamics=InteractionPipelineSpec(operators=[grow_and_walk(r_d=0.05)]),
   )
   result = run_model(spec, showprogress=False)

Mutations
---------

``birth_death`` and ``go_or_grow.growth`` take a ``mutation`` parameter.
A mutation is an event: a daughter mutates with a probability, and then its
traits change by random effects. With ``new_family=True`` every daughter
that mutates founds a new family, so that ``lgca.muller_plot()`` shows the
clones; without ``mutation``, every daughter founds one.

.. code-block:: python

   mutation = {
       "probability": 0.01,  # per daughter; default 1
       "traits": {
           "r_b": {"value": 1.1, "operation": "multiply"},  # the same effect for every mutant
           "kappa": {"distribution": "normal", "scale": 0.2},  # a random effect, added
       },
   }
   entry = {"name": "birth_death", "parameters": {"birth_rate": "r_b", "death_rate": 0.02,
                                                  "mutation": mutation, "new_family": True}}

An effect is

- ``{"distribution": name, ...}``: a draw from any distribution of
  :class:`numpy.random.Generator` with its parameters, e.g. ``"normal"``
  (``scale``), ``"exponential"`` (``scale``), ``"gamma"``, ``"lognormal"``,
  or ``"choice"`` (``a``, ``p``) for discrete steps;
- ``{"value": v}``: the same effect for every mutating daughter;
- a function of your own, for any other distribution of effects (below).

``"operation"`` says how the effect changes the trait: ``"add"`` (default),
``"subtract"``, ``"multiply"`` or ``"set"`` (the effect becomes the trait). ``"bounds": [low, high]`` keeps the trait
within limits, either of which may be ``None``: by default values beyond a
bound are set to it (``"at_bounds": "clip"``); ``"redraw"`` draws the effect
again, which truncates the distribution. A number instead of an effect is
short for a normal change with that standard deviation, and a dict of traits
without ``"traits"`` for a mutation with probability 1, e.g.
``{"kappa": 0.2}``.

Several kinds of mutation, each with its own probability, are a list; they
apply one after the other, and a daughter can acquire several. Drivers and
passengers with exponential effects, as in ``birthdeath_cancerdfe``:

.. code-block:: python

   mutation = [
       {"probability": 0.1, "traits": {"r_b": {"distribution": "exponential", "scale": 0.001,
                                               "operation": "subtract"}}},
       {"probability": 1e-4, "traits": {"r_b": {"distribution": "exponential", "scale": 0.02,
                                                "bounds": [None, 1.0]}}},
   ]

A distribution of effects of your own is a function ``f(rng, size,
**parameters)`` that returns ``size`` effects, registered with
:func:`lgca.mutation_effect` so that model files can name it:

.. code-block:: python

   from lgca import mutation_effect

   @mutation_effect
   def mostly_small(rng, size, scale=0.01, large=0.1, p_large=0.05):
       """Small exponential effects, and now and then a large one."""
       return np.where(rng.random(size) < p_large, large, rng.exponential(scale, size))

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30)),
       state=StateSpec(density=0.2, restchannels=1, identity_based=True, traits={"r_b": 0.2}),
       time=TimeSpec(steps=20, seed=3),
       dynamics=InteractionPipelineSpec(operators=[
           {"name": "birth_death", "parameters": {
               "birth_rate": "r_b", "death_rate": 0.05,
               "mutation": {"probability": 0.2, "traits": {
                   "r_b": {"function": "mostly_small", "p_large": 0.1, "bounds": [0, 1]}}}}},
           {"name": "random_walk"},
       ]),
   )
   result = run_model(spec, showprogress=False)

Differences from the legacy code
--------------------------------

The stacks reproduce the legacy interactions in distribution
(``tests/research_models_test.py``). Where they differ, the difference is
small or the legacy code had a flaw:

- traits that the legacy code stored per family (``family_props["r_b"]``,
  ``["kappa"]``) are cell traits. They only change when a mutated daughter
  founds a family, so every cell of a family has its family's value;
- ``aggregation`` and ``steric_repulsion`` come after growth, so they see
  the density after the births and deaths of the step; the legacy code used
  the density at its start. Both conventions are valid (see
  :doc:`/concepts/interactions`, "Order"), and the difference is small;
- ``go_or_grow_kappa_chemo`` averages the density over the node and its
  neighbours, as ``go_or_grow_kappa`` does; the legacy code divided their sum
  by the number of neighbours, one node fewer;
- in ``birthdeath_discrete`` the birth rate stays below ``a_max``; the
  legacy code let one step up exceed it;
- ``excitable_medium`` keeps the activators within the velocity channels
  after every fast reaction, as the legacy two-species version did;
- in a pipeline the legacy operators saw empty ghost nodes where the
  boundary conditions say otherwise, e.g. no neighbours across a periodic
  edge; the rules see the neighbours.

Speed
-----

``benchmarks/research_models.py`` times every model against the legacy
interaction function (kept in ``tests/legacy``), 20 steps on a 100 x 100
lattice after 50 steps of warm-up, each measurement in a fresh process:

.. code-block:: text

   uv run python benchmarks/research_models.py

.. list-table:: Milliseconds per step on one machine (hexagonal lattice; the excitable media square)
   :header-rows: 1
   :widths: 40 20 20 20

   * - Model
     - legacy
     - rules
     - speed-up
   * - ``go_or_grow_kappa``
     - 168
     - 9.3
     - 18x
   * - ``go_or_grow_glioblastoma``
     - 155
     - 9.0
     - 17x
   * - ``evo_steric``
     - 101
     - 7.5
     - 13x
   * - ``birthdeath_discrete``
     - 171
     - 13
     - 13x
   * - ``birthdeath_cancerdfe``
     - 194
     - 18
     - 11x
   * - ``go_or_grow_kappa_chemo``
     - 60
     - 8.2
     - 7x
   * - ``go_and_grow_mutations``
     - 47
     - 13
     - 3.6x
   * - ``excitable_medium`` (one, two species)
     - 3.4, 4.9
     - 1.6, 2.8
     - 2.1x, 1.8x
   * - ``birth_death`` + ``random_walk`` (identity-based, with and without
       volume exclusion) against ``ib.birthdeath``, ``nove_ib.birthdeath``
     - 116, 164
     - 11, 11
     - 11x, 16x
   * - ``birth_death`` + ``random_walk`` (classical) against
       ``classical.birthdeath``
     - 1.6
     - 1.4
     - 1.1x
   * - ``birth_death`` + ``random_walk`` (two species, without volume
       exclusion) against ``multispecies.birthdeath``
     - 6.4
     - 4.5 to 7.9
     - 0.8x to 1.4x

The legacy growth rules also moved the cells, so they are compared with
``birth_death`` followed by ``random_walk``. The last row depends on the
state of the memory allocator: the rules work on arrays of cells per channel,
and when the C library returns their memory to the system after every step,
the next step pays for fresh pages (7.9 ms). With the allocator keeping its
memory (e.g. ``MALLOC_TRIM_THRESHOLD_=268435456``) it takes 4.5 ms.
