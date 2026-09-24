Custom interactions
===================

A new interaction is a Python function of the lattice state. There are two
forms:

- :func:`lgca.interaction` for a step of its own: cells are born or die
  (``birth_death``), change species (``phenotype_switch``) or are rearranged
  over the channels of their node (``reorientation``);
- :func:`lgca.reorientation_term` for a directional cue that joins the other
  cues of a :class:`~lgca.pipeline.ReorientationSpec` in one random decision.

Both register the rule under a name, so a model refers to it like to a
built-in interaction. :func:`lgca.testing.check_interaction` checks a rule on
every lattice and model family it claims to support. The code on this page
runs as written; it needs these imports:

.. code-block:: python

   import numpy as np

   from lgca import interaction, reorientation_term
   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec, ReorientationSpec, ReorientationTermSpec
   from lgca.testing import check_interaction

A growth rule
-------------

.. code-block:: python

   @interaction(kind="birth_death", families=("classical", "nove"))
   def logistic_growth(state, r_b=0.2, r_d=0.05):
       """Cells divide with probability r_b * (1 - density / capacity) and die with probability r_d.

       Parameters
       ----------
       r_b : float
           Division probability of a cell on an empty node.
       r_d : float
           Death probability of a cell.
       """
       crowding = np.clip(state.density / state.capacity, 0, 1)
       state.remove_cells(r_d)
       state.divide_cells(r_b * (1 - crowding))


   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(30, 30)),
       state=StateSpec(density=0.5, restchannels=1),
       time=TimeSpec(steps=50, seed=1),
       dynamics=InteractionPipelineSpec(operators=[
           logistic_growth(r_b=0.3),
           {"name": "classical.random_walk"},
       ]),
   )
   result = run_model(spec, showprogress=False)
   print(logistic_growth)

The decorator reads the rule's parameters and defaults from its signature and
its description from the docstring, which ``print(logistic_growth)`` and
:func:`~lgca.plugins.describe_plugin` show. Calling the rule with parameters,
``logistic_growth(r_b=0.3)``, gives the entry for ``operators=[...]``; in a
model file the same entry reads ``{"name": "logistic_growth", "parameters":
{"r_b": 0.3}}``. Parameters without a default are required.

``families`` names the model families the rule is written for:
``"classical"`` with volume exclusion (at most one cell per channel and
species) and ``"nove"`` without. Models of other families are rejected when
they are built. ``geometries=`` restricts the lattices in the same way.
Identity-based models, whose cells carry labels and traits, are the families
``"ib"`` (with volume exclusion) and ``"nove_ib"`` (without); see
`Rules for individual cells`_.

What a rule sees and does
-------------------------

The rule receives a :class:`~lgca.lattice_state.LatticeState`: the interior
of the lattice, without ghost nodes, and always with a species axis. The same
rule therefore works for one and for several species.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Attribute or method
     - Meaning
   * - ``state.counts``
     - Cells per channel, shape ``dims + (n_species, K)``; read-only.
   * - ``state.density``, ``state.species_density``
     - Cells per node, and per node and species.
   * - ``state.flux``
     - Sum of the cell velocities per node, shape ``dims + (d,)``.
   * - ``state.neighbor_sum(a)``, ``state.gradient(a)``
     - Sum over the neighbours and centred gradient of ``a``; the boundary
       conditions supply the values beyond the edge.
   * - ``state.field(name)``
     - A field from ``StateSpec.fields``.
   * - ``state.capacity``
     - Cells per node at which a node counts as crowded. It is not enforced;
       the only hard limit is volume exclusion.
   * - ``state.rng``, ``state.step``
     - The model's random generator and the time step. Use only
       ``state.rng`` for random numbers, so runs are reproducible.
   * - ``state.remove_cells(p)``
     - Every cell dies with probability ``p``.
   * - ``state.divide_cells(p, channels=...)``
     - Every cell divides with probability ``p``; the daughter goes to a free
       channel of the given set.
   * - ``state.add_cells(n, channels=...)``
     - ``n`` new cells per node and species, in free channels.
   * - ``state.switch_phenotype(rates, channels=...)``
     - A cell of species ``a`` becomes species ``b`` with probability
       ``rates[a][b]``.
   * - ``state.shuffle_cells(channels, species=...)``
     - Cells move to random channels of their node.
   * - ``state.counts = new``
     - Replace the whole state; the new state is checked.

Probabilities and numbers of cells can be one value, one value per node
(``dims``), per species (``(n_species,)`` or ``dims + (n_species,)``) or, for
probabilities, per channel. Operations act on every cell separately: without
volume exclusion, a channel with five cells loses each of them independently,
not all five at once. ``channels`` is ``"all"``, ``"rest"``, ``"velocity"``,
channel indices, or one of these per species, e.g. ``{0: "velocity", 1:
"rest"}``.

After the rule, the state is checked against the rule's kind: a
reorientation must keep the number of cells of each species at every node, a
phenotype switch the number of cells.

A phenotype switch
------------------

In classical models a phenotype is a species. A phenotype switch moves cells
from one species to another and needs a model with several species:

.. code-block:: python

   @interaction(kind="phenotype_switch", families=("classical", "nove"), n_species=2)
   def crowding_switch(state, rate=0.3):
       """Cells of species 0 turn into species 1 on crowded nodes."""
       rates = np.zeros(state.dims + (2, 2))
       rates[..., 0, 1] = rate * np.clip(state.density / state.capacity, 0, 1)
       state.switch_phenotype(rates)

Each cell keeps its channel (``channels="same"``); a switch into a channel
that the new species already occupies fails. The two-species form of the
built-in go-or-grow model uses this to switch between migrating cells
(species 0, in velocity channels) and resting cells (species 1, in rest
channels); see :mod:`lgca.builtin_rules`. With one species, moving and
resting are channel types, and the switch is a reorientation between them
(``go_or_rest``).

A movement bias
---------------

A reorientation term returns a field computed from the state, and its
coupling says how the field scores a candidate arrangement ``s'`` of a
node's cells: ``"flux"`` (a vector per node; cells move along it),
``"nematic"`` (a tensor; cells move along its axis), ``"rest"`` (a number;
cells rest) or ``"channels"`` (a weight per channel). The sampler picks
``s'`` with probability proportional to ``exp(Σ beta · score)`` over all
terms of the ``ReorientationSpec``.

.. code-block:: python

   @reorientation_term(coupling="flux")
   def away_from_centre(state):
       """Cells move away from the centre of the lattice."""
       positions = np.stack(np.meshgrid(*map(np.arange, state.dims), indexing="ij"), axis=-1)
       return positions - (np.array(state.dims) - 1) / 2


   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(40, 40)),
       state=StateSpec(density=1, restchannels=1),
       time=TimeSpec(steps=30, seed=2),
       dynamics=InteractionPipelineSpec(operators=[
           ReorientationSpec(terms=[away_from_centre(beta=0.1),
                                    ReorientationTermSpec("polar_alignment", beta=1.0)]),
       ]),
   )

Calling the term with ``beta=`` (and its parameters) gives a
:class:`~lgca.pipeline.ReorientationTermSpec`; ``species=`` restricts it to
one species. The built-in terms, from chemotaxis to nematic alignment, are
written the same way in :mod:`lgca.builtin_rules`. Like the sampler, terms
work for classical models with volume exclusion.

A deterministic rearrangement
-----------------------------

A reorientation need not be random. In the HPP lattice gas, two cells that
meet head-on leave at right angles, which keeps their momentum:

.. code-block:: python

   @interaction(kind="reorientation", families="classical", geometries="square", conserves="momentum")
   def hpp_collision(state):
       """Two cells meeting head-on leave at right angles."""
       counts = state.counts.copy()
       velocity = counts[..., :4]  # east, north, west, south
       horizontal = np.all(velocity == [1, 0, 1, 0], axis=-1)
       vertical = np.all(velocity == [0, 1, 0, 1], axis=-1)
       velocity[horizontal] = [0, 1, 0, 1]
       velocity[vertical] = [1, 0, 1, 0]
       state.counts = counts

``conserves="momentum"`` declares that the sum of the cell velocities at
every node is kept; it is checked after every step, like the conservation
law of the kind.

Rules for individual cells
--------------------------

In identity-based models every cell has a label and traits, such as its own
birth rate or drug resistance. ``StateSpec(traits=...)`` gives the initial
cells their traits, one value for all or one per cell, and daughters inherit
the traits of their mother. ``state.cells`` holds one entry per cell, and its
operations act on all cells at once:

.. code-block:: python

   @interaction(kind="birth_death", families=("ib", "nove_ib"))
   def selection(state, r_d=0.2, r_b=0.2, std=0.05):
       """Cells die with probability r_d * (1 - resistance) and divide; daughters mutate."""
       cells = state.cells
       dying = state.rng.random(len(cells)) < r_d * (1 - cells["resistance"])
       cells.kill(dying)
       crowding = state.density[cells.node] / state.capacity
       dividing = state.rng.random(len(cells)) < r_b * (1 - crowding)
       daughters = cells.divide(dividing, channels="all")
       resistance = cells["resistance"][daughters] + state.rng.normal(0, std, len(daughters))
       cells.set_trait(daughters, "resistance", np.clip(resistance, 0, 1))

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(20, 20)),
       state=StateSpec(density=2, restchannels=1, volume_exclusion=False, capacity=8,
                       identity_based=True, traits={"resistance": 0.1}),
       time=TimeSpec(steps=50, seed=1),
       dynamics=InteractionPipelineSpec(operators=[selection(), {"name": "channel_random_walk"}]),
   )
   lgca = run_model(spec, showprogress=False).lgca

``cells.node`` indexes lattice arrays, so ``state.density[cells.node]`` is the
density at each cell's node, and ``cells["resistance"]`` gives a trait per
cell. The operations are ``kill``, ``divide`` (daughters inherit all traits;
``new_family=True`` starts a family per daughter for lineage plots such as
``lgca.muller_plot()``), ``set_trait``, ``move`` (to other channels of the
node) and ``pick`` (at most a given number of cells per node, e.g. as many as
there are free channels). With volume exclusion a cell only enters a free
channel; when more cells compete for the free channels of a node, the
successful ones are chosen at random. The operations on cell numbers work as
well: ``remove_cells``, ``divide_cells`` and ``shuffle_cells`` pick random
cells and keep their labels. ``add_cells`` and ``switch_phenotype`` do not
exist for identity-based models, and ``state.counts`` can be read but not
assigned, because a number of cells does not say which cell went where.

Built-in rules take the name of a trait wherever a parameter may differ
between cells: ``go_or_rest(kappa="kappa", theta=0.6)`` gives every cell its
own switch steepness and all cells the same threshold, and
``go_or_grow.growth(r_b=0.2, mutation={"kappa": 0.2})`` lets the daughters'
``kappa`` mutate. Reorientation terms take ``trait=`` to scale a cue per
cell, e.g. ``ReorientationTermSpec("polar_alignment", beta=1.0,
trait="alignment")``; see :doc:`/concepts/interactions`.

Testing a rule
--------------

.. code-block:: python

   report = check_interaction(logistic_growth, {"r_b": 0.0, "r_d": 0.2}, expected_growth=-0.2)
   print(report)
   check_interaction(crowding_switch)
   check_interaction(hpp_collision)
   check_interaction(selection, traits={"resistance": 0.1})

:func:`~lgca.testing.check_interaction` runs the rule for a few steps on
small seeded models of every declared geometry and family, with one and two
species and with periodic and reflecting boundaries. It checks that states
stay valid, that the conservation laws hold, that changes to ghost nodes do
not leak into the lattice and that the same seed gives the same result.
``expected_growth`` compares the measured change of the number of cells per
step with the expected one, and ``traits`` gives the cells of
identity-based models their initial traits. It raises with a readable report, so one line
makes a test::

   def test_logistic_growth():
       check_interaction(logistic_growth)

Sharing a rule
--------------

Keep rules in a module of your project, e.g. ``interactions.py``, and import
it before running or loading a model that uses them: model files refer to
rules by name and never execute code. Defining a rule again in the same
module, for example by re-running a notebook cell, replaces the earlier
definition.

Operator classes
----------------

Rules that need more control, such as caches prepared once before the first
step or their own validation, can subclass
:class:`~lgca.operator_base.InteractionOperator` and register a
:class:`~lgca.operator_base.PluginInfo` with
:func:`~lgca.plugins.register_plugin`. ``apply(context, step)`` then works on
the model directly (``context.lgca``); a
:class:`~lgca.lattice_state.LatticeState` built from ``context.lgca`` gives the
same operations, and :meth:`~lgca.lattice_state.LatticeState.commit` writes
the result back. The built-in interactions in :mod:`lgca.pipeline` are
written this way.
