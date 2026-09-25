Interactions
============

One time step of an LGCA applies the interactions of the model, in the order
they are listed, and then moves the cells along their velocity channels
(propagation). This page explains the three kinds of interaction, how models
with one and several species are treated, and how directional cues combine.

Three kinds of interaction
--------------------------

Every interaction is one of three kinds, defined by what it keeps:

``birth_death``
   changes the number of cells: birth, death, division.
``phenotype_switch``
   changes what a cell is. In classical models a phenotype is a species, and a
   switch moves cells from one species to another; the number of cells at
   every node stays the same. A classical model with one species has nothing
   to switch to, so it rejects a phenotype switch. In identity-based models a
   switch changes the parameters of individual cells.
``reorientation``
   rearranges the cells of a node over its channels and keeps the number of
   cells of each species at the node. Boltzmann sampling of combined cues
   (below) is the usual way to write one, but any rule that keeps these
   numbers is a reorientation, including deterministic ones such as the HPP
   collision rule, and moving cells between velocity and rest channels.

Rules written with :func:`lgca.interaction` declare their kind and are checked
against it after every step; see :doc:`/how_to/custom_interactions`.

Species
-------

Classical models can have several species, e.g. two cell types. With volume exclusion, each channel holds at
most one cell of each species. Rules see the channel states as an array of
shape ``dims + (n_species, K)``, also when there is only one species, so the
same rule works for one and for several species. Arrays of single-species
models keep their shape ``dims + (K,)`` outside the rules, in ``lgca.nodes``,
recordings and model files.

The library does not restrict where a species may sit. A model in which
some cells must stay in rest channels, like the two-species form of
go-or-grow, uses interactions that keep them there.

Every part of the dynamics can be given to some species only. The parameter
``species`` (an index or a list) of ``birth_death``, ``go_or_rest``,
``random_walk``, the directional cues and ``ReorientationSpec(parameters=
{"species": ...})`` chooses the species whose cells take part; the other
cells are left as they are, but still count, e.g. for crowding.
``phenotype_switch`` has a rate for every pair of species. Cues computed from
the cells (alignment, aggregation, steric repulsion, persistence) sense all
cells by default; ``sensed_species`` chooses whose cells they read. Species 0
following a signal and species 1 aligning with its own kind:

.. code-block:: python

   operators = [
       {"name": "chemotaxis", "parameters": {"beta": 1.0, "field": "signal", "species": 0}},
       {"name": "polar_alignment", "parameters": {"beta": 2.0, "species": 1, "sensed_species": 1}},
   ]

A single cue is a ``ReorientationSpec`` with that one term, so the same
model is ``ReorientationSpec(terms=[ReorientationTermSpec("chemotaxis", beta=1.0,
species=0, parameters={"field": "signal"}), ReorientationTermSpec("polar_alignment",
beta=2.0, species=1, sensed_species=1)])``. The two forms agree when each cue
senses only the species it moves; a cue that senses other species sees them
after the operators before it (see `Order`_).

Order
-----

The pipeline applies the interactions in the order of
``InteractionPipelineSpec.operators``, followed by propagation. The order is
part of the model: division before reorientation is a different model from
reorientation before division, and the classical go-or-grow model is the
sequence ``go_or_rest`` (cells move between velocity and rest channels),
``go_or_grow.growth`` (death, division of resting cells) and
``random_walk`` over the velocity channels. The run metadata
records the schedule as ``result.metadata["schedule"]``.

Each operator sees the state that the previous one left. A cue computed
from the cells, such as ``aggregation`` or ``steric_repulsion``, therefore
sees the density after the births and deaths of the step when it is listed
after growth, and the density at the start of the step when it is listed
before. Both are valid models; the order says which one is meant. Some
legacy interactions computed such a cue at the start of the step but applied
it after growth, also to the daughters; with separate operators the cue
follows the order instead.

Growth
------

``birth_death`` is the growth rule of every model family. In a time step,
every cell dies with probability ``death_rate`` and, independently, tries to
divide with probability ``birth_rate``. Both are decided on the state at the
start of the step: a dying cell may still divide, and a daughter does not die
in the step it is born. The daughter needs room:

- with volume exclusion, it goes to a random channel of its species at the
  node and survives only if that channel was empty, which happens with
  probability ``1 - n_s / K`` for ``n_s`` cells of its species;
- a capacity, ``StateSpec.capacity``, scales the division probability by
  ``1 - n / capacity``, with ``n`` all cells at the node. Models without
  volume exclusion always have one. With volume exclusion it is optional, a
  soft limit in addition to the channels, e.g. to make species compete for
  space.

For one species the expected change of a node is
``birth_rate * n * (1 - n / capacity) - death_rate * n`` (``capacity = K``
with volume exclusion): logistic growth. For death before division, list two
operators, one with only ``death_rate`` and then one with only
``birth_rate``. ``crowding=False`` removes the crowding factors: cells divide
with ``birth_rate`` into free channels, and the capacity becomes a hard limit
on the cells per node.

In identity-based models the rates can differ between cells: the name of a
trait, e.g. ``"birth_rate": "r_b"``, gives every cell its own value.
Daughters inherit all traits of their mother. A mutation is an event: a
daughter mutates with some probability, and then its traits change by random
effects drawn from any distribution; with ``new_family=True`` every mutated
daughter founds a new family, for lineage plots. See
:doc:`/how_to/research_models` for the options and for published models
built this way. In classical models with several species, a
``mutation_matrix`` gives the species of the daughters.

Switching that responds to the surroundings
-------------------------------------------

The probabilities of switches (the rates of ``phenotype_switch``, the events
of ``trait_switch`` and of mutations) are numbers or responses to cues of the
cell's surroundings, in the form of the go-or-grow switch:

.. code-block:: python

   {"max": 0.2, "cues": [
       {"name": "density", "kappa": 5.0, "theta": 0.5},
       {"name": "field", "field": "signal", "kappa": -2.0, "theta": 1.0},
   ]}

is ``p = max * (1 + tanh(Σ_k kappa_k (c_k - theta_k))) / 2``, with ``c_k``
the value of cue ``k`` at the cell's node: half the maximal probability where
the weighted cues balance their thresholds, more where cues with positive
``kappa`` are large. The cues are the relative ``density`` (of the node or its
neighbourhood), the value and the ``gradient`` of a ``field``, and the
``flux`` of the cells, each for all species or the ``sensed_species``; in
identity-based models ``kappa`` and ``theta`` may name traits (every cell with
its own sensitivity) and ``{"name": "trait", "trait": "age"}`` is a cue per
cell. :func:`lgca.switch_cue` registers cues of your own, functions of the
lattice state with a value per node. With volume exclusion, a switch into
another species succeeds only if the channel it lands in is free, which
lowers the realised rate (see below). Migrating cells (species 0) that turn
resting (species 1) in crowded nodes and return at a constant rate:

.. code-block:: python

   {"name": "phenotype_switch", "parameters": {"rates": [
       [0, {"max": 0.3, "cues": [{"name": "density", "kappa": 6.0, "theta": 0.5}]}],
       [0.05, 0],
   ]}}

Switching traits
----------------

Cells of identity-based models can change their traits at any time, not only
when they divide: ``trait_switch`` applies events written as mutations to all
living cells. Every step each cell has an event with its probability, which
changes its traits by the effects; the operation ``"set"`` switches a trait
to a value. A cell that starts to align with probability 0.02 per step, for a
cue scaled by the trait (``ReorientationTermSpec(..., trait="alignment")``):

.. code-block:: python

   {"name": "trait_switch", "parameters": {"switch": {
       "probability": 0.02, "traits": {"alignment": {"value": 2.0, "operation": "set"}}}}}

A change of random size, e.g. ``{"alignment": {"distribution": "normal",
"scale": 0.1, "bounds": [0, None]}}``, makes the strength drift. Two states
that switch with their own rates, ``k_on`` from 0 to 2 and ``k_off`` back,
are two events, each for the cells in one state (``"when"``), and a rate may
respond to cues, e.g. cells that start to align in crowded nodes:

.. code-block:: python

   {"name": "trait_switch", "parameters": {"switch": [
       {"when": {"alignment": 0}, "traits": {"alignment": {"value": 2.0, "operation": "set"}},
        "probability": {"max": 0.1, "cues": [{"name": "density", "kappa": 6.0, "theta": 0.5}]}},
       {"when": {"alignment": 2}, "probability": 0.02,
        "traits": {"alignment": {"value": 0.0, "operation": "set"}}},
   ]}}

``"when"`` takes a value or a range ``[low, high]`` of a trait (either may be
``None``).

Combining directional cues
--------------------------

Directional cues such as alignment and chemotaxis should normally be expressed
as multiple :class:`lgca.pipeline.ReorientationTermSpec` objects inside one
:class:`lgca.pipeline.ReorientationSpec`. Their weighted scores are added, and
the sampler performs **one sampled reorientation transition** of the complete
channel state.

A single cue does not need the spec: every term in the table below is also
an operator of its own, e.g. ``{"name": "chemotaxis", "parameters": {"beta":
2.0, "field": "signal"}}``, which stands for a ``ReorientationSpec`` with this
one term and works in every model family. ``random_walk`` moves cells to
uniformly random channels (``channels`` and ``species`` restrict it).

This is different from a **sequential pipeline** containing two full
reorientation operators. Sequential operators perform two stochastic state
transitions; the second does not simply add a bias to the first. Use sequential
pipeline operators for different biological phases, not for combining energy
terms that should compete in one decision.

How the cues are sampled
------------------------

Each term gives every channel ``i`` of a node a score ``w_i``, the score of
one cell in that channel: ``g · c_i`` for a vector field ``g`` such as a
gradient, the field value for a rest channel with ``resting_bias``. A channel
state scores the sum over its cells, and the terms add up weighted by their
``beta``. How the cells use these scores depends on the model:

- **With volume exclusion** the cells of each species at a node choose a new
  channel state ``s'`` together, among the states with as many cells:
  ``P(s') ∝ exp(Σ_k beta_k G_k(s'))``. Cells exclude each other, so they
  cannot all take the best channel.
- **Without volume exclusion** every cell chooses its channel on its own,
  ``P(i) ∝ exp(Σ_k beta_k w_ki)``, and any number of cells can share a
  channel. For the random walk and ``polar_alignment`` this is the rule of
  ``nove.random_walk`` and ``nove.dd_alignment``.
- **Identity-based models** first update their cell numbers in the same way
  and then place the node's cells on the occupied channels at random. The
  cues do not depend on the properties of individual cells, so it does not
  matter which cell takes which channel.

For a single cell at a node, the first two rules agree.

A term with a ``trait`` gives every cell of an identity-based model its own
weight, e.g. ``ReorientationTermSpec("polar_alignment", beta=1.0,
trait="alignment")`` for an alignment strength per cell: cell ``a`` scores
``beta * alignment_a * w_i`` in channel ``i``. Without volume exclusion
every cell still chooses its channel on its own. With it, the cells of a
node are no longer interchangeable, and the labelled state follows
``P(σ) ∝ exp(Σ_a s_a w_σ(a))``. It is sampled with a Metropolis chain per
node (all nodes at once) that starts from a random arrangement and swaps the
contents of two channels; ``ReorientationSpec(parameters={"sweeps": 10})``
sets its length, ``sweeps * K`` proposals per node. Placing the cells one
after another would be faster but samples a different distribution.

Supported reorientation terms
-----------------------------

Use :func:`lgca.pipeline.list_reorientation_terms` to retrieve the names in
code. ``J(s')`` is the flux of a candidate state, the sum of the velocities of
its cells. Every term is a field of the lattice state together with a
coupling that turns it into a score; new terms are written the same way with
:func:`lgca.reorientation_term`, and the built-in ones are defined in
:mod:`lgca.builtin_rules`.

.. list-table::
   :header-rows: 1
   :widths: 24 36 20 20

   * - Term (aliases)
     - Score
     - Coupling
     - Input
   * - ``random_walk`` (``uniform``)
     - 0: all states are equally likely.
     - rest
     - none
   * - ``resting_bias``
     - Number of cells in rest channels.
     - rest
     - at least one rest channel
   * - ``persistent_walk`` (``persistent_motion``)
     - ``J(s) · J(s')``: cells keep their direction.
     - flux
     - the node's cells
   * - ``polar_alignment``
     - ``J_nb · J(s')`` with the flux ``J_nb`` of the neighbours
       (``include_center`` adds the node's own flux, ``normalize`` divides by
       the number of cells: density-independent alignment).
     - flux
     - neighbouring cells
   * - ``nematic_alignment`` (``nematic``, ``alignment``)
     - Rewards sharing an axis with neighbouring cells:
       ``Σ_k n_k [(c_k · c_i)² - |c_k|² |c_i|² / d]`` for a cell in channel
       ``i`` (see below).
     - channels
     - neighbouring cells
   * - ``aggregation``
     - ``∇ρ · J(s')``: up the gradient of the cell density.
     - flux
     - cell density
   * - ``chemotaxis``
     - ``∇f · J(s')``: up the gradient of a field.
     - flux
     - ``parameters={"field": name}``
   * - ``contact_guidance``
     - ``(n · c_i)² - |c_i|² / d`` for a cell in channel ``i``, with the unit
       director ``n``: along a director field.
     - channels
     - a vector field, default ``director``

All gradients are derivatives in lattice units, with a lattice spacing of one,
on every geometry. This includes the density gradient used by aggregation, so a
sensitivity ``beta`` has the same meaning on 1D, square, hexagonal, cubic and
Moore lattices.

Gradients are centred differences over the neighbouring nodes, taken with the
model's own ``gradient`` method; on the hexagonal lattice this accounts for
the row offsets, so a linear ramp in physical coordinates has its exact
gradient. At the edge of the lattice, the ghost nodes supply the values
beyond it. For the cell density these follow the boundary condition: wrapped
with periodic boundaries, zero beyond reflecting or absorbing walls. A named
field such as the chemotaxis signal keeps the ghost values stored with it,
which repeat its edge values unless you set them, so a linear ramp has half
its slope across the edge.

Named chemotaxis and contact-guidance fields are read from their current padded
``lgca.<field_name>`` arrays once before each reorientation operator. Update their
physical sites via ``field[lgca.nonborder]`` between steps or in a preceding custom
operator. Derived gradients and normalized directors refresh once per operator,
not per site. Contact guidance is nematic: negating a director preserves its axis;
rotating that axis changes the cue.

Periodic hexagonal evolution requires an even number of rows so opposite-channel
transport is reciprocal across the seam. Odd-row models can be constructed for
static plotting, but stepping (including direct propagation) rejects them before
state or random-number changes.

Phenotype switching with volume exclusion
-----------------------------------------

With volume exclusion, a cell that switches species needs a channel that is
free for its new species. ``phenotype_switch`` (and
``LatticeState.switch_phenotype``, on which it is built) lets every cell try
to switch on its own: a switching cell picks a random channel of its new
species at the node, in the set ``channels`` (all channels by default), and
the switch fails if that channel is occupied. A cell switching with
probability ``p`` into a species with ``n`` cells in ``C`` channels thus
switches with probability ``p (1 - n / C)``, as a cell that divides into a
random neighbouring site of a lattice model.

- Cells that switch into the same species at a node pick distinct channels,
  as daughters do, so they never collide. If they are more than the channels,
  those that pick one are chosen at random and the others fail.
- Occupancy is judged at the start of the step: a channel vacated by a cell
  that switches away is free only in the next step. Two cells of different
  species that would swap their species at a full node both stay.
- With ``channels="same"`` a cell keeps its channel and switches only if it
  is free for the new species; when cells of several species want the same
  channel, one of them, chosen at random, succeeds.

Cells whose switch fails keep their species. Without volume exclusion every
switch succeeds, and a switched cell goes to a random channel of the set.

Polar versus nematic composition
--------------------------------

``polar_alignment`` scores candidate flux dotted with the sum of neighboring
fluxes. Opposite headings cancel. ``nematic_alignment`` instead sums squared
velocity dot products, weighted by numeric neighboring channel counts; opposite
headings reinforce the same axis. Both can be combined with the other cues.
The ``alignment`` interaction of ``get_lgca`` is polar.

The axis cues compare traceless tensors ``c cᵀ - |c|² I / d`` (``d`` the
spatial dimension): a cell moving along the axis scores above a resting cell,
which scores 0, and a cell moving across it below. Averaged over all
directions, moving scores like resting, so the cue orients cells without
making them rest or move more. In 2D these are the tensors of the
``nematic`` and ``contact_guidance`` interactions of earlier versions; in 1D
there is only one axis and the cues have no effect.

For compatibility, existing composed ``alignment`` declarations retain nematic
semantics and emit a deprecation warning. Replace that alias with
``nematic_alignment`` to preserve an old model, or explicitly choose
``polar_alignment`` when directed collective motion is intended. Tutorial 3
uses the explicit nematic name; this differs from tutorial 2's polar mechanism.
