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

Order
-----

The pipeline applies the interactions in the order of
``InteractionPipelineSpec.operators``, followed by propagation. The order is
part of the model: division before reorientation is a different model from
reorientation before division, and the classical go-or-grow model is the
sequence ``go_or_rest`` (cells move between velocity and rest channels),
``go_or_grow.growth`` (death, division of resting cells) and
``channel_random_walk`` over the velocity channels. The run metadata
records the schedule as ``result.metadata["schedule"]``.

Combining directional cues
--------------------------

Directional cues such as alignment and chemotaxis should normally be expressed
as multiple :class:`lgca.pipeline.ReorientationTermSpec` objects inside one
:class:`lgca.pipeline.ReorientationSpec`. Their weighted scores are added, and
the sampler performs **one sampled reorientation transition** of the complete
channel state.

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
     - ``J_nb · J(s')`` with the flux ``J_nb`` of the neighbours.
     - flux
     - neighbouring cells
   * - ``nematic_alignment`` (``nematic``, ``alignment``)
     - Rewards sharing an axis with neighbouring cells.
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
     - ``Σ (d · c_i)²`` over occupied channels: along a director field.
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
free for its new species. The built-in ``phenotype_switch`` samples the
complete new state of each node at once, so switches cannot collide.
``LatticeState.switch_phenotype`` lets every cell try to switch on its own:
a switch into an occupied channel fails, and when several cells compete for
the same free channel, one of them wins at random. The population-dynamics
tutorial demonstrates both.

Polar versus nematic composition
--------------------------------

``polar_alignment`` scores candidate flux dotted with the sum of neighboring
fluxes. Opposite headings cancel. ``nematic_alignment`` instead sums squared
velocity dot products, weighted by numeric neighboring channel counts; opposite
headings reinforce the same axis. Both can be combined with the other cues.
The registered ``classical.alignment`` operator is polar.

For compatibility, existing composed ``alignment`` declarations retain nematic
semantics and emit a deprecation warning. Replace that alias with
``nematic_alignment`` to preserve an old model, or explicitly choose
``polar_alignment`` when directed collective motion is intended. Tutorial 3
uses the explicit nematic name; this differs from tutorial 2's polar mechanism.
