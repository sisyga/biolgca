Interaction composition
=======================

An LGCA time step contains conceptually different processes. In the standard
pipeline they run in the following order:

``birth/death -> phenotype switching -> reorientation -> propagation``

Birth and death may change total particle number. Phenotype switching changes
particle type without changing total particle number. Reorientation redistributes
particles among the channels of a node. Propagation then moves velocity-channel
particles deterministically to neighboring nodes.

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

Supported reorientation terms
-----------------------------

Use :func:`lgca.pipeline.list_reorientation_terms` to retrieve the names in
code. The aliases below intentionally remain available for readable model
specifications and backwards compatibility.

.. list-table::
   :header-rows: 1
   :widths: 22 31 25 22

   * - Term
     - Interpretation
     - Required input
     - Notes
   * - ``aggregation``
     - Bias movement toward increasing local density.
     - Neighbor density and boundary nodes.
     - Density-gradient cue.
   * - ``alignment``
     - Favor channel axes aligned with neighboring particles.
     - Neighbor channel state.
     - Alias of the current nematic alignment score.
   * - ``chemotaxis``
     - Bias movement along a scalar-field gradient.
     - ``parameters={"field": "name"}`` and a matching state field.
     - Directional external cue.
   * - ``contact_guidance``
     - Align movement with a local director field.
     - A vector state field; defaults to ``director``.
     - Nematic external cue.
   * - ``nematic``
     - Align channel axes without distinguishing head from tail.
     - Neighbor channel state.
     - Alias of ``nematic_alignment``.
   * - ``nematic_alignment``
     - Align channel axes without distinguishing head from tail.
     - Neighbor channel state.
     - Canonical descriptive name.
   * - ``persistent_motion``
     - Favor the node's previous movement direction.
     - Current local channel state.
     - Alias of ``persistent_walk``.
   * - ``persistent_walk``
     - Favor the node's previous movement direction.
     - Current local channel state.
     - Persistent directional memory.
   * - ``random_walk``
     - Give every admissible channel state equal score.
     - No additional input.
     - Alias of ``uniform``.
   * - ``resting_bias``
     - Favor admissible states with more occupied rest channels.
     - At least one rest channel for a visible effect.
     - Local motility-state bias.
   * - ``uniform``
     - Give every admissible channel state equal score.
     - No additional input.
     - Unbiased sampler term.

All gradients are derivatives in lattice units, with a lattice spacing of one,
on every geometry. This includes the density gradient used by aggregation, so a
sensitivity ``beta`` has the same meaning on 1D, square, hexagonal, cubic and
Moore lattices.

Prescribed chemotaxis fields are differentiated in physical lattice coordinates,
including hexagonal row staggering and vertical spacing. Interior differences
are centered and edge differences are one-sided; scalar fields do not implicitly
wrap with the particle boundary condition. A singleton axis has zero derivative.
Thus a physical linear ramp has the same gradient at boundary and interior sites.

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

Particle-conserving phenotype switching
---------------------------------------

For a particle-number-conserving switch, the complete channel state ``s`` is
mapped to one admissible state ``s'``. The transition preserves
``N(s') = N(s)``; it is not a sequence of independent channel writes. Sampling
the complete state is important for volume exclusion because separate writes
could collide, merge particles or accidentally create an invalid state.

The same full-state principle is already used by particle-conserving
volume-exclusion interactions such as random walk, alignment and chemotaxis.
The population-dynamics tutorial demonstrates the invariant directly.

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
