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
