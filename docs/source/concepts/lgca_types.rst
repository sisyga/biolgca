Types of LGCA
=============

The simulator implements several lattice-gas cellular automata families. The
two main modelling axes are volume exclusion and identity tracking. A third
axis, ``n_species``, selects classical multi-species classes.

The snippets below assume ``from lgca import get_lgca``.

Classical LGCA
--------------

Classical LGCA enforce volume exclusion: each channel is either empty or
occupied by one particle. Particles do not carry individual properties. The
state array stores boolean-like occupancy values with the channel axis last.

Use this family with:

.. code-block:: python

   lgca = get_lgca(ve=True, ib=False)

Identity-based LGCA
-------------------

Identity-based LGCA also enforce volume exclusion, but occupied channels store
particle IDs instead of plain occupancy. Per-particle properties are stored in
``lgca.props`` and can be recorded during simulations.

Use this family with:

.. code-block:: python

   lgca = get_lgca(ve=True, ib=True)

LGCA without volume exclusion
-----------------------------

NoVE LGCA allow channels to hold counts greater than one. This is useful for
density-based models where individual cells do not need to be distinguished.
Several NoVE interactions assume no rest channels; pass ``restchannels=0`` when
using alignment-style NoVE rules such as ``dd_alignment`` or ``di_alignment``.

Use this family with:

.. code-block:: python

   lgca = get_lgca(ve=False, ib=False, restchannels=0)

Identity-based LGCA without volume exclusion
--------------------------------------------

NoVE identity-based LGCA allow multiple labelled cells per node while
preserving individual or family properties. These models are used for
heterogeneous proliferating populations, including the go-or-grow and
glioblastoma interaction variants.

Use this family with:

.. code-block:: python

   lgca = get_lgca(ve=False, ib=True, interaction="go_or_grow")

Multi-species LGCA
------------------

Multi-species LGCA are selected with ``n_species > 1``. They are currently
classical models, with or without volume exclusion. Identity-based
multi-species models are not implemented. Multi-species state arrays include a
species axis before the channel axis.

Use this family with:

.. code-block:: python

   lgca = get_lgca(n_species=2, ve=True, restchannels=1,
                   interaction="excitable_medium_ms")

The current multi-species interaction library is intentionally small. See
:doc:`/reference/interactions_summary` for supported rules.

Data model summary
------------------

.. list-table::
   :header-rows: 1

   * - Family
     - Channel content
     - Individual properties
     - Species axis
   * - Classical
     - Occupancy
     - No
     - No
   * - Identity-based
     - Particle ID
     - Yes
     - No
   * - NoVE classical
     - Particle count
     - No
     - No
   * - NoVE identity-based
     - Lists or arrays of particle IDs
     - Yes
     - No
   * - Multi-species
     - Occupancy or count by species
     - No
     - Yes
