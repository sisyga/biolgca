Factory reference
=================

The package entry point is :func:`lgca.get_lgca`. It chooses the concrete
simulator class from four main switches:

``geometry``
   Lattice geometry. Supported values and aliases are:

   - ``"1d"``, ``"lin"``, ``"linear"``
   - ``"square"``, ``"sq"``, ``"rect"``, ``"rectangular"``
   - ``"hex"``, ``"hx"``, ``"hexagonal"``
   - ``"cubic"``, ``"cb"``
   - ``"moore"``, ``"moore3d"``

``ib``
   If ``True``, use identity-based particles with individual labels and
   properties. Identity-based multi-species classes are not implemented.

``ve``
   If ``True``, enforce volume exclusion: at most one particle can occupy each
   channel. If ``False``, channels may contain multiple particles or particle
   IDs depending on the LGCA family.

``n_species``
   Number of species. Values greater than one select the multi-species class
   family. Multi-species classes are currently classical, not identity-based.
   Through ``get_lgca`` they support only ``random_walk`` (the default),
   ``excitable_medium_ms`` and ``only_propagation`` with volume exclusion, and
   ``birth``, ``birthdeath``, ``go_or_grow`` and ``only_propagation`` without.
   Other multi-species dynamics are built with a ModelSpec interaction
   pipeline; unsupported names raise ``ValueError`` at construction.

Model class matrix
------------------

.. list-table::
   :header-rows: 1

   * - Family
     - 1D
     - Square
     - Hexagonal
     - Cubic
     - Moore
   * - Classical, volume exclusion
     - ``LGCA_1D``
     - ``LGCA_Square``
     - ``LGCA_Hex``
     - ``LGCA_Cubic``
     - ``LGCA_3dMoore``
   * - Identity-based, volume exclusion
     - ``IBLGCA_1D``
     - ``IBLGCA_Square``
     - ``IBLGCA_Hex``
     - ``IBLGCA_Cubic``
     - ``IBLGCA_Moore``
   * - Classical, no volume exclusion
     - ``NoVE_LGCA_1D``
     - ``NoVE_LGCA_Square``
     - ``NoVE_LGCA_Hex``
     - ``NoVE_LGCA_Cubic``
     - ``NoVE_LGCA_Moore``
   * - Identity-based, no volume exclusion
     - ``NoVE_IBLGCA_1D``
     - ``NoVE_IBLGCA_Square``
     - ``NoVE_IBLGCA_Hex``
     - ``NoVE_IBLGCA_Cubic``
     - ``NoVE_IBLGCA_Moore``
   * - Multi-species, volume exclusion
     - ``MSLGCA_1D``
     - ``MSLGCA_Square``
     - ``MSLGCA_Hex``
     - ``MSLGCA_Cubic``
     - ``MSLGCA_Moore``
   * - Multi-species, no volume exclusion
     - ``MSLGCA_NoVE_1D``
     - ``MSLGCA_NoVE_Square``
     - ``MSLGCA_NoVE_Hex``
     - ``MSLGCA_NoVE_Cubic``
     - ``MSLGCA_NoVE_Moore``

Common keyword arguments
------------------------

``dims``
   Lattice dimensions. An integer expands to equal dimensions for square,
   hexagonal, cubic and Moore geometries. A tuple is interpreted by geometry.

``nodes``
   Custom initial lattice state. If supplied, it overrides ``dims``,
   ``density`` and inferred ``restchannels``.

``density``
   Initial random density used when ``nodes`` is not supplied.

``restchannels``
   Number of resting channels in addition to velocity channels. Some
   interactions require rest channels; some NoVE interactions require none.

``interaction``
   Name of a built-in interaction, or a function ``f(lgca)`` that performs the
   interaction step. Built-in options depend on the selected class family. With
   a function, all other keyword arguments that the factory does not use itself
   are collected in ``lgca.interaction_params`` (see the example below).

``bc``
   Boundary condition. Geometry-specific classes implement periodic,
   reflecting and absorbing variants where available.

``seed``
   Seed passed to the simulator's NumPy random generator for deterministic
   simulations.

Examples
--------

.. code-block:: python

   from lgca import get_lgca

   classical = get_lgca(geometry="hex", interaction="alignment", seed=1)
   identity = get_lgca(geometry="1d", ib=True, interaction="birth", seed=1)
   nove = get_lgca(geometry="square", ve=False, restchannels=0,
                   interaction="dd_alignment", seed=1)
   multispecies = get_lgca(geometry="square", n_species=2, restchannels=1,
                           interaction="excitable_medium_ms", seed=1)

A user-defined interaction function receives the LGCA object and changes its
channel state in place. Ghost nodes are overwritten by the boundary conditions
afterwards, so the function may act on the whole ``lgca.nodes`` array:

.. code-block:: python

   def random_death(lgca):
       p = lgca.interaction_params["r_d"]
       lgca.nodes &= lgca.rng.random(lgca.nodes.shape) >= p

   lgca = get_lgca(geometry="square", density=1, interaction=random_death, r_d=0.05, seed=1)
   lgca.timeevo(timesteps=20, record=True)

Keyword arguments are not checked for typos when ``interaction`` is a
function, because the factory cannot know which parameters the function reads.

Placeholders
------------

Detailed shape diagrams for custom ``nodes`` arrays and boundary-condition
tables are still missing. The current authoritative reference is the class API
documentation and the geometry-specific ``set_dims`` implementations.
