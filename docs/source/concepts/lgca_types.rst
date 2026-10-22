Types of LGCA
=============

BioLGCA models differ along two axes: whether channels obey volume exclusion
(at most one cell per channel) and whether cells are counted (classical) or
tracked individually with labels and traits (identity-based). That gives four
model families. A third axis, ``n_species``, adds several species to classical
models. In a :class:`~lgca.model.ModelSpec` the family is chosen by
``StateSpec(volume_exclusion=..., identity_based=..., n_species=...)``;
:func:`lgca.get_lgca` takes ``ve=``, ``ib=`` and ``n_species=``.

The snippets below assume ``from lgca import get_lgca``.

.. list-table::
   :header-rows: 1

   * -
     - Classical (cells are counted)
     - Identity-based (cells have labels and traits)
   * - **Volume exclusion**
     - ``"classical"``: a channel is empty or holds one cell
     - ``"ib"``: a channel holds the label of its cell, 0 if empty
   * - **No volume exclusion**
     - ``"nove"``: a channel holds any number of cells
     - ``"nove_ib"``: a channel holds a list of labels

The names in quotes are the ``families`` of :func:`lgca.interaction
<lgca.rules.interaction>`.

Classical LGCA
--------------

Classical LGCA with volume exclusion store whether each channel is occupied
by a cell. Cells do not carry individual traits. The state array stores
occupancy values (0 or 1) with the channel axis last.

.. code-block:: python

   lgca = get_lgca(ve=True, ib=False)

LGCA without volume exclusion
-----------------------------

Without volume exclusion a channel holds a number of cells. This is useful
for dense populations whose cells do not need to be distinguished; a
``capacity`` (``StateSpec.capacity``) sets the number of cells per node at
which growth rules count a node as crowded. Several of these interactions
assume no rest channels; pass ``restchannels=0`` when using alignment rules
such as ``dd_alignment`` or ``di_alignment``.

.. code-block:: python

   lgca = get_lgca(ve=False, ib=False, restchannels=0)

Identity-based LGCA
-------------------

In identity-based LGCA every cell has a label and traits, such as its own
birth rate, which daughters inherit and mutations change. Rules read and
change the cells through ``state.cells`` (see
:doc:`/how_to/custom_interactions`); on the model object the traits are
stored in ``lgca.props``, one array per trait indexed by label. These models
are used for evolution and heterogeneous populations, e.g. the go-or-grow
and glioblastoma models with traits that mutate.

With volume exclusion, a channel holds the label of its cell, and label 0
marks an empty channel:

.. code-block:: python

   lgca = get_lgca(ve=True, ib=True)

Without volume exclusion, a node holds any number of cells, each channel a
list of labels; label 0 is then a cell like any other, and the models have
one rest channel:

.. code-block:: python

   lgca = get_lgca(ve=False, ib=True, interaction="go_or_grow")

Multi-species LGCA
------------------

Multi-species LGCA are selected with ``n_species > 1``. They are classical
models, with or without volume exclusion; identity-based models describe
differences between cells by their traits instead. Multi-species state
arrays include a species axis before the channel axis.

.. code-block:: python

   lgca = get_lgca(n_species=2, ve=True, restchannels=1,
                   interaction="excitable_medium_ms")

See :doc:`/reference/interactions_summary` for the rules that support
several species.

Data model summary
------------------

.. list-table::
   :header-rows: 1

   * - Family
     - Channel content
     - Traits per cell
     - Species axis
   * - Classical
     - Occupancy (0 or 1)
     - No
     - with ``n_species > 1``
   * - Classical without volume exclusion
     - Number of cells
     - No
     - with ``n_species > 1``
   * - Identity-based
     - Label of the cell, 0 if empty
     - Yes
     - No
   * - Identity-based without volume exclusion
     - List of labels
     - Yes
     - No

The lattice arrays of the model object, ``lgca.nodes`` and
``lgca.cell_density``, include one layer of ghost nodes that implement the boundary conditions; ``lgca.nodes[lgca.nonborder]``
is the lattice itself (see :doc:`/getting_started`).
