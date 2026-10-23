Known limitations
=================

What BioLGCA does not do yet, and what is planned.

Restarting identity-based models
--------------------------------

A run of an identity-based model cannot yet be restarted from saved data:
the ``from_npz`` initializer reads classical states only, because the labels
of the cells and their traits would have to be restored together. Rerun such
a model from its model specification and seed instead: the same seed gives the
same run.

Initial conditions
------------------

Four initial conditions are available: a random density, a filled region, an
explicit array of nodes and an array read from a file (see
:doc:`/how_to/model_specs_and_plugins`, "Initial conditions"). A filled region
fills all channels of all species. To place cells in particular channels, or
cells of one species only, pass an explicit ``nodes`` array.

Boundary conditions
-------------------

Periodic, reflecting and absorbing boundaries are tested on every lattice. A
table of which boundaries every geometry and model family supports, inflow
included, is still to be written.

Plots
-----

:doc:`/how_to/observers_and_plotting` describes the plots and animations.
Still missing: a choice between absolute and relative colour scales in all
plots, plots of the traits and families of identity-based models beyond the
Muller plot, and a style for publication figures.

Parameter studies and the explorer
----------------------------------

Sweeps run in parallel on one machine (:doc:`/how_to/studying_a_model`), not
yet on a cluster. The live explorer (:doc:`/how_to/exploring`) shows one
window of plots at a time.

Analysing evolution
-------------------

BioLGCA records the population size of each family and its ancestry. It has
no functions yet for diversity indices or the entropy of a population, and no
recorder of the trait distribution over time. Until then, record the nodes and
read the traits of the cells (see :doc:`/how_to/observers_and_plotting`).

Sharing rules as a package
--------------------------

Rules are shared as a Python module that is imported before a model runs
(:doc:`/how_to/custom_interactions`). A worked example of distributing rules
as an installable package is planned. Model files will not import code
themselves.

The removed wetting interaction
-------------------------------

The former ``wetting`` interaction was removed because it relied on
undocumented model attributes and silently changed the interaction radius.
The model of cell–cell adhesion, matrix confinement and jamming in breast
cancer invasion (Ilina et al., *Nat. Cell Biol.* 2020,
https://doi.org/10.1038/s41556-020-0552-6) is available in
:doc:`/zoo/jamming`, composed from adhesion, pressure, matrix confinement,
degradation and influx rules.
