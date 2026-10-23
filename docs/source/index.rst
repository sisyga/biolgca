BioLGCA
*******

`BioLGCA <https://github.com/sisyga/biolgca>`_ is a Python package for
simulating lattice-gas cellular automata in biological contexts. It supports
classical and identity-based models, with and without volume exclusion, and
models of several species, on one-, two- and three-dimensional lattices.

Who it is for
-------------

BioLGCA is for researchers and students in mathematical and theoretical
biology who model collective cell migration, tissue growth, pattern formation
or the evolution of cell populations, and for courses that teach these
topics. LGCA describe cells by their numbers and directions of movement on a
lattice, so a model is a short list of local stochastic rules that can be
simulated on large lattices and analysed with mean-field and stability
methods.

General-purpose frameworks for cell-based models couple cells to fields in
many ways: Morpheus and CompuCell3D (cellular Potts models), PhysiCell
(off-lattice agents) and Chaste (several cell-based model types, including an
on-lattice cellular automaton). They do not offer the velocity-channel state
space of LGCA. In this state space, alignment, chemotaxis and go-or-grow
switching are rules for the reorientation of channel states, and mean-field
equations can be derived systematically from the rules. BioLGCA provides this
model class as a library: the rules of the BIO-LGCA literature as composable building blocks,
a checked contract for new rules (:doc:`how_to/custom_interactions`), model
files that rerun from their seed, parameter sweeps, and a :doc:`zoo/index` that
reproduces published LGCA models with these rules.

Start with the notebooks
------------------------

The maintained :doc:`tutorials/index` are the primary learning path. They show
every model specification and interaction in editable notebook cells, beginning with a
random walk and progressing to combined directional cues, population dynamics,
evolution and a reproducible student project.

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then
get BioLGCA and launch JupyterLab:

.. code-block:: bash

   git clone https://github.com/sisyga/biolgca.git
   cd biolgca
   uv sync
   uv run jupyter lab

:doc:`getting_started` explains the installation and alternatives to uv.
Open ``docs/source/tutorials/01_fundamentals.ipynb`` for the first lesson.

A first model in Python
-----------------------

:func:`lgca.get_lgca` builds one of the standard models that come with
BioLGCA by name:

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(geometry="square", dims=(20, 20), density=0.15,
                   interaction="random_walk", seed=1)
   lgca.timeevo(timesteps=30, showprogress=False)
   lgca.plot_density()

For a study, a :class:`lgca.model.ModelSpec` writes out every part of the
model (the lattice, state, time, rules and recorded data), so that rules can
be combined, models saved as files and parameters swept:

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(20, 20), boundary="periodic"),
       state=StateSpec(density=0.15, restchannels=0),
       time=TimeSpec(steps=30, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "random_walk"}],
       ),
       analysis=AnalysisSpec(observers=[DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   result.lgca.plot_density()

Where to go next:

* :doc:`getting_started` introduces both ways;
* :doc:`reference/factory_reference` lists the options of ``get_lgca``;
* :doc:`how_to/model_specs_and_plugins` explains shareable JSON/YAML models
  and composed pipelines;
* :doc:`how_to/exploring` runs a model live with sliders;
* :doc:`how_to/studying_a_model` sweeps parameters;
* the :doc:`zoo/index` has published models to rerun and extend.

Scientific background
---------------------

LGCA are cellular automata with an extended channel state space. Cells
occupy velocity and optional rest channels, so the state records both
cell number and movement direction. See the
`BIO-LGCA overview <https://en.wikipedia.org/wiki/BIO-LGCA>`_ and the method
paper `Deutsch et al. 2021 <https://doi.org/10.1371/journal.pcbi.1009066>`_.

Citing BioLGCA
--------------

If you use BioLGCA in published work, please cite:

* Syga S, Nava-Sedeño JM, Deutsch A (2026). A novel cellular automaton approach
  for modeling genotypic and phenotypic heterogeneity in cell systems.
  *The European Physical Journal Special Topics*.
  `doi:10.1140/epjs/s11734-026-02186-1 <https://doi.org/10.1140/epjs/s11734-026-02186-1>`_

The BIO-LGCA modelling class is described in:

* Deutsch A, Nava-Sedeño JM, Syga S, Hatzikirou H (2021). BIO-LGCA: A cellular
  automaton modelling class for analysing collective cell migration.
  *PLoS Computational Biology* 17(6): e1009066.
  `doi:10.1371/journal.pcbi.1009066 <https://doi.org/10.1371/journal.pcbi.1009066>`_

The repository's ``CITATION.cff`` contains the same references in a
machine-readable form.

Questions and contributions
---------------------------

Ask questions, report bugs and propose ideas on the
`GitHub issue tracker <https://github.com/sisyga/biolgca/issues>`_, which has
a template for each. The
`contributing guide <https://github.com/sisyga/biolgca/blob/master/CONTRIBUTING.md>`_
explains the development setup, the checks and how to propose a new
interaction; everyone taking part follows the
`code of conduct <https://github.com/sisyga/biolgca/blob/master/CODE_OF_CONDUCT.md>`_.
BioLGCA is distributed under the BSD 3-clause license.

.. toctree::
   :maxdepth: 4
   :hidden:

   getting_started
   tutorials/index
   how_to/index
   concepts/index
   example_gallery
   zoo/index
   reference/index
