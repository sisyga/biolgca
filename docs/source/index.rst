BioLGCA
*******

`BioLGCA <https://github.com/sisyga/biolgca>`_ is a Python package for
simulating lattice-gas cellular automata in biological contexts. It supports
classical, identity-based, volume-exclusion-free and multi-species models on
one-, two- and three-dimensional lattices.

Start with the notebooks
------------------------

The maintained :doc:`tutorials/index` are the primary learning path. They show
every model specification and interaction in editable cells, beginning with a
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

For a study, a :class:`lgca.model.ModelSpec` writes every part of the model
out, the lattice, state, time, rules and recorded data, so that rules can be
combined, models saved as files and parameters swept:

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

:doc:`getting_started` introduces both, :doc:`reference/factory_reference` lists
the options of ``get_lgca``, :doc:`how_to/model_specs_and_plugins` explains
shareable JSON/YAML models and composed pipelines, and
:doc:`how_to/studying_a_model` parameter sweeps.

Scientific background
---------------------

LGCA are cellular automata with an extended channel state space. Particles or
cells occupy velocity and optional rest channels, so the state records both
cell number and movement direction. See the
`BIO-LGCA overview <https://en.wikipedia.org/wiki/BIO-LGCA>`_ and the method
paper `Deutsch et al. 2021 <https://doi.org/10.1371/journal.pcbi.1009066>`_.

Citing BioLGCA
--------------

If you use BioLGCA in published work, please cite:

* Deutsch A, Nava-Sedeño JM, Syga S, Hatzikirou H (2021). BIO-LGCA: A cellular
  automaton modelling class for analysing collective cell migration.
  *PLoS Computational Biology* 17(6): e1009066.
  `doi:10.1371/journal.pcbi.1009066 <https://doi.org/10.1371/journal.pcbi.1009066>`_
* Syga S, Nava-Sedeño JM, Deutsch A (2026). A novel cellular automaton approach
  for modeling genotypic and phenotypic heterogeneity in cell systems.
  *The European Physical Journal Special Topics*.
  `doi:10.1140/epjs/s11734-026-02186-1 <https://doi.org/10.1140/epjs/s11734-026-02186-1>`_

The repository's ``CITATION.cff`` contains the same references in a
machine-readable form.

Questions and contributions
---------------------------

Issues are tracked on the
`GitHub issue tracker <https://github.com/sisyga/biolgca/issues>`_. BioLGCA is
distributed under the BSD 3-clause license.

.. toctree::
   :maxdepth: 4
   :hidden:

   getting_started
   tutorials/index
   how_to/index
   concepts/index
   example_gallery
   reference/index
