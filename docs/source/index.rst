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

After installing the repository, launch JupyterLab:

.. code-block:: bash

   python -m pip install -e .
   jupyter lab

Open ``docs/source/tutorials/01_fundamentals.ipynb`` for the first lesson.

A reproducible model in Python
------------------------------

BioLGCA separates lattice setup, state, time, interactions and analysis in a
:class:`lgca.model.ModelSpec`:

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(20, 20), boundary="periodic"),
       state=StateSpec(density=0.15, restchannels=0),
       time=TimeSpec(steps=30, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.random_walk"}],
       ),
       analysis=AnalysisSpec(observers=[DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   result.lgca.plot_density()

The :doc:`how_to/model_specs_and_plugins` guide explains shareable JSON/YAML
models and composed pipelines. The traditional :func:`lgca.get_lgca` factory
remains supported and is documented in :doc:`reference/factory_reference`.

Scientific background
---------------------

LGCA are cellular automata with an extended channel state space. Particles or
cells occupy velocity and optional rest channels, so the state records both
cell number and movement direction. See the
`BIO-LGCA overview <https://en.wikipedia.org/wiki/BIO-LGCA>`_ and the method
paper `Deutsch et al. 2021 <https://doi.org/10.1371/journal.pcbi.1009066>`_.

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
