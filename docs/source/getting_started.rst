Getting started
===============

Installation
------------

Clone the repository and install BioLGCA into an active Python environment:

.. code-block:: bash

   git clone https://github.com/sisyga/biolgca.git
   cd biolgca
   python -m pip install -e .

A normal installation includes NumPy, SciPy, tqdm, Matplotlib and JupyterLab,
so it is sufficient for the maintained tutorials and ordinary one- and
two-dimensional analysis.

Launch JupyterLab from the repository root:

.. code-block:: bash

   jupyter lab

Then open ``docs/source/tutorials/01_fundamentals.ipynb``. The six
:doc:`tutorials/index` notebooks progress from a first random walk to a
reproducible student project.

Your first model specification
------------------------------

The tutorials use :class:`lgca.model.ModelSpec`. Its sections make the lattice,
initial state, time horizon, interaction pipeline and recorded data explicit:

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, PopulationRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(20, 20), boundary="periodic"),
       state=StateSpec(density=0.15, restchannels=0),
       time=TimeSpec(steps=30, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.random_walk"}],
       ),
       analysis=AnalysisSpec(
           observers=[DensityRecorder(), PopulationRecorder()],
       ),
   )

   result = run_model(spec, showprogress=False)
   print(result.lgca.n_t)
   result.lgca.plot_density()

The explicit seed makes stochastic comparisons repeatable. The interaction
entry is visible and can be replaced or composed; lessons 2--4 show the
supported patterns.

Saving and sharing a model
--------------------------

JSON is the canonical, versioned ModelSpec format. YAML is optional authoring
syntax for the same data model. Model files contain data and registered names,
never arbitrary import paths or Python code.

The installed command can export, validate and run curated starting points:

.. code-block:: bash

   biolgca examples list
   biolgca examples export random_walk model.json
   biolgca validate model.json
   biolgca run model.json --output runs/random-walk-001

The run directory contains ``model.resolved.json``, ``metadata.json`` and the
requested observer outputs. See :doc:`how_to/model_specs_and_plugins` for
pipeline composition, input-path rules and model-file details.

Optional dependencies
---------------------

Only specialized or contributor workflows use extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
     - Install command
   * - ``yaml``
     - YAML model-file syntax
     - ``python -m pip install -e ".[yaml]"``
   * - ``plot3d``
     - Mayavi-based three-dimensional plotting
     - ``python -m pip install -e ".[plot3d]"``
   * - ``test``
     - Test runner
     - ``python -m pip install -e ".[test]"``
   * - ``docs``
     - Sphinx/MyST-NB documentation build
     - ``python -m pip install -e ".[docs]"``
   * - ``dev``
     - Development, tests and documentation tools
     - ``python -m pip install -e ".[dev]"``

The old ``plot2d`` extra remains as an empty compatibility name; Matplotlib is
now installed normally. ``plot`` and ``plotting`` remain aliases for the
optional Mayavi stack.

Legacy interactive factory
--------------------------

The :func:`lgca.get_lgca` factory remains supported for existing code and quick
interactive experiments. New teaching and reproducible projects use ModelSpec
because it makes interactions, seeds and recorded data reviewable. See
:doc:`reference/factory_reference` for the complete factory matrix.

Developer checks
----------------

Run the test suite and strict documentation build from the repository root:

.. code-block:: bash

   conda run -n biolgca python -m pytest -q
   conda run -n biolgca python docs/build.py

The documentation build executes every maintained notebook from a clean kernel
and treats cell exceptions and Sphinx warnings as failures.
