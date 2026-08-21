Getting started
===============

Installation
------------

Clone the repository and install the package into your active Python
environment:

.. code-block:: bash

   git clone https://github.com/sisyga/biolgca.git
   cd biolgca
   python -m pip install -e .

The core package depends on ``numpy``, ``scipy`` and ``tqdm``. Optional extras
are declared in ``pyproject.toml``:

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
     - Install command
   * - ``test``
     - Test runner dependencies
     - ``python -m pip install -e ".[test]"``
   * - ``docs``
     - Sphinx documentation build
     - ``python -m pip install -e ".[docs]"``
   * - ``yaml``
     - Optional YAML model-file syntax
     - ``python -m pip install -e ".[yaml]"``
   * - ``plot2d``
     - 2D plotting with Matplotlib
     - ``python -m pip install -e ".[plot2d]"``
   * - ``plot3d``
     - 3D plotting with Mayavi
     - ``python -m pip install -e ".[plot3d]"``
   * - ``plot`` or ``plotting``
     - Matplotlib plus the optional Mayavi 3D stack
     - ``python -m pip install -e ".[plot]"``
   * - ``dev``
     - Development, testing, docs and 2D plotting
     - ``python -m pip install -e ".[dev]"``

The legacy ``requirements.txt``, ``documentation_requirements.txt`` and
``plotting-requirements.txt`` files are still present for older workflows, but
the extras above are the preferred installation interface.

For ordinary 1D and 2D work, prefer ``plot2d``. The ``plot``/``plotting``
umbrella also installs Mayavi and is intended only when both renderer stacks
are needed.

Choose a workflow
-----------------

BioLGCA has two supported entry paths:

1. Use :func:`lgca.get_lgca` for interactive exploration in Python or a
   notebook.
2. Use a versioned ModelSpec file and the ``biolgca`` command for simulations
   that must be reviewed, saved and shared.

Both paths use the same lattice implementations. The ModelSpec path adds
strict validation, registered interaction names and resolved run provenance;
it does not require a GUI or XML configuration.

Creating a simulator
--------------------

Use :func:`lgca.get_lgca` to choose a model family, geometry, initial condition
and interaction rule:

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(
       geometry="1d",
       ib=True,
       ve=True,
       interaction="random_walk",
       dims=50,
       seed=1,
   )
   lgca.timeevo(timesteps=50, record=True, showprogress=False)

The :doc:`factory_reference` page lists the supported factory switches and
class families.

Declarative simulations
-----------------------

For reproducible model setup, use :class:`lgca.model.ModelSpec`. A model spec
keeps lattice setup, interaction dynamics and observer-based logging in
separate sections:

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(20, 20), boundary="reflecting"),
       state=StateSpec(density=0.2, restchannels=0),
       time=TimeSpec(steps=50, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.random_walk"}],
       ),
       analysis=AnalysisSpec(observers=[DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   print(result.lgca.dens_t.shape)

See :doc:`model_specs_and_plugins` for composed interaction phases and
:doc:`observers_and_plotting` for recorders, snapshots and movies.

Shareable command-line workflow
-------------------------------

The installed ``biolgca`` command exports curated templates and runs the same
strict ModelSpec pipeline used from Python:

.. code-block:: bash

   biolgca examples list
   biolgca examples export random_walk model.json
   biolgca validate model.json
   biolgca run model.json --output runs/random-walk-001

``validate`` parses the complete model, resolves registered interactions,
checks initializers and compiles the pipeline without evolving a trajectory or
writing observer outputs. ``run`` creates an explicit directory containing
``model.resolved.json``, ``metadata.json`` and any configured CSV or plot
outputs. It refuses an existing directory unless ``--overwrite`` is supplied.

JSON is the canonical ModelSpec format and every file carries
``schema_version: 1``. YAML is optional authoring syntax for exactly the same
data model. Model files contain data and registered names, never import paths
or arbitrary Python code.

Input resources such as ``from_npz`` states are relative to the model file and
must remain inside its directory. Observer output paths are relative to the run
directory. Absolute paths and ``..`` escapes are rejected; the
``--trusted-paths`` option is reserved for intentionally trusted local models.

Initial-condition presets
-------------------------

Random initialization remains the shortest form:

.. code-block:: json

   "state": {"density": 0.2, "restchannels": 1}

A portable model can instead request a centered, left-aligned or corner region:

.. code-block:: json

   "state": {
     "restchannels": 1,
     "initializer": {
       "name": "region",
       "parameters": {
         "placement": "center",
         "extent": [20, 20],
         "density": 1.0
       }
     }
   }

Large numeric channel states belong in NPZ rather than JSON. Store an array
named ``nodes`` with the exact spatial-plus-channel shape and reference it with
``{"name": "from_npz", "parameters": {"path": "states/start.npz"}}``.
Identity-based NPZ checkpoints are deliberately rejected for now because
particle properties must be restored together with their labels.

A model file is configuration, and a run directory contains resolved
configuration, provenance and requested outputs. Neither is a general restart
checkpoint or a serialized LGCA object. Keep restart/checkpoint data separate
from the shareable model file until a backend-specific checkpoint contract is
available.

Running tests
-------------

From the repository root:

.. code-block:: bash

   python -m pytest -q

Building the documentation
--------------------------

Install the docs extra and build the HTML documentation locally:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python docs/build.py

The build command treats warnings as errors and removes stale generated
autosummary pages before invoking Sphinx.

Read the Docs uses the same Sphinx configuration from ``docs/source/conf.py``
and installs the ``docs`` extra through ``.readthedocs.yaml``.

Notebook examples
-----------------

The repository includes notebook examples for broader tours of the API:

- :download:`BioLGCA notebook <../../BioLGCA.ipynb>`
- :download:`Evolutionary LGCA notebook <../../Evolutionary LGCA.ipynb>`
