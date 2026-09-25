Getting started
===============

Installation
------------

BioLGCA uses `uv <https://docs.astral.sh/uv/>`_ to manage Python and all
dependencies. The repository contains a lock file, ``uv.lock``, that pins every
package version, so everyone who runs ``uv sync`` works in the same environment
as the test suite.

`Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_ once,
then clone the repository and create the environment:

.. code-block:: bash

   git clone https://github.com/sisyga/biolgca.git
   cd biolgca
   uv sync

``uv sync`` creates ``.venv/`` with the Python version named in
``.python-version``, installs BioLGCA in editable mode and adds the test and
documentation tools. If your uv does not download Python automatically, which
is the case for some Linux distribution packages, run ``uv python install``
first.

The installation includes NumPy, SciPy, tqdm, Matplotlib and JupyterLab, so it
is sufficient for the maintained tutorials and ordinary one- and
two-dimensional analysis. Launch JupyterLab from the repository root:

.. code-block:: bash

   uv run jupyter lab

Then open ``docs/source/tutorials/01_fundamentals.ipynb``. The six
:doc:`tutorials/index` notebooks progress from a first random walk to a
reproducible student project. Run your own scripts the same way, for example
``uv run python my_simulation.py``, or activate ``.venv`` as usual.

Without uv, BioLGCA installs into any Python 3.11+ environment with
``python -m pip install -e .``. This resolves the newest compatible package
versions instead of the locked ones.

Your first model
----------------

:func:`lgca.get_lgca` builds one of the standard models that come with
BioLGCA from a few choices: the lattice, the initial density of cells, the
interaction and the seed. ``timeevo`` runs it, and ``lgca.data`` holds what
it recorded:

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(geometry="square", dims=(20, 20), density=0.15,
                   interaction="random_walk", seed=1)
   lgca.timeevo(timesteps=30, recordN=True, showprogress=False)
   print(lgca.data["population"])
   lgca.plot_density()

``lgca.interactions`` lists the standard models of a lattice: movement
(random walk, alignment, aggregation, chemotaxis, ...), growth (birth and
death, go-or-grow) and, for identity-based models (``ib=True``), published
research models. Lessons 1 and 2 of the :doc:`tutorials/index` work this way;
:doc:`reference/factory_reference` lists all options.

Your first model specification
------------------------------

Each standard model is a list of rules. A :class:`lgca.model.ModelSpec` writes
the whole model out: the lattice, initial state, time horizon, rules and
recorded data. Its parts can be combined freely, saved as a file and varied
systematically (lesson 3 onwards):

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, PopulationRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(20, 20), boundary="periodic"),
       state=StateSpec(density=0.15, restchannels=0),
       time=TimeSpec(steps=30, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "random_walk"}],
       ),
       analysis=AnalysisSpec(
           observers=[DensityRecorder(), PopulationRecorder()],
       ),
   )

   result = run_model(spec, showprogress=False)
   print(result.data["population"])
   result.lgca.plot_density()

The explicit seed makes stochastic comparisons repeatable. The rules are
visible and can be replaced or composed; lessons 3 and 4 show the supported
patterns, and :doc:`how_to/studying_a_model` how to sweep parameters and
seeds.

Saving and sharing a model
--------------------------

JSON is the canonical, versioned ModelSpec format. YAML is optional authoring
syntax for the same data model. Model files contain data and registered names,
never arbitrary import paths or Python code.

The installed command can export, validate and run curated starting points.
Prefix each command with ``uv run`` unless ``.venv`` is activated:

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

Two specialized features are optional extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
     - Install command
   * - ``yaml``
     - YAML model-file syntax
     - ``uv sync --extra yaml``
   * - ``plot3d``
     - Mayavi-based three-dimensional plotting
     - ``uv sync --extra plot3d``

With pip, use ``python -m pip install -e ".[yaml]"`` and so on. Contributor
tools are dependency groups rather than extras: ``test``, ``docs`` and ``dev``
(both). ``uv sync`` installs ``dev`` by default; ``uv sync --no-default-groups
--group docs`` installs only the documentation tools.

Developer checks
----------------

Run the test suite and strict documentation build from the repository root:

.. code-block:: bash

   uv run pytest -q
   uv run python docs/build.py

The documentation build executes every maintained notebook from a clean kernel
and treats cell exceptions and Sphinx warnings as failures. After changing
dependencies in ``pyproject.toml``, run ``uv lock`` and commit the updated
``uv.lock``.
