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
   * - ``plot2d``
     - 2D plotting with Matplotlib
     - ``python -m pip install -e ".[plot2d]"``
   * - ``plot3d``
     - 3D plotting with Mayavi
     - ``python -m pip install -e ".[plot3d]"``
   * - ``plot`` or ``plotting``
     - All plotting extras
     - ``python -m pip install -e ".[plot]"``
   * - ``dev``
     - Development, testing, docs and 2D plotting
     - ``python -m pip install -e ".[dev]"``

The legacy ``requirements.txt``, ``documentation_requirements.txt`` and
``plotting-requirements.txt`` files are still present for older workflows, but
the extras above are the preferred installation interface.

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
   python -m sphinx -b html docs/source docs/_build/html

Read the Docs uses the same Sphinx configuration from ``docs/source/conf.py``
and installs the ``docs`` extra through ``.readthedocs.yaml``.

Notebook examples
-----------------

The repository includes notebook examples for broader tours of the API:

- :download:`BioLGCA notebook <../../BioLGCA.ipynb>`
- :download:`Evolutionary LGCA notebook <../../Evolutionary LGCA.ipynb>`
