Getting started
===============

This guide summarises the basic installation steps and explains how to build the
documentation yourself. More detailed usage examples can be found in the
:doc:`examples` section.

Dependencies
------------

The core package depends on `numpy`, `scipy`, and `tqdm`.

To install the package from a local checkout, run:

.. code-block:: bash

   pip install -e .

For development work, install the test and documentation extras:

.. code-block:: bash

   pip install -e ".[test,docs]"

For plotting support, install the relevant optional extra:

.. code-block:: bash

   pip install -e ".[plot2d]"
   pip install -e ".[plot3d]"

Use
---

Import :func:`lgca.get_lgca` to create a simulator instance:

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(ib=True, geometry="1d", interaction="random_walk")
   lgca.timeevo(timesteps=50)
   lgca.plot_density()

The :download:`BioLGCA notebook <../../BioLGCA.ipynb>` gives a broader tour of
the available model arguments.

Building the documentation
--------------------------

The online documentation is built automatically by Read the Docs. The build
process is configured in the :file:`.readthedocs.yaml` file at the repository
root. After installing the docs extra, reproduce the HTML build locally with:

.. code-block:: bash

   python -m sphinx -b html docs/source docs/_build/html
