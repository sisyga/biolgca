.. biolgca documentation master file, created by
   sphinx-quickstart on Mon Jan 17 11:03:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to the biolgca Documentation!
=====================================

`biolgca <https://github.com/sisyga/biolgca>`_ is a Python package for simulating different types of *lattice-gas 
cellular automata (LGCA)* in the biological context.

LGCA are a subclass of cellular automata with an extended state space that allows 
each particle/cell to have a direction. For a more detailed 
introduction see the `Wikipedia article <https://en.wikipedia.org/wiki/BIO-LGCA>`_. 
They present a mesoscopic modelling framework to analyse collective phenomena, e.g. cell migration. 
Use cases are demonstrated in this 
`paper <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009066>`_.

The package is intended for use in ongoing research as well as to exemplify the unique 
advantages of the framework.

.. note::

   This project is under active development and the documentation is under construction.

biolgca is licensed under the BSD 3-clause license.


.. toctree::
   :maxdepth: 3
   :hidden:

   user_guide
   examples
   full_api

.. To Do: read https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html?highlight=automodule for structure
               https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard numpy style docstrings!
			   https://www.python.org/dev/peps/pep-0257/ docstring PEP
   references https://github.com/matplotlib/matplotlib/blob/main/doc/api/cm_api.rst
              https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/cm.py
			  https://matplotlib.org/stable/api/cm_api.html
			  https://docutils.sourceforge.io/docs/user/rst/quickref.html reSt cheat sheet
			  
	Synchon:
	Adding to Jörn’s first point: I developed https://gitlab.com/synchon/hawala during my research project and it is 
	one command away from publishing it to PyPI. Maybe that can help you out.
	Adding to Andreas’s last point: https://joss.theoj.org/ is a good way to publish your open-source software. 
	I was involved in one publication here, and I would recommend it.
	I meant the structure that it has like the usage of tools like poetry and tox for linting (via isort), formatting 
	(via black) and testing (via pytest) and so on. I must admit the python packaging space has been a mess for long 
	but it’s getting better. It might be overwhelming at the start. A working example might help you. Let me know if 
	you have some doubts.
	I developed an earlier prototype and reviewed https://joss.theoj.org/papers/10.21105/joss.02342

	need autosummary to generate automodule and autofunction and link them:
	https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202
	https://github.com/sphinx-doc/sphinx/issues/7912
	https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion main reference from here
	https://alabaster.readthedocs.io/en/latest/customization.html How to customize alabaster
	.function {
    border-bottom: 3px solid #d0d0d0;
    padding-bottom: 10px;
    padding-top: 10px;
	background-color: #e8e8e8;
    }
	.method .field-even {
    padding-top: 0px;
	padding-bottom: 0px;
    }
    .method .field-odd {
        padding-top: 0px;
    	padding-bottom: 0px;
    }