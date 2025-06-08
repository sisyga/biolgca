.. _interaction_chapter:

Built-in Interactions
=====================

This section lists the interaction functions that ship with the package.
Interactions specify the behaviour of the LGCA.  During a timestep the
interaction step updates the state of each node and is followed by a
propagation step moving particles along the lattice.

To specify your own interaction function, please see the tutorial (source code -> BioLGCA.ipynb, dedicated page planned). Built-in interactions can be found below.


Interactions for classical LGCA
-------------------------------

.. automodule:: lgca.interactions
   :members:
   :noindex:
   
   
Interactions for identity-based LGCA
------------------------------------

.. automodule:: lgca.ib_interactions
   :members:
   :noindex:

   
Interactions for LGCA without volume exclusion
----------------------------------------------

.. automodule:: lgca.nove_interactions
   :members:
   :noindex:

