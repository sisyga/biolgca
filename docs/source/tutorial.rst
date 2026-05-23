********
Tutorial
********

This tutorial introduces the main API by walking through a typical simulation
workflow. Detailed, runnable examples are provided in the notebooks
:download:`BioLGCA.ipynb <../../BioLGCA.ipynb>` and
:download:`Evolutionary LGCA.ipynb <../../Evolutionary LGCA.ipynb>`. For more
code snippets see also :doc:`examples`.



Retrieving the correct LGCA
---------------------------
Use :func:`lgca.get_lgca` to select the model family and lattice geometry. See
:doc:`factory_reference` for the full model matrix.

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(
       geometry="hex",
       ib=False,
       ve=True,
       dims=(20, 20),
       interaction="random_walk",
       seed=1,
   )


Simulation
----------
Use :meth:`lgca.base.LGCA_base.timestep` for a single update or
:meth:`lgca.base.LGCA_base.timeevo` for repeated updates:

.. code-block:: python

   lgca.timestep()
   lgca.timeevo(timesteps=50, record=True, showprogress=False)


.. _adding_own_interactions:

Customisation
-------------
Interaction rules are callables that receive the LGCA instance and mutate
``lgca.nodes``. Built-in interactions are selected by name with the
``interaction`` argument. Use ``lgca.print_interactions()`` to list valid names
for a constructed simulator.

Custom interaction registration is still under active development. For now,
assign a callable to ``lgca.interaction`` and store any parameters in
``lgca.interaction_params``. The callable should refresh dynamic fields when it
changes density-dependent state and should leave propagation to
``lgca.timestep()`` unless propagation has been disabled intentionally.

