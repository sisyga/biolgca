********
Tutorial
********

This tutorial introduces the main API by walking through a typical simulation
workflow. Tested, runnable starting points are provided in
:doc:`example_gallery`. The older :download:`BioLGCA.ipynb
<../../BioLGCA.ipynb>` and :download:`Evolutionary LGCA.ipynb
<../../Evolutionary LGCA.ipynb>` notebooks remain available as legacy
factory-API tours, but are not executed by CI. For more code snippets see also
:doc:`examples`.

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


Legacy-style simulation
-----------------------
Use :meth:`lgca.base.LGCA_base.timestep` for a single update or
:meth:`lgca.base.LGCA_base.timeevo` for repeated updates:

.. code-block:: python

   lgca.timestep()
   lgca.timeevo(timesteps=50, record=True, showprogress=False)


Declarative ModelSpec simulation
--------------------------------
The declarative API separates model setup, dynamics, observers and plotting.
Use :class:`lgca.model.ModelSpec` when a model should be reproducible and
self-describing.

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, NodeRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30), boundary="periodic"),
       state=StateSpec(density=0.25, restchannels=1),
       time=TimeSpec(steps=50, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.go_or_grow", "parameters": {"r_b": 0.2}}],
       ),
       analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   print(result.metadata["schedule"])
   result.lgca.plot_density()


Combining interaction phases
----------------------------
Pipelines execute birth/death, phenotype switching and reorientation before the
deterministic propagation step. This mirrors the conservation-law structure of
LGCA dynamics: particle-number changing events happen first, phenotype/species
changes next, mass-preserving reorientation last.

.. code-block:: python

   import numpy as np

   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import (
       InteractionPipelineSpec,
       PhenotypeSwitchSpec,
       ReorientationSpec,
       ReorientationTermSpec,
   )

   signal = np.linspace(0.0, 1.0, 30)[:, None] + np.zeros((30, 30))

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30)),
       state=StateSpec(
           density=0.2,
           restchannels=1,
           n_species=2,
           fields={"signal": signal},
       ),
       time=TimeSpec(steps=25, seed=2),
       dynamics=InteractionPipelineSpec(
           operators=[
               {"name": "birth_death", "parameters": {"birth_rate": [0.01, 0.005]}},
               PhenotypeSwitchSpec(
                   name="phenotype_switch",
                   parameters={"rates": [[0.0, 0.02], [0.01, 0.0]]},
               ),
               ReorientationSpec(
                   terms=[
                       ReorientationTermSpec(name="nematic_alignment", beta=1.0),
                       ReorientationTermSpec(
                           name="chemotaxis",
                           beta=0.4,
                           parameters={"field": "signal"},
                       ),
                   ],
               ),
           ],
       ),
   )

   result = run_model(spec, showprogress=False)


.. _adding_own_interactions:

Customisation
-------------
Built-in interactions can still be selected by name with the ``interaction``
argument. The plugin API exposes the same rules through canonical names such as
``"classical.go_or_grow"``, ``"ib.birthdeath_discrete"`` and
``"nove_ib.go_or_grow_glioblastoma"``. Use
:func:`lgca.plugins.list_plugins` and :func:`lgca.plugins.describe_plugin` to
inspect available operators and their parameter metadata.

Custom interaction callables are still possible by assigning
``lgca.interaction`` and ``lgca.interaction_params`` directly. For new reusable
rules, prefer adding an :class:`lgca.plugins.InteractionOperator` plugin so it
can participate in ModelSpec validation, schedule descriptions and registry
coverage reports.
