.. _examples_chapter:

Examples
========

This section collects small tutorials and notebooks showing how to set up
simulations and customise the LGCA simulator. The Sphinx pages are intentionally
short; the notebooks contain longer exploratory workflows.

For a Morpheus-style list of runnable, curated ``ModelSpec`` examples by
modeling question, see :doc:`example_gallery`.

Legacy factory example
----------------------

.. code-block:: python

   from lgca import get_lgca

   lgca = get_lgca(geometry="hex", interaction="alignment", bc="refl", seed=1)
   lgca.timeevo(timesteps=132, record=True, showprogress=False)
   lgca.plot_flux()

Declarative plugin example
--------------------------

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, NodeRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30), boundary="periodic"),
       state=StateSpec(density=0.25, restchannels=1),
       time=TimeSpec(steps=50, seed=2),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "classical.go_or_grow", "parameters": {"r_b": 0.2}}],
       ),
       analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   print(result.metadata["schedule"])
   result.lgca.plot_density()

Observer and movie example
--------------------------

.. code-block:: python

   from pathlib import Path

   from lgca import get_lgca
   from lgca.plotting import AnimationObserver, PlotSnapshotObserver
   from lgca.simulation import Schedule, SimulationRunner

   lgca = get_lgca(geometry="square", dims=(30, 30), interaction="random_walk", seed=6)

   SimulationRunner(
       lgca,
       timesteps=50,
       observers=[
           PlotSnapshotObserver(
               kind="density",
               schedule=Schedule(steps={0, 25, 50}),
               output_dir=Path("outputs/snapshots"),
           ),
           AnimationObserver(
               kind="density",
               schedule=Schedule(every=2),
               save_path=Path("outputs/density.mp4"),
               save_kwargs={"fps": 10},
           ),
       ],
       showprogress=False,
   ).run()

Repository examples
-------------------

- :download:`BioLGCA notebook <../../BioLGCA.ipynb>`
- :download:`Evolutionary LGCA notebook <../../Evolutionary LGCA.ipynb>`
- :download:`Multispecies NoVE tumor growth demo <../../multispecies_nove_tumor_growth_demo.py>`

.. toctree::
   :maxdepth: 2
   
   tutorial
   
