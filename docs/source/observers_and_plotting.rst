Observers and plotting
======================

Simulation, logging and plotting are now separate concerns. Existing
``timeevo`` keyword flags still work, but internally they are implemented with
observers. New code can pass observers explicitly through
:class:`lgca.simulation.SimulationRunner` or through
:class:`lgca.model.ModelSpec`.

Recording with observers
------------------------

.. code-block:: python

   from lgca import get_lgca
   from lgca.simulation import DensityRecorder, NodeRecorder, SimulationRunner

   lgca = get_lgca(
       geometry="square",
       dims=(30, 30),
       interaction="alignment",
       seed=2,
   )

   runner = SimulationRunner(
       lgca,
       timesteps=50,
       observers=[NodeRecorder(), DensityRecorder()],
       showprogress=False,
   )
   runner.run()

   print(lgca.nodes_t.shape)
   print(lgca.dens_t.shape)

Observer schedules
------------------

Observers can run every step, every ``n`` steps or at an explicit set of steps.

.. code-block:: python

   from lgca.simulation import DensityRecorder, Schedule

   density_every_ten = DensityRecorder(schedule=Schedule(every=10))
   density_at_steps = DensityRecorder(schedule=Schedule(steps={0, 25, 50}))

Built-in recorder observers include:

.. list-table::
   :header-rows: 1

   * - Observer
     - Output
   * - ``NodeRecorder``
     - ``lgca.nodes_t``
   * - ``DensityRecorder``
     - ``lgca.dens_t``
   * - ``ChannelDensityRecorder``
     - ``lgca.channel_pop_t``
   * - ``PopulationRecorder``
     - ``lgca.n_t``
   * - ``PerTypeRecorder``
     - ``lgca.velcells_t`` and ``lgca.restcells_t``
   * - ``OrderParameterRecorder``
     - NoVE entropy and alignment time series
   * - ``FamilyPopulationRecorder``
     - ``lgca.fam_pop_t`` for supported family-tracking runs

Plotting module
---------------

The :mod:`lgca.plotting` module provides small dispatch helpers and plotting
observers. The existing geometry-specific plotting methods remain available on
LGCA instances.

.. code-block:: python

   from lgca import get_lgca
   from lgca.plotting import plot

   lgca = get_lgca(geometry="hex", interaction="alignment", seed=4)
   lgca.timeevo(timesteps=25, showprogress=False)

   plot(lgca, kind="density")

Snapshot and movie observers
----------------------------

Use plotting observers when plot creation should be part of the simulation
specification instead of a post-processing step.

.. code-block:: python

   from pathlib import Path

   from lgca import get_lgca
   from lgca.plotting import AnimationObserver, PlotSnapshotObserver
   from lgca.simulation import Schedule, SimulationRunner

   lgca = get_lgca(geometry="square", dims=(30, 30), interaction="random_walk", seed=6)

   snapshots = PlotSnapshotObserver(
       kind="density",
       schedule=Schedule(steps={0, 25, 50}),
       output_dir=Path("outputs/snapshots"),
   )
   movie = AnimationObserver(
       kind="density",
       schedule=Schedule(every=2),
       save_path=Path("outputs/density.mp4"),
       save_kwargs={"fps": 10},
   )

   SimulationRunner(
       lgca,
       timesteps=50,
       observers=[snapshots, movie],
       showprogress=False,
   ).run()

   print(snapshots.paths)
   print(movie.animation)

The plotting observers require the optional plotting dependencies and use the
same backend methods as ``lgca.plot_density()``, ``lgca.animate_density()`` and
the related geometry-specific helpers.

Density/species selection is shared between static plots and animations.
Square scalar fields use image artists, so large 2D plots do not create one
Python polygon per lattice site; hexagonal renderers retain geometry-aware
collections.
