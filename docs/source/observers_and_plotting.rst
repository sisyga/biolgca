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

For lightweight Python callbacks, wrap a function in
:class:`lgca.simulation.CallbackObserver`. The callback receives the LGCA and
the absolute simulation step. Use :func:`functools.partial` or a closure for
additional arguments:

.. code-block:: python

   from functools import partial

   from lgca.simulation import CallbackObserver, Schedule, SimulationRunner

   def report_population(lgca, step, sink):
       sink.append((step, int(lgca.cell_density[lgca.nonborder].sum())))

   samples = []
   callback = CallbackObserver(
       partial(report_population, sink=samples),
       schedule=Schedule(every=10),
   )
   SimulationRunner(
       lgca, timesteps=50, observers=[callback], showprogress=False
   ).run()

Callbacks are trusted Python code and therefore cannot be embedded in a
portable JSON or YAML ModelSpec.

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

Dense and sparse recorder results
---------------------------------

The default schedule records every step, including step zero, and preserves
the familiar ``timesteps + 1`` leading dimension. A sparse schedule stores
only the selected samples. Each array recorder publishes the corresponding
absolute step vector:

.. list-table::
   :header-rows: 1

   * - Values
     - Steps
   * - ``nodes_t``
     - ``nodes_steps``
   * - ``dens_t``
     - ``dens_steps``
   * - ``n_t``
     - ``n_steps``
   * - ``channel_pop_t``
     - ``channel_pop_steps``
   * - ``velcells_t`` / ``restcells_t``
     - ``velcells_steps`` / ``restcells_steps``
   * - ``fam_pop_t``
     - ``fam_pop_steps``

Always pair a sparse result with its step vector instead of interpreting its
row index as simulation time. Implicit animation of a sparse history is
rejected; pass an explicit history when custom timing is intended.

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

For multi-species models, density plots aggregate all species by default.
Select one species explicitly with ``species=<zero-based index>``. Static and
animated density paths use the same selection rule:

.. code-block:: python

   plot(lgca, kind="density")             # aggregate density
   plot(lgca, kind="density", species=1)  # species 1 only
   lgca.animate_density(species=1)

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
