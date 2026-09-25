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
the local step within this run, starting at zero. Use :func:`functools.partial` or a closure for
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
     - Name in ``result.data``
     - Attribute of the model
   * - ``NodeRecorder``
     - ``"nodes"``
     - ``lgca.nodes_t``
   * - ``DensityRecorder``
     - ``"density"``
     - ``lgca.dens_t``
   * - ``ChannelDensityRecorder``
     - ``"channel_population"``
     - ``lgca.channel_pop_t``
   * - ``PopulationRecorder``
     - ``"population"`` (or ``"n"``)
     - ``lgca.n_t``
   * - ``PerTypeRecorder``
     - ``"moving"`` and ``"resting"``
     - ``lgca.velcells_t`` and ``lgca.restcells_t``
   * - ``OrderParameterRecorder``
     - ``"entropy"``, ``"normalized_entropy"``, ``"polar_alignment"``,
       ``"mean_alignment"``
     - NoVE entropy and alignment time series
   * - ``FamilyPopulationRecorder``
     - ``"family_population"``
     - ``lgca.fam_pop_t`` for supported family-tracking runs
   * - ``ScalarTimeSeriesRecorder``
     - the names of its metrics
     - a CSV file
   * - ``FieldRecorder(["oxygen"])``
     - the field names, e.g. ``"oxygen"``
     - the history of a field, e.g. one updated by a ``pde`` operator

Recorded data of a model run
----------------------------

A model run with :func:`lgca.model.run_model` returns its recordings by name
in ``result.data``, each with the steps at which it was recorded:

.. code-block:: python

   result = run_model(spec, showprogress=False)
   print(list(result.data))               # what the run recorded
   population = result.data["population"]  # one value per recorded step
   steps = result.data.steps("population")

A name that was not recorded raises a ``KeyError`` that lists the recorded
names. The arrays are those on the model (``lgca.n_t``, ...) at the end of the
run; a later run of the same model does not change them.

Identity-based models without volume exclusion record their cells in a
compact form: ``NodeRecorder`` stores the label, node and channel of every
cell at every recorded time in ``lgca.cells_t``, and ``lgca.nodes_t`` builds
the familiar lists of labels per channel from it when it is first read.
Analyses of individual cells are faster on ``cells_t`` directly, e.g. the mean
trait of the cells at every recorded time:

.. code-block:: python

   kappa = np.asarray(lgca.props["kappa"])
   mean_kappa = [kappa[cells.label].mean() for cells in lgca.cells_t]
   density = np.bincount(lgca.cells_t[-1].index, minlength=lgca.cell_density[lgca.nonborder].size)

Dense and sparse recorder results
---------------------------------

The default schedule records every step, including step zero, and preserves
the familiar ``timesteps + 1`` leading dimension. A sparse schedule stores
only the selected samples. Each array recorder publishes the corresponding
local step vector:

.. list-table::
   :header-rows: 1

   * - Values
     - Steps
   * - ``nodes_t`` / ``cells_t``
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
row index as simulation time. Both direct ``animate_config``, ``animate_flux``,
``animate_flow`` and ``animate_density`` methods and the ``animate`` facade use
the paired recorded times, including nonuniform schedules. Explicit arrays use
``steps=...`` when supplied and dense indices otherwise. Renderers receive these
times explicitly and own their title artists.

Density ``channels=...`` selection uses node history and its ``nodes_steps``,
even if a differently sampled density history also exists. Missing node history
raises an actionable error. Explicit density arrays are already reduced: select
their channels before passing them. AnimationObserver captures selected density
channels directly from each observed state.

Square/hexagonal Matplotlib renderers are exercised with configuration, flux,
flow and density data across classical, identity and NoVE backends. Cubic and
Moore Mayavi configuration/density/flux animations use the same resolver and
label each frame with its recorded time. They open an interactive window and
return the Mayavi ``Animator``; pass ``show=False`` inside an application that
already runs a GUI event loop. With the ``plot3d`` extra installed, the test
suite renders every 3D plot offscreen. Linear models retain their space-time
plot methods.

On a compiled model, the operator clock continues across runs while observer
schedules and callbacks restart at local zero. ``result.metadata['runtime']``
records ``start_step``, ``end_step`` and ``sample_time_origin='local'``; the CLI
persists them in ``metadata.json``. Add ``start_step`` to a recorded step vector
to reconstruct cumulative time, and omit the duplicate shared endpoint when
joining consecutive histories. Copy each history before the next run replaces
the arrays. Result metadata is a snapshot and is not changed by later runs.
Direct runners and ``timeevo`` expose the same offset as
``lgca.recording_start_step`` (and the endpoint as ``recording_end_step``).

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

Figures, panels and animations
------------------------------

Each 1D and 2D plot opens a new figure, so consecutive plot calls in a script
do not draw over each other. An empty current figure is used instead, which
keeps ``plt.figure(figsize=...)`` followed by a plot call working. To combine
plots in one figure, pass the target axes with ``ax=``:

.. code-block:: python

   import matplotlib.pyplot as plt

   lgca.timeevo(timesteps=25, record=True, showprogress=False)

   fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
   lgca.plot_density(ax=left)
   lgca.plot_flux(ax=right)

With constrained layout, colour bars stay inside their panel.

Animations of recorded runs (``animate_density``, ``animate_flux``,
``animate_config`` and ``animate_flow``) play in a Jupyter notebook when the
call is the last line of a cell. ``save_path`` writes a movie file; GIF files
need no extra software, video formats such as ``.mp4`` need ffmpeg.
``save_kwargs`` passes options such as ``dpi`` to
:meth:`matplotlib.animation.Animation.save`:

.. code-block:: python

   lgca.animate_flux(save_path="flux.gif", save_kwargs={"dpi": 80})

Live animations (``live_animate_*``) simulate while they draw. They need an
interactive Matplotlib backend, such as a desktop window or ``%matplotlib widget``
with the ``ipympl`` package in JupyterLab.

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

Three-dimensional plots and movies
----------------------------------

Cubic and Moore lattices are drawn with Mayavi (``uv sync --extra plot3d``).
The same observers work for them. Snapshots that are saved and closed, and all
movies, are rendered offscreen, so a run does not open windows:

.. code-block:: python

   lgca = get_lgca(geometry="cubic", dims=20, interaction="go_or_grow", ve=False,
                   capacity=8, restchannels=1, density=0.5, seed=6)

   snapshots = PlotSnapshotObserver(
       kind="density_cubes",
       schedule=Schedule(steps={0, 50}),
       output_dir=Path("outputs/snapshots"),
   )
   movie = AnimationObserver(
       kind="density",
       schedule=Schedule(every=2),
       save_path=Path("outputs/density_3d.mp4"),
       save_kwargs={"fps": 10},
       smooth=1.0,
   )

   SimulationRunner(lgca, timesteps=50, observers=[snapshots, movie], showprogress=False).run()

After a run with recorded history, the animation methods can also write a
movie directly:

.. code-block:: python

   lgca.timeevo(timesteps=50, record=True, showprogress=False)
   lgca.animate_density(save_path="outputs/density_3d_direct.mp4")
   lgca.animate_flux(save_path="outputs/flux_3d.gif", size=(1000, 800))

Video files such as ``.mp4`` need ffmpeg; ``.gif`` files use Pillow.
``save_kwargs`` takes the options of Matplotlib's ``Animation.save`` (``fps``,
``writer``, ``bitrate``, ``codec``, ...). The resolution is the figure size in
pixels, set with ``size=(width, height)``. Without ``save_path``, an
AnimationObserver on a 3D model returns a Mayavi ``Animator`` that plays after
``mlab.show()``.

Density/species selection is shared between static plots and animations.
Square scalar fields use image artists, so large 2D plots do not create one
Python polygon per lattice site; hexagonal renderers retain geometry-aware
collections.
