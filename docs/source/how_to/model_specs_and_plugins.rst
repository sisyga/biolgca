Model specs and interaction plugins
===================================

The :func:`lgca.get_lgca` API remains supported. New code can also describe
a complete simulation with :class:`lgca.model.ModelSpec`, which keeps the
model setup, the dynamics, logging and plotting apart. The design is inspired
by the model declarations of Morpheus: the dynamics are named plugins with
metadata, validation and an explicit order of execution, and propagation
remains the deterministic phase in which cells move on the lattice.

Kinds of interaction and their order
------------------------------------

Every interaction is one of three kinds, defined by what it conserves:

``birth_death``
   changes the number of cells (birth, death, division).
``phenotype_switch``
   changes what a cell is. In classical models a phenotype is a species, and a
   switch moves cells from one species to another, keeping the number of cells
   at each node. A classical model with one species has nothing to switch to,
   so a phenotype switch there is rejected. In identity-based models, a switch
   changes a cell's own parameters.
``reorientation``
   rearranges the cells of a node over its channels and keeps the number of
   cells of each species at the node. Boltzmann sampling of combined scores
   (:class:`~lgca.pipeline.ReorientationSpec`) is the usual way to write one,
   but any rule that keeps these numbers is a reorientation, including moving
   cells between velocity and rest channels (``go_or_rest``).

A time step applies the operators in the order they are listed, then moves the
cells (propagation). The order is part of the model: division before
reorientation is a different model from reorientation before division. Two
reorientation operators in a row are two independent random decisions; cues
that should compete in one decision belong as terms of one
``ReorientationSpec``. The run metadata records the schedule as
``result.metadata["schedule"]``.

Minimal ModelSpec
-----------------

.. code-block:: python

   from lgca.model import (
       AnalysisSpec,
       Description,
       ModelSpec,
       SpaceSpec,
       StateSpec,
       TimeSpec,
       run_model,
   )
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, NodeRecorder

   spec = ModelSpec(
       description=Description(title="native random-walk model"),
       space=SpaceSpec(geometry="hex", dims=(20, 20), boundary="reflecting"),
       state=StateSpec(density=0.2, restchannels=0),
       time=TimeSpec(steps=100, seed=1),
       dynamics=InteractionPipelineSpec(
           operators=[{"name": "random_walk"}],
       ),
       analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   print(result.metadata["schedule"])
   result.lgca.plot_density()

The result exposes the final LGCA object, the compiled pipeline, metadata and
the recorded data by name: ``result.data["nodes"]`` from ``NodeRecorder`` and
``result.data["density"]`` from ``DensityRecorder``, with
``result.data.steps("density")`` (see :doc:`observers_and_plotting`).

A model runs its own copy of the spec it was built from, ``result.spec`` (or
``model.spec`` of :func:`~lgca.model.build_model`): changing the dicts, lists
or arrays of ``spec`` afterwards, or building further models from it, does not
change a model already built. To run a changed spec, build a new model, e.g.
from :func:`lgca.study.vary` (the new spec shares the parts it does not change
with the old one: change it with ``vary`` rather than in place). Operator
objects in the spec are templates too; the model's own copies are
``result.pipeline.operators`` (see :doc:`custom_interactions`). Observers,
rules and functions are shared: the observers record the run for you. The
model keeps its copy as long as it exists, so initial nodes, traits and fields
given as arrays take their memory twice, in ``result.spec`` and in the lattice.

Portable model files
--------------------

JSON is the canonical portable ModelSpec v1 representation. YAML is an
optional authoring syntax for the same data model; install it with
``pip install biolgca[yaml]``. Both formats contain only data:

* registered interactions use ``name`` and ``parameters``;
* composed reorientation uses the stable ``type: reorientation`` form;
* analysis contains only the documented built-in observer types;
* an optional ``state.initializer`` contains a registered initializer name and
  parameters.

Every portable file carries the top-level ``schema_version``, currently
``1``. A file of an unknown, later version is rejected rather than guessed at;
a file without a version, from earlier releases, is read as version 1 before
it is validated. For reproducibility, save and archive the canonical JSON;
YAML is a convenience and gives the same ModelSpec.

Model files never contain Python import paths and never execute modules. A
model file may name a rule of your own once a trusted Python script has
imported and registered it; loaded on its own, a model file finds only the
rules that come with BioLGCA. Observers written in Python stay in Python:
saving one raises an error that says so.

.. code-block:: python

   from lgca.model import load_model_spec, run_model, save_model_spec

   save_model_spec(spec, "model.json")
   loaded = load_model_spec("model.json")
   result = run_model(loaded, showprogress=False)

Arrays with more than 100 elements, such as initial nodes or fields, go to
an array file next to the model file: ``save_model_spec(spec, "model.json")``
also writes ``model.arrays.npz``, and the model file refers to its arrays by
name. Keep the two files together; ``load_model_spec("model.json")`` reads the
arrays from the file next to it. ``save_model_spec(...,
max_inline_array=None)`` writes every array into the model file. Tuples are
written as lists.

Model files that use your own rules run from the command line with
``--plugins``, which imports trusted modules before the model is built:

.. code-block:: console

   $ biolgca run model.json --plugins my_project.rules --output run/

The packaged ``model-spec-v1.schema.json`` is for editors and for validation
during development. Loading a model uses BioLGCA's own strict parser, so only
contributors and tools need ``jsonschema``.

Initial conditions
------------------

:class:`~lgca.model.StateSpec` takes at most one of three initial
conditions; without any, cells are placed at random with a mean of 0.1 cells
per node.

``density``
   The mean number of cells per node, everywhere. With volume exclusion each
   channel is occupied with probability ``density / K``; without it, channel
   populations are Poisson distributed.

``initializer``
   A named initial condition (:func:`lgca.initializers.list_initializers`):

   - ``{"name": "region", "parameters": {"extent": 5, "density": 4}}`` fills
     a block of ``extent`` nodes per axis (one number, or one per axis) with
     ``density`` cells per node (default 1) and leaves the rest empty.
     ``"placement"`` puts the block in the ``"center"`` (default), at the
     ``"left"`` edge of the first axis, or in the ``"corner"`` at the
     origin. The region fills all channels of all species at random; for
     cells in particular channels or of one species, give ``nodes``.
   - ``{"name": "from_npz", "parameters": {"path": "state.npz", "key":
     "nodes"}}`` reads a ``nodes`` array (see below) from an NPZ file, found
     relative to the model file. Not yet for identity-based models.

``nodes``
   The channels of every node, without ghost nodes:

   .. list-table::
      :header-rows: 1
      :widths: 30 30 40

      * - Model
        - Shape
        - Values
      * - classical, volume exclusion
        - ``dims + (K,)``; ``dims + (n_species, K)`` for several species
        - 0 or 1 per channel
      * - classical, no volume exclusion
        - as above
        - cells per channel
      * - identity-based, volume exclusion
        - ``dims + (K,)``
        - the label of the cell, 0 for an empty channel; labels start at 1
      * - identity-based, no volume exclusion
        - ``dims + (K,)``, one rest channel
        - cells per channel; the cells get labels from 0

   ``K`` is the number of velocity channels of the geometry plus
   ``restchannels``; the velocity channels come first. In the two-species
   go-or-grow model, for example, resting cells are species 1 in the rest
   channels:

   .. code-block:: python

      import numpy as np
      from lgca.model import StateSpec

      nodes = np.zeros((60, 60, 2, 12), dtype=int)  # hex: 6 velocity + 6 rest channels
      nodes[30, 30, 1, 6:] = 1                       # one node of resting cells
      StateSpec(nodes=nodes, restchannels=6, n_species=2)

``traits`` gives the initial cells of identity-based models their traits, one
number for all or one per cell in the order of their labels. The two
identity-based families differ here: with volume exclusion
label 0 means "empty" and ``lgca.props`` holds an unused entry for it, while
without volume exclusion label 0 is a real cell. Select living cells by the
labels found on the lattice, not by ``label > 0``. The
:class:`~lgca.simulation.NodeRecorder` records the labels in every channel
(with volume exclusion an integer array, without it a list of labels per
channel); the traits of the living cells follow as
``lgca.props[name][labels]``.

Configuration, outputs and checkpoints
--------------------------------------

A ModelSpec is configuration, not a serialized simulator or trajectory. Keep
it small enough to review and to keep under version control. Initial channel
states can live in an NPZ file next to it, read by the ``from_npz``
initializer. Such paths are relative to the model file and cannot leave its
directory unless trusted paths are enabled explicitly.

``biolgca run`` writes ``model.resolved.json``, ``metadata.json`` and the
outputs of the observers into the run directory you name. The resolved model
is the configuration that was compiled; the metadata hold the provenance of
the run and the resolved schedule of operators. These files are not a
checkpoint to restart from: in particular, ``from_npz`` cannot yet read the
cell traits of identity-based models, and the labels alone do not give them
back.

Composed dynamics
-----------------

Composed pipelines make the conservation laws visible. The following example
uses a multispecies volume-exclusion model with cell turnover, phenotype
switching and a reorientation sampler that combines nematic alignment and
chemotaxis.

.. code-block:: python

   import numpy as np

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import (
       InteractionPipelineSpec,
       PhenotypeSwitchSpec,
       ReorientationSpec,
       ReorientationTermSpec,
   )
   from lgca.simulation import DensityRecorder

   signal = np.linspace(0.0, 1.0, 30)[:, None] + np.zeros((30, 30))

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(30, 30), boundary="reflecting"),
       state=StateSpec(
           density=0.25,
           restchannels=1,
           n_species=2,
           fields={"signal": signal},
       ),
       time=TimeSpec(steps=50, seed=3),
       dynamics=InteractionPipelineSpec(
           operators=[
               {
                   "name": "birth_death",
                   "parameters": {
                       "birth_rate": [0.01, 0.005],
                       "death_rate": [0.002, 0.002],
                   },
               },
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
       analysis=AnalysisSpec(observers=[DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)

Identity-based family dynamics
------------------------------

Identity-based models give every cell its own traits, which daughters
inherit and mutations change; mutated daughters can found families for
lineage analyses. For example, in the glioblastoma model driver mutations found
clones with a higher birth rate and a new switch steepness
(see :doc:`research_models`):

.. code-block:: python

   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import DensityRecorder, NodeRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(40, 40), boundary="periodic"),
       state=StateSpec(
           density=0.6,
           restchannels=1,
           volume_exclusion=False,
           identity_based=True,
           capacity=8,
       ),
       time=TimeSpec(steps=100, seed=5),
       dynamics=InteractionPipelineSpec(
           operators=[
               {
                   "name": "go_or_grow_glioblastoma",
                   "parameters": {
                       "r_b": 0.2,
                       "r_d": 0.01,
                       "r_m": 0.001,
                       "fitness_increase": 1.1,
                       "kappa": 5.0,
                       "theta": 0.5,
                   },
               }
           ],
       ),
       analysis=AnalysisSpec(observers=[NodeRecorder(), DensityRecorder()]),
   )

   result = run_model(spec, showprogress=False)
   birth_rates = result.lgca.props["r_b"]  # one value per cell label
   families = result.lgca.props["family"]

Growth rules of identity-based models combine freely: every daughter inherits
all traits of its mother, whichever rule made it.

Plugin registry
---------------

The registry provides the machine-readable interaction catalogue: names,
parameters with defaults, the pipeline phase and the conservation law of every
registered interaction.

.. code-block:: python

   from lgca.plugins import describe_plugin, interaction_coverage_table, list_plugins

   for plugin in list_plugins(kind="interaction"):
       print(plugin.name, plugin.operator_kind, plugin.description)

   info = describe_plugin("go_or_grow_kappa_chemo")
   print(info.parameters)

   rows = interaction_coverage_table()

Use the plugin metadata to build interfaces, validation reports or
provenance tables.

:doc:`custom_interactions` explains how to write and register rules, with a
complete example and the conservation laws they keep. Model files name
registered rules only; importing your own Python modules is left to a trusted
launcher, e.g. ``biolgca run --plugins my_project.rules``.

CLI measurement files
---------------------

``biolgca run`` saves the numeric recordings in ``measurements.npz`` next to
``metadata.json``, under the names of the Python attributes: ``nodes_t``,
``dens_t``, ``n_t``, ``channel_pop_t``, ``velcells_t``, ``restcells_t``,
``fam_pop_t``, and the four order-parameter arrays. Each comes with its
``*_steps`` array (the order parameters share ``order_parameter_steps``).
CSV snapshot and scalar observers write their own CSV files. Node histories
of identity-based models without volume exclusion hold Python lists, so the
command line rejects them before the run; record the channel density or the
density instead.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   with np.load("run/measurements.npz", allow_pickle=False) as data:
       plt.plot(data["n_steps"], data["n_t"])
   plt.xlabel("Simulation step")
   plt.ylabel("Population")

In 1D history plots of sparse recordings, each row is labelled with its time
step, and the distance between rows counts samples, not time. Property and
family histories and ``lgca.plotting.animate(..., steps=...)`` use the
recorded time steps. History data passed without ``steps`` is taken to be
recorded at every step.

The command line archives the model as ``model.resolved.json``, with relative
observer paths, and its large arrays in ``model.resolved.arrays.npz``. It
copies the NPZ input of an initializer to ``resources/initial_state.npz``
before the run, runs from the copy, and points the archived model to it. Move
the run directory as a whole; its model can then be validated or rerun into a
new output directory without ``--trusted-paths``.

``metadata["provenance"]`` of every model (and ``metadata.json``) records where
a result came from: the versions of Python and the packages, the platform, the
operators with their parameters including defaults, the SHA-256 hashes of the
source code of every rule, term and reaction the model runs (rules change more
often than versions), and of the files it read. The command line adds the hash
of the model file, the git commit of BioLGCA when it runs from a checkout, and
the hashes of the archived files. ``biolgca run`` and ``biolgca validate`` warn
when an archived model has changed since the run that wrote it, as its results
then belong to another model. Likewise a model file records a hash of every array
in its array file and warns when it is loaded after the array file changed,
e.g. because a ``model.yaml`` saved next to ``model.json`` wrote its own
arrays to the shared ``model.arrays.npz``.

A model returned by ``build_model`` or ``run_model`` keeps its compiled
dynamics: ``lgca.timestep()``, ``lgca.timeevo(...)`` and live animations
advance the same pipeline and keep its random state and the accumulated
timings of its operators. ``compiled.step()`` advances one step. Each run that
records starts the schedule of its observers at step zero again and replaces
the history arrays of the previous run. The metadata of the result (and of the
command line) record the steps the run covered as ``runtime.start_step`` and
``runtime.end_step``, with ``sample_time_origin='local'``. To join continued
runs, add the start step to each recorder's step array and drop the endpoint
the two runs share. The metadata of a result do not change when the model runs
on. Runners used directly set ``lgca.recording_start_step`` and
``lgca.recording_end_step`` for the same purpose.

Between runs, ``compiled.reconfigure({"birth_rate": 0.3})`` changes the
dynamics of the running model, e.g. a rate after a burn-in, as the sliders of
:func:`lgca.explore` do; paths are those of :func:`lgca.study.vary`. The change
is tried on a copy of the model first and is applied only if that works.
``metadata["reconfigurations"]`` records the step and the old and new values,
and ``compiled.initial_spec`` keeps the spec the model was built from, so the
run can be repeated. A step that raises is rolled back: the model is then as
before the step and can go on, e.g. after a reconfiguration.

``state.capacity`` supplies the crowding scale of the rules: the carrying
capacity of models without volume exclusion and, with volume exclusion, an
optional soft limit on all cells of a node in addition to the channels. An
operator capacity that conflicts with it is rejected. With volume exclusion,
a node still holds at most ``K`` cells per species. The metadata record
the number of channels as ``channel_capacity``, apart from the capacities of
the growth operators, ``growth_capacities``; with one ``birth_death``
operator, ``capacity`` is the one it uses.

Recording and temporary-memory budgets
--------------------------------------

``lgca.simulation.estimate_recording_bytes(lgca, timesteps, observers)``
estimates the memory the recordings need (their numeric buffers and arrays
of sample times) before any of it is allocated. A run whose estimate exceeds
512 MiB is rejected before its observers are set up. To fit, shorten the run,
record less often (``Schedule(every=...)`` or explicit sample steps), choose a
smaller dtype for densities, or write snapshots to files with
``CSVSnapshotObserver``; no sample is dropped silently. To allow more
memory on purpose, pass ``compiled.run(max_recording_bytes=your_budget)`` or
the same keyword to ``SimulationRunner``; ``None`` turns the check off.

The estimate is a lower bound: Python maps of sample indices, object and list
payloads, family histories that grow during the run, the model state and the
buffers of renderers cost extra.
