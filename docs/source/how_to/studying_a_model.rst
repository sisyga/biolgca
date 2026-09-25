Studying a model
================

A model is studied by running it for many parameter values and seeds and
comparing a measurement across the runs. :mod:`lgca.study` does this with two
functions: :func:`~lgca.study.vary` changes values of a
:class:`~lgca.model.ModelSpec`, and :func:`~lgca.study.sweep` runs every
combination of values and seeds and returns a :class:`pandas.DataFrame` with one
row per run.

Changing a model by name
------------------------

:func:`~lgca.study.vary` returns a copy of a model with some values replaced.
Each value is named by its path in the model:

.. code-block:: python

   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.study import sweep, vary

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(40, 40)),
       state=StateSpec(density=0.2, restchannels=6),
       time=TimeSpec(steps=100, seed=1),
       dynamics=InteractionPipelineSpec(operators=[
           {"name": "go_or_rest", "parameters": {"kappa": 5.0}},
           {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2}},
           {"name": "random_walk", "parameters": {"channels": "velocity"}},
       ]),
   )

   variant = vary(spec, {
       "time.steps": 200,
       "dynamics.operators[0].parameters.kappa": -2.0,
   })

A path is a sequence of names separated by dots, with ``[i]`` for the
``i``-th entry of a list. Three shortcuts keep paths short:

- ``[name]`` selects the entry with that name: ``dynamics.operators[go_or_rest]``,
  or ``dynamics.operators[0].terms[chemotaxis]`` in a ``ReorientationSpec``;
- the parameters of an operator or term can be named directly:
  ``dynamics.operators[go_or_rest].kappa``;
- a single name such as ``"kappa"`` or ``"steps"`` stands for the one place
  of the model with that name, among the space, state and time and the
  parameters of all operators and terms, including parameters left at their
  defaults. If the name occurs in several places, e.g. ``"density"`` for the
  initial density and the density cue of ``go_or_rest``, the error lists
  their paths.

Sweeps
------

:func:`~lgca.study.sweep` runs the model for every combination of the values
in ``grid``, once per seed, and measures every run:

.. code-block:: python

   from lgca.study import final_population

   table = sweep(
       spec,
       grid={"kappa": [-4, 0, 4], "r_b": [0.1, 0.2]},
       seeds=range(10),
       measure={"population": final_population},
   )
   table.groupby(["kappa", "r_b"]).population.agg(["mean", "sem"])

The table has a column per varied value, named by its last name (``kappa``)
or, if two share one, by the shortest end of the path that tells them apart
(``chemotaxis.beta``); then ``seed`` and a column per measure. The rows are in
the order of the grid, with the seeds innermost, and every run uses exactly the
seed in its row, so a sweep can be repeated and extended. ``table.attrs`` holds
the full paths and the BioLGCA version. ``grid`` may also be a list of
combinations, e.g. ``[{"kappa": 2, "r_b": 0.1}, {"kappa": 4, "r_b": 0.3}]``.

A measure is a function of the run's :class:`~lgca.model.ModelRunResult` that
returns a number, an array or a :class:`pandas.Series` indexed by step, or the
name of a recording in ``result.data`` (see :doc:`observers_and_plotting`),
e.g. ``"population"``, which measures its whole time series. A named
recording is added to the model if it has no such recorder.

Time series in long tables
--------------------------

With ``long=True`` the table has one row per run and recorded step, the form
that plotting libraries expect for curves with error bands:

.. code-block:: python

   long = sweep(spec, grid={"kappa": [-4, 4]}, seeds=range(10),
                measure={"population": "population"}, long=True)
   curves = long.groupby(["kappa", "step"]).population.agg(["mean", "sem"])

Time series from several measures are aligned on the column ``step`` (steps
missing in one of them are empty), and numbers are repeated on every row of
their run. Without ``long`` a time series is stored as one array per row.

Running in parallel
-------------------

``n_jobs`` sets how many runs happen at the same time. The results do not
depend on it: every run has its own seed and its own copy of the model's
observers.

- ``backend="processes"`` (the default) runs in worker processes, one run per
  processor core at a time; on 8 cores, a sweep runs about six times faster.
  The workers start fresh and import what they need, on every operating
  system: measures must be functions defined with ``def`` at the top level of
  a module (not lambdas or functions defined in a notebook), and your own
  rules must be registered in a module that the workers import, named with
  ``plugins=["my_project.rules"]``.
- ``backend="threads"`` accepts any function and the rules defined in a
  notebook, but threads run only partly in parallel (about 1.5 to 2 times
  faster).

Because the workers import the script that started the sweep, a script must
run the sweep under a main guard, or every worker would start the sweep again:

.. code-block:: python

   if __name__ == "__main__":
       table = sweep(spec, grid={"kappa": [-4, 0, 4]}, seeds=range(10), n_jobs=4)
       table.to_csv("runs.csv")

Files that observers would write, CSV snapshots and time series, are
discarded in a sweep; measure what you need instead.

From the command line
---------------------

The ``biolgca`` command sweeps a model file into a directory with
``table.csv`` and ``sweep.json`` (the model, the grid, the seeds and the
version):

.. code-block:: console

   $ biolgca sweep model.json --vary kappa=-4,0,4 --vary r_b=0.1,0.2 --seeds 0:10 \
         --measure population --long --n-jobs 4 --output runs/

``--vary`` takes a path or a short name and comma-separated values (numbers,
``true``/``false``, or words); ``--seeds`` a range ``0:10`` or a list
``1,2,3``; ``--measure`` names recordings, and without it the table has the
population at the end of every run. ``--plugins my_project.rules`` imports
trusted modules with your own rules first; ``biolgca run`` and
``biolgca validate`` take it as well. Modules are never imported because a
model file names them.
