Fields and multiscale models
============================

A field is a quantity with one value per node, such as the concentration of
oxygen, a nutrient, a growth factor, a chemoattractant or a drug. Cells read
fields through cues (chemotaxis, switching probabilities, birth and death
rates). A field changes when the ``pde`` operator updates it: one LGCA step
of a reaction–advection–diffusion equation, in its listed place among the
cell operators, with the cells as sources and sinks. Tutorials 7 (growth
limited by oxygen) and 8 (aggregation toward a secreted signal) build
complete models; this page collects the options.

A field and its equation
------------------------

Declare the field in ``StateSpec.fields``, with an array of one value per
node or a number for a uniform start, and list a ``pde`` operator that
updates it:

.. code-block:: python

   from lgca.fields import PDESpec
   from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
   from lgca.pipeline import InteractionPipelineSpec
   from lgca.simulation import FieldRecorder

   spec = ModelSpec(
       space=SpaceSpec(geometry="square", dims=(40, 40), boundary="reflecting"),
       state=StateSpec(density=0.3, restchannels=1, fields={"signal": 0.0}),
       time=TimeSpec(steps=20, seed=1),
       dynamics=InteractionPipelineSpec(operators=[
           PDESpec(field="signal", diffusion=1.0, decay=0.05, cells=[{"production": 0.1}]),
           {"name": "chemotaxis", "parameters": {"field": "signal", "beta": 2.0}},
       ]),
       analysis=AnalysisSpec(observers=[FieldRecorder(["signal"])]),
   )
   result = run_model(spec, showprogress=False)
   result.data["signal"].shape  # (21, 40, 40): the field at every recorded step

``PDESpec(...)`` and ``{"name": "pde", "parameters": {...}}`` are the same
operator. The equation is

.. math::

   \partial_t c = D\,\Delta c - \nabla\cdot(v\,c) + P - L\,c,

with these parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Meaning
     - Default
   * - ``field``
     - the name of the field in ``StateSpec.fields``
     - required
   * - ``diffusion``
     - :math:`D`, in nodes² per step
     - 0
   * - ``decay``
     - a loss rate per step everywhere
     - 0
   * - ``production``
     - :math:`P` independent of the cells: a number, or the name of a field
       (e.g. a map of vessels)
     - 0
   * - ``cells``
     - secretion and uptake by the cells (below)
     - none
   * - ``reactions``
     - reactions of your own (below)
     - none
   * - ``advection``
     - the velocity :math:`v`: a vector, or the name of a field of vectors
     - none
   * - ``boundary``
     - what lies beyond the edge of the lattice (below)
     - ``"no_flux"`` or ``"periodic"``
   * - ``solver``
     - ``"implicit"``, ``"explicit"`` or ``"steady"`` (below)
     - ``"implicit"``

The Laplacian uses the neighbourhood of the cells, the velocity channels of
the lattice, and is exact for quadratic functions on every lattice; it is
available as :func:`lgca.fields.laplacian`. The operator changes no cells, and
the operators after it read the new values. Several fields need one operator
each.

Units
-----

Parameters are in lattice units: lengths in nodes, times in LGCA steps. A
diffusion coefficient :math:`D` in µm²/s becomes :math:`D\,\tau/\varepsilon^2`
for nodes of size :math:`\varepsilon` and steps of duration :math:`\tau`, and a
rate :math:`k` per second becomes :math:`k\,\tau`. Oxygen in tissue
(:math:`D \approx 2\cdot 10^3` µm²/s) with nodes of 20 µm and one step per hour
has :math:`D \approx 1.8\cdot 10^4` nodes² per step. A field whose diffusion
over one step reaches far beyond the colony is at a steady state with the
cells for all practical purposes; use the steady solver for it.

Cells as sources and sinks
--------------------------

Each entry of ``cells`` is one term:

- ``{"production": r}``: every cell adds ``r`` per step (secretion);
- ``{"uptake": r}``: every cell removes ``r c`` per step, so a node with
  ``n`` cells has the loss rate ``r n``;
- ``{"uptake": r, "saturation": K, "n": 1}``: saturating uptake,
  ``r cⁿ / (Kⁿ + cⁿ)`` per cell (Michaelis–Menten for ``n = 1``), at most
  ``r``.

A term can be limited to some cells with ``"species"`` (an index or a list)
and ``"channels"`` (``"all"``, ``"rest"``, ``"velocity"``), e.g. only resting
cells consume. In identity-based models ``r`` may name a cell trait, so every
cell secretes or consumes at its own rate:

.. code-block:: python

   consumers = PDESpec(field="oxygen", diffusion=100.0,
                       cells=[{"uptake": "consumption", "saturation": 0.05, "channels": "rest"}],
                       boundary={"value": 1.0}, solver="steady")

Boundaries
----------

- ``"periodic"``: the default, and the only choice, on periodic lattices.
- ``"no_flux"``: nothing crosses the edge; the default on other lattices.
- ``{"value": c}``: the field is ``c`` beyond the edge, e.g. oxygen supplied
  by the surrounding tissue.
- One condition per side on 1D, square and cubic lattices, e.g. a gradient
  from left to right: ``{"x-": {"value": 1.0}, "x+": {"value": 0.0},
  "default": "no_flux"}``.

The boundary of the field is independent of that of the cells, except that a
periodic lattice needs a periodic field. The model stores the field with its
ghost nodes filled by the condition, so gradients at the edge are right.

Choosing a solver
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - ``solver``
     - What it does
     - Use it for
   * - ``"implicit"``
     - one backward Euler step (``solver_options={"substeps": 4}`` for more);
       stable for any :math:`D`, keeps the field non-negative, first order
       in time
     - fields that change on the time scale of the cells, e.g. a chemokine
   * - ``"explicit"``
     - :func:`scipy.integrate.solve_ivp` over the step (RK45 by default) with
       error control
     - slow fields, and accurate transients; it warns when a field is too
       fast for it
   * - ``"steady"``
     - solves :math:`D\,\Delta c - \nabla\cdot(v\,c) + P - L\,c = 0` at every
       step and when the model is built
     - fields much faster than the cells: oxygen, nutrients, growth factors

A steady state needs something that removes the field: decay, uptake, a
reaction's loss rate or a fixed value at the boundary; otherwise the operator
says so. The linear systems are solved by SciPy; the steady solver uses an
algebraic multigrid preconditioner (pyamg) and costs about as much as one
go-or-grow step on lattices of 100² to 400² nodes (``benchmarks/fields.py``).
``solver_options`` sets tolerances and the method; see
:class:`~lgca.fields.PDESpec`. After a run, ``result.metadata["fields"]``
reports per field how it was solved, e.g. the number of iterations.

Advection
---------

``advection=[vx, vy]`` carries the field with a flow of that velocity, in
nodes per step (Cartesian components, also on hexagonal lattices), e.g.
interstitial flow. The name of a field of shape ``dims + (d,)`` gives a
velocity per node, read at every step. The scheme takes the upwind value on
every face: it conserves the total and keeps the field non-negative, at the
price of a numerical diffusion of about ``|v| / 2``, so it is accurate where
:math:`D` is large against that.

.. code-block:: python

   downstream = PDESpec(field="signal", diffusion=1.0, decay=0.05, advection=[0.5, 0.0],
                        cells=[{"production": 0.1}])

Reactions of your own
---------------------

A reaction is a function of the lattice state and the field's values that
returns a production and a loss rate, both non-negative with one value per
node (or a number). Register it with :func:`lgca.reaction` and use it by name,
with its parameters:

.. code-block:: python

   import lgca


   @lgca.reaction
   def autocatalysis(state, c, rate=1.0, K=1.0):
       """Production that rises with the field and saturates; lost where cells are crowded."""
       return rate * c**2 / (K**2 + c**2), 0.01 * state.density


   pattern = PDESpec(field="signal", diffusion=1.0, decay=0.1,
                     reactions=[{"name": "autocatalysis", "rate": 0.5}])

``state.field("other")`` reads another field, so fields can react with each
other. They are updated one after another, in the order of their operators
(operator splitting, first order in time). Terms that depend on the field
are iterated to convergence in the implicit and steady solvers. A model file
that uses a reaction needs the module that registers it imported first:
``biolgca run model.json --plugins my_reactions``, or ``plugins=`` of
:func:`lgca.study.sweep`.

Cells that respond to fields
----------------------------

- ``chemotaxis`` moves cells up the gradient of a field, alone or as a term
  of a :class:`~lgca.pipeline.ReorientationSpec`.
- Switching probabilities respond to the ``field`` and ``gradient`` cues
  (:mod:`lgca.switching`). They set the rates of ``phenotype_switch``, the
  events of ``trait_switch`` and of mutations, the probability of resting of
  ``go_or_rest`` and ``resting``, and ``birth_rate`` and ``death_rate`` of
  ``birth_death``.
- The Hill form suits concentrations: division that needs oxygen and death
  that rises where it runs out,

.. code-block:: python

   growth = {"name": "birth_death", "parameters": {
       "birth_rate": {"max": 0.1, "hill": [{"name": "field", "field": "oxygen", "K": 0.3, "n": 2}]},
       "death_rate": {"max": 0.05, "hill": [{"name": "field", "field": "oxygen", "K": 0.05, "n": -4}]},
   }}

Recording and plotting
----------------------

``FieldRecorder(["oxygen"], schedule=Schedule(every=10))`` records fields;
``result.data["oxygen"]`` is the history (interior nodes) and
``result.data.steps("oxygen")`` its steps. ``biolgca run`` writes
``field_oxygen`` and ``field_oxygen_steps`` to ``measurements.npz``. The final
field is ``result.lgca.oxygen``, with its ghost nodes.

.. code-block:: python

   import matplotlib.pyplot as plt

   frames = result.data["signal"]
   result.lgca.plot_scalarfield(frames[-1])  # the last frame
   animation = result.lgca.animate_scalarfield(frames, steps=result.data.steps("signal"))
   plt.close("all")

On square and hexagonal lattices ``plot_scalarfield`` draws one frame and
``animate_scalarfield`` the history with one colour scale; on 1D lattices
``plot_scalarfield`` draws a history as a kymograph and one frame as a line.

Model files
-----------

A ``pde`` operator is written into model files like every other operator:

.. code-block:: json

   {"name": "pde", "parameters": {
       "field": "oxygen",
       "diffusion": 20000.0,
       "cells": [{"uptake": 20.0, "saturation": 0.05}],
       "boundary": {"value": 1.0},
       "solver": "steady"
   }}

:func:`~lgca.model.describe_model_graph` shows which operators write and read
each field. :func:`lgca.study.vary` reaches its parameters by path, e.g.
``"dynamics.operators[pde].parameters.cells[0].uptake"``, or by a short name
such as ``"decay"``.
