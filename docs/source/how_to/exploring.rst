Exploring a model live
======================

Before measuring a model over many runs (:doc:`studying_a_model`), it helps
to watch it while changing a parameter: where does alignment set in, how
strong must chemotaxis be for aggregates to form? :func:`lgca.explore` runs a
model in a Jupyter notebook (JupyterLab, Colab or VS Code) with sliders for
the parameters you name, play and pause buttons, and a view that updates as
the model runs.

.. code-block:: python

   import lgca
   from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec
   from lgca.pipeline import InteractionPipelineSpec

   spec = ModelSpec(
       space=SpaceSpec(geometry="hex", dims=(50, 50)),
       state=StateSpec(density=0.5, restchannels=1),
       time=TimeSpec(seed=1),
       dynamics=InteractionPipelineSpec(operators=[
           {"name": "polar_alignment", "parameters": {"beta": 1.0}},
       ]),
   )

   explorer = lgca.explore(spec, {"beta": (0.0, 5.0), "density": (0.1, 3.0)})
   explorer  # the last line of a cell shows the explorer

**Play** runs the model frame by frame until **Pause**, **Step** runs one
frame, and **Reset** builds the model again from its initial state. The
*steps/frame* slider sets how many steps pass between two frames: one frame
takes about 0.1 s to draw, so more steps per frame show slow processes
faster. Only one explorer plays at a time; starting one pauses the others.

Controls
--------

``controls`` maps a name to a control. Names are those of
:func:`lgca.study.vary`: a full path such as
``"dynamics.operators[0].parameters.beta"``, or a short name such as
``"beta"`` when only one place of the model has it. The slider shows the
shortest name that tells the controls apart, e.g. ``chemotaxis.beta``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Control
     - Gives
   * - ``(min, max)`` or ``(min, max, step)``
     - a slider; of integers if all numbers are integers
   * - a list of values, e.g. ``["square", "hex"]``
     - a dropdown
   * - an ipywidgets widget, e.g. ``ipywidgets.FloatLogSlider(min=-3, max=0)``
     - that widget, e.g. for rates over several orders of magnitude

Every control starts at the model's value, or at the parameter's default if
the model leaves it out. What happens when a control changes depends on
where the value lives:

- **Parameters of the dynamics** (operators, reorientation terms, fields'
  equations) change the running model. The lattice keeps its cells, so you
  can watch the model move from one regime to another. A dotted line in the
  time series marks the change. A steady field is solved again at once.
- **Everything else** (``density``, ``dims``, ``restchannels``,
  ``geometry``, ``seed``) builds the model again from a new initial state,
  when the slider is released.

A change is first tried on a copy of the model for one step. If the model
rejects the value, e.g. a rate above 1, the message appears below the view
and the control goes back.

The view
--------

The dropdown *view* chooses what the lattice panel shows: the density
(``"density"``, and ``"density: species i"`` with several species), the flux
(``"flux"``, coloured by its direction) or any field of the model. Square
and hexagonal lattices show the current state. 1D lattices show a kymograph
of the last ``window`` steps (default 100), time running down, for every
view at once, so switching views keeps the history. 3D lattices are not
drawn.

The panel beside the lattice plots measures over time, evaluated after every
step:

.. code-block:: python

   import numpy as np

   def moving(lattice):
       """Cells in velocity channels."""
       nodes = lattice.nodes[lattice.nonborder]
       return nodes[..., :lattice.velocitychannels].sum()

   explorer = lgca.explore(spec, {"beta": (0.0, 5.0)}, view="flux",
                           measure={"moving cells": moving}, steps_per_frame=5)

``measure`` takes ``"population"`` (the default), the name of a field (its
mean over the lattice), a list of these, a mapping of labels to functions of
the LGCA object, or ``None`` for no time series. The explorer ignores the
model's observers and ``time.steps``: it runs until paused.

From exploring to measuring
---------------------------

``explorer.spec`` is the model with the current values of the controls, so
a setting found by exploring is ready for :func:`~lgca.model.run_model` or
:func:`~lgca.study.sweep`. The explorer also works without the buttons,
which is how a notebook can show a state reached after a change:

.. code-block:: python

   from lgca.model import run_model

   explorer.set(beta=3.0)  # as if the slider moved
   explorer.advance(50)  # 50 steps, then draw
   png = explorer.frame()  # the current frame as PNG, e.g. for IPython.display.Image
   explorer.close()  # stop the model and close its widgets

   result = run_model(explorer.spec, showprogress=False)  # the same model, from its initial state

Names that are not Python identifiers go in a mapping:
``explorer.set(**{"chemotaxis.beta": 3.0})``.
