Planned documentation topics
============================

This page records intentionally deferred work. It distinguishes supported
release behavior from ideas that still need an implementation and tests.

Custom interactions
-------------------

The supported callable and registered-plugin paths are described in
:doc:`custom_interactions`, :doc:`interactions_summary` and
:doc:`model_specs_and_plugins`. A future packaging cookbook may add an
end-to-end external distribution example. Arbitrary imports from model files
remain intentionally unsupported.

Boundary conditions
-------------------

Periodic, reflecting and absorbing behavior is covered by the core invariant
suite. A concise user-facing matrix for periodic, reflecting, absorbing and
inflow support by geometry and model family is still needed.

Plotting cookbook
-----------------

The observer and plotting dispatch APIs are described in
:doc:`observers_and_plotting`. Square scalar fields use image artists and
multi-species density selection is shared across static and animated paths.
Remaining work includes consistent absolute/relative normalization, scalable
colorbar tick locators, a documented slicing matrix, 3-D renderer lifecycle
support, property/family plots and publication styling.

Custom initial conditions
-------------------------

Named region and numeric NPZ initializers are documented in
:doc:`getting_started`. A complete advanced guide should still describe raw
``nodes`` shapes for classical, identity-based, NoVE and multi-species models
in each geometry. Identity-based checkpoint loading remains unsupported until
particle properties and labels can be restored atomically.

Runtime control and parameter studies
-------------------------------------

ModelSpec plus explicit run directories provide the stable building blocks for
future parameter scans, but BioLGCA does not yet provide a scan scheduler.
Pause/resume controls, live parameter editing and a second live custom-quantity
window are separate future UI/runtime projects; they are not part of the
portable configuration contract.

Scientific observables and family workflows
-------------------------------------------

Family population and ancestry utilities are available, but a focused
observables API for diversity indices and entropy is still planned. Remaining
identity-based work also includes validated family-aware initialization,
stopping conditions and additional family visualizations.
