Beginner example gallery
========================

The fastest way to learn BioLGCA is to start from a small model that already
runs, then change one parameter at a time. This page collects the curated
``ModelSpec`` examples exposed by :mod:`lgca.examples`.

The layout is inspired by the old :download:`BioLGCA.ipynb <../../BioLGCA.ipynb>`
tour and by the Morpheus example gallery: examples are grouped by modeling
question rather than by Python class.

Run a gallery example
---------------------

Use :func:`lgca.examples.run_example` when you want a complete runnable model:

.. code-block:: python

   from lgca.examples import run_example

   result = run_example("random_walk", steps=10, showprogress=False)
   result.lgca.plot_density()

Use :func:`lgca.examples.example_gallery` to browse model cards:

.. code-block:: python

   from lgca.examples import example_gallery

   for card in example_gallery():
       print(card.name, "-", card.question)

Use :func:`lgca.examples.save_example_spec` when you want a JSON or YAML config
that can be shared, edited and loaded again:

.. code-block:: python

   from lgca.examples import save_example_spec
   from lgca.model import load_model_spec, run_model

   path = save_example_spec("chemotaxis", "chemotaxis.json")
   spec = load_model_spec(path)
   result = run_model(spec, showprogress=False)

Model cards
-----------

random_walk
~~~~~~~~~~~

:Category: movement
:Question: How does unbiased cell movement spread a population?
:Concepts: movement, diffusion, density

Start here if students are new to LGCA. The model has no directed cue and is
useful for checking how density spreads under unbiased reorientation and
propagation.

alignment
~~~~~~~~~

:Category: collective motion
:Question: How do local alignment rules create coherent streams?
:Concepts: collective motion, flux, reorientation

This mirrors the alignment example from ``BioLGCA.ipynb``. After running it,
plot flux or density and vary the alignment strength in the model spec.

chemotaxis
~~~~~~~~~~

:Category: guidance
:Question: How does a signal field bias cell movement?
:Concepts: signal field, gradient sensing, reorientation

This is the closest BioLGCA gallery entry to the Morpheus chemotaxis examples:
the model includes a static signal field and a reorientation term that reads
that field.

multispecies_birth_death
~~~~~~~~~~~~~~~~~~~~~~~~

:Category: population dynamics
:Question: How do different birth and death rates change competing populations?
:Concepts: multispecies, birth, death

Use this example to discuss how rates become model assumptions. It is small
enough for tests but exposes a two-species ``birth_death`` operator that can be
edited directly in the saved model spec.

identity_tumor_growth
~~~~~~~~~~~~~~~~~~~~~

:Category: tumor growth
:Question: How can individual cell properties drive go-or-grow tumor expansion?
:Concepts: identity-based LGCA, go-or-grow, tumor growth

This example connects to the go-and-grow and go-or-grow sections in
``BioLGCA.ipynb``. It uses an identity-based NoVE model so students can inspect
how individual properties and population-level density change together.

Suggested classroom workflow
----------------------------

1. Run ``run_example("random_walk", steps=10, showprogress=False)``.
2. Save the same example with ``save_example_spec`` and inspect the JSON file.
3. Change one parameter, seed or lattice size.
4. Re-run and compare density, population or exported CSV observer output.
5. Move to ``alignment`` or ``chemotaxis`` once the setup/run/inspect loop is
   familiar.
