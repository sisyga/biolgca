Beginner example gallery
========================

The fastest way to learn BioLGCA is to start from a small model that already
runs, read the whole model setup, then change one parameter at a time. This
page collects the curated ``ModelSpec`` examples exposed by :mod:`lgca.examples`.

The layout is inspired by the old :download:`BioLGCA.ipynb <../../BioLGCA.ipynb>`
tour and by the Morpheus example gallery: each example has its own source file,
grouped by modeling question rather than by Python class.

Study one example file
----------------------

Each file under ``lgca/examples`` is intended to be readable as a small lesson:
it declares metadata, builds a ``ModelSpec`` in ``build_spec``, and runs it with
``run_model``. The package-level helpers are still useful for tests and scripts,
but the source files are the best place for students to start.

Use :func:`lgca.examples.example_gallery` to browse model cards:

.. code-block:: python

   from lgca.examples import example_gallery

   for card in example_gallery():
       print(card.name, card.source_path, "-", card.question)

Open one file, build its spec, then run it:

.. code-block:: python

   from lgca.examples.chemotaxis import build_spec
   from lgca.model import run_model

   spec = build_spec()
   result = run_model(spec, showprogress=False)
   result.lgca.plot_density()

From the repository root, each teaching file can also be run as a module:

.. code-block:: console

   python -m lgca.examples.random_walk

Use :func:`lgca.examples.save_example_spec` when you want a JSON or YAML config
that can be shared, edited and loaded again:

.. code-block:: python

   from lgca.examples import save_example_spec
   from lgca.model import load_model_spec, run_model

   path = save_example_spec("chemotaxis", "chemotaxis.json")
   spec = load_model_spec(path)
   result = run_model(spec, showprogress=False)

Use :func:`lgca.examples.run_example` only when you want a quick smoke run by
stable name:

.. code-block:: python

   from lgca.examples import run_example

   result = run_example("random_walk", steps=10, showprogress=False)

Model cards
-----------

random_walk
~~~~~~~~~~~

:Category: movement
:Question: How does unbiased cell movement spread a population?
:Concepts: movement, diffusion, density
:Source: ``lgca/examples/random_walk.py``

Start here if students are new to LGCA. The model has no directed cue and is
useful for checking how density spreads under unbiased reorientation and
propagation.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Random walk example
   steps: 1
   operators: classical.random_walk
   observers: DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/random_walk.py
   :language: python
   :caption: lgca/examples/random_walk.py

alignment
~~~~~~~~~

:Category: collective motion
:Question: How do local alignment rules create coherent streams?
:Concepts: collective motion, flux, reorientation
:Source: ``lgca/examples/alignment.py``

This mirrors the alignment example from ``BioLGCA.ipynb``. After running it,
plot flux or density and vary the alignment strength in the model spec.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Alignment example
   steps: 1
   operators: classical.alignment
   observers: DensityRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/alignment.py
   :language: python
   :caption: lgca/examples/alignment.py

chemotaxis
~~~~~~~~~~

:Category: guidance
:Question: How does a signal field bias cell movement?
:Concepts: signal field, gradient sensing, reorientation
:Source: ``lgca/examples/chemotaxis.py``

This is the closest BioLGCA gallery entry to the Morpheus chemotaxis examples:
the model includes a static signal field and a reorientation term that reads
that field.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Chemotaxis example
   steps: 1
   operators: reorientation.boltzmann
   observers: DensityRecorder
   fields: signal

.. literalinclude:: ../../lgca/examples/chemotaxis.py
   :language: python
   :caption: lgca/examples/chemotaxis.py

multispecies_birth_death
~~~~~~~~~~~~~~~~~~~~~~~~

:Category: population dynamics
:Question: How do different birth and death rates change competing populations?
:Concepts: multispecies, birth, death
:Source: ``lgca/examples/multispecies_birth_death.py``

Use this example to discuss how rates become model assumptions. It is small
enough for tests but exposes a two-species ``birth_death`` operator that can be
edited directly in the saved model spec.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Multispecies birth-death example
   steps: 1
   operators: birth_death
   observers: DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/multispecies_birth_death.py
   :language: python
   :caption: lgca/examples/multispecies_birth_death.py

identity_tumor_growth
~~~~~~~~~~~~~~~~~~~~~

:Category: tumor growth
:Question: How can individual cell properties drive go-or-grow tumor expansion?
:Concepts: identity-based LGCA, go-or-grow, tumor growth
:Source: ``lgca/examples/identity_tumor_growth.py``

This example connects to the go-and-grow and go-or-grow sections in
``BioLGCA.ipynb``. It uses an identity-based NoVE model so students can inspect
how individual properties and population-level density change together.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Identity tumor growth example
   steps: 1
   operators: nove_ib.go_or_grow
   observers: DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/identity_tumor_growth.py
   :language: python
   :caption: lgca/examples/identity_tumor_growth.py

Suggested classroom workflow
----------------------------

1. Open ``lgca/examples/random_walk.py`` and identify the space, state, time,
   dynamics and analysis blocks.
2. Run ``from lgca.examples.random_walk import run`` followed by
   ``run(steps=10, showprogress=False)``.
3. Change one parameter, seed or lattice size in ``build_spec``.
4. Re-run and compare density, population or exported CSV observer output.
5. Save the same example with ``save_example_spec`` and inspect the JSON file.
6. Move to ``alignment`` or ``chemotaxis`` once the setup/run/inspect loop is
   familiar.
