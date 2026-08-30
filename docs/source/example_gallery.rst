Example gallery
===============

The maintained :doc:`tutorials/index` teach how to construct and combine
interactions in visible notebook cells. This gallery is the reference catalog
to use afterward: it collects complete, tested ``ModelSpec`` examples exposed
by :mod:`lgca.examples`.

The gallery is inspired by the old :download:`BioLGCA.ipynb <../../BioLGCA.ipynb>`
and :download:`Evolutionary LGCA.ipynb <../../Evolutionary LGCA.ipynb>` tours,
and by the Morpheus habit of keeping examples discoverable by modeling question.
Each example has its own source file under ``lgca/examples``. The source files
are useful recipes and regression-tested starting points, but they do not
replace the explanatory notebooks.

Study one example file
----------------------

Each example file declares metadata, builds a ``ModelSpec`` in ``build_spec``,
and runs it with ``run_model``. The package-level helpers are useful for tests,
scripts and project scaffolding once you have worked through the tutorials.

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

From the repository root, each teaching file can be run as a module:

.. code-block:: console

   python -m lgca.examples.random_walk

The same files also run directly from the examples folder:

.. code-block:: console

   cd lgca/examples
   python random_walk.py --steps 1

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

Use this to discuss polar alignment: local flux biases the next channel choice.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Alignment example
   steps: 1
   operators: classical.alignment
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/alignment.py
   :language: python
   :caption: lgca/examples/alignment.py

aggregation
~~~~~~~~~~~

:Category: collective motion
:Question: How does density-biased movement create clusters?
:Concepts: aggregation, density, rest channels
:Source: ``lgca/examples/aggregation.py``

This mirrors the notebook aggregation example with a square lattice, moderate
density and three rest channels.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Aggregation example
   steps: 1
   operators: classical.aggregation
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/aggregation.py
   :language: python
   :caption: lgca/examples/aggregation.py

nematic_interaction
~~~~~~~~~~~~~~~~~~~

:Category: collective motion
:Question: How does axis alignment differ from polar alignment?
:Concepts: nematic alignment, orientation, flux
:Source: ``lgca/examples/nematic_interaction.py``

Use this after alignment to compare head-tail symmetric orientation with polar
motion.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Nematic interaction example
   steps: 1
   operators: classical.nematic
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/nematic_interaction.py
   :language: python
   :caption: lgca/examples/nematic_interaction.py

chemotaxis
~~~~~~~~~~

:Category: guidance
:Question: How does a signal field bias cell movement?
:Concepts: signal field, gradient sensing, reorientation
:Source: ``lgca/examples/chemotaxis.py``

This example includes an explicit static signal field and a reorientation term
that reads the field.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Chemotaxis example
   steps: 1
   operators: reorientation.boltzmann
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: signal

.. literalinclude:: ../../lgca/examples/chemotaxis.py
   :language: python
   :caption: lgca/examples/chemotaxis.py

persistent_movement
~~~~~~~~~~~~~~~~~~~

:Category: movement
:Question: How does directional memory change a single-cell trajectory?
:Concepts: persistent motion, single-cell initial state, reflecting boundary
:Source: ``lgca/examples/persistent_movement.py``

This notebook-scale example starts with one cell on a small reflecting lattice.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Persistent movement example
   steps: 1
   operators: classical.persistent_walk
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/persistent_movement.py
   :language: python
   :caption: lgca/examples/persistent_movement.py

contact_guidance
~~~~~~~~~~~~~~~~

:Category: guidance
:Question: How does an oriented scaffold bias movement?
:Concepts: contact guidance, director field, single-cell initial state
:Source: ``lgca/examples/contact_guidance.py``

The guiding director field is built explicitly so students can see the coupling
between state fields and movement.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Nematic contact guidance example
   steps: 1
   operators: reorientation.boltzmann
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: director

.. literalinclude:: ../../lgca/examples/contact_guidance.py
   :language: python
   :caption: lgca/examples/contact_guidance.py

go_and_grow
~~~~~~~~~~~

:Category: tumor growth
:Question: How does local birth expand a seeded population?
:Concepts: birth, rest channels, tumor growth
:Source: ``lgca/examples/go_and_grow.py``

This is the classical go-and-grow notebook example expressed as a birth model
with a seeded center cell.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Go-and-grow example
   steps: 1
   operators: classical.birth
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/go_and_grow.py
   :language: python
   :caption: lgca/examples/go_and_grow.py

go_or_grow
~~~~~~~~~~

:Category: tumor growth
:Question: How does switching between motion and birth change expansion?
:Concepts: go-or-grow, phenotype switching, rest channels
:Source: ``lgca/examples/go_or_grow.py``

This keeps the short 15-step notebook scale and exposes the switching
parameter in the model spec.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Go-or-grow example
   steps: 1
   operators: classical.go_or_grow
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/go_or_grow.py
   :language: python
   :caption: lgca/examples/go_or_grow.py

identity_go_and_grow
~~~~~~~~~~~~~~~~~~~~

:Category: evolution
:Question: How does an identity-based lineage grow from one seed?
:Concepts: identity-based LGCA, birth-death, lineage properties
:Source: ``lgca/examples/identity_go_and_grow.py``

This ports the notebook's one-dimensional identity-based go-and-grow setup.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Identity-based go-and-grow example
   steps: 1
   operators: ib.birthdeath
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/identity_go_and_grow.py
   :language: python
   :caption: lgca/examples/identity_go_and_grow.py

excitable_medium
~~~~~~~~~~~~~~~~

:Category: pattern formation
:Question: How can local excitation create propagating waves?
:Concepts: excitable medium, rest channels, wave propagation
:Source: ``lgca/examples/excitable_medium.py``

The initial condition marks two lattice regions, matching the notebook's
wave-propagation demonstration.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Excitable medium example
   steps: 1
   operators: classical.excitable_medium
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/excitable_medium.py
   :language: python
   :caption: lgca/examples/excitable_medium.py

custom_rest_or_align
~~~~~~~~~~~~~~~~~~~~

:Category: custom dynamics
:Question: How can a notebook interaction rule become a reusable model spec?
:Concepts: custom interaction, alignment, rest channels
:Source: ``lgca/examples/custom_rest_or_align.py``

Use this when students are ready to inspect a custom interaction function and
see how it can still live inside the ``ModelSpec`` runtime.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Custom rest-or-align example
   steps: 1
   operators: custom.rest_or_align
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/custom_rest_or_align.py
   :language: python
   :caption: lgca/examples/custom_rest_or_align.py

evolutionary_go_and_grow
~~~~~~~~~~~~~~~~~~~~~~~~

:Category: evolution
:Question: How does heritable birth-rate variation change growth?
:Concepts: identity-based LGCA, birth-death, evolution
:Source: ``lgca/examples/evolutionary_go_and_grow.py``

This follows the proof-of-principle setup from ``Evolutionary LGCA.ipynb``.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Evolutionary go-and-grow example
   steps: 1
   operators: ib.birthdeath
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/evolutionary_go_and_grow.py
   :language: python
   :caption: lgca/examples/evolutionary_go_and_grow.py

evolutionary_go_or_grow
~~~~~~~~~~~~~~~~~~~~~~~

:Category: evolution
:Question: How does switching affect evolutionary expansion?
:Concepts: identity-based LGCA, go-or-grow, evolution
:Source: ``lgca/examples/evolutionary_go_or_grow.py``

This ports the evolutionary notebook's one-dimensional go-or-grow setup.

Example output from ``run(steps=1)``:

.. code-block:: text

   title: Evolutionary go-or-grow example
   steps: 1
   operators: ib.go_or_grow
   observers: NodeRecorder, DensityRecorder, PopulationRecorder
   fields: -

.. literalinclude:: ../../lgca/examples/evolutionary_go_or_grow.py
   :language: python
   :caption: lgca/examples/evolutionary_go_or_grow.py

multispecies_birth_death
~~~~~~~~~~~~~~~~~~~~~~~~

:Category: population dynamics
:Question: How do different birth and death rates change competing populations?
:Concepts: multispecies, birth, death
:Source: ``lgca/examples/multispecies_birth_death.py``

Use this to discuss how species-specific rates become explicit model
assumptions.

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

This two-dimensional identity-based NoVE model is a compact tumor-growth
teaching example.

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
