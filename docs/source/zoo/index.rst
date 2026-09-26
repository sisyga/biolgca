Model zoo
=========

Published LGCA models, written with the rules of this version so that they
can be read, rerun, changed and explored. Each entry is a module of
:mod:`lgca.zoo` and a notebook that reproduces a result of its paper at a
size that runs in about a minute, and states what differs from the paper.
``build_spec(full=True)`` gives the paper's lattice and run length.

.. code-block:: python

   from lgca import zoo

   for card in zoo.catalogue():
       print(card.title, "-", card.question)

   allee = zoo.load("allee_effect")
   spec = allee.build_spec(density=0.3)  # a lgca.model.ModelSpec
   zoo.parameter_table("allee_effect")  # the paper's symbols and their paths in the spec

.. list-table::
   :header-rows: 1
   :widths: 22 38 25 15

   * - Entry
     - Question
     - Lattice and model
     - Reference
   * - :doc:`allee_effect`
     - Why do small tumour cell populations die out although every cell can
       divide?
     - square, classical with volume exclusion; go-or-grow
     - Böttger et al. 2015
   * - :doc:`phenotypic_plasticity`
     - How does the switch between migrating and dividing evolve in a growing
       tumour, and where do the strategies end up?
     - hexagonal and 1D, identity-based without volume exclusion; evolving
       go-or-grow switch
     - Syga et al. 2024
   * - :doc:`jamming`
     - How do cell–cell adhesion and confinement by the matrix decide whether
       cancer cells invade as a jammed sheet, a fluid sheet or single cells?
     - hexagonal, classical with volume exclusion; adhesion, alignment,
       matrix confinement and degradation
     - Ilina et al. 2020

The single mechanisms behind these models are in the :doc:`../example_gallery`
and the :doc:`../tutorials/index`.

.. toctree::
   :maxdepth: 1
   :hidden:

   allee_effect
   phenotypic_plasticity
   jamming
