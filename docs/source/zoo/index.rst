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
   * - :doc:`evolving_front`
     - How does a population whose proliferation rate evolves invade empty
       space, and where do the fastest-dividing cells end up?
     - square strip, identity-based without volume exclusion (evolutionary
       LGCA); evolving division rate
     - Syga et al. 2026
   * - :doc:`phenotypic_plasticity`
     - How does the switch between migrating and dividing evolve in a growing
       tumour, and where do the strategies end up?
     - hexagonal and 1D, identity-based without volume exclusion; evolving
       go-or-grow switch
     - Syga et al. 2024
   * - :doc:`clonal_go_or_grow`
     - When driver mutations change both how fast cells divide and when
       they switch between migrating and dividing, which clones win in a
       growing tumour, and where?
     - hexagonal, identity-based without volume exclusion; go-or-grow with
       clones
     - new model
   * - :doc:`excitable_media`
     - Do spiral waves survive when an excitable medium consists of a small
       number of discrete individuals?
     - hexagonal, two species, classical with volume exclusion; excitable
       birth and death
     - Syga et al. 2019
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
   evolving_front
   phenotypic_plasticity
   clonal_go_or_grow
   excitable_media
   jamming
