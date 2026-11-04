---
title: 'BioLGCA: biological lattice-gas cellular automata in Python'
tags:
  - Python
  - cellular automata
  - lattice-gas cellular automata
  - collective cell migration
  - tumour growth
  - evolution
  - mathematical biology
authors:
  - name: Simon Syga
    orcid: 0000-0001-9955-9012
    affiliation: 1
affiliations:
  - name: Technische Universität Dresden, Germany
    index: 1
    ror: 042aqky30
date: 4 November 2026
bibliography: paper.bib
---

# Summary

Cells in tissues move, align with their neighbours, follow chemical signals,
divide, die and pass on heritable traits. How these individual behaviours
produce collective patterns, such as invading tumour fronts, migrating
sheets or travelling waves, is a central question of mathematical biology.
Biological lattice-gas cellular automata (BIO-LGCA) are a class of
agent-based models built for this question [@deutsch2018; @deutsch2021].
Cells live on the nodes of a regular lattice, and every node has one channel
per direction of movement plus optional rest channels. The state of a node
therefore records how many cells it holds and where they are heading. Each
time step applies local, stochastic interaction rules at every node, then
moves the cells along their channels. Because the rules are local and the
state is a fixed set of channels, simulations of large lattices are fast,
and the models can be analysed with mean-field and linear stability methods
[@hatzikirou2008; @deutsch2021].

BioLGCA is a Python library for building, running and sharing such models.
It covers classical models, in which cells are counted, and identity-based
models, in which every cell carries a label and heritable traits; models with
and without volume exclusion; one or several cell types; and 1D, square,
hexagonal and two 3D lattices. Fields such as oxygen or a chemoattractant can
evolve together with the cells. BioLGCA is released under the BSD-3-Clause
licence, installed with `pip install biolgca` and archived on Zenodo
[@biolgca_zenodo].

# Statement of need

BIO-LGCA have been used to study the Allee effect in tumour initiation
[@boettger2015], jamming transitions in breast cancer invasion [@ilina2020],
the evolution of phenotypic plasticity [@syga2024] and the evolution of
heterogeneity at invasion fronts [@syga2026]. In each of these studies the
model was written as research code for one paper. Combining two mechanisms,
for example alignment with go-or-grow switching, meant editing that code, and
reproducing a published model meant finding and running it.

BioLGCA turns these models into a library with three aims. First, the
mechanisms of the BIO-LGCA literature are composable building blocks:
random walk, alignment, chemotaxis, contact guidance, aggregation, birth
and death, go-or-grow and phenotype switching, mutations of traits, and fields
that diffuse, decay and react with the cells. Second, a model is a
declarative specification that can be saved as a JSON or YAML file and
rerun from its seed, swept over parameters, or run from the command line.
Third, writing a new rule takes a Python function, and the library checks
that the rule obeys the conservation laws of its kind.

The target users are researchers and students in mathematical and
theoretical biology who model collective migration, tissue growth, pattern
formation and the evolution of cell populations, and teachers of courses on
these topics. Eight tutorials, how-to guides and an API reference are
published at <https://biolgca.readthedocs.io>.

# State of the field

Several mature frameworks simulate cell-based models. Morpheus
[@starruss2014] and CompuCell3D [@swat2012] implement the cellular Potts
model, in which each cell occupies many lattice sites; PhysiCell
[@ghaffarizadeh2018] simulates off-lattice agents coupled to diffusing
substrates; Chaste [@cooper2020] offers several cell-based model types,
including an on-lattice cellular automaton; and Mesa [@terhoeven2025] is a
general framework for agent-based models in Python. None of them provides
the velocity-channel state space of LGCA. In this state space, alignment,
chemotaxis and go-or-grow switching are rules for the reorientation of the
cells at a node, and mean-field equations follow systematically from the
rules [@deutsch2021]. Writing an LGCA on top of a general agent-based
framework would lose both the fast array representation of the channels and
this link to analysis.

BioLGCA was therefore built as its own library rather than as an extension.
Its first version grew from 2018 alongside the research that used it.
Version 2.0 keeps the standard models of that version, under their original
names, and runs them on the same rules as new models, so that published
results remain reproducible with current code.

# Software design

A model in BioLGCA is a `ModelSpec`: the lattice (geometry, size,
boundaries), the initial state (density, rest channels, cell types, traits,
fields), the time (steps and seed), the dynamics as an ordered list of
operators, and the observers that record data. A specification is plain
data, so it can be validated, saved, compared and changed by path, as in
this sweep over the switching parameter of go-or-grow:

```python
from lgca.examples import go_or_grow
from lgca.study import sweep

table = sweep(go_or_grow.build_spec(), grid={"kappa": [-4, 0, 4]},
              seeds=range(10), measure={"population": "population"})
```

`build_model` compiles a specification into a model whose step applies the
operators in their listed order and then moves the cells. Operators are of
four kinds, from narrow to wide: `field` (changes fields only),
`reorientation` (moves cells between the channels of their node),
`phenotype_switch` (also changes cell types or traits) and `birth_death`
(may also create and remove cells). The kind is a contract. Rules act on a
`LatticeState`, which holds the channels of all nodes as one NumPy array
[@harris2020], through per-cell operations, such as "every cell dies with
probability $p$", which exist once for models with volume exclusion and
once without. Each operation checks the kind before it changes anything,
so a reorientation rule that loses cells is caught. Users register new
rules with the `@interaction` and `@reorientation_term` decorators, and
`check_interaction` runs a rule on every lattice and model family it claims
to support. Reorientation terms such as alignment and chemotaxis combine
into one Boltzmann-weighted decision per node, which is how the BIO-LGCA
literature defines them.

A step is applied as a whole or not at all: if a rule raises an error, the
model rolls back to the state before the step. Fields are updated by a
`pde` operator that solves diffusion, decay, production, uptake and custom
reactions with implicit, explicit or steady-state solvers on the
neighbourhood of each lattice, using SciPy [@virtanen2020] and algebraic
multigrid from PyAMG [@bell2023]. Every run records provenance: the versions
of Python and of the packages, the parameters including defaults, and
hashes of the source code of every rule and of the input files, so a model
file can be rerun and checked.

Around this core, `lgca.study` runs parameter sweeps into pandas tables,
`lgca.explore` runs a model live in a Jupyter notebook with sliders for its
parameters, and the `biolgca` command validates, runs and sweeps model
files. The model zoo (`lgca.zoo`) reproduces published LGCA models with the
current rules; each entry has a card with the paper, the result reproduced
and what differs, and a notebook (\autoref{fig:zoo}). The quick-start function
`get_lgca` builds the standard models of version 1 by name and compiles
them into the same pipeline. About 2900 tests run in continuous integration
on Linux, Windows and macOS, with Python 3.11 to 3.14 and the minimum
supported versions of the dependencies.

![Results of the model zoo and examples, each reproduced by a notebook in the
documentation. (a, b) Spiral waves in an LGCA model of an excitable medium and
in its mean-field equations [@syga2019]. (c) A tumour spheroid on a hexagonal
lattice whose cells take up oxygen: the outer rim proliferates while the core
becomes hypoxic. (d) Density of an invading population whose proliferation
rate evolves, and (e) its accelerating front compared with Fisher–KPP
predictions [@syga2026].\label{fig:zoo}](figure.png)

# Research impact statement

The first version of BioLGCA was used for the models of five publications:
discrete excitable media [@syga2019], jamming transitions in breast cancer
invasion [@ilina2020], the BIO-LGCA modelling class [@deutsch2021], the
evolution of phenotypic plasticity [@syga2024] and the evolution of
heterogeneity at invasion fronts [@syga2026]. It was also the simulation
tool of a doctoral thesis [@syga2023], six master's theses
[@syga2018; @keidel2020; @guttner2021; @paulat2022; @mullermax2022;
@nussbaumer2025] and a student research project [@jain2020] at TU Dresden.
Four of these publications are reproduced in the model zoo with the current
rules, together with the Allee effect of go-or-grow [@boettger2015], a model
of tumour evolution in the spirit of @noble2022 and a model of mutational
meltdown from @syga2023. The zoo notebooks, the tutorials and the saved
model files serve as reproducible materials for new studies and for
teaching.

# AI usage disclosure

Since 2025 the package has been developed with the help of generative AI
coding assistants (OpenAI Codex, GitHub Copilot and Anthropic's Claude
Code), which wrote code, tests and documentation under the direction of the
maintainer. Codex work was merged from `codex/` branches, Copilot work is
authored by `copilot-swe-agent[bot]`, and commits written with Claude Code
since September 2026 carry a `Co-Authored-By` trailer. The results are
checked independently of the code that produced them: the test suite
compares the new rules statistically with the frozen interaction functions
of the code before 2.0 (`tests/legacy/`) and with analytic expectations,
every rule is checked against the conservation laws of its kind, and each
zoo notebook reproduces a result of its paper at reduced size and states
what differs.

# Acknowledgements

Bianca Güttner co-authored the code before version 2.0. Code was also
contributed by Franziska Keidel, Marcus Krellner, Harish P. Jain and Mateo
Larralde. The models build on the BIO-LGCA work of Andreas Deutsch, Josué
Manik Nava-Sedeño, Haralampos Hatzikirou and colleagues.

SS is funded by the European Union (ERC, subLethal, 101054921,
<https://erc.europa.eu/>). Views and opinions expressed are however those
of the authors only and do not necessarily reflect those of the European
Union or the European Research Council Executive Agency. Neither the
European Union nor the granting authority can be held responsible for them.
SS was supported by the Graduate Academy of TUD Dresden University of
Technology, funded by the Excellence Initiative of the German Federal and
State Governments.

# References
