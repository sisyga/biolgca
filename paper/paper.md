---
title: 'biolgca: a Python package for biological lattice-gas cellular automata'
tags:
  - Python
  - cellular automata
  - lattice-gas cellular automata
  - collective cell migration
  - tumor growth
authors:
  - name: Simon Syga
    orcid: 0000-0001-9955-9012
    affiliation: 1
affiliations:
  - name: Center for Interdisciplinary Digital Sciences, TUD Dresden University of Technology, Germany
    index: 1
    ror: 042aqky30
date: 17 November 2026
bibliography: paper.bib
---

# Summary

Cells in tissues move collectively or as single cells, follow chemical
signals, divide, die and pass on heritable traits. How do these individual
behaviors produce collective patterns such as invading tumor fronts or
traveling waves? Biological lattice-gas cellular automata (BIO-LGCA) are
mathematical models built to answer this question [@deutsch2018;
@deutsch2021], and `biolgca` is a Python package for building, running and
sharing such models. A cellular automaton divides space into the nodes of a
regular lattice and updates all nodes in discrete time steps by local
stochastic rules; in a BIO-LGCA, every cell at a node also has a direction of
movement and can carry individual properties such as heritable traits.

A user of `biolgca` picks a lattice in one, two or three dimensions, combines
ready-made rules such as alignment, chemotaxis (movement along a chemical
gradient), birth and death or the mutation of heritable traits, adds rules of
their own as Python functions, and gets a model that can be saved as a file
and rerun from its seed. Fields such as oxygen or a chemoattractant can evolve
together with the cells. `biolgca` is released under the BSD-3-Clause license,
installed with `pip install biolgca`, imported as `lgca`, and archived on
Zenodo [@biolgca_zenodo].

# Statement of need

BIO-LGCA treat cells as particles that move on a lattice and interact with
their neighbors through simple rules, which keeps simulations of large cell
populations cheap. They have been used to study growth in small tumors
[@boettger2015], the invasion of breast cancer cells [@ilina2020] and the
evolution of traits in tumor cell populations [@syga2024; @syga2026].
Without a software package, however, every such study must first
implement the lattice, its boundary conditions and the movement of the cells
before any biology is modeled, and the resulting code is hard for
others to rerun or to extend. To the author's knowledge, `biolgca` is the first
general-purpose open-source package for BIO-LGCA.

`biolgca` addresses these obstacles. It provides the main mechanisms of the
BIO-LGCA literature as rules that combine in one model:
random walk, alignment, chemotaxis, contact guidance, aggregation, birth and
death, switching between migration and proliferation, mutations of heritable
traits, and fields that diffuse, decay and react with the cells
(\autoref{fig:examples}). A model is saved as a JSON file that others can
rerun from its seed. Users can add rules of their own, and the package checks
that each changes only what it declares to change.

The target users are researchers, students and teachers who model or teach
collective migration, tissue growth, pattern formation and the evolution of
cell populations. Eight tutorials, how-to guides and an API
reference are published at <https://biolgca.readthedocs.io>.

# State of the field

Several frameworks simulate cell-based models. Morpheus [@starruss2014] and
CompuCell3D [@swat2012] are built around the cellular Potts model, in which
each cell occupies many nodes. PhysiCell [@ghaffarizadeh2018] simulates
off-lattice agents coupled to diffusing substrates, and Chaste [@cooper2020]
implements several cell-based model types. Closest to `biolgca` are two cellular
automaton simulators: the Hybrid Automata Library [@bravo2020], a Java library
of on- and off-lattice agents coupled to diffusing fields, and the stochastic
mesoscopic tumor simulator of @jimenezsanchez2021.

None of these packages implements the update that defines a lattice-gas
cellular automaton (LGCA) [@frisch1986]. Every node of the lattice has one slot, called a velocity
channel, for each direction in which a cell can move. A time step first
applies a chain of stochastic rules to the cells at every node, by which they
turn, divide, die or change their type, and then moves every cell to the
neighboring node that its channel points to. Because a node stores the
directions of its cells, alignment and chemotaxis become rules for how the
cells at a node turn. Equations for the average behavior of the population can
also be derived from the rules [@deutsch2021]. An LGCA could be written on top of
an agent library, but it would lack the array state, the rule library and the
rule checks described below; `biolgca` was therefore written as a separate package.

# Software design

Four design choices shape `biolgca`, each with a trade-off.

*One family of models.* `biolgca` implements LGCA only. Cells are points on a
regular lattice (1D, square, hexagonal, or cubic with 6 or 26 neighbors), and
all nodes are updated at once. The state of a model is therefore held in NumPy
arrays [@harris2020], and every rule is an array operation. The price is
resolution: cell shape and mechanics, which cellular Potts and off-lattice
models resolve, are outside this family. Cells are stored either as counts,
which is fast, or as individuals with heritable traits, which makes lineages
and evolution observable but costs time and memory in proportion to the number
of cells.

*Python.* `biolgca` is written in Python so that students can read the built-in
rules and write their own. The price is speed.
Array operations limit this cost: a three-dimensional tumor spheroid of about
370,000 cells with an oxygen field runs ten simulated days in about three
minutes on one CPU core. A single simulation
is not parallelized; sweeps over parameters and seeds can run in parallel
processes.

*Models as data.* A model is a specification: the lattice, the initial state,
the number of steps and the seed, an ordered list of rules with their
parameters, and the quantities to record. This separation of a model from the
code that runs it is inspired by Morpheus [@starruss2014]. Because a
specification is data, it can be validated, saved, compared and swept; this
sweep runs a shipped model with three values of `kappa` and ten seeds each:

```python
from lgca.examples import go_or_grow
from lgca.study import sweep

table = sweep(go_or_grow.build_spec(), grid={"kappa": [-4, 0, 4]},
              seeds=range(10), measure={"population": "population"})
```

The price is indirection: a rule must be registered under a name before a
specification can refer to it. Because results depend on the code behind these
names, every run records its provenance: the versions of `biolgca`, Python,
NumPy and SciPy, all parameters including defaults, and hashes of the source
code of its rules.

*Checked rules.* An error in a user-written rule is hard to spot in a
stochastic simulation: a rule meant to reorient cells might lose some of them. Every rule therefore declares which of four things it may
change: fields such as oxygen, the directions of cells, their types and
traits, or their number. Rules change the state through operations that the
package provides, such as "every cell dies with probability $p$". These
operations check the declaration, and a rule that oversteps it triggers a
warning, which the test helper `check_interaction` turns into an error. The
operations work with and without a limit on the number of cells per node, so a
rule written once runs in both kinds of model.

`biolgca` requires Python 3.11 or later. It builds on NumPy, SciPy
[@virtanen2020], PyAMG [@bell2023], pandas [@mckinney2010] and Matplotlib
[@hunter2007], among others.

![Typical simulations with `biolgca`, each from a model that ships with the
package. (a) Aggregation on a hexagonal lattice. (b) Alignment on a hexagonal
lattice: cells form streams that share a direction; white nodes are empty.
(c) A tumor spheroid whose cells take up oxygen, after ten days: a
proliferating rim surrounds quiescent cells and a necrotic core. (d) A spiral
wave in an excitable medium [@syga2019].\label{fig:examples}](figure.png)

# Research impact statement

`biolgca` has been developed since 2018. An early version accompanied the paper
that defined the BIO-LGCA model class [@deutsch2021], and the package produced
the simulations of four further publications [@syga2019; @ilina2020;
@syga2024; @syga2026]; three of the five papers published a copy of the code
[@deutsch2021; @ilina2020; @syga2024]. At TU Dresden it was the simulation tool
of a doctoral thesis [@syga2024thesis], six master's theses, one of them with
public code [@mullermax2023], and a student research project; five of the
master's theses and the project were written by students other than the author. At Khalifa University, a senior project on
tumor growth fronts built on it [@herbawi2024].

Because most of the present code was written since June 2025, the model zoo
that ships with `biolgca` reproduces results of six published works with the
current code [@boettger2015; @syga2019; @ilina2020; @syga2024thesis;
@syga2024; @syga2026]. Each entry states which result it reproduces and what
differs from the publication. A further entry builds on the model of
@noble2022. A glioblastoma model is being developed with the current package.
The record of the package's earlier versions suggests that `biolgca` can be the
basis for many more publications and student projects.

# AI usage disclosure

The author has used generative AI coding assistants to develop `biolgca` since
June 2025: OpenAI Codex (June 2025 to September 2026, starting with
o3-based models) and Anthropic's Claude Code (since September 2026 with Opus
5.5, and with Fable 5.2 for code review).
Earlier code was written without AI tools. Under the author's direction, the
assistants optimized and generalized that hand-written code, generated new
code, tests and documentation, and reviewed the result, so that most of the
present code was written with them. Claude also drafted and edited this paper
from the author's outline, comments and corrections. The author decided the
scope and design of the package and reviewed, edited and validated all
AI-assisted output. Tests compare the rules with analytic expectations and
with frozen copies of their earlier implementations, and the model zoo
reproduces published results.

# Acknowledgements

Bianca Güttner co-authored parts of the code. Code was also contributed by
Franziska Keidel, Marcus Krellner, Harish P. Jain and Mateo Larralde. The
models build on the BIO-LGCA work of Andreas Deutsch, Josué Manik
Nava-Sedeño, Haralampos Hatzikirou and colleagues.

SS is funded by the European Union (ERC, subLethal, 101054921,
<https://erc.europa.eu/>). Views and opinions expressed are however those
of the authors only and do not necessarily reflect those of the European
Union or the European Research Council Executive Agency. Neither the
European Union nor the granting authority can be held responsible for them.
SS was supported by the Graduate Academy of TUD Dresden University of
Technology, funded by the Excellence Initiative of the German Federal and
State Governments. The funders had no role in the design of the software or
in the writing of this paper.

# References
