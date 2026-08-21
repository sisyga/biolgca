# Welcome to biolgca!
[![CI](https://github.com/sisyga/biolgca/actions/workflows/ci.yml/badge.svg)](https://github.com/sisyga/biolgca/actions/workflows/ci.yml)

`biolgca` is a Python package for simulating different types of **lattice-gas 
cellular automata (LGCA)** in the biological context.

#### LGCA
LGCA are a subclass of cellular 
automata with an extended state space that allows each particle/cell to have a direction. 
For a more detailed 
introduction see the [Wikipedia article](https://en.wikipedia.org/wiki/BIO-LGCA). 
They present a 
mesoscopic modelling framework to analyse collective phenomena, e.g. cell migration. 
Use cases are demonstrated in this 
[paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009066).

#### Content 
The package is intended for use in ongoing research as well as to exemplify the unique 
advantages of the framework. It is under active development. The types of LGCA 
currently supported are:
- classical LGCA (volume exclusion, all particles/cells have the same properties)
- identity-based LGCA (volume exclusion, particles/cells can have individual properties)
- classical LGCA without volume exclusion (all particles/cells have the same properties)
- identity-based LGCA without volume exclusion (particles/cells can have individual properties)
- classical multi-species LGCA with or without volume exclusion

These can be simulated in 1D, 2D square, 2D hexagonal, **3D cubic**, and **3D Moore** lattices.
Classical multi-species LGCA are available by passing `n_species > 1`.
A [library of interaction rules](docs/source/interactions_summary.rst) is already implemented.
Built-in rules are available through the legacy `interaction=` argument and as
native interaction plugins for declarative `ModelSpec` pipelines. Adding a
custom interaction rule or customising other parts of the simulation (e.g. the
interaction radius) is easy.

Current analysis possibilities include observer-based logging plus plots of:
- density (+ animation)
- flux (+ animation)
- flow (+ animation)
- full state space (+ animation)
- scalar field
- vector field

The internal state of the LGCA is always accessible for computational analysis.

# Quick Start
Clone the repository and install the package with the ordinary 1-D/2-D plotting
stack:

```bash
git clone https://github.com/sisyga/biolgca.git
cd biolgca
python -m pip install -e ".[plot2d]"
```

BioLGCA has two supported starting points.

## 1. Explore interactively

Use `get_lgca(...)` in Python or a notebook when you want to change parameters
and inspect a simulation directly:

```python
from lgca import get_lgca

lgca = get_lgca(
    geometry="hex",
    interaction="alignment",
    bc="reflecting",
    seed=1,
)
lgca.timeevo(timesteps=50, record=True, showprogress=False)
lgca.plot_density()
```

## 2. Save and share a reproducible simulation

Use a versioned ModelSpec file and the installed CLI when a simulation must be
reviewed, rerun or shared:

```bash
biolgca examples export random_walk model.json
biolgca validate model.json
biolgca run model.json --output runs/random-walk-001
```

JSON is the canonical ModelSpec format; optional YAML is an authoring syntax for
the same data model. The run directory contains the resolved model, runtime
metadata and configured observer outputs. Existing output directories are
rejected unless `--overwrite` is explicit.

# Table of Contents
- [Example usage](#example-usage)
- [Declarative simulations](#declarative-simulations)
- [Getting started](#getting-started)
- [Running tests and building docs](#running-tests-and-building-docs)
- [Questions/Contribute](#questionscontribute)
- [License](#license)

# Example usage
#### Timestepping and flux plot
```python
from lgca import get_lgca
# specify the alignment interaction
# LGCA geometry: hexagonal lattice, no resting channels and reflecting boundary conditions
# nodes will be initialised randomly
lgca = get_lgca(geometry='hex', interaction='alignment', bc='refl')
# simulate for 132 timesteps without viewing the result
# record the full channel configuration of the lattice at all timesteps
lgca.timeevo(timesteps=132, record=True)
# plot the flux to see clusters of aligned particles moving in the same direction
lgca.plot_flux()
```
![Alignment interaction](docs/images/alignment_small.png)
#### Live simulation and density profile

```python
from lgca import get_lgca

# specify the excitable medium interaction with interaction parameter N=20
# LGCA geometry: 20 resting channels and reflecting boundary conditions
lgca = get_lgca(interaction='excitable_medium', restchannels=20, N=20, bc='refl')
# initialise a custom lattice configuration
lgca.nodes[...] = 0
lgca.nodes[:lgca.lx // 2, :, :lgca.velocitychannels] = 1  # 6 moving particles per node in the left half of the lattice
lgca.nodes[:, :lgca.ly // 2, lgca.velocitychannels:] = 1  # 6 resting particles per node in the lower half of the lattice
# view a live simulation: density profile of the velocity channels only
lgca.live_animate_density(channels=slice(0, lgca.velocitychannels), vmax=lgca.velocitychannels)
# the following image results from stopping the simulation after k=87 timesteps
```
![Excitable media](docs/images/excitable_medium_small.png)
#### Identity-based LGCA and 1D plots

```python
from lgca import get_lgca

# identity-based LGCA: each cell can have own properties
# specify the "Go and grow" interaction: cancer cells proliferate and migrate,
# where the birth rate mutates
# LGCA geometry: 1D lattice, 6 resting channels
lgca = get_lgca(interaction='go_and_grow', ib=True, geometry='lin', restchannels=6)
# initialise a custom lattice configuration
lgca.nodes[...] = 0
lgca.update_dynamic_fields()
lgca.nodes[lgca.dims[0] // 2, :] = 1  # 1 fully filled node at the center
# simulate for 200 timesteps and record the full lattice configuration
lgca.timeevo(timesteps=200, record=True)
# plot cell density to see the tumour growth over time
lgca.plot_density(colorbarwidth=0.2)  # plot on the left
# plot the birth rate to inspect its increase
lgca.plot_prop_spatial(propname='r_b')  # plot on the right
```
![Go and grow density](docs/images/go_and_grow_density_small.png) ![Go and grow birth rate](./docs/images/go_and_grow_rb_small.png)

# Declarative simulations
For reproducible simulations, `ModelSpec` separates lattice setup, interaction
plugins, observer-based logging, and plotting:

```python
from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import DensityRecorder

spec = ModelSpec(
    space=SpaceSpec(geometry="hex", dims=(20, 20), boundary="reflecting"),
    state=StateSpec(density=0.2, restchannels=0),
    time=TimeSpec(steps=100, seed=1),
    dynamics=InteractionPipelineSpec(
        operators=[{"name": "classical.alignment", "parameters": {"beta": 2.0}}],
    ),
    analysis=AnalysisSpec(observers=[DensityRecorder()]),
)

result = run_model(spec, showprogress=False)
result.lgca.plot_density()
```

See the [ModelSpec and plugin guide](docs/source/model_specs_and_plugins.rst)
and the [observer/plotting guide](docs/source/observers_and_plotting.rst).

Model specs can also be exported, validated, shared, and run without writing a
Python launcher:

```bash
biolgca examples list
biolgca examples export random_walk model.json
biolgca validate model.json
biolgca run model.json --output runs/random-walk-001
```

The run directory contains the resolved model, runtime metadata, and configured
observer outputs. Existing run directories are rejected unless
`--overwrite` is explicit. Imported NPZ resources stay inside the model
directory and generated files stay inside the run directory by default.

Portable files carry `schema_version: 1` and contain only data plus registered
plugin names; they never execute import paths or arbitrary Python. A ModelSpec
is configuration, not a serialized trajectory. `model.resolved.json` and
`metadata.json` capture reproducible configuration and provenance, but they are
not restart checkpoints. Large numeric initial states can be stored in a
companion NPZ file; identity-based checkpoints are not yet supported because
particle properties must be restored together with their labels.

Portable initial conditions support random density through `state.density`, a
named `region` initializer, and numeric states loaded with `from_npz`.

# Getting started
#### Dependencies
`biolgca` depends on `numpy`, `scipy`, and `tqdm` for running simulations.
For ordinary 1-D/2-D plotting, install the Matplotlib-only extra:
```bash
python -m pip install -e ".[plot2d]"
```

The `plot3d` extra installs Mayavi. The `plot`/`plotting` umbrella installs both
renderer stacks and is intended only when 3-D plotting is required.

For development, install test, lint, and documentation dependencies:
```bash
python -m pip install -e ".[dev]"
```

YAML model files are optional; JSON works with the core installation. Enable
YAML with:

```bash
python -m pip install -e ".[yaml]"
```

#### Installation
Clone the repository and install it into your active Python environment:
```bash
git clone https://github.com/sisyga/biolgca.git
cd biolgca
pip install -e .
```

#### Use
To use the package, simply import the `get_lgca` function:
```python
from lgca import get_lgca
```
It will return an instance of the correct class of LGCA according to the passed 
arguments, e.g. a 1D identity-based LGCA (`IBLGCA_1D`), already initialised with 
the specified initial conditions. This can be used to simulate 
the automaton with the specified interaction and inspect the results.
```python
# request the LGCA
lgca = get_lgca(ib=True, geometry='1d', interaction='random_walk')
# simulate for 50 timesteps
lgca.timeevo(timesteps=50)
# plot the development of the particle/cell density over time
lgca.plot_density()
```
The [Tutorial](./BioLGCA.ipynb) guides you through the argument options.

# Running tests and building docs
Run the test-suite from the repository root with
```bash
python -m pytest -q
```
Build the HTML documentation in `docs/_build` using
```bash
python docs/build.py
```
The build is strict and removes previously generated autosummary pages first,
so deleted or renamed APIs cannot leave stale documentation behind.

# Questions/Contribute
The structure of the package and its functionalities are detailed in the
[documentation](docs/source/). There you will also find coding examples and API pages.

Issues are tracked on the [GitHub page](https://github.com/sisyga/biolgca/issues).
We collect both bugs and feature ideas there.

For guidelines how to add code to and maintain the repo visit the [Wiki](https://github.com/sisyga/biolgca/wiki).

For remaining questions you can contact us via E-mail.<br>
Simon Syga: `simon.syga@tu-dresden.de`<br>
Bianca Güttner: `bianca.guettner@nct-dresden.de`

# License
BSD 3-clause license (see LICENSE file or [online resource](https://opensource.org/licenses/BSD-3-Clause)).

Copyright (C) 2018-2026 Technische Universität Dresden, contact: simon.syga@tu-dresden.de.
