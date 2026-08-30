# BioLGCA

[![CI](https://github.com/sisyga/biolgca/actions/workflows/ci.yml/badge.svg)](https://github.com/sisyga/biolgca/actions/workflows/ci.yml)

BioLGCA is a Python package for lattice-gas cellular automata in biological
contexts. It supports reproducible simulations of cell migration, collective
movement, population dynamics, phenotypic switching and spatial evolution.

LGCA represent cells in velocity and optional rest channels at each lattice
site. This mesoscopic state records cell number and movement direction while
remaining computationally accessible for custom analysis. See the
[BIO-LGCA overview](https://en.wikipedia.org/wiki/BIO-LGCA) and the
[method paper](https://doi.org/10.1371/journal.pcbi.1009066).

## Install and start the tutorials

Clone the repository and install it into an active Python environment:

```bash
git clone https://github.com/sisyga/biolgca.git
cd biolgca
python -m pip install -e .
jupyter lab
```

A normal installation includes Matplotlib and JupyterLab. Open
[`docs/source/tutorials/01_fundamentals.ipynb`](docs/source/tutorials/01_fundamentals.ipynb)
and continue through the six maintained lessons:

1. [LGCA fundamentals and random movement](docs/source/tutorials/01_fundamentals.ipynb)
2. [Collective movement](docs/source/tutorials/02_collective_movement.ipynb)
3. [Combining directional cues](docs/source/tutorials/03_combining_interactions.ipynb)
4. [Population dynamics and phenotypic switching](docs/source/tutorials/04_population_dynamics.ipynb)
5. [Evolutionary LGCA](docs/source/tutorials/05_evolutionary_lgca.ipynb)
6. [Developing a reproducible student project](docs/source/tutorials/06_student_project.ipynb)

Every notebook constructs its `ModelSpec` and interaction pipeline in visible
cells. The notebooks execute in CI and are intended to be copied and modified
for student projects.

## First reproducible simulation

`ModelSpec` separates lattice geometry, initial state, time, dynamics and
analysis:

```python
from lgca.model import AnalysisSpec, ModelSpec, SpaceSpec, StateSpec, TimeSpec, run_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import DensityRecorder, PopulationRecorder

spec = ModelSpec(
    space=SpaceSpec(geometry="square", dims=(20, 20), boundary="periodic"),
    state=StateSpec(density=0.15, restchannels=0),
    time=TimeSpec(steps=30, seed=1),
    dynamics=InteractionPipelineSpec(
        operators=[{"name": "classical.random_walk"}],
    ),
    analysis=AnalysisSpec(
        observers=[DensityRecorder(), PopulationRecorder()],
    ),
)

result = run_model(spec, showprogress=False)
result.lgca.plot_density()
```

To combine directional cues in one reorientation decision, place several
terms inside one `ReorientationSpec`:

```python
from lgca.pipeline import ReorientationSpec, ReorientationTermSpec

combined = ReorientationSpec(
    terms=[
        ReorientationTermSpec(name="alignment", beta=1.0),
        ReorientationTermSpec(
            name="chemotaxis",
            beta=0.5,
            parameters={"field": "signal"},
        ),
    ]
)
```

See the [interaction-composition explanation](docs/source/concepts/interactions.rst)
and [ModelSpec guide](docs/source/how_to/model_specs_and_plugins.rst).

## Save and share simulations

JSON is the canonical, versioned ModelSpec format. The installed command can
export, validate and run models without a custom launcher:

```bash
biolgca examples list
biolgca examples export random_walk model.json
biolgca validate model.json
biolgca run model.json --output runs/random-walk-001
```

The run directory contains the resolved model, runtime metadata and configured
observer outputs. YAML model files are available through the optional `yaml`
extra.

## Supported models and analysis

BioLGCA includes:

- classical and identity-based LGCA with volume exclusion;
- classical and identity-based LGCA without volume exclusion;
- classical multi-species LGCA with or without volume exclusion;
- linear, square, hexagonal, cubic and three-dimensional Moore lattices; and
- observer-based recording plus density, flux, flow, state, field, property and
  family-population plots.

The [example gallery](docs/source/example_gallery.rst) catalogs tested source
examples after the tutorial path. Three-dimensional Mayavi plotting remains an
optional `plot3d` dependency.

## Legacy factory API

Existing code can continue to use `get_lgca` for direct interactive setup:

```python
from lgca import get_lgca

lgca = get_lgca(
    geometry="hex",
    interaction="alignment",
    bc="reflecting",
    seed=1,
)
lgca.timeevo(timesteps=50, record=True, showprogress=False)
lgca.plot_flux()
```

The [factory reference](docs/source/reference/factory_reference.rst) documents
this compatibility API. Historical teaching notebooks are retained under
[`notebooks/legacy/`](notebooks/legacy/) but are not the maintained learner
path.

## Development and documentation

Install contributor dependencies and run the project gates from the repository
root:

```bash
python -m pip install -e ".[dev]"
conda run -n biolgca python -m pytest -q
conda run -n biolgca python docs/build.py
```

The strict documentation build executes all six maintained notebooks from
clean kernels and treats cell exceptions and Sphinx warnings as failures.

Issues and feature ideas are tracked on
[GitHub](https://github.com/sisyga/biolgca/issues). User-facing changes are
recorded in [CHANGELOG.md](CHANGELOG.md).

## License

BioLGCA is distributed under the BSD 3-clause license. See [LICENSE.txt](LICENSE.txt).

Copyright (C) 2018-2026 Technische Universität Dresden.
