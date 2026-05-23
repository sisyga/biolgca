# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**biolgca** is a Python package for simulating lattice-gas cellular automata (LGCA) in biological contexts. It provides
a mesoscopic modeling framework to analyze collective phenomena like cell migration.

### Core Architecture

The package follows a hierarchical class structure:

- **Base classes** (`lgca/base.py`): Abstract classes defining common LGCA behavior
    - `LGCA_base`: Classical LGCA with volume exclusion
    - `IBLGCA_base`: Identity-based LGCA (particles have individual properties)
    - `NoVE_LGCA_base`: LGCA without volume exclusion
    - `NoVE_IBLGCA_base`: Identity-based LGCA without volume exclusion

- **Geometry-specific classes**: Implementations for different lattice types
    - `lgca_1d.py`: 1D linear lattices
    - `lgca_square.py`: 2D square lattices
    - `lgca_hex.py`: 2D hexagonal lattices
    - `lgca_cubic.py`: 3D cubic lattices
    - `lgca_3dmoore.py`: 3D Moore neighborhood lattices

- **Entry point**: `get_lgca()` function in `__init__.py` selects appropriate class based on geometry, identity-based (
  ib), and volume exclusion (ve) parameters.

### Key Components

- **Interactions** (`interactions.py`): Library of interaction rules (random_walk, alignment, birth, etc.)
- **Plotting** (`plots.py`, `square_plotting.py`): Visualization functions for density, flux, flow analysis
- **Extensions**: Geometry-specific extensions (`base_extensions.py`, `square_ext.py`, `cubic_ext.py`)

## Development Commands

### Testing

```bash
pytest
```

Runs the full test suite from repository root.

### Documentation

```bash
sphinx-build -b html docs/source docs/_build
```

Builds HTML documentation in `docs/_build/`.

### Dependencies

Core requirements: `numpy`, `scipy`, `sympy`, `tqdm`

```bash
pip install -r requirements.txt
```

Optional plotting dependencies:

```bash
pip install -r plotting-requirements.txt
```

## Key Usage Patterns

### Creating LGCA Instances

```python
from lgca import get_lgca

# Basic usage - returns appropriate class instance
lgca = get_lgca(geometry='hex', interaction='alignment', bc='refl')

# Identity-based with custom properties
lgca = get_lgca(ib=True, geometry='square', interaction='go_and_grow')

# Without volume exclusion
lgca = get_lgca(ve=False, geometry='cubic')
```

### Supported Geometries

- `'1d'`, `'lin'`, `'linear'` → 1D lattices
- `'square'`, `'sq'`, `'rect'` → 2D square lattices
- `'hex'`, `'hexagonal'`, `'hx'` → 2D hexagonal lattices
- `'cubic'`, `'cb'` → 3D cubic lattices
- `'moore'`, `'moore3d'` → 3D Moore lattices

### Simulation Workflow

```python
# Initialize and run simulation
lgca.timeevo(timesteps=100, record=True)

# Analysis and visualization
lgca.plot_density()
lgca.plot_flux()
lgca.live_animate_density()
```

## Important Notes

- **No package manager files**: This is a pure Python package without setup.py/pyproject.toml
- **Manual installation**: Clone repository and add to Python path if needed
- **Interaction library**: Pre-built interactions in `interactions.py` - check existing ones before creating new ones
- **Geometry inheritance**: Each geometry inherits base functionality but may override specific methods
- **State access**: Internal LGCA state (`lgca.nodes`) is always accessible for computational analysis
- **Boundary conditions**: 'periodic', 'reflecting', 'absorbing', 'inflow' (support varies by geometry)