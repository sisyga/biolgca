# AGENTS.md – Quick Start for Developers

This document provides guidelines and context for developers working on the `biolgca` project.

## Project Overview

The `biolgca` package is a Python library for simulating lattice-gas cellular automata (LGCA) in biological contexts. It supports various types of LGCA models in different lattice geometries.

### Key Files and Folders

- **`lgca/`**: Core package directory
  - `__init__.py`: Package initialization, includes factory methods
  - `base.py`: Base classes for LGCA models
  - `lgca_1d.py`, `lgca_square.py`, `lgca_hex.py`, `lgca_cubic.py`: Implementation of LGCA models for different geometries
  - `interactions.py`: Library of interaction rules
  - `ib_interactions.py`: Identity-based interaction rules
  - `plots.py`: Visualization utilities

- **`tests/`**: Test suite
  - Run tests with `pytest` from the project root
  - Files follow naming convention `*_test.py`

- **`docs/`**: Documentation
  - Source files in `docs/source/`
  - Generated with Sphinx

## Code Style

### Python Style Guidelines

- Follow PEP 8 conventions
- Use NumPy-style docstrings for functions and classes
- Class names are CamelCase, methods and functions are snake_case
- Variable names should be descriptive and indicate their purpose
- Use type hints where appropriate

## Testing

- Use the dedicated Anaconda environment for this project: `conda run -n biolgca python -m pytest -q`
- Run the test command from the project root before proposing a PR
- Test files should be placed in the `tests/` directory
- Use parameterized testing when testing similar functionality across different models
- Tests should be deterministic (use fixed random seeds where appropriate)

## Documentation

- Document all public APIs with NumPy-style docstrings
- Include examples in docstrings where helpful
- Keep the examples in docstrings and documentation synchronized with the current API

## Contributing

### PR Instructions

- **Title format**: `[Fix|Feat|Docs] <one-line summary>`
- **Body requirements**: Must include a "Testing Done" section
- Include reference to issues being addressed (if applicable)
