"""Renderer-independent extraction and validation of plotting data."""

from __future__ import annotations

import numpy as np


def validate_species(lgca, species):
    """Validate and normalize an optional species index."""

    if species is None:
        return None
    n_species = getattr(lgca, "n_species", 1)
    if isinstance(species, bool) or not isinstance(species, (int, np.integer)):
        raise ValueError("species must be an integer index")
    if not 0 <= int(species) < n_species:
        raise ValueError(f"species must be between 0 and {n_species - 1}")
    return int(species)


def select_density(lgca, density=None, channels=slice(None), species=None):
    """Return a spatial density array for aggregate or one species."""

    spatial_ndim = len(lgca.dims)
    if density is None:
        nodes = lgca.nodes[lgca.nonborder]
        if nodes.ndim == spatial_ndim + 2:
            species = validate_species(lgca, species)
            if species is None:
                density = nodes[..., channels].sum(axis=(-2, -1))
            else:
                density = nodes[..., species, channels].sum(-1)
        else:
            if species is not None:
                raise ValueError("species selection requires a multispecies LGCA")
            density = nodes[..., channels].sum(-1)
    else:
        density = np.asarray(density)
        if density.ndim == spatial_ndim + 1:
            species = validate_species(lgca, species)
            density = density.sum(-1) if species is None else density[..., species]
        elif species is not None:
            raise ValueError("species can only select a species axis in density data")

    density = np.asarray(density)
    if density.shape != tuple(lgca.dims):
        raise ValueError(f"density must have spatial shape {tuple(lgca.dims)}, got {density.shape}")
    return density


def select_density_history(lgca, density_history, species=None):
    """Return a time-by-space aggregate or per-species density history."""

    density_history = np.asarray(density_history)
    spatial_ndim = len(lgca.dims)
    if density_history.ndim == spatial_ndim + 2:
        species = validate_species(lgca, species)
        density_history = (
            density_history.sum(-1)
            if species is None
            else density_history[..., species]
        )
    elif species is not None:
        raise ValueError("species can only select a species axis in density data")
    expected = tuple(lgca.dims)
    if density_history.ndim != spatial_ndim + 1 or density_history.shape[1:] != expected:
        raise ValueError(
            f"density history must have shape (time, {', '.join(map(str, expected))}), "
            f"got {density_history.shape}"
        )
    return density_history


def select_scalar_field(lgca, field):
    """Return an unpadded scalar field with the model's spatial shape."""

    field = np.asanyarray(field)
    if field.shape != tuple(lgca.dims):
        field = field[lgca.nonborder]
    if field.shape != tuple(lgca.dims):
        raise ValueError(
            f"scalar field must have spatial shape {tuple(lgca.dims)}, got {field.shape}"
        )
    return field
