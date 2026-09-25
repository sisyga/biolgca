"""Renderer-independent extraction and validation of plotting data."""

from __future__ import annotations

import numpy as np


def history_steps(lgca, length, attribute, steps=None, *, implicit=False):
    """Return validated sample times; explicit data defaults to dense steps."""
    if steps is None and implicit:
        steps = getattr(lgca, attribute, None)
    values = np.arange(length) if steps is None else np.asarray(steps)
    if (values.shape != (length,) or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0)):
        raise ValueError("steps must contain one strictly increasing finite time per sample")
    return values


def resolve_animation_history(lgca, data_argument, data=None, steps=None,
                              channels=slice(None)):
    """Resolve frames and paired times without dropping channel selection.

    Explicit arrays default to dense times. Density arrays are already reduced
    and cannot be channel-selected; implicit selection requires node history.
    """
    implicit = data is None
    if implicit and data_argument == "field_t":
        raise ValueError("pass the recorded field as data, e.g. data=result.data['signal'] with "
                         "steps=result.data.steps('signal')")
    selected = channels != slice(None)
    attribute = "dens_t" if data_argument == "density_t" else "nodes_t"
    if implicit:
        if data_argument == "density_t" and selected:
            attribute = "nodes_t"
        if not hasattr(lgca, attribute):
            raise RuntimeError(f"Animation requires recorded {attribute}; "
                               "use NodeRecorder for channel selection or DensityRecorder for total density")
        data = getattr(lgca, attribute)
        if data_argument == "density_t" and selected:
            data = np.asarray(data)[..., channels]
            if data.dtype == object:
                data = np.fromiter((len(cell) for cell in data.flat), dtype=int,
                                   count=data.size).reshape(data.shape)
            elif hasattr(lgca, "occupied"):
                data = data > 0
            data = data.sum(axis=-1)
    elif data_argument == "density_t" and selected:
        raise ValueError("Explicit density data is already reduced; select channels before passing data")
    data = np.asarray(data)
    if not len(data):
        raise ValueError("Animation requires at least one sampled frame")
    times = history_steps(lgca, len(data), "dens_steps" if attribute == "dens_t" else "nodes_steps",
                          steps, implicit=implicit)
    return data, times


def label_history_axis(ax, steps, offset=0):
    """Label image sample rows with their actual, possibly nonuniform times."""
    indices = np.unique(np.linspace(0, len(steps) - 1, min(8, len(steps))).astype(int))
    ax.set_yticks(indices + offset, labels=[str(steps[index]) for index in indices])
    # Rows of a sparse history are samples, not equally spaced times.
    dense = len(steps) < 2 or np.all(np.diff(np.asarray(steps)) == 1)
    ax.set_ylabel("Time step $k$" if dense else "Recorded time step $k$")


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
