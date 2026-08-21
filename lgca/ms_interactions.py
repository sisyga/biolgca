"""Interaction rules for classical multi-species LGCA."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from lgca.interactions import tanh_switch


def _validate_species_vector(lgca, name, value):
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(lgca.n_species, float(values))
    if values.shape != (lgca.n_species,):
        raise ValueError(f"{name} must be a scalar or have shape ({lgca.n_species},).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return values


def _validate_mutation_matrix(lgca, mutation_matrix):
    matrix = np.asarray(mutation_matrix, dtype=float)
    expected = (lgca.n_species, lgca.n_species)
    if matrix.shape != expected:
        raise ValueError(f"mutation_matrix must have shape {expected}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("mutation_matrix must contain finite numeric values.")
    if np.any(matrix < 0):
        raise ValueError("mutation_matrix entries must be non-negative.")
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("mutation_matrix rows must sum to 1.")
    return matrix


def mutation_matrix_from_trait_bins(traits, std):
    """Discretize a Gaussian mutation kernel over trait-centered species bins."""
    traits = np.asarray(traits, dtype=float)
    if traits.ndim != 1:
        raise ValueError("traits must be one-dimensional.")
    if not np.all(np.isfinite(traits)):
        raise ValueError("traits must contain finite numeric values.")
    if std < 0:
        raise ValueError("std must be non-negative.")
    if std == 0:
        return np.eye(len(traits))

    sort_idx = np.argsort(traits)
    sorted_traits = traits[sort_idx]
    if np.any(np.diff(sorted_traits) <= 0):
        raise ValueError("traits must contain distinct values to derive a mutation kernel.")

    edges = np.empty(len(traits) + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = 0.5 * (sorted_traits[:-1] + sorted_traits[1:])

    rows_sorted = ndtr((edges[1:][None, :] - traits[:, None]) / std)
    rows_sorted -= ndtr((edges[:-1][None, :] - traits[:, None]) / std)
    matrix = np.empty_like(rows_sorted)
    matrix[:, sort_idx] = rows_sorted
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def _resolve_mutation_matrix(lgca, trait_name=None, std_name=None):
    params = lgca.interaction_params
    if "mutation_matrix" in params:
        return params["mutation_matrix"]
    if trait_name is not None and std_name in params:
        return mutation_matrix_from_trait_bins(params[trait_name], params[std_name])
    return np.eye(lgca.n_species)


def _sample_offspring_by_species(rng, births_by_parent, mutation_matrix):
    """Sample offspring phenotypes, avoiding work for deterministic identity mutation."""

    births_by_parent = np.asarray(births_by_parent, dtype=np.int64)
    mutation_matrix = np.asarray(mutation_matrix, dtype=float)
    if np.array_equal(mutation_matrix, np.eye(mutation_matrix.shape[0])):
        return births_by_parent.copy()
    offspring_by_parent = rng.multinomial(births_by_parent, mutation_matrix)
    return offspring_by_parent.sum(axis=-2)


def _offspring_by_species(lgca, births_by_parent):
    return _sample_offspring_by_species(
        lgca.rng,
        births_by_parent,
        lgca.interaction_params["mutation_matrix"],
    )


def _redistribute_species_counts(lgca, species_counts, channel_weights=None):
    channel_weights = lgca.channel_weights if channel_weights is None else channel_weights
    return lgca.rng.multinomial(
        np.asarray(species_counts, dtype=np.int64),
        channel_weights,
    ).astype(lgca.nodes.dtype)


def birth(lgca):
    """Multispecies NoVE birth interaction with species-level birth rates."""
    species_counts = lgca.nodes.sum(axis=-1).astype(np.int64)
    total = species_counts.sum(axis=-1)
    rho = total / lgca.interaction_params["capacity"]
    birth_prob = np.clip(
        lgca.interaction_params["r_b"] * (1 - rho[..., None]),
        0,
        1,
    )
    births_by_parent = lgca.rng.binomial(species_counts, birth_prob)
    final_counts = species_counts + _offspring_by_species(lgca, births_by_parent)
    lgca.nodes = _redistribute_species_counts(lgca, final_counts)


def birthdeath(lgca):
    """Multispecies NoVE birth--death interaction with species-level birth rates."""
    species_counts = lgca.nodes.sum(axis=-1).astype(np.int64)
    total = species_counts.sum(axis=-1)
    deaths = lgca.rng.binomial(species_counts, lgca.interaction_params["r_d"])
    survivors = species_counts - deaths
    rho = total / lgca.interaction_params["capacity"]
    birth_prob = np.clip(
        lgca.interaction_params["r_b"] * (1 - rho[..., None]),
        0,
        1,
    )
    births_by_parent = lgca.rng.binomial(species_counts, birth_prob)
    final_counts = survivors + _offspring_by_species(lgca, births_by_parent)
    lgca.nodes = _redistribute_species_counts(lgca, final_counts)


def go_or_grow(lgca):
    """Multispecies NoVE go-or-grow interaction with species-level kappa."""
    newnodes = np.zeros_like(lgca.nodes)
    velocity_weights = np.full(lgca.velocitychannels, 1 / lgca.velocitychannels)
    n_m = lgca.nodes[..., :lgca.velocitychannels].sum(axis=-1).astype(np.int64)
    n_r = lgca.nodes[..., lgca.velocitychannels:].sum(axis=-1).astype(np.int64)
    total = n_m.sum(axis=-1) + n_r.sum(axis=-1)
    rho = total / lgca.interaction_params["capacity"]

    n_m -= lgca.rng.binomial(n_m, lgca.interaction_params["r_d"])
    n_r -= lgca.rng.binomial(n_r, lgca.interaction_params["r_d"])

    switch_prob = tanh_switch(
        rho[..., None],
        kappa=lgca.interaction_params["kappa"],
        theta=lgca.interaction_params["theta"],
    )
    moving_to_rest = lgca.rng.binomial(n_m, switch_prob)
    rest_to_moving = lgca.rng.binomial(n_r, 1 - switch_prob)
    n_m = n_m + rest_to_moving - moving_to_rest
    n_r = n_r + moving_to_rest - rest_to_moving

    post_death_total = n_m.sum(axis=-1) + n_r.sum(axis=-1)
    birth_prob = np.clip(
        lgca.interaction_params["r_b"]
        * (1 - post_death_total / lgca.interaction_params["capacity"]),
        0,
        1,
    )
    births_by_parent = lgca.rng.binomial(n_r, birth_prob[..., None])
    n_r += _offspring_by_species(lgca, births_by_parent)

    newnodes[..., :lgca.velocitychannels] = lgca.rng.multinomial(n_m, velocity_weights)
    newnodes[..., lgca.velocitychannels] = n_r.astype(lgca.nodes.dtype)

    lgca.nodes = newnodes


def excitable_medium_ms(lgca):
    """Excitable medium for two species.

    Species 0 remains in its rest channels while species 1 undergoes a
    birth--death reaction followed by a random walk.
    """
    if getattr(lgca, "n_species", 1) != 2:
        raise ValueError("excitable_medium_ms requires a multi-species LGCA with exactly two species.")
    if not np.issubdtype(lgca.nodes.dtype, np.bool_):
        raise ValueError("excitable_medium_ms requires volume exclusion.")
    if lgca.restchannels < 1:
        raise ValueError("excitable_medium_ms requires at least one rest channel.")

    n_x = lgca.nodes[..., 1, :lgca.velocitychannels].sum(-1)
    n_y = lgca.nodes[..., 0, lgca.velocitychannels:].sum(-1)

    rho_x = n_x / lgca.velocitychannels
    rho_y = n_y / lgca.restchannels
    p_xp = rho_x ** 2 * (
        1 + (rho_y + lgca.interaction_params["beta"]) / lgca.interaction_params["alpha"]
    )
    p_xm = rho_x ** 3 + rho_x * (
        rho_y + lgca.interaction_params["beta"]
    ) / lgca.interaction_params["alpha"]
    p_yp = rho_x
    p_ym = rho_y

    dn_y = (lgca.rng.random(n_y.shape) < p_yp).astype(np.int8)
    dn_y -= lgca.rng.random(n_y.shape) < p_ym

    for _ in range(lgca.interaction_params["N"]):
        dn_x = (lgca.rng.random(n_x.shape) < p_xp).astype(np.int8)
        dn_x -= lgca.rng.random(n_x.shape) < p_xm
        n_x += dn_x
        n_x = np.clip(n_x, 0, lgca.velocitychannels)
        rho_x = n_x / lgca.velocitychannels
        p_xp = rho_x ** 2 * (
            1 + (rho_y + lgca.interaction_params["beta"]) / lgca.interaction_params["alpha"]
        )
        p_xm = rho_x ** 3 + rho_x * (
            rho_y + lgca.interaction_params["beta"]
        ) / lgca.interaction_params["alpha"]

    n_y += dn_y
    n_y = np.clip(n_y, 0, lgca.restchannels)

    newnodes = np.zeros_like(lgca.nodes)
    v_idx = np.arange(lgca.velocitychannels)
    r_idx = np.arange(lgca.restchannels)
    newnodes[..., 1, :lgca.velocitychannels] = (
        v_idx < n_x[..., None]
    ).astype(lgca.nodes.dtype)
    newnodes[..., 0, lgca.velocitychannels:] = (
        r_idx < n_y[..., None]
    ).astype(lgca.nodes.dtype)
    newnodes[..., 1, :lgca.velocitychannels] = lgca.rng.permuted(
        newnodes[..., 1, :lgca.velocitychannels], axis=-1
    )
    lgca.nodes = newnodes
