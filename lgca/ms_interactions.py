"""Interaction rules for classical multi-species LGCA."""

from __future__ import annotations

import numpy as np


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

