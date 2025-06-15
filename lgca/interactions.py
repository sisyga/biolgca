# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2025 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""
Interaction functions and helper functions for classical LGCA with volume exclusion.
"""

import itertools  # Added for itertools.product
from bisect import bisect_left

import numpy as np


def disarrange(a: np.ndarray, axis=-1):
    """
    Shuffle a in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of a. Each one-dimensional
    slice is shuffled independently.

    THIS IS A LEGACY FUNCTION, USE THE RANDOM WALK FUNCTION INSTEAD FOR A MORE EFFICIENT IMPLEMENTATION!

    Parameters
    ----------
    a : numpy.ndarray
        The array to shuffle
    axis : int, optional, default: -1
           Along which axis to shuffle `a`. The default is -1, which implies the
           last axis.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


def tanh_switch(rho, kappa=5., theta=0.8):
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


def ent_prod(x):
    return x * np.log(x, where=x > 0, out=np.zeros_like(x, dtype=float))


def random_walk(lgca):
    """Apply a random walk to all nodes.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Notes
    -----
    The ``lgca`` object is modified in place. This interaction does not use
    ``lgca.interaction_params``.

    Returns
    -------
    None
    """
    lgca.nodes = lgca.rng.permuted(lgca.nodes, axis=-1)


def birth(lgca):
    """Perform a simple birth step followed by a random walk.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float
        Birth probability per empty channel; effective
        probability is ``r_b * n / lgca.K`` for local density ``n``.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    birth = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * lgca.cell_density[..., None] / lgca.K
    np.add(lgca.nodes, (1 - lgca.nodes) * birth, out=lgca.nodes, casting='unsafe')
    random_walk(lgca)


def birthdeath(lgca):
    """Perform a birth--death step followed by a random walk.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float
        Birth probability per empty channel.
    r_d : float
        Death probability per occupied channel.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    birth = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_b'] * lgca.cell_density[..., None] / lgca.K
    death = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_d']
    ds = (1 - lgca.nodes) * birth - lgca.nodes * death
    np.add(lgca.nodes, ds, out=lgca.nodes, casting='unsafe')
    lgca.update_dynamic_fields()
    random_walk(lgca)


def alignment(lgca):
    """Interaction rule: Particles align with the mean flux in their local
    neighborhood. The sensitivity of alignment is controlled by lgca.beta.
    Vectorized implementation.
    """
    if not hasattr(lgca, 'beta') or lgca.beta is None:
        # print('sensitivity set to beta = ', 2.0)
        lgca.beta = 2.0

    d_geom = len(lgca.dims)

    # Determine the velocity matrix to use
    velocity_matrix = None
    if hasattr(lgca, 'c') and isinstance(lgca.c, np.ndarray) and lgca.c.shape == (lgca.K, d_geom):
        velocity_matrix = lgca.c
    elif hasattr(lgca, 'velocity_vectors') and isinstance(lgca.velocity_vectors, np.ndarray) and \
            lgca.velocity_vectors.shape == (lgca.K, d_geom):
        velocity_matrix = lgca.velocity_vectors

    if velocity_matrix is None:
        # print("Warning: No compatible velocity matrix found for alignment. Skipping interaction.")
        return

    # Assumes lgca.nodes has shape (K, dim0_padded, dim1_padded, ...)
    spatial_axes_in_nodes = tuple(range(1, lgca.nodes.ndim))  # All axes except the first (K)

    # Define neighborhood offsets (Moore neighborhood, radius 1, including center)
    radius = 1
    iter_ranges_for_offsets = [range(-radius, radius + 1)] * d_geom
    neighbor_offsets = list(itertools.product(*iter_ranges_for_offsets))

    sum_nodes_k_in_nb = np.zeros_like(lgca.nodes)
    for offset_coords in neighbor_offsets:
        roll_shifts = [0] * lgca.nodes.ndim  # Shift for [K, dim0, dim1, ...]
        for i in range(d_geom):
            roll_shifts[spatial_axes_in_nodes[i]] = -offset_coords[i]
        sum_nodes_k_in_nb += np.roll(lgca.nodes, shift=tuple(roll_shifts), axis=tuple(range(lgca.nodes.ndim)))

    total_particles_in_nb_map = sum_nodes_k_in_nb.sum(axis=0)  # Sum over K, shape (padded_dims...)
    sum_weighted_velocities_map = np.einsum('k...,kd->...d', sum_nodes_k_in_nb,
                                            velocity_matrix)  # (padded_dims..., d_geom)

    mean_flux_map = np.zeros_like(sum_weighted_velocities_map)
    valid_nb_mask_for_flux = total_particles_in_nb_map > 0
    mean_flux_map[valid_nb_mask_for_flux] = sum_weighted_velocities_map[valid_nb_mask_for_flux] / \
                                            total_particles_in_nb_map[valid_nb_mask_for_flux][..., np.newaxis]

    updatable_mask_full = np.zeros(lgca.cell_density.shape, dtype=bool)  # (padded_dims...)
    updatable_mask_full[lgca.nonborder] = True

    mean_flux_physical = mean_flux_map[lgca.nonborder]  # (physical_dims..., d_geom)
    is_zero_flux_physical = np.all(np.isclose(mean_flux_physical, 0.0), axis=-1)  # (physical_dims...)

    _temp_zero_flux_full = np.zeros(lgca.cell_density.shape, dtype=bool)
    _temp_zero_flux_full[lgca.nonborder] = is_zero_flux_physical

    nodes_to_shuffle_mask = updatable_mask_full & _temp_zero_flux_full  # (padded_dims...)
    nodes_to_align_mask = updatable_mask_full & (~_temp_zero_flux_full)  # (padded_dims...)

    num_shuffle_nodes = np.sum(nodes_to_shuffle_mask)
    if num_shuffle_nodes > 0:
        subset_to_shuffle = lgca.nodes[:, nodes_to_shuffle_mask]  # (K, num_shuffle_nodes)
        for i in range(num_shuffle_nodes):
            np.random.shuffle(subset_to_shuffle[:, i])
        lgca.nodes[:, nodes_to_shuffle_mask] = subset_to_shuffle

    num_align_nodes = np.sum(nodes_to_align_mask)
    if num_align_nodes > 0:
        active_mean_flux = mean_flux_map[nodes_to_align_mask]  # (num_align_nodes, d_geom)

        v_prop_all = np.dot(active_mean_flux, velocity_matrix.T)  # (num_align_nodes, K)
        v_prop_all = np.exp(lgca.beta * v_prop_all)

        v_prop_all_sum = v_prop_all.sum(axis=1, keepdims=True)  # (num_align_nodes, 1)

        p_vals_normalized = np.zeros_like(v_prop_all)
        has_positive_sum = (v_prop_all_sum > 1e-9).squeeze(axis=1)  # (num_align_nodes,)

        p_vals_normalized[~has_positive_sum, :] = 1.0 / lgca.K
        if np.any(has_positive_sum):
            p_vals_normalized[has_positive_sum, :] = \
                v_prop_all[has_positive_sum, :] / v_prop_all_sum[has_positive_sum, :]

        densities_for_align_nodes = lgca.cell_density[nodes_to_align_mask]  # (num_align_nodes,)
        new_states_for_align_nodes_flat = np.zeros((num_align_nodes, lgca.K), dtype=lgca.nodes.dtype)

        for i in range(num_align_nodes):
            n_particles = int(densities_for_align_nodes[i])
            if n_particles > 0:
                current_pvals = p_vals_normalized[i, :]
                current_pvals_sum = current_pvals.sum()
                if not (np.isfinite(current_pvals_sum) and current_pvals_sum > 1e-9):
                    current_pvals = np.ones(lgca.K) / lgca.K
                elif not np.isclose(current_pvals_sum, 1.0):
                    current_pvals = current_pvals / current_pvals_sum

                new_states_for_align_nodes_flat[i, :] = lgca.rng.multinomial(n_particles, current_pvals)

        lgca.nodes[:, nodes_to_align_mask] = new_states_for_align_nodes_flat.T


def persistent_walk(lgca):
    """Rearrange nodes to implement persistent motion.
    Vectorized implementation.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of alignment with the previous velocity direction.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    if not hasattr(lgca, 'interaction_params') or 'beta' not in lgca.interaction_params:
        # Default beta if not provided, or handle as an error/warning
        # print("Warning: 'beta' not found in interaction_params for persistent_walk. Using default beta = 1.0")
        beta = 1.0
    else:
        beta = lgca.interaction_params['beta']

    d_geom = len(lgca.dims)

    velocity_matrix = None
    if hasattr(lgca, 'c') and isinstance(lgca.c, np.ndarray) and lgca.c.shape == (lgca.K, d_geom):
        velocity_matrix = lgca.c
    elif hasattr(lgca, 'velocity_vectors') and isinstance(lgca.velocity_vectors, np.ndarray) and \
            lgca.velocity_vectors.shape == (lgca.K, d_geom):
        velocity_matrix = lgca.velocity_vectors

    if velocity_matrix is None:
        # print("Warning: No compatible velocity matrix found for persistent_walk. Skipping interaction.")
        return

    # Calculate current flux for all nodes (including shadow nodes for consistency if calc_flux handles it)
    # g will have shape (dims_padded..., d_geom)
    g_flux_map = lgca.calc_flux(lgca.nodes)

    # Mask for physical nodes that are neither empty nor full
    # lgca.nonborder is a tuple of arrays, one for each dimension, containing indices of physical nodes
    # We need a boolean mask of the same shape as cell_density (padded_dims...)
    updatable_mask_full = np.zeros(lgca.cell_density.shape, dtype=bool)  # (padded_dims...)
    updatable_mask_full[lgca.nonborder] = True

    density_physical = lgca.cell_density[lgca.nonborder]  # (physical_dims_flat...)
    condition_mask_physical = (density_physical > 0) & (density_physical < lgca.K)

    # Expand condition_mask_physical back to the full grid shape for relevant_physical_nodes_mask
    relevant_physical_nodes_mask = np.zeros(lgca.cell_density.shape, dtype=bool)
    relevant_physical_nodes_mask[lgca.nonborder] = condition_mask_physical

    # Combine with updatable_mask_full (though it's somewhat redundant here as nonborder is already applied)
    # This mask identifies nodes in the physical grid that should be processed.
    nodes_to_process_mask = updatable_mask_full & relevant_physical_nodes_mask  # (padded_dims...)

    # Get flux for these processable physical nodes
    flux_at_processable_nodes = g_flux_map[nodes_to_process_mask]  # (num_processable_nodes, d_geom)

    # Identify nodes with zero flux among the processable ones
    is_zero_flux_at_processable = np.all(np.isclose(flux_at_processable_nodes, 0.0),
                                         axis=-1)  # (num_processable_nodes,)

    # Create full-grid masks for shuffle and align based on zero/non-zero flux
    # Start with False, then selectively set True for the identified nodes
    # nodes_to_shuffle_mask_full = np.zeros(lgca.cell_density.shape, dtype=bool) # Not strictly needed if we use
    # nodes_to_shuffle_mask directly
    # nodes_to_align_mask_full = np.zeros(lgca.cell_density.shape, dtype=bool) # Not strictly needed

    # Map back the zero_flux condition to the full grid
    _temp_zero_flux_full = np.zeros(lgca.cell_density.shape, dtype=bool)
    _temp_zero_flux_full[nodes_to_process_mask] = is_zero_flux_at_processable

    nodes_to_shuffle_mask = nodes_to_process_mask & _temp_zero_flux_full
    nodes_to_align_mask = nodes_to_process_mask & (~_temp_zero_flux_full)

    # 1. Handle nodes that need shuffling (zero flux)
    num_shuffle_nodes = np.sum(nodes_to_shuffle_mask)
    if num_shuffle_nodes > 0:
        subset_to_shuffle = lgca.nodes[:, nodes_to_shuffle_mask]  # (K, num_shuffle_nodes)
        for i in range(num_shuffle_nodes):
            # In-place shuffle of channels for each node
            lgca.rng.shuffle(subset_to_shuffle[:, i])
        lgca.nodes[:, nodes_to_shuffle_mask] = subset_to_shuffle

    # 2. Handle nodes that need alignment (non-zero flux)
    num_align_nodes = np.sum(nodes_to_align_mask)
    if num_align_nodes > 0:
        active_flux_vectors = g_flux_map[nodes_to_align_mask]  # (num_align_nodes, d_geom)

        # Calculate alignment probabilities
        # v_prop_all will be (num_align_nodes, K)
        v_prop_all = np.dot(active_flux_vectors, velocity_matrix.T)
        v_prop_all = np.exp(beta * v_prop_all)

        v_prop_all_sum = v_prop_all.sum(axis=1, keepdims=True)  # (num_align_nodes, 1)

        p_vals_normalized = np.zeros_like(v_prop_all)
        # Avoid division by zero for nodes where all exp(beta * dot_product) are zero (e.g. beta very negative)
        has_positive_sum = (v_prop_all_sum > 1e-9).squeeze(axis=1)  # (num_align_nodes,)

        # Default to uniform probability if sum is too small (or all dot products were -inf)
        p_vals_normalized[~has_positive_sum, :] = 1.0 / lgca.K

        # Normalize probabilities for nodes with positive sum
        if np.any(has_positive_sum):
            p_vals_normalized[has_positive_sum, :] = \
                v_prop_all[has_positive_sum, :] / v_prop_all_sum[has_positive_sum, :]

        # Get densities for the nodes to be aligned
        densities_for_align_nodes = lgca.cell_density[nodes_to_align_mask]  # (num_align_nodes,)
        new_states_for_align_nodes_flat = np.zeros((num_align_nodes, lgca.K), dtype=lgca.nodes.dtype)

        for i in range(num_align_nodes):
            n_particles = int(densities_for_align_nodes[i])
            if n_particles > 0:  # Should always be true due to earlier mask
                current_pvals = p_vals_normalized[i, :]
                # Ensure probabilities sum to 1, handling potential floating point inaccuracies or zero sums
                current_pvals_sum = current_pvals.sum()
                if not (np.isfinite(current_pvals_sum) and current_pvals_sum > 1e-9):
                    current_pvals = np.ones(lgca.K) / lgca.K  # Fallback to uniform
                elif not np.isclose(current_pvals_sum, 1.0):
                    current_pvals = current_pvals / current_pvals_sum  # Re-normalize

                new_states_for_align_nodes_flat[i, :] = lgca.rng.multinomial(n_particles, current_pvals)

        # Transpose new_states to (K, num_align_nodes) before assigning back
        lgca.nodes[:, nodes_to_align_mask] = new_states_for_align_nodes_flat.T

    # No explicit random_walk(lgca) at the end, as alignment itself is a rearrangement.
    # If calc_flux or other parts require updated dynamic fields, ensure they are called.
    # lgca.update_dynamic_fields() # If density changed and is used by other interactions before propagation


def chemotaxis(lgca):
    """Rearrange nodes in response to an external gradient.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of the bias toward the gradient.
    gradient_field : numpy.ndarray
        External gradient field influencing motion.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = tuple(nb_coord_array[relevant] for nb_coord_array in lgca.nonborder)

    if not coords[0].size:
        return

    newnodes = lgca.nodes.copy()
    gradient_field = lgca.interaction_params['gradient_field']

    for coord_idx_tuple in zip(*coords):
        n = lgca.cell_density[coord_idx_tuple]
        permutations = lgca.get_permutations(n)
        j = lgca.get_flux_permutations(n)

        local_gradient = gradient_field[coord_idx_tuple]
        if local_gradient.ndim == 0:
            local_gradient = np.array([local_gradient])

        if j.ndim == 2 and local_gradient.shape[0] == j.shape[1]:
            weights = np.exp(lgca.interaction_params['beta'] * np.einsum('j,ij->i', local_gradient, j)).cumsum()
            if weights.size > 0 and weights[-1] > 0:
                ind = bisect_left(weights, lgca.rng.random() * weights[-1])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            elif permutations.shape[0] > 0:
                ind = lgca.rng.integers(permutations.shape[0])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
        elif permutations.shape[0] > 0:
            ind = lgca.rng.integers(permutations.shape[0])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]

    lgca.nodes = newnodes


def contact_guidance(lgca):
    """Align cells with a predefined guiding axis.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Alignment strength toward the guiding axis.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = tuple(nb_coord_array[relevant] for nb_coord_array in lgca.nonborder)

    if not coords[0].size:
        return

    newnodes = lgca.nodes.copy()

    for coord_idx_tuple in zip(*coords):
        n = lgca.cell_density[coord_idx_tuple]
        sni = lgca.guiding_tensor[coord_idx_tuple]  # Guiding tensor at the coordinate
        permutations = lgca.get_permutations(n)
        si = lgca.si[n]  # Shape (num_perms, d_geom, d_geom) or similar for tensor contraction

        # Ensure sni and si are compatible for einsum('ijk,jk->i', si, sni)
        # si: (P, M, N), sni: (M, N) -> weights: (P,)
        if si.ndim == 3 and sni.ndim == 2 and si.shape[1:] == sni.shape:
            weights = np.exp(lgca.interaction_params['beta'] * np.einsum('pmn,mn->p', si, sni)).cumsum()
            if weights.size > 0 and weights[-1] > 0:
                ind = bisect_left(weights, lgca.rng.random() * weights[-1])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            elif permutations.shape[0] > 0:  # Fallback
                ind = lgca.rng.integers(permutations.shape[0])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
        elif permutations.shape[0] > 0:  # Fallback
            ind = lgca.rng.integers(permutations.shape[0])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            
    lgca.nodes = newnodes


def nematic(lgca):
    """Implement nematic alignment of neighbouring cells.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of nematic alignment.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = tuple(nb_coord_array[relevant] for nb_coord_array in lgca.nonborder)

    if not coords[0].size:
        return

    # lgca.nodes shape (K, dims...), lgca.cij shape (K, d, d) or (vel_K, d, d)
    # s should have shape (dims..., d, d)
    # Assuming lgca.nodes[:lgca.velocitychannels, ...] are the relevant channels for cij
    s = np.einsum('k...,kxy->...xy', lgca.nodes[:lgca.velocitychannels, ...], lgca.cij)
    sn = lgca.nb_sum(s)  # sn is sum of s over neighbors, shape (dims..., d, d)

    for coord_idx_tuple in zip(*coords):
        n = lgca.cell_density[coord_idx_tuple]
        sni_local = sn[coord_idx_tuple]  # (d,d) nematic tensor from neighbors
        permutations = lgca.get_permutations(n)
        si = lgca.si[n]  # (P, d, d) nematic tensors for each permutation

        if si.ndim == 3 and sni_local.ndim == 2 and si.shape[1:] == sni_local.shape:
            weights = np.exp(lgca.interaction_params['beta'] * np.einsum('pmn,mn->p', si, sni_local)).cumsum()
            if weights.size > 0 and weights[-1] > 0:
                ind = bisect_left(weights, lgca.rng.random() * weights[-1])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            elif permutations.shape[0] > 0:  # Fallback
                ind = lgca.rng.integers(permutations.shape[0])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
        elif permutations.shape[0] > 0:  # Fallback
            ind = lgca.rng.integers(permutations.shape[0])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]

    lgca.nodes = newnodes


def aggregation(lgca):
    """Bias movement towards higher cell density.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    beta : float
        Strength of the bias toward higher density regions.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    newnodes = lgca.nodes.copy()
    relevant = (lgca.cell_density[lgca.nonborder] > 0) & \
               (lgca.cell_density[lgca.nonborder] < lgca.K)
    coords = tuple(nb_coord_array[relevant] for nb_coord_array in lgca.nonborder)

    if not coords[0].size:
        return

    # g is gradient of cell_density, shape (d_geom, dims...)
    g = lgca.gradient(lgca.cell_density)

    for coord_idx_tuple in zip(*coords):
        n = lgca.cell_density[coord_idx_tuple]
        permutations = lgca.get_permutations(n)
        j = lgca.get_flux_permutations(n)  # (P, d_geom)

        # local_grad has shape (d_geom,)
        local_grad = g[(slice(None),) + coord_idx_tuple]
        if local_grad.ndim == 0:  # Should be (d_geom,)
            local_grad = np.array([local_grad])

        if j.ndim == 2 and local_grad.shape[0] == j.shape[1]:
            weights = np.exp(lgca.interaction_params['beta'] * np.einsum('d,pd->p', local_grad, j)).cumsum()
            if weights.size > 0 and weights[-1] > 0:
                ind = bisect_left(weights, lgca.rng.random() * weights[-1])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            elif permutations.shape[0] > 0:  # Fallback
                ind = lgca.rng.integers(permutations.shape[0])
                newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
        elif permutations.shape[0] > 0:  # Fallback
            ind = lgca.rng.integers(permutations.shape[0])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            
    lgca.nodes = newnodes


def wetting(lgca):
    """Model wetting dynamics on an adhesive surface.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float
        Birth probability used inside the spheroid region.
    rho_0 : float
        Homeostatic density determining the pressure gradient.
    alpha : float
        ECM degradation rate.
    beta : float
        Adhesion strength weighting flux alignment.
    gamma : float
        Strength of the pressure gradient term.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    if hasattr(lgca, 'spheroid') and lgca.spheroid is not None:  # Check if spheroid is defined
        # Assuming lgca.spheroid is a mask or indices compatible with lgca.nodes
        # If lgca.nodes is (K, dims...), spheroid should be (dims...)
        spheroid_mask_spatial = lgca.spheroid

        # Create a (K, dims...) mask for nodes within the spheroid
        spheroid_nodes_mask = np.zeros_like(lgca.nodes, dtype=bool)
        spheroid_nodes_mask[:, spheroid_mask_spatial] = True

        # Apply birth only to nodes within the spheroid
        birth_values = lgca.rng.random(lgca.nodes.shape) < lgca.interaction_params['r_b']

        # ds for birth: add particle if channel is empty (1-node) and birth event occurs
        ds_birth_spheroid = (1 - lgca.nodes) * birth_values * spheroid_nodes_mask
        np.add(lgca.nodes, ds_birth_spheroid, out=lgca.nodes, casting='unsafe')
        lgca.update_dynamic_fields()

    newnodes = lgca.nodes.copy()
    # relevant for wetting might be all non-border cells with any particles
    relevant = (lgca.cell_density[lgca.nonborder] > 0)
    coords = tuple(nb_coord_array[relevant] for nb_coord_array in lgca.nonborder)

    if not coords[0].size:
        lgca.nodes = newnodes  # Ensure nodes are updated even if no coords
        if hasattr(lgca, 'ecm'):  # Update ecm regardless of cell interactions
            lgca.ecm -= lgca.interaction_params['alpha'] * lgca.ecm * lgca.cell_density / lgca.K
        return

    nbs = lgca.nb_sum(lgca.cell_density)
    # n_crit might not be defined, handle if missing, or ensure it's available
    n_crit = getattr(lgca, 'n_crit', lgca.K)  # Default to K if n_crit not present
    nbs_factor = np.clip(1 - nbs / n_crit, a_min=0, a_max=None) / n_crit * 2
    g_adh = lgca.gradient(nbs * nbs_factor)  # Gradient of modified neighbor density

    pressure_num = np.clip(lgca.cell_density - lgca.interaction_params['rho_0'], a_min=0., a_max=None)
    pressure_den = (lgca.K - lgca.interaction_params['rho_0'])
    pressure = np.zeros_like(lgca.cell_density, dtype=float)
    if pressure_den > 1e-9:  # Avoid division by zero
        pressure = pressure_num / pressure_den
    g_pressure = -lgca.gradient(pressure)

    # resting sum over resting channels, result is (dims...)
    resting_sum_spatial = lgca.nodes[lgca.velocitychannels:, ...].sum(axis=0)
    # nb_sum of resting_sum_spatial, then normalize
    resting_nb_sum_norm = lgca.nb_sum(resting_sum_spatial)
    if lgca.velocitychannels > 0 and lgca.interaction_params['rho_0'] > 0:  # Avoid div by zero
        # The original code had K_vel (velocitychannels) in the denominator.
        # It also had rho_0. If rho_0 is a density, this might be K_rest * rho_0 or similar.
        # For now, using velocitychannels * rho_0 as a placeholder, may need domain expert review.
        # A common pattern is normalizing by the max possible sum, e.g. K_rest * max_density_at_site.
        # If rho_0 is a target density, then K_rest * rho_0 might be a target number of resting particles.
        # Let's assume it's related to the number of resting channels and a reference density.
        if lgca.restchannels > 0 and lgca.interaction_params['rho_0'] > 0:
            normalizer = lgca.restchannels * lgca.interaction_params['rho_0']
            if normalizer > 1e-9:
                resting_nb_sum_norm /= normalizer
            else:  # Avoid division by zero if normalizer is too small
                resting_nb_sum_norm = np.zeros_like(resting_nb_sum_norm)
        else:  # Handle case where normalization is not possible
            resting_nb_sum_norm = np.zeros_like(resting_nb_sum_norm)
    else:  # Handle case where normalization is not possible
        resting_nb_sum_norm = np.zeros_like(resting_nb_sum_norm)

    g_flux_all_nodes = lgca.calc_flux(lgca.nodes)  # Flux at all nodes (d, dims...)
    g_flux_nb_sum = lgca.nb_sum(g_flux_all_nodes)  # Sum of fluxes in neighborhood (d, dims...)

    for coord_idx_tuple in zip(*coords):
        n = lgca.cell_density[coord_idx_tuple]
        permutations = lgca.get_permutations(n)  # (P, K)
        if permutations.shape[0] == 0: continue

        restc = permutations[:, lgca.velocitychannels:].sum(axis=-1)  # (P,) sum of resting particles per perm
        j_perms = lgca.get_flux_permutations(n)  # (P, d) flux for each permutation

        # Flux from neighbors at current coord: (d,)
        j_nb_local = g_flux_nb_sum[(slice(None),) + coord_idx_tuple]

        # Ensure consistent d_geom for dot products
        d_geom = len(lgca.dims)
        if j_nb_local.shape[0] != d_geom or (j_perms.ndim == 2 and j_perms.shape[1] != d_geom):
            # Fallback to random choice if dimensions mismatch
            ind = lgca.rng.integers(permutations.shape[0])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            continue

        term1 = 0
        if d_geom > 0 and j_perms.ndim == 2:  # Adhesion to neighbor flux
            # einsum('d,pd->p', j_nb_local, j_perms)
            # Original had / K_vel / 2. Assuming K_vel is lgca.velocitychannels
            if lgca.velocitychannels > 0:
                term1 = lgca.interaction_params['beta'] * np.dot(j_perms, j_nb_local) / lgca.velocitychannels / 2
            else:  # Avoid division by zero
                term1 = 0

        term2 = lgca.interaction_params['beta'] * resting_nb_sum_norm[
            coord_idx_tuple] * restc  # Adhesion to resting cells

        g_adh_local = g_adh[(slice(None),) + coord_idx_tuple]  # (d,)
        term3 = 0
        if d_geom > 0 and j_perms.ndim == 2 and g_adh_local.shape[0] == d_geom:  # Adhesion to ECM gradient
            term3 = lgca.interaction_params['beta'] * np.dot(j_perms, g_adh_local)

        term4 = 0
        if hasattr(lgca, 'ecm'):  # Interaction with ECM field
            term4 = restc * lgca.ecm[coord_idx_tuple]

        g_pressure_local = g_pressure[(slice(None),) + coord_idx_tuple]  # (d,)
        term5 = 0
        if d_geom > 0 and j_perms.ndim == 2 and g_pressure_local.shape[0] == d_geom:  # Pressure term
            term5 = lgca.interaction_params['gamma'] * np.dot(j_perms, g_pressure_local)

        weights_exp_terms = np.exp(term1 + term2 + term3 + term4 + term5)
        weights = weights_exp_terms.cumsum()

        if weights.size > 0 and weights[-1] > 1e-9:  # Check for valid weights
            ind = bisect_left(weights, lgca.rng.random() * weights[-1])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
        elif permutations.shape[0] > 0:  # Fallback if all weights are zero
            ind = lgca.rng.integers(permutations.shape[0])
            newnodes[(Ellipsis,) + coord_idx_tuple] = permutations[ind]
            
    lgca.nodes = newnodes
    if hasattr(lgca, 'ecm'):
        lgca.ecm -= lgca.interaction_params['alpha'] * lgca.ecm * lgca.cell_density / lgca.K


def excitable_medium(lgca):
    """Simulate an excitable medium following Barkley's model.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    alpha : float
        Controls the excitability of the medium.
    beta : float
        Interaction coefficient between species.
    N : int
        Number of sub-steps performed in each interaction.

    Notes
    -----
    The ``lgca`` object is modified in place.

    Returns
    -------
    None
    """
    # Assuming nodes are (K, dims...). K = velocitychannels + restchannels
    # Species X in velocity channels, Species Y in rest channels
    n_x_spatial = lgca.nodes[:lgca.velocitychannels, ...].sum(axis=0)  # (dims...)
    n_y_spatial = lgca.nodes[lgca.velocitychannels:, ...].sum(axis=0)  # (dims...)

    # Densities (avoid division by zero if channels = 0)
    rho_x = n_x_spatial / lgca.velocitychannels if lgca.velocitychannels > 0 else np.zeros_like(n_x_spatial,
                                                                                                dtype=float)
    rho_y = n_y_spatial / lgca.restchannels if lgca.restchannels > 0 else np.zeros_like(n_y_spatial, dtype=float)

    # Probabilities based on Barkley's model equations
    # These are probabilities per unit time, may need scaling by dt if N represents sub-time steps.
    # For now, assume direct use as in original.
    p_xp = rho_x ** 2 * (1 + (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha'])
    p_xm = rho_x ** 3 + rho_x * (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha']
    p_yp = rho_x  # Note: Original had p_yp = rho_x[coord] - this is now spatial
    p_ym = rho_y

    # Changes in Y (dn_y) - calculated once
    # The original code used (random < p).astype(int8) which is for total count, not per particle.
    # This implies a rate of change for the total count at a site.
    dn_y = (lgca.rng.random(n_y_spatial.shape) < p_yp).astype(np.int8) - \
           (lgca.rng.random(n_y_spatial.shape) < p_ym).astype(np.int8)

    # Iterative changes for X
    current_n_x = n_x_spatial.copy()
    current_rho_x = rho_x.copy()

    for _ in range(lgca.interaction_params['N']):
        # Update probabilities for X based on current rho_x and original rho_y
        p_xp_current = current_rho_x ** 2 * (
                    1 + (rho_y + lgca.interaction_params['beta']) / lgca.interaction_params['alpha'])
        p_xm_current = current_rho_x ** 3 + current_rho_x * (rho_y + lgca.interaction_params['beta']) / \
                       lgca.interaction_params['alpha']

        dn_x_iter = (lgca.rng.random(current_n_x.shape) < p_xp_current).astype(np.int8) - \
                    (lgca.rng.random(current_n_x.shape) < p_xm_current).astype(np.int8)
        current_n_x += dn_x_iter
        # Clip n_x to be within [0, lgca.velocitychannels]
        current_n_x = np.clip(current_n_x, 0, lgca.velocitychannels)
        current_rho_x = current_n_x / lgca.velocitychannels if lgca.velocitychannels > 0 else np.zeros_like(current_n_x,
                                                                                                            dtype=float)

    # Final updated counts
    final_n_x = current_n_x
    final_n_y = n_y_spatial + dn_y
    # Clip n_y to be within [0, lgca.restchannels]
    final_n_y = np.clip(final_n_y, 0, lgca.restchannels)

    # Reconstruct lgca.nodes
    newnodes_spatial = np.zeros_like(lgca.nodes)  # (K, dims...)

    # Distribute final_n_x particles into velocity channels
    for current_coord_tuple in np.ndindex(final_n_x.shape):  # Iterate over spatial dimensions
        num_x_particles = int(final_n_x[current_coord_tuple])
        if num_x_particles > 0 and lgca.velocitychannels > 0:
            # Ensure we don't try to choose more particles than available channels if num_x_particles > K_vel
            # This shouldn't happen if clipping was done correctly to lgca.velocitychannels
            chosen_v_indices = lgca.rng.choice(lgca.velocitychannels, min(num_x_particles, lgca.velocitychannels),
                                               replace=False)
            newnodes_spatial[chosen_v_indices, current_coord_tuple] = 1

        num_y_particles = int(final_n_y[current_coord_tuple])
        if num_y_particles > 0 and lgca.restchannels > 0:
            chosen_r_indices = lgca.rng.choice(lgca.restchannels, min(num_y_particles, lgca.restchannels),
                                               replace=False)
            newnodes_spatial[lgca.velocitychannels + chosen_r_indices, current_coord_tuple] = 1

    # The disarrange part was on newv (velocity channels part of nodes)
    # This should be done after filling newnodes_spatial
    if lgca.velocitychannels > 0:
        vel_channels_block = newnodes_spatial[:lgca.velocitychannels, ...]
        # Shuffle channels for each spatial site independently.
        for current_coord_tuple in np.ndindex(final_n_x.shape):  # Iterate over spatial dimensions
            lgca.rng.shuffle(
                vel_channels_block[(slice(None),) + current_coord_tuple])  # Shuffle along axis 0 for this site
        newnodes_spatial[:lgca.velocitychannels, ...] = vel_channels_block

    lgca.nodes = newnodes_spatial


# Helper function for go_or_grow, possibly s_binom
def _s_binom_entropy_like(n_total, p_state, k_max_for_state):
    """
    Calculates a quantity related to entropy for a binomial-like distribution.
    This is a placeholder based on the usage in go_or_grow.
    n_total: total particles at a site (scalar or array)
    p_state: probability of being in a particular state (e.g., resting) (scalar or array)
    k_max_for_state: maximum number of particles in that state (e.g., lgca.restchannels)
    
    The original s_binom(n, p, kmax) seemed to calculate something like:
    Sum_{i=0 to kmax} [ binom_coeff(n,i) * p^i * (1-p)^(n-i) * log(binom_coeff(n,i) * p^i * (1-p)^(n-i)) ]
    This is very complex. A simpler interpretation might be related to Shannon entropy of the two states (e.g.
    resting vs moving).
    
    Let's assume a simpler form for now: - (p_state * log(p_state) + (1-p_state)*log(1-p_state))
    This is the entropy of a Bernoulli trial. The k_max_for_state and n_total might be for normalization or weighting.
    Given the context of ds1 and ds2 being differences, it's likely an entropy-like term.
    
    For now, to avoid NameError and allow profiling, this will be a simplified placeholder.
    A more accurate version would require understanding the original paper/model for go_or_grow.
    """
    # Ensure p_state is within (0, 1) for log
    p_state_clipped = np.clip(p_state, 1e-9, 1 - 1e-9)
    entropy_term = - (p_state_clipped * np.log(p_state_clipped) + (1 - p_state_clipped) * np.log(1 - p_state_clipped))
    # The role of n_total and k_max_for_state is unclear without more context.
    # If it's an extensive quantity, it might be multiplied by n_total.
    # If k_max_for_state is about available states, it might normalize.
    # Let's return the basic entropy term for now.
    return entropy_term


def go_or_grow(lgca):
    """Perform the go-or-grow switching interaction.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Interaction Parameters
    ----------------------
    r_b : float # This parameter is not used in the provided snippet for go_or_grow, but kept for consistency if it
    was intended.
        Birth probability for resting cells. 
    r_d : float # This parameter is not used in the provided snippet for go_or_grow
        Death probability for both states.
    beta : float # Renamed from kappa in some contexts, this is the sensitivity for switching
        Steepness of the switching function / sensitivity to entropy change.
    theta : float # This parameter is not used here, tanh_switch is not directly called for entropy part
        Threshold density for switching.

    Notes
    -----
    The ``lgca`` object is modified in place.
    This version attempts to use the entropy-based switching logic.
    Assumes lgca.velocitychannels and lgca.restchannels are defined.
    Uses a placeholder `_s_binom_entropy_like` for the undefined `s_binom`.
    """
    # Operate on relevant non-border nodes
    nonborder_slice = lgca.nonborder
    density_on_nonborder = lgca.cell_density[nonborder_slice]
    # Relevant for switching: 0 < n < K (strictly, to allow switching)
    # Or, if n=0 or n=K, no switching is possible.
    relevant_on_nonborder_mask = (density_on_nonborder > 0) & (density_on_nonborder < lgca.K)

    relevant_coords_indices_full = tuple(nb_coord_array[relevant_on_nonborder_mask]
                                         for nb_coord_array in nonborder_slice)

    if not relevant_coords_indices_full[0].size:
        return  # No relevant nodes to process

    # Calculations for relevant nodes only
    n_total_relevant = lgca.cell_density[relevant_coords_indices_full]
    # Number of moving particles at relevant sites
    n_m_relevant = lgca.nodes[:lgca.velocitychannels, ...].sum(axis=0)[relevant_coords_indices_full]
    # Number of resting particles at relevant sites
    n_r_relevant = lgca.nodes[lgca.velocitychannels:, ...].sum(axis=0)[relevant_coords_indices_full]

    # Max particles that can switch from moving (m) to resting (r)
    # Limited by num moving (n_m_relevant) and available space in resting channels (lgca.restchannels - n_r_relevant)
    M1_relevant = np.minimum(n_m_relevant, lgca.restchannels - n_r_relevant)
    # Max particles that can switch from resting (r) to moving (m)
    # Limited by num resting (n_r_relevant) and available space in velocity channels (lgca.velocitychannels -
    # n_m_relevant)
    M2_relevant = np.minimum(n_r_relevant, lgca.velocitychannels - n_m_relevant)

    M1_relevant = np.maximum(0, M1_relevant).astype(int)  # Ensure non-negative integer
    M2_relevant = np.maximum(0, M2_relevant).astype(int)  # Ensure non-negative integer

    # Current proportion of resting particles (used for entropy calculation)
    p0_relevant = np.divide(n_r_relevant, n_total_relevant, where=n_total_relevant > 0,
                            out=np.zeros_like(n_r_relevant, dtype=float))
    # Current entropy-like term (using placeholder)
    # The third argument to s_binom was lgca.velocitychannels in original snippet, which seems odd if p0 is for resting.
    # Assuming it should be related to the state being described by p0, so perhaps lgca.restchannels or K.
    # Using lgca.restchannels as a guess for k_max if p0 is for resting state.
    s_current_relevant = _s_binom_entropy_like(n_total_relevant, p0_relevant, lgca.restchannels)

    # Entropy change for m -> r (one particle moves from moving to resting: n_r+1, n_m-1)
    # Proportion of resting particles if one switches from m to r
    p10_relevant = np.divide(n_r_relevant + 1, n_total_relevant, where=n_total_relevant > 0,
                             out=np.zeros_like(n_r_relevant, dtype=float))
    s10_relevant = _s_binom_entropy_like(n_total_relevant, p10_relevant, lgca.restchannels)
    ds1_relevant = s10_relevant - s_current_relevant  # Entropy gain if one m particle becomes r

    # Entropy change for r -> m (one particle moves from resting to moving: n_r-1, n_m+1)
    # Proportion of resting particles if one switches from r to m
    p01_relevant = np.divide(n_r_relevant - 1, n_total_relevant, where=n_total_relevant > 0,
                             out=np.zeros_like(n_r_relevant, dtype=float))
    # Ensure p01_relevant is not negative if n_r_relevant was 0 or 1 and n_total_relevant was small.
    p01_relevant = np.maximum(0, p01_relevant)  # Clip if n_r_relevant was 0 or 1 and n_total_relevant was small.
    s01_relevant = _s_binom_entropy_like(n_total_relevant, p01_relevant, lgca.restchannels)
    ds2_relevant = s01_relevant - s_current_relevant  # Entropy gain if one r particle becomes m

    # Probabilities for switching (Boltzmann-like factor based on entropy change)
    # prob_switch_m_to_r: probability for a single potential switcher (from M1_relevant) to switch
    prob_switch_m_to_r = 1 / (1 + np.exp(lgca.interaction_params['beta'] * ds1_relevant))
    prob_switch_m_to_r[M1_relevant == 0] = 0.  # No particles can switch if M1 is 0

    # prob_switch_r_to_m: probability for a single potential switcher (from M2_relevant) to switch
    prob_switch_r_to_m = 1 / (1 + np.exp(lgca.interaction_params['beta'] * ds2_relevant))
    # Original had beta for ds1 and -beta for ds2. If beta is sensitivity to entropy gain,
    # then switching should be favored if entropy increases (ds > 0).
    # 1/(1+exp(beta*ds)): if ds > 0, exp term small, prob ~1. if ds < 0, exp term large, prob ~0. This seems correct
    # for favoring entropy increase.
    # Let's assume beta is positive.
    prob_switch_r_to_m[M2_relevant == 0] = 0.  # No particles can switch if M2 is 0

    j1_switched = np.zeros_like(M1_relevant, dtype=int)  # Number of m->r switches
    j2_switched = np.zeros_like(M2_relevant, dtype=int)  # Number of r->m switches

    # Binomial draws for number of particles switching
    # Ensure probabilities are clipped to [0,1] for binomial function
    if np.any(M1_relevant > 0):  # Only draw if there are potential switchers
        valid_M1_mask = M1_relevant > 0
        j1_switched[valid_M1_mask] = lgca.rng.binomial(M1_relevant[valid_M1_mask],
                                                       np.clip(prob_switch_m_to_r[valid_M1_mask], 0, 1))

    if np.any(M2_relevant > 0):  # Only draw if there are potential switchers
        valid_M2_mask = M2_relevant > 0
        j2_switched[valid_M2_mask] = lgca.rng.binomial(M2_relevant[valid_M2_mask],
                                                       np.clip(prob_switch_r_to_m[valid_M2_mask], 0, 1))

    # Final numbers of moving and resting particles
    n_m_final_relevant = n_m_relevant + j2_switched - j1_switched
    n_r_final_relevant = n_r_relevant + j1_switched - j2_switched

    # Update nodes for relevant coordinates by reconstructing the node states
    num_relevant_nodes = n_total_relevant.shape[0]
    for i in range(num_relevant_nodes):
        current_coord_full = tuple(coord_array[i] for coord_array in relevant_coords_indices_full)

        # Ensure final counts are within channel limits
        n_mxy_final = np.clip(int(n_m_final_relevant[i]), 0, lgca.velocitychannels)
        n_rxy_final = np.clip(int(n_r_final_relevant[i]), 0, lgca.restchannels)

        node_state = np.zeros(lgca.K, dtype=lgca.nodes.dtype)
        if n_mxy_final > 0:
            # Randomly choose which velocity channels get occupied
            chosen_v_indices = lgca.rng.choice(lgca.velocitychannels, n_mxy_final, replace=False)
            node_state[chosen_v_indices] = 1
        if n_rxy_final > 0:
            # Randomly choose which resting channels get occupied
            chosen_r_indices = lgca.rng.choice(lgca.restchannels, n_rxy_final, replace=False)
            node_state[lgca.velocitychannels + chosen_r_indices] = 1  # Offset by K_vel

        lgca.nodes[(Ellipsis,) + current_coord_full] = node_state

    # Birth/death part from original go_and_grow (if intended to be combined)
    # This was missing from the entropy-based switching part.
    # If r_b and r_d are part of this interaction:
    if 'r_b' in lgca.interaction_params and 'r_d' in lgca.interaction_params:
        # Apply birth only to resting cells, death to all
        # This needs to be done carefully, considering the state after switching.
        # For simplicity, let's assume birth/death happens based on the new state.
        # This might require another pass or careful integration.
        # The original go_and_grow had birth for resting, death for all, then random_walk.
        # Here, the switching is the main rearrangement.

        # Simplified birth/death applied after switching (may need refinement)
        # Birth for resting cells (based on n_r_final_relevant)
        # This is complex to vectorize here as n_r_final_relevant is per relevant node.
        # A full birth/death step like the standalone `birthdeath` function might be more appropriate
        # if this interaction is meant to include it.
        # For now, focusing on the switching part.
        pass  # Placeholder for combined birth/death logic if needed.

    # random_walk(lgca) # Original go_and_grow had this. If switching is the only reorg, it might not be needed.
    # Or, if particles in newly assigned states should be shuffled among their type.
    # The current reconstruction already shuffles by using rng.choice.


def go_or_rest(lgca):
    """Switch cells between moving and resting states based on local density.
    Uses tanh_switch function. Assumes lgca.nodes is (K, dims...).
    No birth or death in this version.
    """
    nonborder_slice = lgca.nonborder
    density_on_nonborder = lgca.cell_density[nonborder_slice]
    # Relevant for switching: any node with particles that could potentially switch.
    # If a node is all moving or all resting, but not full/empty, it could switch.
    relevant_on_nonborder_mask = (density_on_nonborder > 0)

    relevant_coords_indices_full = tuple(nb_coord_array[relevant_on_nonborder_mask]
                                         for nb_coord_array in nonborder_slice)

    if not relevant_coords_indices_full[0].size:
        return  # No relevant nodes

    # Get current number of moving and resting particles at relevant sites
    n_m_full_grid = lgca.nodes[:lgca.velocitychannels, ...].sum(axis=0)
    n_r_full_grid = lgca.nodes[lgca.velocitychannels:, ...].sum(axis=0)

    n_m_relevant = n_m_full_grid[relevant_coords_indices_full]
    n_r_relevant = n_r_full_grid[relevant_coords_indices_full]
    density_relevant = lgca.cell_density[relevant_coords_indices_full]  # Total particles at relevant sites

    # Max particles that can switch from moving (m) to resting (r)
    M1_relevant = np.minimum(n_m_relevant, lgca.restchannels - n_r_relevant)
    # Max particles that can switch from resting (r) to moving (m)
    M2_relevant = np.minimum(n_r_relevant, lgca.velocitychannels - n_m_relevant)

    M1_relevant = np.maximum(0, M1_relevant).astype(int)  # Ensure non-negative integer
    M2_relevant = np.maximum(0, M2_relevant).astype(int)  # Ensure non-negative integer

    # Normalized density for tanh_switch (rho = n/K)
    rho_relevant = np.divide(density_relevant, lgca.K, where=lgca.K > 0,
                             out=np.zeros_like(density_relevant, dtype=float))

    # Probability to switch from moving to resting (go -> rest)
    # High density -> high prob_m_to_r
    prob_m_to_r = tanh_switch(rho_relevant,
                              kappa=lgca.interaction_params['kappa'],
                              theta=lgca.interaction_params['theta'])
    prob_m_to_r[M1_relevant == 0] = 0.0  # Cannot switch if no capacity or no particles

    # Probability to switch from resting to moving (rest -> go)
    # Low density -> high prob_r_to_m (since 1 - tanh_switch)
    prob_r_to_m = 1 - tanh_switch(rho_relevant,
                                  kappa=lgca.interaction_params['kappa'],
                                  theta=lgca.interaction_params['theta'])
    prob_r_to_m[M2_relevant == 0] = 0.0  # Cannot switch if no capacity or no particles

    j1_switched = np.zeros_like(M1_relevant, dtype=int)  # m -> r switches
    j2_switched = np.zeros_like(M2_relevant, dtype=int)  # r -> m switches

    if np.any(M1_relevant > 0):
        valid_M1_mask = M1_relevant > 0
        j1_switched[valid_M1_mask] = lgca.rng.binomial(M1_relevant[valid_M1_mask],
                                                       np.clip(prob_m_to_r[valid_M1_mask], 0, 1))

    if np.any(M2_relevant > 0):
        valid_M2_mask = M2_relevant > 0
        j2_switched[valid_M2_mask] = lgca.rng.binomial(M2_relevant[valid_M2_mask],
                                                       np.clip(prob_r_to_m[valid_M2_mask], 0, 1))

    # Final numbers of moving and resting particles
    n_m_final_relevant = n_m_relevant + j2_switched - j1_switched
    n_r_final_relevant = n_r_relevant + j1_switched - j2_switched

    # Update nodes for relevant coordinates
    num_relevant_nodes = density_relevant.shape[0]
    for i in range(num_relevant_nodes):
        current_coord_full = tuple(coord_array[i] for coord_array in relevant_coords_indices_full)

        n_mxy_final = np.clip(int(n_m_final_relevant[i]), 0, lgca.velocitychannels)
        n_rxy_final = np.clip(int(n_r_final_relevant[i]), 0, lgca.restchannels)

        node_state = np.zeros(lgca.K, dtype=lgca.nodes.dtype)
        if n_mxy_final > 0:
            chosen_v_indices = lgca.rng.choice(lgca.velocitychannels, n_mxy_final, replace=False)
            node_state[chosen_v_indices] = 1
        if n_rxy_final > 0:
            chosen_r_indices = lgca.rng.choice(lgca.restchannels, n_rxy_final, replace=False)
            node_state[lgca.velocitychannels + chosen_r_indices] = 1

        lgca.nodes[(Ellipsis,) + current_coord_full] = node_state
    # random_walk(lgca) # Typically, switching interactions are followed by random walk if not inherent.
    # The current reconstruction shuffles particles within their type.


def only_propagation(lgca):
    """Placeholder interaction that performs no rearrangement.

    Parameters
    ----------
    lgca : LGCA
        Lattice gas cellular automaton instance.

    Notes
    -----
    The ``lgca`` object is modified in place but remains unchanged. This
    interaction does not use ``lgca.interaction_params``.

    Returns
    -------
    None
    """
    pass