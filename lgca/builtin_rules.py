"""Built-in interactions written with the :func:`lgca.interaction` decorator.

They work with and without volume exclusion and for any number of species
unless stated otherwise, and serve as worked examples of the decorator.

Go-or-grow
----------
Cells in velocity channels migrate, cells in rest channels rest and divide.
One time step of the classical go-or-grow model is the pipeline ::

    operators=[
        {"name": "go_or_rest", "parameters": {"kappa": 4.0, "theta": 0.75}},
        {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01}},
        {"name": "random_walk", "parameters": {"channels": "velocity"}},
    ]

in this order, followed by propagation: cells switch between moving and
resting (a reorientation between velocity and rest channels), die, and
resting cells divide into rest channels; then the moving cells pick new
directions. This reproduces ``classical.go_or_grow`` and ``nove.go_or_grow``
in distribution.

In the two-species variant, migrating cells are species 0 (in velocity
channels) and resting cells species 1 (in rest channels), and the switch is
a phenotype switch, ``go_or_grow.switch``; ``go_or_grow.growth`` and the
random walk (with ``"species": 0``) work unchanged. It gives both phenotypes
their own counts in recordings. The rules keep each species in its channels,
so the initial state must place migrating cells in velocity channels and
resting cells in rest channels.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import numpy as np

from .base import channel_sum
from .lattice_state import _species_indices, channel_mask, random_occupancy
from .mutations import apply_mutations, parse_mutation
from .rules import interaction, register_single_cue, reorientation_term
from .switching import Probability, choice_probabilities, parse_probability

__all__ = ["birth_death", "go_or_grow_growth", "go_or_grow_switch", "go_or_rest", "phenotype_switch",
           "random_walk", "tanh_switch", "trait_switch"]


def tanh_switch(rho, kappa=5.0, theta=0.8):
    """Probability to rest at relative density ``rho``: ``(1 + tanh(kappa (rho - theta))) / 2``.

    ``theta`` is the relative density at which half of the cells rest, and
    ``kappa`` the steepness of the switch; with ``kappa < 0`` crowded cells move.
    """
    return 0.5 * (1 + np.tanh(kappa * (rho - theta)))


# Terms of the Boltzmann reorientation (ReorientationSpec). J(s') is the flux of
# the candidate state; see lgca.rules.reorientation_term for the couplings.

@reorientation_term(coupling="rest", name="random_walk", aliases="uniform")
def uniform(state):
    """Score 0: all channel states are equally likely."""
    return 0.0


@reorientation_term(coupling="rest", name="resting_bias")
def resting_bias(state):
    """Number of cells in rest channels: cells prefer to rest."""
    return 1.0


_GO_OR_REST = {"cues": [{"name": "density", "kappa": 5.0, "theta": 0.75}]}


@reorientation_term(coupling="rest", name="resting")
def resting(state, probability=None):
    """Cells rest with a probability that responds to cues, as in go-or-grow.

    A cell alone at its node rests with ``probability`` and otherwise moves
    to a random velocity channel. ``probability`` is a switching probability
    (:mod:`lgca.switching`): a number, the go-or-grow switch
    ``{"cues": [{"name": "density", "kappa": 5, "theta": 0.75}]}`` (the
    default), or any other response to cues, also in the Boltzmann form; in
    identity-based models its ``kappa`` and ``theta`` may name traits, so
    that every cell rests by its own sensitivity.

    The score is ``log(p / (1 - p)) + log(v / r)`` per cell in a rest channel,
    with ``v`` velocity and ``r`` rest channels. Without volume exclusion
    every cell rests with probability ``p``, as after ``go_or_rest`` and a
    random walk over the velocity channels. With volume exclusion the cells
    of a node choose a channel state together, ``P(s) ∝ exp(score · cells at
    rest)``; the cells at rest then follow Fisher's noncentral hypergeometric
    distribution, with odds ``p / (1 - p) · v / r`` of a rest channel against
    a velocity channel. ``beta`` scales the score: 0 is a
    random walk over all channels, 1 gives ``probability``.

    Parameters
    ----------
    probability : float or dict
        Probability that a cell alone at its node rests: a number or a response
        to cues (:mod:`lgca.switching`). Default: the go-or-grow switch of the
        node density, kappa 5 and theta 0.75.
    """
    from .rules import CellWeights

    if state.restchannels < 1:
        raise ValueError("resting needs at least one rest channel")
    chance = parse_probability(_GO_OR_REST if probability is None else probability, "probability")
    offset = np.log(state.velocitychannels / state.restchannels)
    if chance.reads_traits:
        if not state.identity_based:
            raise ValueError("the probability of resting reads cell traits, which only identity-based "
                             "models have")
        cells = state.cells
        return CellWeights(cells.label, chance.cell_log_odds(state, np.arange(len(cells))) + offset)
    return chance.log_odds(state) + offset


@reorientation_term(coupling="flux", name="persistent_walk", aliases="persistent_motion")
def persistent_walk(state):
    """J(s) · J(s'): cells keep the direction they had at this node."""
    return state.flux


@reorientation_term(coupling="flux", name="polar_alignment")
def polar_alignment(state, include_center=False, normalize=False):
    """J_nb · J(s'), with J_nb the flux of the neighbouring nodes: cells move with their neighbours.

    Parameters
    ----------
    include_center : bool
        Add the flux of the node's own cells to J_nb.
    normalize : bool
        Divide J_nb by the number of cells it sums over (at least one), so the
        cue measures the mean direction rather than the number of neighbours
        (density-independent alignment).
    """
    flux = state.neighbor_sum(state.flux)
    cells = state.neighbor_sum(state.density)
    if include_center:
        flux, cells = flux + state.flux, cells + state.density
    if normalize:
        flux = flux / np.maximum(cells, 1)[..., None]
    return flux


@reorientation_term(coupling="channels", name="nematic_alignment", aliases=("nematic", "alignment"))
def nematic_alignment(state):
    """Cells share an axis with neighbouring cells; opposite directions count the same.

    A cell in channel i gains Σ_k n_k [(c_k · c_i)² - |c_k|² |c_i|² / d] from the n_k
    neighbouring cells in channel k (d the spatial dimension): the product of the
    traceless tensors c cᵀ - |c|² I / d of the two directions. Moving along the
    neighbours' axis scores above resting (0), moving across it below.
    """
    neighbours = state.neighbor_sum(state.counts[..., :state.velocitychannels].sum(axis=-2))
    c = state.c
    lengths = (c ** 2).sum(0)
    return neighbours @ ((c.T @ c) ** 2 - np.multiply.outer(lengths, lengths) / c.shape[0])


@reorientation_term(coupling="channels", name="steric_repulsion")
def steric_repulsion(state):
    """Cells avoid crowded neighbours: channel i scores minus the cells at the node it points to."""
    return -state.neighbor_values(state.density)


@reorientation_term(coupling="flux", name="aggregation")
def aggregation(state):
    """∇ρ · J(s'): cells move up the gradient of the cell density."""
    return state.gradient(state.density)


@reorientation_term(coupling="flux", name="chemotaxis")
def chemotaxis(state, field):
    """∇f · J(s'): cells move up the gradient of a field.

    Parameters
    ----------
    field : str
        Name of a scalar field in StateSpec.fields. Its gradient is centred
        everywhere; beyond the lattice edge the field keeps the ghost values
        the model stores for it (the edge values unless changed).
    """
    values = state.field(field)
    if values.shape != state.dims:
        raise ValueError(f"state.fields.{field} must have shape {state.dims}, got {values.shape}")
    return state.gradient(field)


@reorientation_term(coupling="flux", name="directed_motion")
def directed_motion(state, field):
    """g · J(s'): cells move along a given vector field g, e.g. a flow or an external gradient.

    Parameters
    ----------
    field : str
        Name of a vector field in StateSpec.fields, shape ``dims + (d,)``; its
        length sets the strength of the bias together with ``beta``.
    """
    values = np.asarray(state.field(field), dtype=float)
    if values.shape != state.dims + (state.c.shape[0],):
        raise ValueError(f"state.fields.{field} must have shape {state.dims + (state.c.shape[0],)}, "
                         f"got {values.shape}")
    return values


@reorientation_term(coupling="channels", name="contact_guidance")
def contact_guidance(state, field="director"):
    """Cells move along the axis of a director field.

    A cell in channel i scores (n · c_i)² - |c_i|² / d, with n the unit director and d
    the spatial dimension: above resting (0) along the axis, below it across. Where
    the director is zero, all channels score 0.

    Parameters
    ----------
    field : str
        Name of a vector field in StateSpec.fields; only its direction matters.
    """
    director = np.asarray(state.field(field), dtype=float)
    if director.shape != state.dims + (state.c.shape[0],):
        raise ValueError(f"state.fields.{field} must have shape {state.dims + (state.c.shape[0],)}, "
                         f"got {director.shape}")
    norm = np.linalg.norm(director, axis=-1, keepdims=True)
    director = np.divide(director, norm, out=np.zeros_like(director), where=norm > 0)
    c = state.c
    return (director @ c) ** 2 - (director ** 2).sum(-1)[..., None] * (c ** 2).sum(0) / c.shape[0]

# Every built-in cue also works alone as an operator: {"name": "chemotaxis", "parameters": {"beta": 2,
# "field": "signal"}} is a ReorientationSpec with this one term.
for _cue, _aliases in ((polar_alignment, ()), (nematic_alignment, ("nematic",)),
                       (persistent_walk, ("persistent_motion",)), (aggregation, ()), (chemotaxis, ()), (directed_motion, ()),
                       (contact_guidance, ()), (resting_bias, ()), (resting, ()), (steric_repulsion, ())):
    register_single_cue(_cue, aliases=_aliases)

@interaction(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), name="birth_death")
def birth_death(state, birth_rate=0.0, death_rate=0.0, crowding=True, mutation_matrix=None, mutation=None,
                new_family=False, channels="all", species=None):
    """Cells die and divide at the same time; crowded nodes have less room for daughters.

    In a time step, every cell dies with probability ``death_rate`` and,
    independently, tries to divide with probability ``birth_rate``; both are
    decided on the state at the start of the step, so a dying cell may still
    divide, and daughters do not die in the step they are born. The daughter
    needs room:

    - with volume exclusion, it goes to a random channel of its species at
      the node (dividing cells pick different channels) and survives only
      if that channel was empty, which happens with probability
      ``1 - n_s / K`` for ``n_s`` cells of its species;
    - a capacity (``StateSpec.capacity``; always set without volume
      exclusion) scales the division probability by ``1 - n / capacity``,
      with ``n`` all cells at the node. With volume exclusion it is an
      optional soft limit in addition to the channels, e.g. to make species
      compete for space.

    The expected change of a node with one species is
    ``birth_rate * n * (1 - n / capacity) - death_rate * n``, with
    ``capacity = K`` under volume exclusion. For death before division,
    list two operators: one with only ``death_rate``, then one with only
    ``birth_rate``.

    With ``channels``, only the cells in a set of channels die and divide,
    and their daughters go to that set: e.g. ``channels="rest"`` for the
    go-or-grow rule that only resting cells divide. Under volume exclusion
    the daughters then need empty channels of the set, ``K`` above becomes
    the size of the set, and ``n_s`` the cells in it.

    Parameters
    ----------
    birth_rate : float, sequence, str or dict
        Probability per time step that a cell tries to divide: one number,
        one per species, in identity-based models the name of a cell trait
        (a value per cell), or a probability that responds to cues
        (:mod:`lgca.switching`), e.g. division that needs oxygen,
        ``{"max": 0.1, "hill": [{"name": "field", "field": "oxygen", "K":
        0.2}]}``; also one such probability per species.
    death_rate : float, sequence, str or dict
        Probability per time step that a cell dies, given like ``birth_rate``;
        e.g. death under hypoxia, ``{"max": 0.05, "hill": [{"name": "field",
        "field": "oxygen", "K": 0.05, "n": -2}]}``.
    crowding : bool
        False: cells divide with ``birth_rate`` regardless of crowding;
        daughters take free channels, and ``state.capacity`` is a hard limit
        on the cells per node. Divisions beyond either fail, chosen at random.
    mutation_matrix : array_like or None
        Classical models with several species: entry ``[a][b]`` is the
        probability that a daughter of species ``a`` belongs to species ``b``
        (rows sum to 1). Default: daughters have their mother's species.
    mutation : dict, list or None
        Identity-based models: how daughters mutate, e.g.
        ``{"probability": 0.01, "traits": {"r_b": {"distribution": "normal",
        "scale": 0.02, "bounds": [0, 1]}}}``; see :mod:`lgca.mutations` for
        distributions, fixed and custom effects and several kinds of mutation.
    new_family : bool
        Identity-based models: daughters that mutate found a new family, for
        lineage analyses such as Muller plots; without ``mutation``, every
        daughter founds one.
    channels : str or list of int
        The channels whose cells die and divide, and where daughters go:
        ``"all"``, ``"velocity"``, ``"rest"`` or channel indices.
    species : int, list of int or None
        Classical models with several species: the species whose cells die
        and divide; the others are left as they are (but count for crowding).
        Default: all.

    Examples
    --------
    Logistic growth, ``{"name": "birth_death", "parameters": {"birth_rate":
    0.2, "death_rate": 0.02}}``; in an identity-based model with a mutating
    birth rate per cell, ``{"birth_rate": "r_b", "death_rate": 0.02,
    "mutation": {"r_b": {"distribution": "normal", "scale": 0.01, "bounds": [0, 0.5],
    "at_bounds": "redraw"}}}``.
    """
    if not isinstance(crowding, (bool, np.bool_)):
        raise TypeError(f"crowding must be True or False, got {crowding!r}")
    if state.identity_based:
        if species is not None:
            _species_indices(species, 1)
        if mutation_matrix is not None and not np.array_equal(np.asarray(mutation_matrix), [[1]]):
            raise ValueError("mutation_matrix needs a classical model with several species; identity-based "
                             "models have one species and change traits with mutation=")
        _birth_death_cells(state, birth_rate, death_rate, crowding, mutation, new_family, channels)
        return
    if mutation or new_family:
        raise ValueError("mutation and new_family change cell traits and families, which only "
                         "identity-based models have")
    n_species, rng = state.n_species, state.rng
    # one probability per species, or per node and species if a probability responds to cues
    death = _species_rates(state, death_rate, "death_rate")
    birth = _species_rates(state, birth_rate, "birth_rate")
    if species is not None:  # the other species neither die nor divide
        still = np.setdiff1d(np.arange(n_species), _species_indices(species, n_species))
        death, birth = death.copy(), birth.copy()  # not the caller's arrays
        death[..., still] = birth[..., still] = 0.0
    matrix = _mutation_matrix(mutation_matrix, n_species)
    in_set = channel_mask(channels, state.K, state.velocitychannels)
    every = in_set.all()
    # deaths and divisions are decided on the state at the start of the step
    nodes = state.counts
    counts = state.species_density
    density = counts[..., 0] if n_species == 1 else counts.sum(axis=-1)
    if not every:  # only the cells in the set take part
        nodes = nodes * in_set
        counts = channel_sum(nodes)
    if crowding and state.has_capacity:
        birth = birth * np.clip(1 - density / state.capacity, 0, 1)[..., None]
    if state.volume_exclusion:  # a cell per channel: one random number decides both, independently
        # booleans and float32 keep the temporaries small, which is what makes this fast
        occupied = nodes.astype(bool)
        b = np.asarray(birth[..., None], dtype=np.float32)
        d = np.asarray(death[..., None], dtype=np.float32)
        draw = rng.random(nodes.shape, dtype=np.float32)
        dying = occupied & ((draw < b * d) | ((draw >= b) & (draw < b + d * (1 - b))))
        births = (occupied & (draw < b)).view(np.uint8) @ np.ones(state.K, dtype=np.int64)
    else:
        deaths = (_by_species(rng, nodes, death, axis=-2) if death.ndim == 1
                  else rng.binomial(nodes, death[..., None]))
        births = _by_species(rng, counts, birth, axis=-1) if birth.ndim == 1 else rng.binomial(counts, birth)
    if matrix is not None:
        births = rng.multinomial(births, matrix).sum(axis=-2)
    if crowding and state.volume_exclusion:  # daughters go to distinct random channels; the empty ones hold them
        size = int(in_set.sum())
        targets = random_occupancy(rng, np.minimum(births, size), size)
        if not every:
            placed = np.zeros(occupied.shape, dtype=bool)
            placed[..., in_set] = targets
            targets, occupied = placed, state.counts.astype(bool)
        state.counts = (occupied & ~dying) | (targets & ~occupied)
        return
    if state.volume_exclusion:
        deaths = dying.astype(np.int64)
    if not crowding:
        births = _at_most(rng, births, np.maximum(state.capacity - density, 0))
    state.add_cells(births, channels=in_set)  # into channels that were free at the start of the step
    state.counts = state.counts - deaths


def _birth_death_cells(state, birth_rate, death_rate, crowding, mutation, new_family, channels):
    """birth_death for identity-based models: death and division per cell, decided at once."""
    cells, rng = state.cells, state.rng
    in_set = channel_mask(channels, state.K, state.velocitychannels)
    dying = rng.random(len(cells)) < _cell_rates(state, death_rate, "death_rate")
    birth = _cell_rates(state, birth_rate, "birth_rate")
    density = state.density
    if crowding and state.has_capacity:
        birth = birth * np.clip(1 - density / state.capacity, 0, 1)[cells.node]
    dividing = rng.random(len(cells)) < birth
    if not in_set.all():  # only the cells in the set take part
        taking_part = in_set[cells.channel]
        dying &= taking_part
        dividing &= taking_part
    if crowding and state.volume_exclusion:
        occupied = density if in_set.all() else (
            np.bincount(cells.index[in_set[cells.channel]], minlength=density.size).reshape(density.shape))
        attempts = np.bincount(cells.index[dividing], minlength=density.size).reshape(density.shape)
        dividing = cells.pick(dividing, _into_empty(rng, attempts, occupied, int(in_set.sum())))
    elif not crowding:
        dividing = cells.pick(dividing, np.maximum(state.capacity - density, 0))
    daughters = _divide_and_mutate(state, dividing, in_set, mutation, new_family)
    cells.kill(np.concatenate([dying, np.zeros(len(daughters), dtype=bool)]))


def _into_empty(rng, attempts, occupied, channels):
    """How many of ``attempts`` daughters, sent to distinct random channels of ``channels``, find them empty."""
    attempts = np.minimum(attempts, channels)
    occupied = np.broadcast_to(occupied, attempts.shape)
    found = np.zeros_like(attempts)
    trying = np.nonzero(attempts)  # most nodes have no dividing cell
    # hypergeometric draws by inverting the cumulative distribution of a table
    cumulative = _hypergeometric_table(channels)[occupied[trying], attempts[trying]]
    found[trying] = (rng.random((len(trying[0]), 1)) >= cumulative).sum(axis=-1)
    return found


@functools.lru_cache(maxsize=16)
def _hypergeometric_table(channels):
    """P(at most k of a distinct random channels are empty | o occupied), indexed [o, a, k]."""
    from math import comb

    table = np.zeros((channels + 1, channels + 1, channels + 1))
    for occupied in range(channels + 1):
        for attempts in range(channels + 1):
            for empty in range(attempts + 1):
                table[occupied, attempts, empty] = (comb(channels - occupied, empty)
                                                    * comb(occupied, attempts - empty) / comb(channels, attempts))
    cumulative = np.cumsum(table, axis=-1)
    cumulative[..., -1] = 1.0  # rounding
    return cumulative[..., :-1]  # the last bound is never exceeded


def _by_species(rng, number, p, axis):
    """``rng.binomial(number, p)`` for ``p`` given per species (``axis`` of ``number``)."""
    drawn = np.empty(number.shape, dtype=np.int64)
    for species, value in enumerate(np.ravel(p)):
        index = (slice(None),) * (axis % number.ndim) + (species,)
        drawn[index] = _thinned(rng, number[index], value)
    return drawn


def _thinned(rng, number, p):
    """``rng.binomial(number, p)`` for one probability; with few cells from one random number per cell."""
    flat = np.ascontiguousarray(number).reshape(-1)
    if flat.sum() > 4 * flat.size:
        return rng.binomial(number, p)
    rows = np.repeat(np.arange(flat.size), flat)
    return np.bincount(rows[rng.random(len(rows)) < p], minlength=flat.size).reshape(number.shape)


def _per_species(value, name, n_species, probability=True):
    rates = np.asarray(_number(value, name), dtype=float)
    if rates.ndim == 0:
        rates = np.full(n_species, float(rates))
    if rates.shape != (n_species,):
        raise ValueError(f"{name} must be one number or one per species ({n_species}), got {value!r}")
    if probability and np.any((rates < 0) | (rates > 1)):
        raise ValueError(f"{name} must be probabilities, got {value!r}")
    return rates


def _species_rates(state, value, name):
    """Probabilities per species, shape ``(n_species,)``, or ``dims + (n_species,)`` if one responds to cues."""
    n_species = state.n_species
    responds = isinstance(value, (Mapping, Probability))
    entries = [value] * n_species if responds else value
    if not isinstance(entries, (list, tuple)) or not any(isinstance(entry, (Mapping, Probability))
                                                         for entry in entries):
        return _per_species(value, name, n_species)
    if len(entries) != n_species:
        raise ValueError(f"{name} must be one probability or one per species ({n_species}), got {value!r}")
    rates = np.empty(state.dims + (n_species,))
    for index, entry in enumerate(entries):
        where = name if responds else f"{name}[{index}]"
        rates[..., index] = parse_probability(entry, where).nodes(state)
    return rates


def _cell_rates(state, value, name):
    """A probability per cell: a number, the named trait, or a probability that responds to cues."""
    cells = state.cells
    if isinstance(value, (Mapping, Probability)):
        return np.broadcast_to(parse_probability(value, name).cells(state, np.arange(len(cells))), len(cells))
    return _per_cell(cells, value, name)


def _mutation_matrix(value, n_species):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (n_species, n_species):
        raise ValueError(f"mutation_matrix must have shape ({n_species}, {n_species}), got {matrix.shape}")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("mutation_matrix rows must be probabilities that sum to 1")
    return None if np.array_equal(matrix, np.eye(n_species)) else matrix / matrix.sum(axis=1, keepdims=True)


def _at_most(rng, births, room):
    """Keep at most ``room`` of the births at each node, chosen uniformly among all species' births."""
    total = births.sum(axis=-1)
    kept = np.empty_like(births)
    left, wanted = np.minimum(total, room), total
    for species in range(births.shape[-1]):
        kept[..., species] = rng.hypergeometric(births[..., species], wanted - births[..., species], left)
        left, wanted = left - kept[..., species], wanted - births[..., species]
    return kept


def _divide_and_mutate(state, dividing, channels, mutation, new_family):
    """The selected cells divide; the daughters mutate, and (new_family) the mutated ones found families."""
    if not isinstance(new_family, (bool, np.bool_)):
        raise TypeError(f"new_family must be True or False, got {new_family!r}; daughters found a new "
                        f"family when they mutate (see mutation)")
    cells = state.cells
    mutations = parse_mutation(mutation)
    daughters = cells.divide(dividing, channels=channels)
    mutated = apply_mutations(state, daughters, mutations)
    if new_family:  # without mutation, every daughter founds one (a neutral lineage marker)
        cells.found_families(daughters[mutated] if mutations else daughters)
    return daughters


@interaction(kind="phenotype_switch", families=("classical", "nove"), name="phenotype_switch",
             aliases="species_switch")
def phenotype_switch(state, rates, channels="all"):
    """Cells switch species: a cell of species a becomes species b with probability rates[a][b] per step.

    The rates are numbers or probabilities that respond to cues of the
    surroundings, e.g. cells that turn migratory in crowded nodes
    (:mod:`lgca.switching`)::

        "rates": [[0, {"max": 0.2, "cues": [{"name": "density", "kappa": 5, "theta": 0.5}]}],
                  [0.05, 0]]

    or weights in the Boltzmann form, e.g. three species where species 0
    turns into 1 in crowded nodes and into 2 along a signal::

        "rates": [[0, {"rate": 0.02, "cues": [{"name": "density", "beta": 4}]},
                      {"rate": 0.01, "cues": [{"name": "field", "field": "signal", "beta": 2}]}],
                  [0.05, 0, 0], [0.05, 0, 0]]

    The number of cells at every node stays the same. A switching cell goes
    to a random channel of its new species in ``channels``; with volume
    exclusion the switch fails if that channel is occupied, so a cell
    switching into a species with n cells in C channels succeeds with
    probability 1 - n/C. Cells switching into the same species at a node pick
    distinct channels, and occupancy is judged before the switch (see
    :meth:`LatticeState.switch_phenotype`).

    Parameters
    ----------
    rates : list of lists
        ``n_species x n_species`` switching probabilities per time step, each
        a number or a response to cues; the diagonal is ignored, and the
        (maximal) probabilities of each row must sum to at most 1. A row may
        instead give weights in the Boltzmann form, ``{"rate": r, "cues":
        [...]}``: a cell of species a becomes b with probability
        ``w_ab / (1 + Σ_b' w_ab')``, staying having weight 1, with no bound on
        the sum.
    channels : str or list of int
        Where switched cells go: a random channel of their new species in this
        set (``"all"``, ``"velocity"``, ``"rest"`` or indices), or ``"same"``:
        they keep their channel. With volume exclusion the channel must be free
        for the new species.
    """
    n_species = state.n_species
    if n_species < 2:
        raise ValueError("phenotype_switch moves cells between species and needs n_species >= 2")
    rows = rates.tolist() if isinstance(rates, np.ndarray) else rates
    if not isinstance(rows, (list, tuple)) or len(rows) != n_species or any(
            not isinstance(row, (list, tuple)) or len(row) != n_species for row in rows):
        raise ValueError(f"rates must be a {n_species} x {n_species} matrix, got {rates!r}")
    table = [[parse_probability(entry, f"rates[{a}][{b}]") for b, entry in enumerate(row)]
             for a, row in enumerate(rows)]
    probabilities = [_switch_row(state, a, row) for a, row in enumerate(table)]
    if all(np.ndim(p) == 0 for row in probabilities for p in row):
        matrix = np.array(probabilities, dtype=float)
    else:
        matrix = np.zeros(state.dims + (n_species, n_species))
        for a, row in enumerate(probabilities):
            for b, p in enumerate(row):
                matrix[..., a, b] = p
    state.switch_phenotype(matrix, channels=channels)


def _switch_row(state, a, row):
    """The probabilities that a cell of species ``a`` becomes each species (0 on the diagonal)."""
    others = [b for b in range(len(row)) if b != a]
    boltzmann = [b for b in others if row[b].boltzmann]
    result = [0.0] * len(row)
    if boltzmann:
        mixed = [b for b in others if not row[b].boltzmann and not (row[b].constant and row[b].max == 0)]
        if mixed:
            raise ValueError(f"rates[{a}] mixes the Boltzmann form ('rate') with probabilities in "
                             f"rates[{a}]{mixed}; write every switch of a row as {{'rate': ...}} (or 0)")
        for b, p in zip(boltzmann, choice_probabilities([row[b].log_weight(state) for b in boltzmann])):
            result[b] = p
        return result
    if sum(row[b].max for b in others) > 1 + 1e-12:
        raise ValueError("the switching probabilities of each species must sum to at most 1")
    for b in others:
        result[b] = row[b].nodes(state)
    return result


@interaction(kind="phenotype_switch", families=("ib", "nove_ib"), name="trait_switch")
def trait_switch(state, switch, new_family=False):
    """Cells change their traits by events, as daughters do by mutations, at any time.

    Every time step each cell has an event with its probability, and the
    event changes the cell's traits by its effects. The events and effects are
    written as mutations (:mod:`lgca.mutations`), with the operation ``"set"``
    to switch a trait to a value, e.g. a cell that starts to align::

        {"name": "trait_switch", "parameters": {"switch": {
            "probability": 0.02, "traits": {"alignment": {"value": 2.0, "operation": "set"}}}}}

    The probability may respond to cues of the cell's surroundings (see
    :mod:`lgca.switching`), and ``"when"`` limits an event to cells with some
    values of a trait, e.g. cells that align more in crowded nodes and stop
    at a constant rate::

        "switch": [
            {"when": {"alignment": 0}, "traits": {"alignment": {"value": 2.0, "operation": "set"}},
             "probability": {"max": 0.1, "cues": [{"name": "density", "kappa": 6, "theta": 0.5}]}},
            {"when": {"alignment": 2}, "probability": 0.02,
             "traits": {"alignment": {"value": 0.0, "operation": "set"}}},
        ]

    Parameters
    ----------
    switch : dict or list
        The events: ``{"probability": p, "traits": {name: effect, ...}}``,
        or a list of them, applied in order.
    new_family : bool
        Cells that switch found a new family.
    """
    events = parse_mutation(switch, name="switch")
    if not events:
        raise ValueError("trait_switch needs a switch, e.g. {'probability': 0.01, 'traits': {'alignment': 0.1}}")
    if not isinstance(new_family, (bool, np.bool_)):
        raise TypeError(f"new_family must be True or False, got {new_family!r}")
    cells = state.cells
    switched = apply_mutations(state, np.arange(len(cells)), events)
    if new_family:
        cells.found_families(switched)


_MIGRATING, _RESTING = 0, 1
_WHEN_FULL = ("legacy", "reject")


@interaction(kind="reorientation", families=("classical", "nove", "ib", "nove_ib"), name="random_walk")
def random_walk(state, channels="all", species=None):
    """Cells move to uniformly random channels of their node, within a set of channels.

    In identity-based models the cells keep their labels; cells outside the
    set keep their channels.

    Parameters
    ----------
    channels : str or list of int
        Channels the moving cells are spread over: ``"all"``, ``"velocity"``,
        ``"rest"`` or channel indices. Cells in other channels stay put.
    species : int, list of int or None
        Species that move; the others stay in their channels. Default: all.
    """
    state.shuffle_cells(channels, species=species)


@interaction(kind="reorientation", families=("classical", "nove", "ib", "nove_ib"), name="go_or_rest")
def go_or_rest(state, kappa=5.0, theta=0.75, when_full="legacy", density="node", species=None):
    """Moving cells start resting on crowded nodes, resting cells start moving on sparse ones.

    A moving cell starts resting with probability ``tanh_switch(rho, kappa,
    theta)``, a resting cell starts moving with the complementary
    probability; ``rho`` is the relative density of all cells at the node.

    Parameters
    ----------
    kappa : float, sequence or str
        Steepness of the switch. With kappa > 0 crowded cells rest, with
        kappa < 0 they move. One number, one per species, or in
        identity-based models the name of a cell trait (a value per cell).
    theta : float, sequence or str
        Relative density at which half of the cells rest, given like
        ``kappa``. The relative density is the number of cells at the node
        over its capacity: the number of channels K with volume exclusion,
        ``StateSpec.capacity`` (the crowding scale) without.
    when_full : {"legacy", "reject"}
        With volume exclusion, what happens when a switching cell finds no free
        channel. "legacy" reproduces the original rule: the number of switching
        cells is drawn from the cells that fit into free channels. "reject":
        every cell tries to switch, and switches into full channels fail.
    density : {"node", "neighbourhood"}
        The cells that set the relative density: those at the node, or the
        mean over the node and its neighbours.
    species : int, list of int or None
        Classical models with several species: the species whose cells switch;
        the others keep their channels (but count for the density). Default:
        all.
    """
    _check_mode(when_full)
    if state.restchannels < 1:
        raise ValueError("go_or_rest needs at least one rest channel for resting cells")
    still = None if species is None else np.setdiff1d(np.arange(state.n_species),
                                                      _species_indices(species, state.n_species))
    if state.identity_based:
        _go_or_rest_cells(state, kappa, theta, when_full, density)
        return
    kappa = _per_species(kappa, "kappa", state.n_species, probability=False)
    theta = _per_species(theta, "theta", state.n_species, probability=False)
    rest = tanh_switch(_relative_density(state, density)[..., None], kappa, theta)  # per species
    velocity, rng = state.velocitychannels, state.rng
    cells = state.counts
    moving, resting = cells[..., :velocity], cells[..., velocity:]
    if state.volume_exclusion:
        n_m, n_r = moving.sum(-1), resting.sum(-1)
        free_rest, free_velocity = state.restchannels - n_r, velocity - n_m
        if when_full == "legacy":
            to_rest = rng.binomial(np.minimum(n_m, free_rest), rest)
            to_move = rng.binomial(np.minimum(n_r, free_velocity), 1 - rest)
        else:
            to_rest = np.minimum(rng.binomial(n_m, rest), free_rest)
            to_move = np.minimum(rng.binomial(n_r, 1 - rest), free_velocity)
        new_moving = moving - _pick(rng, moving == 1, to_rest) + _pick(rng, moving == 0, to_move)
        new_resting = resting - _pick(rng, resting == 1, to_move) + _pick(rng, resting == 0, to_rest)
    else:
        leaving_moving = rng.binomial(moving, rest[..., None])
        leaving_resting = rng.binomial(resting, (1 - rest)[..., None])
        new_moving = moving - leaving_moving + _spread(rng, leaving_resting.sum(-1), velocity)
        new_resting = resting - leaving_resting + _spread(rng, leaving_moving.sum(-1), state.restchannels)
    new = np.concatenate((new_moving, new_resting), axis=-1)
    if still is not None and len(still):
        new[..., still, :] = cells[..., still, :]
    state.counts = new


@interaction(kind="phenotype_switch", families=("classical", "nove"), n_species=2, name="go_or_grow.switch")
def go_or_grow_switch(state, kappa=5.0, theta=0.75, when_full="legacy", density="node"):
    """Migrating cells start resting in crowded nodes, resting cells start migrating in sparse ones.

    The two-species form of go_or_rest: migrating cells are species 0 (in
    velocity channels), resting cells species 1 (in rest channels).

    Parameters
    ----------
    kappa : float
        Steepness of the switch. With kappa > 0 crowded cells rest, with
        kappa < 0 they migrate.
    theta : float
        Relative density at which half of the cells rest: cells at the node
        over K with volume exclusion, over ``StateSpec.capacity`` without.
    when_full : {"legacy", "reject"}
        With volume exclusion, what happens when a switching cell finds no free
        channel. "legacy" reproduces the original rule: the number of switching
        cells is drawn from the cells that fit into free channels. "reject":
        every cell tries to switch into a random channel of its new kind, and
        fails if it is occupied (as in ``phenotype_switch``).
    density : {"node", "neighbourhood"}
        The cells that set the relative density: those at the node, or the
        mean over the node and its neighbours.
    """
    _check_layout(state)
    _check_mode(when_full)
    rest = tanh_switch(_relative_density(state, density), kappa, theta)
    if state.volume_exclusion and when_full == "legacy":
        counts = state.counts.copy()
        velocity = state.velocitychannels
        n_m = counts[..., _MIGRATING, :].sum(-1)
        n_r = counts[..., _RESTING, :].sum(-1)
        to_rest = state.rng.binomial(np.minimum(n_m, state.restchannels - n_r), rest)
        to_move = state.rng.binomial(np.minimum(n_r, velocity - n_m), 1 - rest)
        rng = state.rng
        counts[..., _MIGRATING, :] -= _pick(rng, counts[..., _MIGRATING, :] == 1, to_rest)
        counts[..., _RESTING, :] -= _pick(rng, counts[..., _RESTING, :] == 1, to_move)
        free_rest = state.counts[..., _RESTING, :] == 0
        free_rest[..., :velocity] = False
        free_velocity = state.counts[..., _MIGRATING, :] == 0
        free_velocity[..., velocity:] = False
        counts[..., _RESTING, :] += _pick(rng, free_rest, to_rest)
        counts[..., _MIGRATING, :] += _pick(rng, free_velocity, to_move)
        state.counts = counts
        return
    rates = np.zeros(state.dims + (2, 2))
    rates[..., _MIGRATING, _RESTING] = rest
    rates[..., _RESTING, _MIGRATING] = 1 - rest
    state.switch_phenotype(rates, channels={_MIGRATING: "velocity", _RESTING: "rest"})


@interaction(kind="birth_death", families=("classical", "nove", "ib", "nove_ib"), name="go_or_grow.growth")
def go_or_grow_growth(state, r_b=0.2, r_d=0.01, r_d_resting=None, when_full="legacy", mutation=None,
                      new_family=False):
    """Cells die, and resting cells divide into free rest channels.

    Resting cells are the cells in rest channels, or species 1 in the
    two-species form of go-or-grow.

    Parameters
    ----------
    r_b : float or str
        Division probability of a resting cell. Without volume exclusion it is
        scaled by 1 - density / capacity. In identity-based models, the name
        of a cell trait gives every cell its own value.
    r_d : float or str
        Death probability of a moving cell, and of a resting cell unless
        r_d_resting is given; or the name of a cell trait.
    r_d_resting : float, str or None
        Death probability of a resting cell. Default: r_d.
    when_full : {"legacy", "reject"}
        With volume exclusion, how divisions meet full rest channels. "legacy"
        reproduces the original rule: the number of divisions is drawn from as
        many resting cells as there are free rest channels. "reject": every
        resting cell tries to divide, and divisions into full channels fail.
    mutation : dict, list or None
        Identity-based models: how daughters mutate, e.g. ``{"kappa": 0.2}``
        for a normal change with standard deviation 0.2 in every daughter;
        see :mod:`lgca.mutations`.
    new_family : bool
        Identity-based models: daughters that mutate found a new family;
        without ``mutation``, every daughter founds one.
    """
    _check_mode(when_full)
    if state.identity_based:
        _growth_cells(state, r_b, r_d, r_d_resting, when_full, mutation, new_family)
        return
    r_b, r_d = _number(r_b, "r_b"), _number(r_d, "r_d")
    r_d_resting = None if r_d_resting is None else _number(r_d_resting, "r_d_resting")
    if state.n_species == 1:
        if state.restchannels < 1:
            raise ValueError("go-or-grow needs at least one rest channel for resting cells")
        resting = np.arange(state.K) >= state.velocitychannels
        resting_species = 0
    elif state.n_species == 2:
        _check_layout(state)
        resting = np.array([np.zeros(state.K, dtype=bool), np.ones(state.K, dtype=bool)])
        resting_species = _RESTING
    else:
        raise ValueError(f"go_or_grow.growth works with one or two species, not {state.n_species}")
    resting = np.broadcast_to(resting, (state.n_species, state.K))
    crowding = state.density / _node_capacity(state)
    death = np.where(resting, r_d if r_d_resting is None else r_d_resting, r_d)
    state.remove_cells(np.broadcast_to(death, state.dims + death.shape))
    if state.volume_exclusion and when_full == "legacy":
        n_r = (state.counts * resting).sum(axis=(-2, -1))
        births = np.zeros(state.dims + (state.n_species,), dtype=np.int64)
        births[..., resting_species] = state.rng.binomial(np.minimum(n_r, state.restchannels - n_r), r_b)
        state.add_cells(births, channels={resting_species: "rest"})
        return
    rate = r_b if state.volume_exclusion else np.clip(r_b * (1 - crowding), 0, 1)
    division = np.where(resting, np.asarray(rate, dtype=float)[..., None, None], 0.0)
    state.divide_cells(np.broadcast_to(division, state.dims + resting.shape),
                       channels={resting_species: "rest"})


def _go_or_rest_cells(state, kappa, theta, when_full, density):
    """go_or_rest for identity-based models: every cell switches with its own probability."""
    cells, rng = state.cells, state.rng
    rho = _relative_density(state, density)[cells.node]
    rest = tanh_switch(rho, _per_cell(cells, kappa, "kappa"), _per_cell(cells, theta, "theta"))
    resting = cells.in_channels("rest")
    to_rest = ~resting & (rng.random(len(cells)) < rest)
    to_move = resting & (rng.random(len(cells)) < 1 - rest)
    if state.volume_exclusion:
        # only as many cells as there were free channels before the switch
        n_r = state.counts[..., 0, state.velocitychannels:].sum(-1)
        n_m = state.counts[..., 0, :state.velocitychannels].sum(-1)
        free_rest, free_velocity = state.restchannels - n_r, state.velocitychannels - n_m
        if when_full == "legacy":  # the cells that fit try to switch
            to_rest = cells.pick(~resting, free_rest) & to_rest
            to_move = cells.pick(resting, free_velocity) & to_move
        else:  # every cell tries; the successful ones are chosen at random
            to_rest = cells.pick(to_rest, free_rest)
            to_move = cells.pick(to_move, free_velocity)
    cells.move(to_rest, "rest")
    cells.move(to_move, "velocity")


def _growth_cells(state, r_b, r_d, r_d_resting, when_full, mutation, new_family):
    """go_or_grow.growth for identity-based models: death and division per cell."""
    cells, rng = state.cells, state.rng
    crowding = state.density / _node_capacity(state)
    resting = cells.in_channels("rest")
    death = _per_cell(cells, r_d, "r_d")
    if r_d_resting is not None:
        death = np.where(resting, _per_cell(cells, r_d_resting, "r_d_resting"), death)
    alive = rng.random(len(cells)) >= death
    cells.kill(~alive)
    resting = resting[alive]
    birth = _per_cell(cells, r_b, "r_b")
    if state.volume_exclusion:
        free_rest = state.restchannels - state.counts[..., 0, state.velocitychannels:].sum(-1)
        if when_full == "legacy":  # as many resting cells as there are free rest channels try
            dividing = cells.pick(resting, free_rest) & (rng.random(len(cells)) < birth)
        else:
            dividing = resting & (rng.random(len(cells)) < birth)
    else:
        rate = np.clip(birth * (1 - crowding[cells.node]), 0, 1)
        dividing = resting & (rng.random(len(cells)) < rate)
    _divide_and_mutate(state, dividing, "rest", mutation, new_family)


def _per_cell(cells, value, name):
    """A parameter per cell: the named trait, or one number for all cells."""
    if isinstance(value, str):
        return cells[value]
    return np.full(len(cells), _number(value, name))


def _number(value, name):
    if isinstance(value, str):
        # a model mismatch rather than a wrong type: the same value is valid in identity-based models
        raise ValueError(f"{name}={value!r} names a cell trait, which only identity-based models have")  # noqa: TRY004
    return value


def go_or_grow_layout(state):
    """Move migrating cells to velocity channels and resting cells to rest channels.

    For random initial states of the two-species form, e.g. in
    :func:`lgca.testing.check_interaction`: cells in the wrong channels are
    dropped. States with one species are left as they are.
    """
    if state.n_species == 1:
        return
    counts = state.counts.copy()
    velocity = state.velocitychannels
    counts[..., _MIGRATING, velocity:] = 0
    counts[..., _RESTING, :velocity] = 0
    state.counts = counts


def _relative_density(state, density):
    """Cells over capacity at every node, or averaged over the node and its neighbours."""
    if density == "node":
        cells = state.density
    elif density == "neighbourhood":
        cells = (state.density + state.neighbor_sum(state.density)) / (state.velocitychannels + 1)
    else:
        raise ValueError(f"density must be 'node' or 'neighbourhood', got {density!r}")
    return cells / _node_capacity(state)


def _node_capacity(state):
    """Cells a node can hold: its channels with volume exclusion, the model capacity without."""
    return state.K if state.volume_exclusion else state.capacity


def _check_mode(when_full):
    if when_full not in _WHEN_FULL:
        raise ValueError(f"when_full must be one of {_WHEN_FULL}, got {when_full!r}")


def _spread(rng, number, channels):
    """Spread ``number`` cells per node uniformly over ``channels`` channels."""
    return rng.multinomial(number, np.full(channels, 1 / channels))


def _check_layout(state):
    velocity = state.velocitychannels
    if state.restchannels < 1:
        raise ValueError("go-or-grow needs at least one rest channel for resting cells")
    if np.any(state.counts[..., _MIGRATING, velocity:]) or np.any(state.counts[..., _RESTING, :velocity]):
        raise ValueError("go-or-grow keeps migrating cells (species 0) in velocity channels and resting "
                         "cells (species 1) in rest channels; the state has cells elsewhere")


def _pick(rng, candidates, number):
    """Indicator of ``number`` uniformly chosen channels among ``candidates`` at every node."""
    scores = rng.random(candidates.shape)
    scores[~candidates] = np.inf
    ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1)
    return (ranks < number[..., None]).astype(np.int64)
