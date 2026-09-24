"""The lattice as an interaction sees it, and the operations it may apply.

A :class:`LatticeState` hides ghost nodes, dtypes and the differences between
classical model families. Its channel states always have a species axis,
``dims + (n_species, K)``, so a rule written for one species works unchanged
for several, with or without volume exclusion.

Rules change the state through operations with a defined meaning per cell
("every cell dies with probability p"), which are implemented once for models
with volume exclusion (at most one cell per channel and species) and without
it (any number of cells per channel). The changes reach the model when
:meth:`LatticeState.commit` writes them back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["LatticeState"]

_KINDS = (None, "birth_death", "phenotype_switch", "reorientation")


class LatticeState:
    """Interior channel states of a classical LGCA with a species axis.

    Parameters
    ----------
    lgca : LGCA object
        A classical model with or without volume exclusion and with one or
        several species. The state of an identity-based model can be read
        (``counts`` are its cells per channel) but not changed.
    step : int, default=0
        Current time step, available as :attr:`step`.
    capacity : int, optional
        Cells per node at which a node counts as crowded, the scale that rules
        use in e.g. ``density / capacity``. It is not enforced: the only hard
        limit is volume exclusion, one cell per channel and species. Defaults
        to ``n_species * K`` with volume exclusion and to the model's
        ``capacity`` without it.
    kind : {None, "birth_death", "phenotype_switch", "reorientation"}
        Kind of the interaction that uses the state. :meth:`commit` checks its
        conservation law: a reorientation keeps the number of cells of each
        species at every node, a phenotype switch the number of cells at every
        node.

    Notes
    -----
    Randomness comes from :attr:`rng`, the model's generator, so seeded runs
    are reproducible. Probabilities and cell numbers passed to operations
    broadcast against the lattice: a scalar, an array of shape ``dims``, one
    value per species (shape ``(n_species,)`` or ``dims + (n_species,)``) or,
    for probabilities, one value per channel (``dims + (n_species, K)``).

    Examples
    --------
    >>> from lgca import get_lgca
    >>> lgca = get_lgca(geometry="square", dims=(10, 10), restchannels=1, density=0.5, seed=1)
    >>> state = LatticeState(lgca, kind="birth_death")
    >>> removed = state.remove_cells(0.1 * state.density / state.K)
    >>> added = state.divide_cells(0.2, channels="rest")
    >>> state.commit()
    """

    def __init__(self, lgca, *, step: int = 0, capacity: int | None = None, kind: str | None = None):
        from .ib_base import IBLGCA_base
        from .nove_ib_base import NoVE_IBLGCA_base

        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self._lgca = lgca
        self._kind = kind
        self.fields_read: set[str] = set()  # names passed to field() or gradient(), for dependencies
        self._step = int(step)
        self._dims = tuple(int(size) for size in lgca.dims)
        self._identity = isinstance(lgca, IBLGCA_base)
        interior = np.asarray(lgca.nodes[lgca.nonborder])
        self._labels = None
        if self._identity:  # cell labels: keep a copy, count the cells per channel
            self._labels = _copy_labels(interior)
            interior = lgca._channel_counts(interior)
        self._has_species_axis = interior.ndim == len(self._dims) + 2
        if not self._has_species_axis:
            interior = interior[..., None, :]
        self._counts = interior.astype(np.int64)
        self._ve = not isinstance(lgca, NoVE_IBLGCA_base) if self._identity else lgca.nodes.dtype == bool
        n_species, channels = self._counts.shape[-2:]
        if capacity is None:
            capacity = n_species * channels if self._ve else getattr(lgca, "capacity", channels)
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        self._capacity = int(capacity)
        self._initial = {"birth_death": None,
                         "phenotype_switch": self.density,
                         "reorientation": self.species_density,
                         None: None}[kind]

    def __repr__(self) -> str:
        exclusion = "with" if self._ve else "without"
        return (f"LatticeState({self.geometry}, dims={self._dims}, n_species={self.n_species}, "
                f"K={self.K}, {exclusion} volume exclusion, step={self._step})")

    # ------------------------------------------------------------------ views

    @property
    def counts(self) -> np.ndarray:
        """Cells per channel, shape ``dims + (n_species, K)`` (read-only).

        Assign a new array to replace the whole state; the assignment is
        checked (non-negative integers, one cell per channel and species with
        volume exclusion).
        """
        view = self._counts.view()
        view.flags.writeable = False
        return view

    @counts.setter
    def counts(self, value) -> None:
        if self._identity:
            raise TypeError("state.counts of an identity-based model can be read but not assigned: an "
                            "array of cell numbers does not say which cell went where. Change the "
                            "state with its operations, e.g. shuffle_cells, which move the cells' labels.")
        value = np.asarray(value)
        if value.shape == self._dims + (self.K,) and self.n_species == 1:
            value = value[..., None, :]
        if value.shape != self._counts.shape:
            raise ValueError(f"counts must have shape {self._counts.shape}, got {value.shape}")
        if value.dtype == bool:
            value = value.astype(np.int64)
        if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
            raise ValueError("counts must be finite numbers")
        if np.any(value < 0) or np.any(value != np.round(value)):
            raise ValueError("counts must be non-negative integers")
        value = value.astype(np.int64)
        if self._ve and np.any(value > 1):
            raise ValueError("with volume exclusion a channel holds at most one cell of each species")
        self._counts = value

    @property
    def density(self) -> np.ndarray:
        """Cells per node, shape ``dims``."""
        return self._counts.sum(axis=(-2, -1))

    @property
    def species_density(self) -> np.ndarray:
        """Cells per node and species, shape ``dims + (n_species,)``."""
        return self._counts.sum(axis=-1)

    @property
    def flux(self) -> np.ndarray:
        """Sum of the velocities of the cells at each node, shape ``dims + (d,)``."""
        velocities = self._counts[..., :self.velocitychannels].sum(axis=-2)
        return velocities @ np.asarray(self.c, dtype=float).T

    def neighbor_sum(self, values) -> np.ndarray:
        """Sum of ``values`` over each node's neighbours, excluding the node.

        ``values`` has shape ``dims`` or ``dims + (...)``. Outside the lattice,
        values wrap around with periodic boundaries and are zero otherwise,
        as there are no cells beyond reflecting or absorbing walls.
        """
        values = np.asarray(values)
        if values.shape[:len(self._dims)] != self._dims:
            raise ValueError(f"values must start with the lattice shape {self._dims}, got {values.shape}")
        return self._lgca.nb_sum(self._pad(values))[self._lgca.nonborder]

    def gradient(self, values) -> np.ndarray:
        """Gradient of a scalar field in lattice units, shape ``dims + (d,)``.

        ``values`` is an array of shape ``dims`` or the name of a field (see
        :meth:`field`). Differences are centred everywhere, using ghost nodes
        beyond the lattice edge: an array gets the ghost values of
        :meth:`neighbor_sum` (wrapped with periodic boundaries, zero
        otherwise), a named field keeps the ghost values the model stores for
        it. This is the convention of the model's own ``gradient`` method.
        """
        if isinstance(values, str):
            padded = np.asarray(self._padded_field(values), dtype=float)
        else:
            values = np.asarray(values, dtype=float)
            if values.shape != self._dims:
                raise ValueError(f"values must have the lattice shape {self._dims}, got {values.shape}")
            padded = self._pad(values)
        if padded.shape != self._lgca.nodes.shape[:len(self._dims)]:
            raise ValueError("gradient needs a scalar field with one value per node")
        return self._lgca.gradient(padded)[self._lgca.nonborder]

    def field(self, name: str) -> np.ndarray:
        """A named field from ``StateSpec.fields``, shape ``dims + (...)`` (read-only)."""
        values = np.asarray(self._padded_field(name))
        if values.shape[:len(self._dims)] != self._dims:
            values = values[self._lgca.nonborder]
        values = values.view()
        values.flags.writeable = False
        return values

    def _padded_field(self, name):
        """The field as the model stores it, with ghost nodes, or padded like cells."""
        if not hasattr(self._lgca, name):
            raise KeyError(f"state.fields.{name} does not exist; declare the field in StateSpec.fields")
        self.fields_read.add(name)
        values = np.asarray(getattr(self._lgca, name))
        padded = self._lgca.nodes.shape[:len(self._dims)]
        if values.shape[:len(self._dims)] == padded:
            return values
        if values.shape[:len(self._dims)] != self._dims:
            raise ValueError(f"field {name!r} has shape {values.shape}, which does not match the lattice")
        return self._pad(values)

    @property
    def dims(self) -> tuple[int, ...]:
        """Lattice shape without ghost nodes."""
        return self._dims

    @property
    def n_species(self) -> int:
        return self._counts.shape[-2]

    @property
    def K(self) -> int:
        """Channels per node and species."""
        return self._counts.shape[-1]

    @property
    def velocitychannels(self) -> int:
        return int(self._lgca.velocitychannels)

    @property
    def restchannels(self) -> int:
        return self.K - self.velocitychannels

    @property
    def capacity(self) -> int:
        """Cells per node at which a node counts as crowded (not enforced)."""
        return self._capacity

    @property
    def volume_exclusion(self) -> bool:
        return self._ve

    @property
    def c(self) -> np.ndarray:
        """Velocity vectors of the velocity channels, shape ``(d, velocitychannels)``."""
        return np.asarray(self._lgca.c)

    @property
    def rng(self) -> np.random.Generator:
        return self._lgca.rng

    @property
    def step(self) -> int:
        return self._step

    @property
    def geometry(self) -> str:
        return self._lgca.geometry

    @property
    def boundary(self) -> str:
        return self._lgca.bc

    @property
    def kind(self) -> str | None:
        return self._kind

    # ------------------------------------------------------------- operations

    def remove_cells(self, p) -> np.ndarray:
        """Every cell dies independently with probability ``p``.

        Returns the number of removed cells per node and species.
        """
        self._require_classical("remove_cells")
        removed = self.rng.binomial(self._counts, self._probability(p, "p"))
        self._counts -= removed
        return removed.sum(axis=-1)

    def divide_cells(self, p, channels: Any = "all") -> np.ndarray:
        """Every cell divides with probability ``p``; the daughter has its species.

        Daughters go to free channels of the set ``channels`` (see
        :meth:`add_cells`). With volume exclusion, a division fails when its
        species has no free channel left in the set; which divisions fail is
        random. Without volume exclusion,
        ``channels="same"`` puts each daughter in its mother's channel.

        Returns the number of added cells per node and species.
        """
        self._require_classical("divide_cells")
        daughters = self.rng.binomial(self._counts, self._probability(p, "p"))
        if isinstance(channels, str) and channels == "same":
            if self._ve:
                raise ValueError("channels='same' needs a model without volume exclusion: "
                                 "with it, the mother's channel is occupied")
            self._counts += daughters
            return daughters.sum(axis=-1)
        return self.add_cells(daughters.sum(axis=-1), channels=channels)

    def add_cells(self, n, channels: Any = "all") -> np.ndarray:
        """Add ``n`` cells per node and species to free channels.

        ``channels`` is ``"all"``, ``"rest"``, ``"velocity"``, a sequence of
        channel indices or a boolean mask of length ``K``, or a mapping from
        species to one of these, e.g. ``{0: "velocity", 1: "rest"}`` (species
        left out use all channels); the new cells are spread uniformly over
        the free channels of this set. With volume
        exclusion, a species gets at most as many cells as it has free
        channels in the set. :attr:`capacity` is not enforced; rules that
        should slow down near it scale their rates, e.g. with
        ``1 - density / capacity``.

        Returns the number of added cells per node and species.
        """
        self._require_classical("add_cells")
        wanted = self._number(n, "n")
        allowed = self._channel_sets(channels)
        if not self._ve:
            added = self._spread(wanted, allowed)
            self._counts += added
            return wanted
        free = (self._counts == 0) & allowed
        granted = np.minimum(wanted, free.sum(axis=-1))
        self._counts += self._choose(free, granted)
        return granted

    def switch_phenotype(self, rates, channels: Any = "same") -> np.ndarray:
        """Every cell of species ``a`` becomes species ``b`` with probability ``rates[a][b]``.

        ``rates`` has shape ``(n_species, n_species)`` or
        ``dims + (n_species, n_species)``; the diagonal is ignored and each
        row's other entries must sum to at most one. The number of cells at
        every node is kept.

        With ``channels="same"`` a cell keeps its channel. Otherwise switched
        cells move to free channels of the given set (see :meth:`add_cells`).
        With volume exclusion, a switch needs a channel that is free for the
        new species before the switch; when several cells compete for fewer
        free channels, the successful ones are chosen at random and the others
        keep their species.

        Returns the number of switches per node, shape
        ``dims + (n_species, n_species)``, from species ``a`` (row) to ``b``.
        """
        self._require_classical("switch_phenotype")
        transition = self._transition_matrix(rates)
        same = isinstance(channels, str) and channels == "same"
        allowed = None if same else self._channel_sets(channels)
        if self._ve:
            return self._switch_ve(transition, allowed)
        return self._switch_nove(transition, allowed)

    def shuffle_cells(self, channels: Any = "all", species: int | Sequence[int] | None = None) -> None:
        """Rearrange each node's cells uniformly at random within a channel set.

        Cells of each species in the set ``channels`` are placed on its
        channels anew, uniformly at random; cells outside the set stay put.
        ``species`` restricts the shuffle to some species. A random walk is
        ``shuffle_cells()``, and moving cells that keep resting cells in place
        use ``shuffle_cells("velocity")``.

        In identity-based models the cells keep their labels: the labelled
        cells in the set are placed on the new positions in random order.
        """
        allowed = self._channel_sets(channels)
        selected = np.zeros(self.n_species, dtype=bool)
        selected[slice(None) if species is None else np.atleast_1d(species)] = True
        mask = selected[:, None] & allowed
        number = np.where(mask, self._counts, 0).sum(axis=-1)
        cleared = np.where(mask, 0, self._counts)
        if self._ve:
            cleared += self._choose(np.broadcast_to(mask, cleared.shape), number)
        else:
            cleared += self._spread(number, allowed)
        self._counts = cleared
        if self._identity:  # one species
            self._labels = place_labels(self._labels, cleared[..., 0, :], mask[0], self.rng)

    def commit(self) -> None:
        """Check the state and write it into the model's interior nodes.

        Raises ``ValueError`` if the state breaks the conservation law of its
        kind. Ghost nodes are left to the model's boundary conditions.
        """
        if self._kind == "reorientation":
            changed = np.any(self.species_density != self._initial, axis=-1)
            what = "the number of cells of each species"
        elif self._kind == "phenotype_switch":
            changed = self.density != self._initial
            what = "the number of cells"
        else:
            changed = None
        if changed is not None and np.any(changed):
            first = tuple(int(index) for index in np.argwhere(changed)[0])
            raise ValueError(f"a {self._kind} must keep {what} at every node; it changed at "
                             f"{int(changed.sum())} nodes, first at {first}")
        lgca = self._lgca
        if self._identity:
            lgca.nodes[lgca.nonborder] = _copy_labels(self._labels)
        else:
            interior = self._counts if self._has_species_axis else self._counts[..., 0, :]
            lgca.nodes[lgca.nonborder] = interior.astype(lgca.nodes.dtype)
        lgca.update_dynamic_fields()

    # --------------------------------------------------------------- helpers

    def _require_classical(self, operation):
        if self._identity:
            raise TypeError(f"{operation} is not available for identity-based models yet; of the "
                            "operations, they support shuffle_cells. Their state can be read, "
                            "e.g. by reorientation terms.")

    def _pad(self, values):
        """Embed interior values in the padded lattice according to the boundary."""
        width = int(self._lgca.r_int)
        ndim = len(self._dims)
        # inflow boundaries are reflecting in x and periodic in y
        wrapped = {"periodic": range(ndim), "inflow": range(1, ndim)}.get(self.boundary, ())
        for axis in range(ndim):
            pad_width = [(0, 0)] * values.ndim
            pad_width[axis] = (width, width)
            values = np.pad(values, pad_width, mode="wrap" if axis in wrapped else "constant")
        return values

    def _probability(self, p, name):
        values = self._broadcastable(p, name, per_channel=True).astype(float, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain probabilities between 0 and 1, not NaN or infinity")
        if np.any((values < 0) | (values > 1)):
            hint = ("; without volume exclusion a node can hold more than its capacity, so a rate "
                    "proportional to density / capacity may need np.minimum(..., 1)"
                    if not self._ve and values.max() > 1 else "")
            raise ValueError(f"{name} must contain probabilities between 0 and 1, got values from "
                             f"{values.min():.3g} to {values.max():.3g}{hint}")
        return values

    def _number(self, n, name):
        values = np.asarray(n)
        if values.dtype == bool or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{name} must contain non-negative integers")
        values = self._broadcastable(values, name, per_channel=False)
        if np.any(values < 0) or np.any(values != np.round(values)):
            raise ValueError(f"{name} must contain non-negative integers")
        shape = self._dims + (self.n_species, 1)
        return np.broadcast_to(values, shape)[..., 0].astype(np.int64)

    def _broadcastable(self, value, name, per_channel):
        """Reshape a scalar or per-node, per-species or per-channel array to broadcast
        against ``dims + (n_species, K)``."""
        value = np.asarray(value)
        dims, species = self._dims, (self.n_species,)
        shapes = {(): value, dims: value[..., None, None], dims + species: value[..., None]}
        if species != dims:
            shapes[species] = value[..., None]
        if per_channel:
            shapes[dims + species + (self.K,)] = value
        if value.shape in shapes:
            return shapes[value.shape]
        expected = ["()", str(dims), str(dims + species)]
        if species != dims:
            expected.append(str(species))
        if per_channel:
            expected.append(str(dims + species + (self.K,)))
        raise ValueError(f"{name} has shape {value.shape}; expected one of {', '.join(expected)}")

    def _channel_mask(self, channels):
        velocity = self.velocitychannels
        if isinstance(channels, str):
            masks = {"all": np.ones(self.K, dtype=bool),
                     "rest": np.arange(self.K) >= velocity,
                     "velocity": np.arange(self.K) < velocity}
            if channels not in masks:
                raise ValueError(f"channels must be 'all', 'rest', 'velocity', a sequence of channel "
                                 f"indices or a boolean mask, got {channels!r}")
            mask = masks[channels]
        else:
            channels = np.asarray(channels)
            if channels.dtype == bool:
                if channels.shape != (self.K,):
                    raise ValueError(f"a channel mask must have length K={self.K}")
                mask = channels
            else:
                if channels.ndim != 1 or np.any((channels < 0) | (channels >= self.K)):
                    raise ValueError(f"channel indices must lie between 0 and {self.K - 1}")
                mask = np.zeros(self.K, dtype=bool)
                mask[channels] = True
        if not mask.any():
            raise ValueError(f"the channel set {channels!r} is empty in this model")
        return mask

    def _channel_sets(self, channels):
        """Allowed channels per species, shape ``(n_species, K)``."""
        if isinstance(channels, dict):
            unknown = [species for species in channels if not 0 <= int(species) < self.n_species]
            if unknown:
                raise ValueError(f"channels names species {unknown}, but the model has {self.n_species}")
            return np.stack([self._channel_mask(channels.get(species, "all"))
                             for species in range(self.n_species)])
        return np.broadcast_to(self._channel_mask(channels), (self.n_species, self.K))

    def _spread(self, number, allowed):
        """Distribute ``number`` cells per node and species uniformly over allowed channels."""
        weights = allowed / allowed.sum(axis=-1, keepdims=True)
        return self.rng.multinomial(number, weights)

    def _choose(self, free, number):
        """Occupy ``number`` uniformly chosen channels among the ``free`` ones (volume exclusion)."""
        if not np.any(number):
            return np.zeros(self._counts.shape, dtype=np.int64)
        scores = self.rng.random(self._counts.shape)
        scores[~np.broadcast_to(free, scores.shape)] = np.inf
        ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1)
        return (ranks < number[..., None]).astype(np.int64)

    def _transition_matrix(self, rates):
        n_species = self.n_species
        rates = np.asarray(rates, dtype=float)
        pair = (n_species, n_species)
        if rates.shape not in (pair, self._dims + pair):
            raise ValueError(f"rates must have shape {pair} or {self._dims + pair}, got {rates.shape}")
        if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
            raise ValueError("rates must contain probabilities between 0 and 1")
        off_diagonal = rates * (1 - np.eye(n_species))
        leaving = off_diagonal.sum(axis=-1)
        if np.any(leaving > 1 + 1e-12):
            raise ValueError("the switching probabilities of each species must sum to at most 1")
        return off_diagonal + np.eye(n_species) * np.clip(1 - leaving, 0, 1)[..., None]

    def _switch_nove(self, transition, allowed):
        counts = self._counts
        pvals = np.broadcast_to(transition[..., :, None, :], counts.shape + (self.n_species,))
        moves = self.rng.multinomial(counts, pvals)  # dims + (from, K, to)
        switched = moves.sum(axis=-2) * (1 - np.eye(self.n_species, dtype=np.int64))
        if allowed is None:
            self._counts = np.swapaxes(moves.sum(axis=-3), -1, -2)
        else:
            stay = np.einsum("...aka->...ak", moves)
            self._counts = stay + self._spread(switched.sum(axis=-2), allowed)
        return switched

    def _switch_ve(self, transition, allowed):
        counts = self._counts
        n_species = self.n_species
        occupied = counts == 1
        cumulative = np.cumsum(np.broadcast_to(transition[..., :, None, :], counts.shape + (n_species,)),
                               axis=-1)
        draws = self.rng.random(counts.shape)
        target = np.minimum((draws[..., None] >= cumulative).sum(axis=-1), n_species - 1)
        own = np.arange(n_species)[:, None]
        free_before = counts == 0
        removed = np.zeros_like(counts)
        added = np.zeros_like(counts)
        switched = np.zeros(self._dims + (n_species, n_species), dtype=np.int64)
        for species in range(n_species):
            candidates = occupied & (target == species) & (own != species)
            if not candidates.any():
                continue
            scores = self.rng.random(counts.shape)
            scores[~candidates] = np.inf
            if allowed is None:
                # one winner per channel among the species that want its free slot
                winner = np.argmin(scores, axis=-2)[..., None, :] == own
                accepted = candidates & winner & free_before[..., species, None, :]
                added[..., species, :] = accepted.any(axis=-2)
            else:
                slots = free_before[..., species, :] & allowed[species]
                flat = scores.reshape(self._dims + (-1,))
                ranks = np.argsort(np.argsort(flat, axis=-1), axis=-1).reshape(counts.shape)
                accepted = candidates & (ranks < slots.sum(axis=-1)[..., None, None])
                number = accepted.sum(axis=(-2, -1))
                added[..., species, :] = self._choose_slots(slots, number)
            removed += accepted
            switched[..., species] = accepted.sum(axis=-1)
        self._counts = counts - removed + added
        return switched

    def _choose_slots(self, free, number):
        """Pick ``number`` uniformly chosen free channels at every node for one species."""
        scores = self.rng.random(free.shape)
        scores[~free] = np.inf
        ranks = np.argsort(np.argsort(scores, axis=-1), axis=-1)
        return (ranks < number[..., None]).astype(np.int64)


def _copy_labels(labels):
    """Copy an array of labels; for lists of labels (no volume exclusion) copy the lists too."""
    labels = np.asarray(labels)
    if labels.dtype != object:
        return labels.copy()
    copied = np.empty(labels.shape, dtype=object)
    for index, channel in np.ndenumerate(labels):
        copied[index] = list(channel)
    return copied


def place_labels(labels, counts, channels, rng):
    """Place each node's labelled cells in the ``channels`` on new cell numbers, in random order.

    ``labels`` has shape ``dims + (K,)``: a label per channel, 0 for empty
    (volume exclusion), or a list of labels per channel (no volume exclusion).
    ``counts`` are the new cell numbers per channel, with as many cells in the
    ``channels`` (a boolean mask of length ``K``) of every node as before;
    cells in other channels stay where they are. Returns the new labels.
    """
    channels = np.asarray(channels, dtype=bool)
    if labels.dtype != object:
        # labelled cells of the set first, in random order; the k-th new cell gets the k-th label
        keys = np.where(channels & (labels > 0), rng.random(labels.shape), np.inf)
        shuffled = np.take_along_axis(labels, np.argsort(keys, axis=-1), axis=-1)
        occupied = (counts > 0) & channels
        rank = np.maximum(np.cumsum(occupied, axis=-1) - 1, 0)
        placed = np.where(occupied, np.take_along_axis(shuffled, rank, axis=-1), 0)
        return np.where(channels, placed, labels).astype(labels.dtype)
    flat = labels.reshape(-1, labels.shape[-1])
    moving = flat[:, channels]
    per_node = np.fromiter((sum(len(channel) for channel in node) for node in moving), dtype=np.int64,
                           count=len(moving))
    cells = np.fromiter((label for node in moving for channel in node for label in channel),
                        dtype=np.intp, count=int(per_node.sum()))
    node = np.repeat(np.arange(len(moving)), per_node)
    cells = cells[np.lexsort((rng.random(len(cells)), node))]
    new_counts = np.asarray(counts).reshape(flat.shape)[:, channels]
    parts = np.split(cells, np.cumsum(new_counts.ravel())[:-1])
    placed = np.empty(moving.size, dtype=object)
    for index, part in enumerate(parts):
        placed[index] = part.tolist()
    result = flat.copy()
    result[:, channels] = placed.reshape(moving.shape)
    return result.reshape(labels.shape)
