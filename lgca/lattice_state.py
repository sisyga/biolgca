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

import copy
import functools
import itertools
from collections.abc import Sequence
from math import comb
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
        ``capacity`` without it; :attr:`has_capacity` tells whether the model
        sets one.
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
        # lgca.nonborder as slices: the interior as a view, without copying through index arrays
        self._interior = tuple(slice(lgca.r_int, lgca.r_int + size) for size in self._dims)
        self._identity = isinstance(lgca, IBLGCA_base)
        self._cells = None
        self._ghost_cells = None
        table = lgca._cell_table() if isinstance(lgca, NoVE_IBLGCA_base) else None
        if table is not None:  # the model holds a cell table: no lists to read
            interior = self._cells_from_table(lgca, *table)
        else:
            interior = np.asarray(lgca.nodes[self._interior])
            if self._identity:  # labelled cells: a table with one entry per cell
                self._cells = _cells_from_nodes(self, interior)
                interior = lgca._channel_counts(interior)
        self._has_species_axis = interior.ndim == len(self._dims) + 2
        if not self._has_species_axis:
            interior = interior[..., None, :]
        self._ve = not isinstance(lgca, NoVE_IBLGCA_base) if self._identity else lgca.nodes.dtype == bool
        # with volume exclusion counts are 0 or 1: small integers keep the temporaries of rules small
        self._dtype = np.int8 if self._ve else np.int64
        if not self._ve:
            _check_representable(interior)
        self._counts = interior.astype(self._dtype)
        n_species, channels = self._counts.shape[-2:]
        self._capacity_set = capacity is not None or not self._ve
        if capacity is None:
            capacity = n_species * channels if self._ve else getattr(lgca, "capacity", channels)
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        self._capacity = int(capacity)
        self._initial = (self.density if kind == "phenotype_switch"
                         else self.species_density if kind == "reorientation" else None)
        self._initial_cells = (None if self._cells is None or kind not in ("reorientation", "phenotype_switch")
                               else (self._cells.index.copy(), self._cells.label.copy()))

    def __repr__(self) -> str:
        exclusion = "with" if self._ve else "without"
        return (f"LatticeState({self.geometry}, dims={self._dims}, n_species={self.n_species}, "
                f"K={self.K}, {exclusion} volume exclusion, step={self._step})")

    # ------------------------------------------------------------------ views

    @property
    def counts(self) -> np.ndarray:
        """Cells per channel, shape ``dims + (n_species, K)`` (read-only).

        Integers: ``int8`` with volume exclusion (0 or 1), else ``int64``;
        sums and products with other arrays give the wider type. Assign a new
        array to replace the whole state; the assignment is checked
        (non-negative integers, one cell per channel and species with volume
        exclusion).
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
            self._counts = value.astype(self._dtype)
            return
        if not np.issubdtype(value.dtype, np.integer):
            if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                raise ValueError("counts must be finite numbers")
            if np.any(value != np.round(value)):
                raise ValueError("counts must be non-negative integers")
        if np.any(value < 0):
            raise ValueError("counts must be non-negative integers")
        if self._ve and np.any(value > 1):
            raise ValueError("with volume exclusion a channel holds at most one cell of each species")
        self._counts = value.astype(self._dtype, copy=False)

    @property
    def density(self) -> np.ndarray:
        """Cells per node, shape ``dims``."""
        species = self.species_density
        return species[..., 0].copy() if species.shape[-1] == 1 else species.sum(axis=-1)

    @property
    def species_density(self) -> np.ndarray:
        """Cells per node and species, shape ``dims + (n_species,)``."""
        return self._counts @ np.ones(self._counts.shape[-1], dtype=np.int64)  # faster than sum(-1)

    @property
    def flux(self) -> np.ndarray:
        """Sum of the velocities of the cells at each node, shape ``dims + (d,)``."""
        velocities = self._counts[..., :self.velocitychannels].sum(axis=-2)
        return velocities @ np.asarray(self.c, dtype=float).T

    def sensing(self, species) -> LatticeState:
        """The state with only the cells of ``species`` (an index or a list of indices), to read.

        For cues that sense some species only, e.g. cells that align with
        their own species: ``counts``, ``density``, ``flux`` and the other
        views of the returned state count only these cells. It cannot be
        committed.
        """
        chosen = _species_indices(species, self.n_species)
        view = copy.copy(self)
        view._counts = self._counts[..., chosen, :]
        view._cells = None
        view._sensing_only = True
        return view

    @property
    def cells(self):
        """The cells of an identity-based model, one entry per cell (:class:`~lgca.cells.Cells`).

        Gives each cell's label, node, channel and traits, and operations on
        individual cells (``kill``, ``divide``, ``move``, ``set_trait``).
        """
        if self._cells is None:
            raise TypeError("state.cells exists in identity-based models; in classical models, cells "
                            "have no labels, so rules use state.counts and the operations")
        return self._cells

    def neighbor_sum(self, values) -> np.ndarray:
        """Sum of ``values`` over each node's neighbours, excluding the node.

        ``values`` has shape ``dims`` or ``dims + (...)``. Outside the lattice,
        values wrap around with periodic boundaries and are zero otherwise,
        as there are no cells beyond reflecting or absorbing walls.
        """
        values = np.asarray(values)
        if values.shape[:len(self._dims)] != self._dims:
            raise ValueError(f"values must start with the lattice shape {self._dims}, got {values.shape}")
        return self._lgca.nb_sum(self._pad(values))[self._interior]

    def neighbor_values(self, values) -> np.ndarray:
        """``values`` at the neighbour each velocity channel points to, shape ``dims + (velocitychannels,)``.

        ``values`` has shape ``dims``. Beyond the lattice edge they wrap
        around with periodic boundaries and are zero otherwise, as in
        :meth:`neighbor_sum`, which is their sum over the channels.
        """
        values = np.asarray(values, dtype=float)
        if values.shape != self._dims:
            raise ValueError(f"values must have the lattice shape {self._dims}, got {values.shape}")
        return self._lgca.channel_weight(self._pad(values))[self._interior]

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
        if padded.shape != np.shape(self._lgca.cell_density)[:len(self._dims)]:
            raise ValueError("gradient needs a scalar field with one value per node")
        return self._lgca.gradient(padded)[self._interior]

    def field(self, name: str) -> np.ndarray:
        """A named field from ``StateSpec.fields``, shape ``dims + (...)`` (read-only)."""
        values = np.asarray(self._padded_field(name))
        if values.shape[:len(self._dims)] != self._dims:
            values = values[self._interior]
        values = values.view()
        values.flags.writeable = False
        return values

    def _padded_field(self, name):
        """The field as the model stores it, with ghost nodes, or padded like cells."""
        if not hasattr(self._lgca, name):
            raise KeyError(f"state.fields.{name} does not exist; declare the field in StateSpec.fields")
        self.fields_read.add(name)
        values = np.asarray(getattr(self._lgca, name))
        padded = np.shape(self._lgca.cell_density)[:len(self._dims)]
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
    def has_capacity(self) -> bool:
        """Whether the model sets a capacity (``StateSpec.capacity``).

        Models without volume exclusion always have one. With volume
        exclusion it is optional: a soft limit on all cells of a node, in
        addition to the channels, e.g. for competition between species.
        """
        return self._capacity_set

    @property
    def volume_exclusion(self) -> bool:
        return self._ve

    @property
    def identity_based(self) -> bool:
        """Whether cells carry labels and traits (see :attr:`cells`)."""
        return self._identity

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
        if self._identity:
            cells = self._cells
            dying = self.rng.random(len(cells)) < self._per_cell(self._probability(p, "p"))
            removed = self._per_node(cells.index[dying])
            cells.kill(dying)
            return removed
        p = self._probability(p, "p")
        if self._ve:  # a channel holds at most one cell: one random number each
            removed = self._counts * (self.rng.random(self._counts.shape) < p)
        else:
            removed = self.rng.binomial(self._counts, p)
        self._counts -= removed
        return removed.sum(axis=-1)

    def divide_cells(self, p, channels: Any = "all") -> np.ndarray:
        """Every cell divides with probability ``p``; the daughter has its species.

        Daughters go to free channels of the set ``channels`` (see
        :meth:`add_cells`). With volume exclusion, a division fails when its
        species has no free channel left in the set; which divisions fail is
        random. Without volume exclusion,
        ``channels="same"`` puts each daughter in its mother's channel.

        In identity-based models the daughters inherit their mother's traits
        (see :meth:`lgca.cells.Cells.divide`).

        Returns the number of added cells per node and species.
        """
        if self._identity:
            cells = self._cells
            dividing = self.rng.random(len(cells)) < self._per_cell(self._probability(p, "p"))
            daughters = cells.divide(dividing, channels=channels)
            return self._per_node(cells.index[daughters])
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
        self._require_classical("add_cells", "new cells need traits; let cells divide instead "
                                "(divide_cells or state.cells.divide)")
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

        With ``channels="same"`` a cell keeps its channel. Otherwise a switched
        cell moves to a random channel of its new species in the given set (see
        :meth:`add_cells`). With volume exclusion, the switch fails if that
        channel is occupied; a single switching cell into a species with ``n``
        cells in ``C`` channels thus succeeds with probability ``1 - n / C``.
        Cells that switch into the same species at a node aim at distinct
        channels; if they are more than the channels, those that aim at one
        are chosen at random and the others fail. With ``channels="same"`` a
        cell succeeds if its channel is free for the new species; cells of
        different species that want the same channel get one random winner.
        Occupancy is judged before the switch: a channel vacated by a cell
        that switches away is not available in the same step. Cells whose
        switch fails keep their species.

        Returns the number of switches per node, shape
        ``dims + (n_species, n_species)``, from species ``a`` (row) to ``b``.
        """
        self._require_classical("switch_phenotype", "they have one species; change the traits of "
                                "cells with state.cells.set_trait")
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
        number = (self._counts * mask) @ np.ones(self.K, dtype=np.int64)
        cleared = self._counts * ~mask
        if self._ve:  # every node draws a state of the set's channels
            for moving in np.flatnonzero(mask.any(axis=-1)):
                where = np.flatnonzero(mask[moving])
                if where[-1] - where[0] + 1 == len(where):  # contiguous: a slice is faster
                    where = slice(where[0], where[-1] + 1)
                cleared[..., moving, where] = random_occupancy(self.rng, number[..., moving], mask[moving].sum())
        else:
            cleared += self._spread(number, allowed)
        if self._identity:  # one species: place the cells of the set on the new cell numbers
            self._place_cells(cleared[..., 0, :], mask[0])
        self._counts = cleared

    def commit(self) -> None:
        """Check the state and write it into the model's interior nodes.

        Raises ``ValueError`` if the state breaks the conservation law of its
        kind. Ghost nodes are left to the model's boundary conditions.
        """
        if getattr(self, "_sensing_only", False):
            raise TypeError("a state from sensing() shows some species only and cannot be committed")
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
        if self._initial_cells is not None and not _same_cells(self._initial_cells, self._cells):
            raise ValueError(f"a {self._kind} must keep the cells of every node; cells were removed "
                             "or added")
        lgca = self._lgca
        if self._ghost_cells is not None:
            cells, (ghost_labels, ghost_slots) = self._cells, self._ghost_cells
            slots = lgca._slot_table()["padded"][cells.index * self.K + cells.channel]
            lgca._set_cell_table(np.concatenate([cells.label, ghost_labels]),
                                 np.concatenate([slots, ghost_slots]))
        elif self._identity:
            lgca.nodes[self._interior] = _nodes_from_cells(self._cells, self._counts[..., 0, :],
                                                           lgca.nodes.dtype)
        else:
            interior = self._counts if self._has_species_axis else self._counts[..., 0, :]
            lgca.nodes[self._interior] = interior
        lgca.update_dynamic_fields()

    # --------------------------------------------------------------- helpers

    def _cells_from_table(self, lgca, labels, slots):
        """Cells from the model's table (labels, padded slots); returns the counts per channel."""
        from .cells import Cells

        K = int(lgca.K)
        inner = lgca._slot_table()["interior"][slots]
        inside = inner >= 0
        self._ghost_cells = (labels[~inside], slots[~inside])  # in flight beyond a wall
        self._cells = Cells(self, labels[inside], inner[inside] // K, inner[inside] % K)
        counts = np.bincount(inner[inside], minlength=int(np.prod(self._dims)) * K)
        return counts.reshape(self._dims + (K,))

    def _require_classical(self, operation, reason):
        if self._identity:
            raise TypeError(f"{operation} is not available in identity-based models: {reason}")

    def _cells_changed(self):
        """Recount the cells per channel after an operation on the cell table."""
        cells = self._cells
        n_slots = int(np.prod(self._dims)) * self.K
        counts = np.bincount(cells.index * self.K + cells.channel, minlength=n_slots)
        self._counts = counts.reshape(self._dims + (1, self.K))

    def _per_cell(self, values):
        """Values that broadcast against ``dims + (1, K)``, looked up for every cell."""
        values = np.broadcast_to(values, self._dims + (1, self.K)).reshape(-1, self.K)
        return values[self._cells.index, self._cells.channel]

    def _per_node(self, index):
        """Number of cells per node and species (one species) from node indices."""
        counts = np.bincount(index, minlength=int(np.prod(self._dims)))
        return counts.reshape(self._dims + (1,))

    def _place_cells(self, counts, channels):
        """Put the cells in the channel set on the new cell numbers ``counts``, in random order."""
        cells = self._cells
        inside = channels[cells.channel]
        K = self.K
        # slots of the set in node order, one per new cell, and the set's cells in node order
        slots = np.repeat(np.arange(counts.size), (counts * channels).ravel())
        moving = np.flatnonzero(inside)
        order = np.argsort(cells.index[moving] + self.rng.random(len(moving)))  # by node, random within
        channel = cells.channel.copy()
        channel[moving[order]] = slots % K
        cells.channel = channel

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
        return channel_mask(channels, self.K, self.velocitychannels)

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
        number = np.asarray(number, dtype=np.int64)
        if number.sum() > 20 * number.size:  # many cells: one multinomial draw per node and species
            weights = allowed / allowed.sum(axis=-1, keepdims=True)
            return self.rng.multinomial(number, weights)
        # every cell picks a channel of its species' set, and the picks are counted
        spread = np.zeros(number.shape + (self.K,), dtype=np.int64)
        for species in range(number.shape[-1]):
            options = np.flatnonzero(allowed[species])
            cells = number[..., species].ravel()
            rows = np.repeat(np.arange(cells.size), cells)
            picks = options[self.rng.integers(0, len(options), len(rows))]
            spread[..., species, :] = np.bincount(rows * self.K + picks,
                                                  minlength=cells.size * self.K).reshape(spread.shape[:-2] + (self.K,))
        return spread

    def _choose(self, free, number):
        """Occupy ``number`` uniformly chosen channels among the ``free`` ones (volume exclusion)."""
        chosen = np.zeros(self._counts.shape, dtype=np.int64)
        number = np.broadcast_to(number, self._counts.shape[:-1])
        rows = np.nonzero(number)  # only the nodes and species that gain cells
        if not len(rows[0]):
            return chosen
        free = np.broadcast_to(free, self._counts.shape)[rows]
        scores = np.where(free, self.rng.random(free.shape), np.inf)
        # the number-th smallest score of each row is the threshold
        threshold = np.take_along_axis(np.sort(scores, axis=-1), (number[rows] - 1)[:, None], axis=-1)
        chosen[rows] = free & (scores <= threshold)
        return chosen

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
        cumulative = np.cumsum(transition, axis=-1)
        draws = self.rng.random(counts.shape)
        target = np.zeros(counts.shape, dtype=np.int64)
        for species in range(n_species - 1):
            target += draws >= cumulative[..., :, species, None]
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
                # the switchers aim at distinct random channels of the set (at most one per
                # channel, the others fail) and succeed where the channel was free before
                options = np.flatnonzero(allowed[species])
                aiming = np.minimum(candidates.sum(axis=(-2, -1)), len(options))
                target_channels = np.zeros(self._dims + (self.K,), dtype=bool)
                target_channels[..., options] = random_occupancy(self.rng, aiming, len(options))
                landed = target_channels & free_before[..., species, :]
                accepted = candidates & self._first(scores, landed.sum(axis=-1))
                added[..., species, :] = landed
            removed += accepted
            switched[..., species] = accepted.sum(axis=-1)
        self._counts = counts - removed + added
        return switched

    def _first(self, scores, number):
        """The ``number`` smallest ``scores`` of every node (over species and channels), as a mask."""
        flat = scores.reshape(self._dims + (-1,))
        ordered = np.sort(flat, axis=-1)
        threshold = np.take_along_axis(ordered, np.maximum(number - 1, 0)[..., None], axis=-1)
        chosen = (flat <= threshold) & (number[..., None] > 0)
        return chosen.reshape(scores.shape)


@functools.lru_cache(maxsize=64)
def occupations(channels, cells):
    """All states of ``channels`` channels with ``cells`` of them occupied, one per row (read-only).

    Cached, and shared by the uniform placement of cells (:func:`random_occupancy`) and the
    Boltzmann sampler of channel subsets.
    """
    rows = list(itertools.combinations(range(channels), cells))
    states = np.zeros((len(rows), channels), dtype=bool)
    states[np.repeat(np.arange(len(rows)), cells), np.asarray(rows, dtype=np.int64).ravel()] = True
    states.flags.writeable = False
    return states


def _enumerable(channels, cells):
    """Whether the states of ``cells`` cells in ``channels`` channels fit the enumeration limits.

    The limits of the models' own enumeration (``LGCA_base.get_permutations``): at most a
    million states and 64 MiB.
    """
    return _enumerable_rows(comb(channels, cells), channels)


def _enumerable_rows(states, channels):
    from .base import _MAX_LAZY_CACHE_BYTES, _MAX_PERMUTATION_CANDIDATES

    return states <= _MAX_PERMUTATION_CANDIDATES and states * channels <= _MAX_LAZY_CACHE_BYTES


@functools.lru_cache(maxsize=8)
def _state_table(channels):
    """All states of ``channels`` channels, by number of cells: table, first rows, sizes; or None.

    None when the ``2 ** channels`` states exceed the enumeration limits (more than 20
    channels).
    """
    if not _enumerable_rows(2 ** channels, channels):
        return None
    sizes = np.array([comb(channels, cells) for cells in range(channels + 1)])
    table = np.concatenate([occupations(channels, cells) for cells in range(channels + 1)])
    return table, np.r_[0, np.cumsum(sizes)[:-1]], sizes


def random_occupancy(rng, number, channels):
    """``number`` cells per entry in uniformly chosen channels of ``channels``, as a boolean array.

    The states with ``n`` cells are enumerated once (and cached), and each entry draws one
    of them with one random number: from one table of all states when they fit the
    enumeration limits (up to 20 channels), else from the states of each number of cells.
    Where even these are too many, e.g. about half-full nodes of the Moore lattice, entries
    rank random keys per channel instead. All ways give every state the same probability.
    """
    number = np.asarray(number, dtype=np.int64)
    table = _state_table(channels)
    if table is not None:
        states, first, sizes = table
        return states[first[number] + (rng.random(number.shape) * sizes[number]).astype(np.int64)]
    placed = np.zeros(number.shape + (channels,), dtype=bool)
    for cells in np.unique(number):
        if cells == 0:
            continue
        where = np.nonzero(number == cells)
        count = len(where[0])
        if _enumerable(channels, int(cells)):
            states = occupations(channels, int(cells))
            placed[where] = states[(rng.random(count) * len(states)).astype(np.int64)]
        else:
            keys = rng.random((count, channels))
            placed[where] = np.argsort(np.argsort(keys, axis=-1), axis=-1) < cells
    return placed


def _species_indices(species, n_species):
    """Species as a sorted array of distinct indices, checked against ``n_species``."""
    chosen = np.atleast_1d(np.asarray(species))
    if chosen.ndim != 1 or len(chosen) == 0 or chosen.dtype == bool or not np.issubdtype(chosen.dtype, np.integer):
        raise ValueError(f"species must be a species index or a list of them, got {species!r}")
    if np.any((chosen < 0) | (chosen >= n_species)):
        raise ValueError(f"species {species!r} does not exist; the model has {n_species} "
                         f"(indices 0 to {n_species - 1})")
    return np.unique(chosen)


def _check_representable(counts):
    """Refuse cell numbers whose sum over a node does not fit the signed int64 of the rules."""
    counts = np.asarray(counts)
    if counts.size == 0 or counts.dtype.kind != "u" or int(counts.max()) < _INT64_LIMIT // counts.shape[-1]:
        return
    totals = counts.reshape(-1, counts.shape[-2] * counts.shape[-1]).astype(object).sum(axis=-1)
    if max(totals) >= _INT64_LIMIT:
        raise ValueError(f"a node holds {max(totals)} cells, but the samplers count the cells of a node "
                         f"as signed int64 (below {_INT64_LIMIT})")


_INT64_LIMIT = 2 ** 63


def channel_mask(channels, K, velocitychannels):
    """Boolean mask of length ``K`` for ``"all"``, ``"rest"``, ``"velocity"``, channel indices or a mask."""
    if isinstance(channels, str):
        masks = {"all": np.ones(K, dtype=bool),
                 "rest": np.arange(K) >= velocitychannels,
                 "velocity": np.arange(K) < velocitychannels}
        if channels not in masks:
            raise ValueError(f"channels must be 'all', 'rest', 'velocity', a sequence of channel "
                             f"indices or a boolean mask, got {channels!r}")
        mask = masks[channels]
    else:
        channels = np.asarray(channels)
        if channels.dtype == bool:
            if channels.shape != (K,):
                raise ValueError(f"a channel mask must have length K={K}")
            mask = channels
        else:
            if channels.ndim != 1 or np.any((channels < 0) | (channels >= K)):
                raise ValueError(f"channel indices must lie between 0 and {K - 1}")
            mask = np.zeros(K, dtype=bool)
            mask[channels] = True
    if not mask.any():
        raise ValueError(f"the channel set {channels!r} is empty in this model")
    return mask


def _cells_from_nodes(state, interior):
    """The table of cells from interior labels: an array (0 empty) or lists of labels per channel."""
    from .cells import Cells

    K = interior.shape[-1]
    flat = interior.reshape(-1)
    if interior.dtype != object:
        slot = np.flatnonzero(flat)
        return Cells(state, flat[slot], slot // K, slot % K)
    lengths = np.fromiter(map(len, flat), dtype=np.int64, count=flat.size)
    labels = np.fromiter((label for channel in flat for label in channel), dtype=np.int64,
                         count=int(lengths.sum()))
    slot = np.repeat(np.arange(flat.size), lengths)
    return Cells(state, labels, slot // K, slot % K)


def _nodes_from_cells(cells, counts, dtype):
    """Interior labels from the table: an array (0 empty) or lists of labels per channel."""
    K = counts.shape[-1]
    slot = cells.index * K + cells.channel
    if dtype != object:
        flat = np.zeros(counts.size, dtype=dtype)
        flat[slot] = cells.label
        return flat.reshape(counts.shape)
    labels = cells.label[np.argsort(slot, kind="stable")].tolist()
    ends = np.cumsum(counts.ravel()).tolist()
    flat = np.empty(counts.size, dtype=object)
    start = 0
    for index, end in enumerate(ends):
        flat[index] = labels[start:end]
        start = end
    return flat.reshape(counts.shape)


def _cells_per_node(cells):
    """Node indices and labels, sorted, to compare which cells sit at which node."""
    order = np.lexsort((cells.label, cells.index))
    return cells.index[order], cells.label[order]


def _same_cells(initial, cells):
    """Whether every node holds the cells it held initially (``initial``: node indices and labels)."""
    index, label = initial
    if np.array_equal(index, cells.index) and np.array_equal(label, cells.label):
        return True  # the usual case: only channels changed
    if len(index) != len(cells.index):
        return False
    order = np.lexsort((label, index))
    return all(np.array_equal(a, b) for a, b in zip((index[order], label[order]), _cells_per_node(cells)))


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
