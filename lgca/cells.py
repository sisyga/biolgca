"""The cells of identity-based models, one entry per cell.

Identity-based models give every cell a label and traits (heritable
properties such as a birth rate). :class:`Cells` holds the living cells as
flat arrays (label, node, channel), so that rules act on all cells at once
instead of looping over nodes. Traits live in ``lgca.props``, one value per
label, as :class:`TraitArray` buffers.

A rule reaches the cells through :attr:`lgca.lattice_state.LatticeState.cells`:

>>> cells = state.cells                                        # doctest: +SKIP
>>> rho = state.density[cells.node] / state.capacity           # doctest: +SKIP
>>> rest = cells.in_channels("rest")                           # doctest: +SKIP
>>> daughters = cells.divide(rest & (state.rng.random(len(cells)) < cells["r_b"]),
...                          channels="rest")                  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Cells", "TraitArray"]


class TraitArray:
    """One value per cell label, in a NumPy buffer with spare capacity.

    Appending copies the buffer only when it is full, and then doubles it, so
    adding daughters costs almost nothing, and reading the traits of many
    cells is one indexing operation. It behaves like a list for code that
    appends values (``append``, ``extend``, ``len``, iteration) and like an
    array for vectorized code (``np.asarray(values)``, ``values[labels]``).

    Examples
    --------
    >>> r_b = TraitArray([0.2, 0.3])
    >>> r_b.append(0.25)
    >>> r_b[np.array([2, 0])]
    array([0.25, 0.2 ])
    >>> len(r_b)
    3
    """

    __slots__ = ("_data", "_size")

    def __init__(self, values=(), dtype=None):
        data = np.array(list(values) if not isinstance(values, np.ndarray) else values, dtype=dtype)
        if data.ndim != 1:
            flat = np.empty(len(values), dtype=object)
            for index, value in enumerate(values):
                flat[index] = value
            data = flat
        self._data = data
        self._size = len(data)

    @property
    def values(self) -> np.ndarray:
        """The values as an array (a view: writing to it changes the traits)."""
        return self._data[:self._size]

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self) -> int:
        return self._size

    def __array__(self, dtype=None, copy=None):
        values = self.values
        if dtype is not None:
            return values.astype(dtype)
        return values.copy() if copy else values

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value) -> None:
        self._fit(value)
        self.values[index] = value

    def __iter__(self):
        return iter(self.values.tolist())

    def __repr__(self) -> str:
        return f"TraitArray({self.values!r})"

    def append(self, value) -> None:
        self._fit(value)
        self._reserve(1)
        self._data[self._size] = value
        self._size += 1

    def extend(self, values) -> None:
        values = np.asarray(values) if self.dtype != object else _object_array(values)
        self._fit(values)
        self._reserve(len(values))
        self._data[self._size:self._size + len(values)] = values
        self._size += len(values)

    def tolist(self) -> list:
        return self.values.tolist()

    def copy(self) -> TraitArray:
        return TraitArray(self.values.copy())

    def _reserve(self, extra: int) -> None:
        needed = self._size + extra
        if needed > len(self._data):
            data = np.empty(max(needed, 2 * len(self._data), 16), dtype=self._data.dtype)
            data[:self._size] = self._data[:self._size]
            self._data = data

    def _fit(self, values) -> None:
        """Widen the dtype if needed, e.g. from integers to floats."""
        if self.dtype == object:
            return
        dtype = np.result_type(self.dtype, np.asarray(values).dtype)
        if dtype != self.dtype:
            self._data = self._data.astype(dtype)


def _object_array(values):
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def trait_array(lgca, name: str) -> TraitArray:
    """``lgca.props[name]`` as a :class:`TraitArray`, converting it in place if needed."""
    if name not in lgca.props:
        known = ", ".join(sorted(lgca.props)) or "none"
        raise KeyError(f"the cells have no trait {name!r} (traits: {known}); declare it with "
                       f"StateSpec(traits={{{name!r}: ...}})")
    values = lgca.props[name]
    if not isinstance(values, TraitArray):
        values = TraitArray(values)
        lgca.props[name] = values
    return values


class Cells:
    """The living cells of an identity-based model, one entry per cell.

    Created by :attr:`lgca.lattice_state.LatticeState.cells`. The arrays
    :attr:`label`, :attr:`channel` and :attr:`node` have one entry per cell;
    ``cells["name"]`` gives the values of a trait. Operations change the
    cells at once and keep the lattice state up to date; the model changes
    when the state is committed.

    Masks select cells: a boolean array with one entry per cell, or an array
    of cell positions. With volume exclusion, a cell can only enter a free
    channel; when more cells compete for a node's free channels than there
    are, the successful ones are chosen at random.
    """

    def __init__(self, state, label, index, channel):
        self._state = state
        self._lgca = state._lgca
        self.label = np.asarray(label, dtype=np.int64)
        self.index = np.asarray(index, dtype=np.int64)
        self.channel = np.asarray(channel, dtype=np.int64)

    # ------------------------------------------------------------------ views

    def __len__(self) -> int:
        return len(self.label)

    def __repr__(self) -> str:
        return f"<{len(self)} cells; traits: {', '.join(self.traits) or 'none'}>"

    def __getitem__(self, name: str) -> np.ndarray:
        """Values of the trait ``name`` for every cell (a copy)."""
        return trait_array(self._lgca, name).values[self.label]

    @property
    def traits(self) -> tuple[str, ...]:
        """Names of the cells' traits."""
        return tuple(self._lgca.props)

    @property
    def node(self) -> tuple[np.ndarray, ...]:
        """Coordinates of each cell's node; indexes arrays of shape ``dims``, e.g. ``state.density[cells.node]``."""
        return np.unravel_index(self.index, self._state.dims)

    def in_channels(self, channels: Any) -> np.ndarray:
        """Mask of the cells in a channel set: ``"rest"``, ``"velocity"``, ``"all"`` or channel indices."""
        return self._state._channel_mask(channels)[self.channel]

    def pick(self, which, number) -> np.ndarray:
        """Choose at most ``number`` cells per node among ``which``, uniformly at random.

        ``number`` is a number or an array of shape ``dims``. Useful to let
        only as many cells try something as there are free channels.
        """
        selected = self._mask(which)
        number = np.broadcast_to(np.asarray(number), self._state.dims).ravel()
        rank = self._rank(selected)
        return selected & (rank < number[self.index])

    # ------------------------------------------------------------- operations

    def kill(self, which) -> int:
        """Remove the selected cells. Returns their number."""
        dead = self._mask(which)
        keep = ~dead
        self.label, self.index, self.channel = self.label[keep], self.index[keep], self.channel[keep]
        self._state._cells_changed()
        return int(dead.sum())

    def set_trait(self, which, name: str, values) -> None:
        """Set the trait ``name`` of the selected cells to ``values`` (one value or one per selected cell)."""
        trait = trait_array(self._lgca, name)
        labels = self.label[self._mask(which)]
        trait[labels] = values

    def move(self, which, channels: Any) -> np.ndarray:
        """Move the selected cells to channels of the set ``channels`` at their node.

        Without volume exclusion every cell moves to a uniformly chosen channel
        of the set. With it, cells move to channels that were free before the
        call; the rest stay where they are. Returns the mask of moved cells.
        """
        moving = self._mask(which)
        allowed = self._state._channel_mask(channels)
        winners, channel = self._place(moving, allowed)
        self.channel = self.channel.copy()
        self.channel[winners] = channel
        self._state._cells_changed()
        return winners

    def divide(self, which, channels: Any = "all", new_family: bool = False) -> np.ndarray:
        """The selected cells divide; each daughter inherits all traits of its mother.

        Daughters go to channels of the set ``channels`` at the mother's node:
        uniformly chosen without volume exclusion, free channels with it (a
        division fails when no free channel is left; which ones fail is
        random). Without volume exclusion, ``channels="same"`` puts the daughter
        in its mother's channel. With ``new_family=True`` every daughter founds
        a new family (see ``lgca.muller_plot``); otherwise it belongs to its
        mother's family.

        Returns the positions of the daughters in the cell arrays, e.g. to
        mutate their traits with :meth:`set_trait`.
        """
        dividing = self._mask(which)
        if isinstance(channels, str) and channels == "same":
            if self._state.volume_exclusion:
                raise ValueError("channels='same' needs a model without volume exclusion: "
                                 "with it, the mother's channel is occupied")
            succeeded, channel = dividing, self.channel[dividing]
        else:
            succeeded, channel = self._place(dividing, self._state._channel_mask(channels))
        mothers = np.flatnonzero(succeeded)
        lgca = self._lgca
        first = int(lgca.maxlabel) + 1
        labels = np.arange(first, first + len(mothers), dtype=np.int64)
        _inherit(lgca, self.label[mothers], labels, new_family)
        lgca.maxlabel = first + len(mothers) - 1
        start = len(self)
        self.label = np.concatenate([self.label, labels])
        self.index = np.concatenate([self.index, self.index[mothers]])
        self.channel = np.concatenate([self.channel, channel])
        self._state._cells_changed()
        return np.arange(start, len(self))

    # --------------------------------------------------------------- helpers

    def _mask(self, which) -> np.ndarray:
        which = np.asarray(which)
        if which.dtype == bool:
            if which.shape != (len(self),):
                raise ValueError(f"a mask of cells must have one entry per cell ({len(self)}), "
                                 f"got shape {which.shape}")
            return which
        mask = np.zeros(len(self), dtype=bool)
        mask[which.astype(np.int64)] = True
        return mask

    def _rank(self, selected, key=None) -> np.ndarray:
        """Rank of each selected cell among the selected cells of its node, in random order."""
        if key is None:
            key = self._state.rng.random(len(self))
        key = np.where(selected, key, np.inf)
        order = np.lexsort((key, self.index))
        index = self.index[order]
        starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
        group_start = np.repeat(starts, np.diff(np.r_[starts, len(index)]))
        rank = np.empty(len(self), dtype=np.int64)
        rank[order] = np.arange(len(self)) - group_start
        return rank

    def _place(self, selected, allowed):
        """Channels of the set ``allowed`` for the selected cells.

        Returns the mask of cells that get a channel and their channels.
        Without volume exclusion every selected cell gets a uniformly chosen
        channel; with it, free channels are handed out in random order.
        """
        rng = self._state.rng
        options = np.flatnonzero(allowed)
        if not self._state.volume_exclusion:
            return selected, options[rng.integers(0, len(options), int(selected.sum()))]
        K = self._state.K
        n_nodes = int(np.prod(self._state.dims))
        occupied = np.zeros(n_nodes * K, dtype=bool)
        occupied[self.index * K + self.channel] = True
        free = ~occupied.reshape(n_nodes, K) & allowed
        # free channels of every node in random order, and the selected cells in random order
        slot_node, slot_channel = np.nonzero(free)
        slot_order = np.lexsort((rng.random(len(slot_node)), slot_node))
        slot_node, slot_channel = slot_node[slot_order], slot_channel[slot_order]
        rank = self._rank(selected)
        winners = selected & (rank < free.sum(axis=1)[self.index])
        # the k-th winner of a node takes the node's k-th free channel
        slot_start = np.searchsorted(slot_node, self.index[winners])
        return winners, slot_channel[slot_start + rank[winners]]


def _inherit(lgca, mothers, daughters, new_family):
    """Append the daughters' trait rows (copies of their mothers') and their families."""
    if len(daughters) == 0:
        return
    if new_family and "family" not in lgca.props:
        lgca.init_families(type="homogeneous", mutation=True)
    for name in list(lgca.props):
        trait = trait_array(lgca, name)
        if len(trait) != daughters[0]:
            raise ValueError(f"trait {name!r} has {len(trait)} values, but the next label is "
                             f"{daughters[0]}; every trait needs one value per label")
        values = trait.values[mothers]
        if name == "family" and new_family:
            values = _found_families(lgca, values)
        trait.extend(values)


def _found_families(lgca, parent_families):
    """Register one new family per daughter, descending from its mother's family."""
    if not hasattr(lgca, "family_props"):
        raise ValueError("new_family=True needs family tracking with mutations; the model tracks "
                         "families without them (init_families(mutation=False))")
    first = int(lgca.maxfamily) + 1
    families = np.arange(first, first + len(parent_families))
    lgca.maxfamily = int(families[-1])
    lgca.family_props["ancestor"].extend(int(parent) for parent in parent_families)
    descendants = lgca.family_props["descendants"]
    for parent, family in zip(parent_families.tolist(), families.tolist()):
        descendants[parent].append(family)
    descendants.extend([] for _ in families)
    return families


def slot_maps(lgca):
    """Where cells go under the boundary conditions and propagation, as lookup tables.

    Slots number the channels of the padded lattice (ghost nodes included),
    ``node * K + channel``. The maps are derived from the classical model
    without volume exclusion of the same geometry: slot IDs pass through its
    ``apply_boundaries()`` and ``propagation()``, interior and ghost slots in
    separate passes; IDs that meet in one slot pass again in random halves,
    until every ID is found or known to leave the lattice. So the
    tables follow the geometry's own transport code, hexagonal offsets and
    reflection included.

    Returns a dict with ``boundary`` and ``propagation`` (the new slot of a
    cell in every slot, -1 if it leaves the lattice; ``propagation`` includes
    the boundary conditions applied before it), ``source`` (for periodic
    boundaries the interior slot a ghost slot copies, else -1), and
    ``interior`` (the interior slot ``node * K + channel`` of every padded
    slot, -1 for ghosts) with its inverse ``padded``.
    """
    import warnings

    from . import get_lgca

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clone = get_lgca(geometry=lgca.geometry, dims=lgca.dims, restchannels=lgca.restchannels, bc=lgca.bc,
                         ve=False, density=0, interaction="only_propagation", r_int=lgca.r_int)
    shape = clone.nodes.shape
    n = int(np.prod(shape))
    spatial = np.zeros(shape[:-1], dtype=bool)
    spatial[lgca.nonborder] = True
    interior = np.broadcast_to(spatial[..., None], shape).ravel()

    def run(ids, *, propagate):
        clone.nodes = ids.reshape(shape).copy()
        clone.apply_boundaries()
        if propagate:
            clone.propagation()
        return clone.nodes.ravel()

    def lookup(propagate):
        destination = np.full(n, -1, dtype=np.int64)
        split = np.random.default_rng(0)  # splits colliding IDs; not the model's random numbers
        for sources in (interior, ~interior):
            unresolved, halves = sources.copy(), np.zeros(n, dtype=np.int64)
            while unresolved.any():
                for half in (0, 1):
                    chosen = unresolved & (halves == half)
                    if not chosen.any():
                        continue
                    count = run(chosen.astype(np.int64), propagate=propagate)
                    out = run(np.where(chosen, np.arange(1, n + 1), 0), propagate=propagate)
                    alone = np.flatnonzero(count == 1)  # slots that received one ID: it can be read
                    found = out[alone] - 1
                    # prefer interior occurrences: periodic ghost nodes hold copies of interior cells
                    order = np.argsort(~interior[alone], kind="stable")
                    alone, found = alone[order], found[order]
                    first = np.unique(found, return_index=True)[1]
                    destination[found[first]] = alone[first]
                    unresolved[found] = False
                    if count.max() <= 1:  # no two IDs met: the IDs not found left the lattice
                        unresolved[chosen] = False
                halves = split.integers(0, 2, n)  # IDs that met go on in random halves
        return destination

    boundary, propagation = lookup(False), lookup(True)
    if lgca.bc == "periodic":  # ghost slots copy interior slots
        source = run(np.where(interior, np.arange(1, n + 1), 0), propagate=False) - 1
        source[interior] = -1
    else:
        source = np.full(n, -1, dtype=np.int64)
    padded = np.flatnonzero(interior)
    index = np.full(n, -1, dtype=np.int64)
    index[padded] = np.arange(len(padded))
    return {"boundary": boundary, "propagation": propagation, "source": source,
            "interior": index, "padded": padded, "shape": shape}
