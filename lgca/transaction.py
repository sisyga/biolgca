"""A step is applied as a whole or not at all.

:class:`StepTransaction` takes a cheap checkpoint of everything a step may
change before the step, and restores it in place if the step raises
(``KeyboardInterrupt`` included), so that a model whose step failed is in the
state before the step and can step again:

- the attributes of the LGCA object (references: covers every attribute that
  a step replaces, such as ``cell_density`` or the cell table of
  identity-based models without volume exclusion, and counters such as
  ``maxlabel``);
- ``nodes``, which operators write in place (one copy, in a buffer reused
  from step to step), or the lists of labels of identity-based models without
  volume exclusion whose boundary has no cell table;
- the declared fields, which operators may write in place;
- the traits: each :class:`~lgca.cells.TraitArray` keeps its buffer and
  length, and records the old values of what ``trait[index] = values``
  overwrites; trait lists of legacy models keep their length;
- the family tree (``family_props``) and the random generator.

Not restored: writes through ``TraitArray.values``, element writes to legacy
trait or family lists, operator-internal state (e.g. statistics and multigrid
hierarchies of field solvers) and Python objects that rules keep themselves.
Rules keep their state on the lattice.
"""

from __future__ import annotations

import weakref

import numpy as np

from .cells import TraitArray

__all__ = ["StepTransaction"]

# transactions of steps outside a compiled model (get_lgca with a function as interaction), kept for their
# buffers; not on the LGCA object, so that copies of it do not copy them
_TRANSACTIONS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class StepTransaction:
    """Checkpoint of an LGCA object before a step; :meth:`rollback` restores it in place.

    Parameters
    ----------
    lgca : LGCA object
        The model whose steps are made atomic.
    fields : iterable of str
        Names of the fields that the step may write in place.

    Examples
    --------
    >>> transaction = StepTransaction(lgca, fields=("u",))        # doctest: +SKIP
    >>> transaction.begin()                                       # doctest: +SKIP
    >>> try:                                                      # doctest: +SKIP
    ...     step(lgca)
    ... except BaseException:
    ...     transaction.rollback()
    ...     raise
    ... else:
    ...     transaction.commit()
    """

    def __init__(self, lgca, fields=()):
        self.lgca = lgca
        self.fields = tuple(fields)
        self._buffers: dict[str, np.ndarray] = {}  # reused from step to step: no allocations
        self._active = False

    @classmethod
    def of(cls, lgca, fields=()) -> StepTransaction:
        """The transaction kept for ``lgca`` (created on first use)."""
        transaction = _TRANSACTIONS.get(lgca)
        if transaction is None or transaction.fields != tuple(fields):
            transaction = _TRANSACTIONS[lgca] = cls(lgca, fields)
        return transaction

    # ---------------------------------------------------------------- begin

    def begin(self) -> None:
        """Take the checkpoint; the step starts after this."""
        from .nove_ib_base import NoVE_IBLGCA_base

        lgca = self.lgca
        self._nodes = self._lists = None
        if isinstance(lgca, NoVE_IBLGCA_base):
            if lgca._cell_table() is None:  # e.g. inflow boundaries: the lists of labels are the state
                nodes = lgca.__dict__["_nodes_array"]
                flat = nodes.reshape(-1)
                lengths = np.fromiter(map(len, flat), dtype=np.int64, count=flat.size)
                labels = np.fromiter((label for channel in flat for label in channel), dtype=np.int64,
                                     count=int(lengths.sum()))
                self._lists = (nodes, labels, lengths)
        else:
            self._nodes = (lgca.nodes, self._copy("nodes", lgca.nodes))
        self._attributes = dict(lgca.__dict__)  # after _cell_table(), which may make the table the state
        self._random = lgca.rng.bit_generator.state
        self._fields = []
        for name in self.fields:
            values = getattr(lgca, name, None)
            if isinstance(values, np.ndarray):
                self._fields.append((values, self._copy("field " + name, values)))
        self._begin_traits(lgca)
        self._active = True

    def _begin_traits(self, lgca):
        self._props = self._family = None
        self._traits = []
        self._journals = []  # (trait, where this transaction's records start, whether it began the journal)
        props = getattr(lgca, "props", None)
        if props is None:
            return
        self._props = (props, dict(props))
        for values in props.values():
            if isinstance(values, TraitArray):
                self._traits.append((values, values._data, values._size))
                journal = getattr(values, "_journal", None)  # not None: a transaction around this one
                began = journal is None
                if began:
                    journal = values._journal = []
                self._journals.append((values, len(journal), began))
            elif isinstance(values, list):
                self._traits.append((values, None, len(values)))
            elif isinstance(values, np.ndarray):
                self._traits.append((values, values.copy(), None))
        family = getattr(lgca, "family_props", None)
        if family is not None:
            lengths = {key: len(value) for key, value in family.items() if isinstance(value, (list, TraitArray))}
            self._family = (family, dict(family), lengths)

    def _copy(self, key, array):
        buffer = self._buffers.get(key)
        if buffer is None or buffer.shape != array.shape or buffer.dtype != array.dtype:
            buffer = self._buffers[key] = np.empty_like(array)
        np.copyto(buffer, array)
        return buffer

    # ------------------------------------------------------------------ end

    def commit(self) -> None:
        """The step succeeded: forget the checkpoint (the buffers stay for the next step)."""
        self._end()

    def rollback(self) -> None:
        """Restore the state of :meth:`begin`, in place: arrays keep their identity."""
        if not self._active:
            return
        lgca = self.lgca
        self._restore_traits()
        lgca.__dict__.clear()  # attributes first: arrays written in place are then the ones of the checkpoint
        lgca.__dict__.update(self._attributes)
        if self._nodes is not None:
            nodes, saved = self._nodes
            np.copyto(nodes, saved)
        elif self._lists is not None:
            nodes, labels, lengths = self._lists
            flat = nodes.reshape(-1)
            for index, part in enumerate(np.split(labels, np.cumsum(lengths)[:-1])):
                flat[index] = part.tolist()
        elif lgca.__dict__.get("_store") is not None:  # the table is the state: lists read in the step are stale
            lgca.__dict__["_nodes_array"] = None
        for values, saved in self._fields:
            np.copyto(values, saved)
        lgca.rng.bit_generator.state = self._random
        # derived arrays (cell_density and the like) are replaced in a step, never written in place, so the
        # restored attributes are the values before the step
        self._end()

    def _restore_traits(self):
        if self._props is None:
            return
        for trait, start, _ in self._journals:
            journal = trait._journal
            for buffer, index, old in reversed(journal[start:]):  # newest first
                buffer[index] = old
            del journal[start:]
        for trait, data, size in self._traits:
            if isinstance(trait, TraitArray):
                trait._data, trait._size = data, size
            elif isinstance(trait, list):
                del trait[size:]
            else:
                trait[...] = data
        props, saved = self._props
        props.clear()
        props.update(saved)
        if self._family is not None:
            family, saved, lengths = self._family
            ancestor, descendants = family.get("ancestor"), family.get("descendants")
            if ancestor is not None and descendants is not None:
                # families founded in the step: newest first, off the list of their parent's descendants
                for new in range(len(ancestor) - 1, lengths.get("ancestor", 0) - 1, -1):
                    parent = int(ancestor[new])
                    if parent < lengths.get("descendants", 0) and descendants[parent] \
                            and descendants[parent][-1] == new:
                        descendants[parent].pop()
            family.clear()
            family.update(saved)
            for key, length in lengths.items():
                values = family[key]
                if isinstance(values, TraitArray):
                    values._size = length
                else:
                    del values[length:]

    def _end(self):
        for trait, start, began in self._journals if self._active else ():
            if began:
                trait._journal = None
        self._active = False
