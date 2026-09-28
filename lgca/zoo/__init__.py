"""Model zoo: published LGCA models written with the current rules, ready to rerun and explore.

Every entry is a module with

- ``CARD``: a :class:`ZooEntry` with the reference, the question and the result reproduced;
- ``PARAMETERS``: the paper's symbols as :class:`Parameter`, with the paths of
  :func:`lgca.study.vary`, so that :func:`lgca.explore` and :func:`lgca.study.sweep` can use them;
- ``build_spec(full=False, ...)``: the model as a :class:`~lgca.model.ModelSpec`, with the paper's
  values as defaults; ``full=True`` gives the paper's lattice and run length, the default a
  smaller model that runs in about a minute;
- functions that measure what the paper measured.

Each entry has a notebook in the documentation (Model zoo) that reproduces the result::

    from lgca import zoo

    for card in zoo.catalogue():
        print(card.name, "-", card.question)

    allee = zoo.load("allee_effect")
    spec = allee.build_spec(density=0.3)
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from ._card import Parameter, ZooEntry

__all__ = ["ENTRIES", "Parameter", "ZooEntry", "catalogue", "load", "parameter_table"]

ENTRIES = ("allee_effect", "evolving_front", "phenotypic_plasticity", "clonal_go_or_grow", "evolution_modes",
           "mutational_meltdown", "excitable_media", "jamming")


def load(name: str) -> ModuleType:
    """The module of the entry ``name``, e.g. ``load("allee_effect").build_spec()``."""
    if name not in ENTRIES:
        raise KeyError(f"the zoo has no entry {name!r}; its entries are {list(ENTRIES)}")
    return import_module(f"{__name__}.{name}")


def catalogue() -> list[ZooEntry]:
    """The cards of all entries, in the order of the documentation."""
    return [load(name).CARD for name in ENTRIES]


def parameter_table(name: str):
    """The parameters of an entry as a :class:`pandas.DataFrame`: symbol, meaning, value and path."""
    import pandas as pd

    rows = [{"symbol": p.symbol, "meaning": p.meaning, "value": p.value, "path": p.path or ""}
            for p in load(name).PARAMETERS.values()]
    return pd.DataFrame(rows).set_index("symbol")
