"""The card of a zoo entry and the parameters of its paper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    """A parameter of a paper: its symbol, meaning and value, and where it lives in the spec.

    ``path`` is a path of :func:`lgca.study.vary`, or ``None`` for a parameter that sets the
    structure of the model (e.g. the number of channels) or its initial state in another way.
    """

    symbol: str
    meaning: str
    value: object
    path: str | None = None


@dataclass(frozen=True)
class ZooEntry:
    """The card of a published model in :mod:`lgca.zoo`.

    Attributes
    ----------
    name : str
        Module name, e.g. ``"allee_effect"``; ``lgca.zoo.load(name)`` returns the module.
    title : str
        Short title.
    question : str
        The biological question the model answers.
    authors : str
        Authors as cited, e.g. ``"Böttger K, Hatzikirou H, Voss-Böhme A, ..."``.
    paper : str
        Title of the paper; for a model without a paper, where it comes from.
    year : int or None
        Year of publication; ``None`` for a model without a paper.
    venue : str
        Journal or book, with volume and pages.
    doi : str or None
        DOI of the paper.
    reproduces : str
        The figure or result reproduced by the notebook.
    mechanisms : tuple of str
        Mechanisms of the model, e.g. ``("go-or-grow", "birth and death")``.
    lattice : str
        Geometry and model family, e.g. ``"square, classical with volume exclusion"``.
    fidelity : str
        ``"same rules"``, ``"simplified: ..."`` or ``"new model"``.
    """

    name: str
    title: str
    question: str
    authors: str
    paper: str
    year: int | None
    venue: str
    doi: str | None
    reproduces: str
    mechanisms: tuple[str, ...]
    lattice: str
    fidelity: str

    @property
    def url(self) -> str | None:
        """Link to the paper."""
        return f"https://doi.org/{self.doi}" if self.doi else None

    @property
    def citation(self) -> str:
        """The reference, e.g. for a notebook or a talk."""
        if self.year is None:
            return f"{self.authors}. {self.paper}. {self.venue}."
        doi = f" https://doi.org/{self.doi}" if self.doi else ""
        return f"{self.authors} ({self.year}). {self.paper}. {self.venue}.{doi}"
