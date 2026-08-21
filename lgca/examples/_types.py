"""Shared types for the curated example gallery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleInfo:
    """Beginner-facing metadata for a curated example."""

    name: str
    title: str
    category: str
    question: str
    concepts: tuple[str, ...]
    source_path: str = ""
    source: str = ""
    portable: bool = True
