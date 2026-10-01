"""Where a result came from: versions, code, parameters and inputs.

``build_model`` records a provenance block in ``model.metadata["provenance"]``:
the versions of Python, the package and its main dependencies, the platform,
a SHA-256 hash of the source of every rule the model runs (rules change more
often than versions), the parameters of every operator with their defaults,
and the hashes of the files the model read. ``lgca.study.sweep`` and the
``biolgca`` command add theirs to the table and to the files they write. A
person who compares two runs compares these blocks; the library itself
compares only hashes it wrote (array files of model files, the archive of a
command-line run) and warns when they differ.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["environment", "pipeline_record", "sha256"]


def sha256(data: bytes) -> str:
    """The SHA-256 hash of ``data``, as hexadecimal text."""
    return hashlib.sha256(data).hexdigest()


def array_hash(array: np.ndarray) -> str:
    """A hash of an array's values, dtype and shape (not of the file that stores it)."""
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}{list(array.shape)}".encode()
    return sha256(header + array.tobytes())


def environment() -> dict[str, Any]:
    """Versions of Python, BioLGCA and its main dependencies, and the platform."""
    import scipy

    from .model import _package_version

    versions = {"biolgca": _package_version(), "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__}
    pandas = sys.modules.get("pandas")  # not imported only for this
    if pandas is not None:
        versions["pandas"] = pandas.__version__
    return {"versions": versions, "platform": platform.platform()}


def vcs_commit() -> dict[str, Any] | None:
    """The git commit of the BioLGCA source, if it runs from a git checkout, and whether files changed."""
    package = Path(__file__).resolve().parent

    def git(*arguments):
        return subprocess.run(["git", "-C", str(package), *arguments], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        changed = bool(git("status", "--porcelain", "--untracked-files=no", "--", "."))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "changed": changed}


def pipeline_record(pipeline) -> dict[str, Any]:
    """The operators of a pipeline with their parameters (defaults included), and the source hashes of
    the rules, terms and reactions they run, by name."""
    sources: dict[str, dict[str, Any]] = {}
    operators = [_operator_record(operator, sources) for operator in pipeline.operators]
    return {"operators": operators, "rules": sources}


def _operator_record(operator, sources) -> dict[str, Any]:
    info = operator.info
    parameters = {name: spec.default for name, spec in getattr(info, "parameter_specs", {}).items()
                  if not spec.required}
    parameters.update(getattr(operator, "parameters", None) or {})
    record = {"name": operator.name, "kind": operator.operator_kind,
              "parameters": {str(name): describe(value) for name, value in parameters.items()}}
    rule = getattr(operator, "rule", None)
    _add_source(sources, operator.name, getattr(rule, "function", None) or type(operator))
    stacked = getattr(operator, "operators", None)
    if stacked:
        record["operators"] = [_operator_record(entry, sources) for entry in stacked]
    terms = getattr(operator, "terms", None)
    if terms:
        record["terms"] = [{"name": term.name, "beta": term.beta,
                            "parameters": {str(name): describe(value) for name, value in term.parameters.items()}}
                           for term in terms]
        for term in terms:
            definition = getattr(term, "definition", None)
            _add_source(sources, term.name, getattr(definition, "function", None))
    for reaction in getattr(operator, "reactions", None) or ():
        _add_source(sources, reaction.name, getattr(reaction, "function", None))
    return record


def _add_source(sources, name, code) -> None:
    if code is None or name in sources:
        return
    sources[name] = {"module": getattr(code, "__module__", None), "sha256": _source_hash(code)}


@functools.cache
def _source_hash(code) -> str | None:
    """The hash of the source of a function or class; cached, as every model of a sweep asks (a rule
    defined again is a new object)."""
    try:
        text = inspect.getsource(code)
    except (OSError, TypeError):  # e.g. defined in an interactive session
        return None
    return sha256(text.encode("utf-8"))


def describe(value) -> Any:
    """A JSON-friendly description of a parameter value: arrays by shape, dtype and hash, functions by
    name, other objects by type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 16 and value.dtype != object:
            return value.tolist()
        return {"array": list(value.shape), "dtype": str(value.dtype),
                "sha256": None if value.dtype == object else array_hash(value)}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if callable(value):
        return f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', type(value).__name__)}"
    return f"<{type(value).__module__}.{type(value).__qualname__}>"
