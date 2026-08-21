"""Named, pure-data initial-condition presets for ModelSpec runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


__all__ = ["apply_initializer", "list_initializers", "resolve_resource_path"]


def list_initializers() -> tuple[str, ...]:
    """Return the initializer names available to portable model files."""

    return tuple(sorted(_INITIALIZERS))


def apply_initializer(
    lgca,
    declaration: Mapping[str, Any],
    *,
    resource_base: str | Path | None = None,
    trusted_paths: bool = False,
) -> None:
    """Apply one validated initializer declaration to an LGCA instance."""

    if not isinstance(declaration, Mapping):
        raise ValueError("model.state.initializer must be a mapping")
    name = declaration.get("name")
    try:
        initializer = _INITIALIZERS[name]
    except (KeyError, TypeError) as exc:
        valid = ", ".join(list_initializers())
        raise ValueError(
            f"model.state.initializer.name must be one of: {valid}"
        ) from exc
    parameters = declaration.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("model.state.initializer.parameters must be a mapping")
    initializer(
        lgca,
        dict(parameters),
        resource_base=resource_base,
        trusted_paths=trusted_paths,
    )


def resolve_resource_path(
    path: str | Path,
    *,
    resource_base: str | Path | None,
    trusted_paths: bool = False,
) -> Path:
    """Resolve an imported resource without allowing accidental path escape."""

    raw = Path(path)
    if resource_base is None:
        raise ValueError(
            "Relative initializer resources require resource_base; pass the model "
            "directory or use trusted_paths=True (CLI: --trusted-paths) with an explicit base."
        )
    base = Path(resource_base).resolve()
    if not trusted_paths and (raw.is_absolute() or ".." in raw.parts):
        raise ValueError(
            "Initializer resource paths must be relative and may not contain '..'; "
            "use trusted_paths=True (CLI: --trusted-paths) only for trusted local models."
        )
    resolved = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    if not trusted_paths and not resolved.is_relative_to(base):
        raise ValueError(
            "Initializer resource path escapes the model directory; use "
            "trusted_paths=True (CLI: --trusted-paths) only for trusted local models."
        )
    return resolved


def _region_initializer(lgca, parameters, **_) -> None:
    allowed = {"placement", "extent", "density"}
    _reject_unknown(parameters, allowed, "region")
    placement = parameters.get("placement", "center")
    if placement not in {"center", "left", "corner"}:
        raise ValueError("region placement must be 'center', 'left', or 'corner'")
    if "extent" not in parameters:
        raise ValueError("region extent is required")
    extent = _normalize_extent(parameters["extent"], tuple(lgca.dims))
    density = parameters.get("density", 1.0)
    if (
        isinstance(density, bool)
        or np.asarray(density).ndim != 0
        or not np.isfinite(float(density))
        or float(density) < 0
    ):
        raise ValueError("region density must be a finite non-negative scalar")

    starts = [(dim - width) // 2 for dim, width in zip(lgca.dims, extent)]
    if placement in {"left", "corner"}:
        starts[0] = 0
    if placement == "corner":
        starts = [0] * len(extent)
    region = tuple(slice(start, start + width) for start, width in zip(starts, extent))
    mask = np.zeros(tuple(lgca.dims), dtype=bool)
    mask[region] = True

    lgca.random_reset(float(density))
    outside = tuple(axis[~mask] for axis in lgca.nonborder)
    if lgca.nodes.dtype == object:
        for coord in np.argwhere(~mask):
            logical_coord = tuple(coord)
            array_coord = tuple(axis[logical_coord] for axis in lgca.nonborder)
            node = lgca.nodes[array_coord]
            for channel in range(node.shape[-1]):
                node[channel] = []
    else:
        lgca.nodes[outside] = 0
    lgca.apply_boundaries()
    lgca.update_dynamic_fields()


def _from_npz_initializer(
    lgca,
    parameters,
    *,
    resource_base,
    trusted_paths,
) -> None:
    allowed = {"path", "key"}
    _reject_unknown(parameters, allowed, "from_npz")
    if "path" not in parameters:
        raise ValueError("from_npz path is required")
    if hasattr(lgca, "props"):
        raise ValueError(
            "from_npz does not yet support identity-based states because particle "
            "properties must be restored together with labels"
        )
    path = resolve_resource_path(
        parameters["path"],
        resource_base=resource_base,
        trusted_paths=trusted_paths,
    )
    key = parameters.get("key", "nodes")
    if not isinstance(key, str):
        raise ValueError("from_npz key must be a string")
    if not path.is_file():
        raise FileNotFoundError(f"Initializer NPZ file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise ValueError(f"Initializer NPZ file has no array {key!r}")
        nodes = np.asarray(archive[key])
    expected = lgca.nodes[lgca.nonborder].shape
    if nodes.shape != expected:
        raise ValueError(
            f"Initializer nodes have shape {nodes.shape}, expected {expected}"
        )
    _validate_node_values(nodes, lgca.nodes.dtype)
    lgca.nodes[lgca.nonborder] = nodes.astype(lgca.nodes.dtype, copy=False)
    lgca.apply_boundaries()
    lgca.update_dynamic_fields()


def _normalize_extent(value, dims: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError("region extent must contain positive integers")
    if np.asarray(value).ndim == 0:
        values = (value,) * len(dims)
    else:
        values = tuple(value)
    if len(values) != len(dims):
        raise ValueError(f"region extent must have {len(dims)} values")
    if any(
        isinstance(width, bool) or int(width) != width or not 0 < int(width) <= dim
        for width, dim in zip(values, dims)
    ):
        raise ValueError(f"region extent must contain positive integers within {dims}")
    return tuple(int(width) for width in values)


def _validate_node_values(nodes: np.ndarray, target_dtype) -> None:
    if np.dtype(target_dtype) == np.dtype(bool):
        if not np.all((nodes == 0) | (nodes == 1)):
            raise ValueError("Volume-exclusion NPZ nodes must contain only 0 or 1")
        return
    if not np.issubdtype(nodes.dtype, np.number):
        raise ValueError("NPZ nodes must use a numeric dtype")
    if np.any(~np.isfinite(nodes)) or np.any(nodes < 0) or np.any(nodes != np.floor(nodes)):
        raise ValueError("No-volume-exclusion NPZ nodes must be non-negative integers")


def _reject_unknown(parameters, allowed, name: str) -> None:
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise ValueError(f"{name} initializer has unknown parameters: {unknown}")


_INITIALIZERS = {
    "region": _region_initializer,
    "from_npz": _from_npz_initializer,
}
