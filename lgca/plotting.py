"""Plotting and animation observers for LGCA simulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .list_utils import _copy_arr_of_lists, get_arr_of_empty_lists
from .plot_data import _reject_sparse_implicit_history
from .simulation import Observer, Schedule

__all__ = [
    "AnimationObserver",
    "MovieObserver",
    "PlotSnapshotObserver",
    "animate",
    "plot",
]


_PLOT_METHODS = {
    "density": "plot_density",
    "config": "plot_config",
    "configuration": "plot_config",
    "flux": "plot_flux",
    "flow": "plot_flow",
    "scalarfield": "plot_scalarfield",
}

_ANIMATION_METHODS = {
    "density": ("animate_density", "density_t"),
    "config": ("animate_config", "nodes_t"),
    "configuration": ("animate_config", "nodes_t"),
    "flux": ("animate_flux", "nodes_t"),
    "flow": ("animate_flow", "nodes_t"),
}


def plot(lgca, kind: str = "density", **kwargs):
    """Create one plot for the current state using the model's renderer."""

    method = getattr(lgca, _resolve_kind(kind, _PLOT_METHODS))
    return method(**kwargs)


def animate(lgca, kind: str = "density", data=None, **kwargs):
    """Create an animation using the model's animation renderer."""

    method_name, data_argument = _resolve_animation(kind)
    method = getattr(lgca, method_name)
    if data is None:
        _reject_sparse_implicit_history(lgca, data_argument)
        return method(**kwargs)
    return method(**{data_argument: data}, **kwargs)


class PlotSnapshotObserver(Observer):
    """Create static plot snapshots during a simulation run."""

    def __init__(
        self,
        kind: str = "density",
        schedule: Schedule | None = None,
        output_dir=None,
        filename: str = "{kind}_{step:05d}.png",
        close: bool | None = None,
        retain_results: bool = False,
        **plot_kwargs,
    ):
        super().__init__(schedule=schedule)
        self.kind = kind
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.filename = filename
        self.retain_results = retain_results
        self.close = (output_dir is not None or not retain_results) if close is None else close
        self.plot_kwargs = plot_kwargs
        self.results = []
        self.paths = []

    def setup(self, lgca, runner) -> None:
        _validate_observer_backend(lgca)
        self.results = []
        self.paths = []
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, lgca, step: int) -> None:
        result = plot(lgca, kind=self.kind, **self.plot_kwargs)
        fig = _figure_from_result(result)
        if self.retain_results:
            self.results.append((step, result))

        if self.output_dir is not None:
            path = self.output_dir / self.filename.format(kind=self.kind, step=step)
            fig.savefig(path)
            self.paths.append(path)

        if self.close:
            _close_figure(fig)


class AnimationObserver(Observer):
    """Collect simulation frames and build an animation at the end."""

    def __init__(
        self,
        kind: str = "density",
        schedule: Schedule | None = None,
        save_path=None,
        save_kwargs: dict | None = None,
        close: bool = False,
        **animation_kwargs,
    ):
        super().__init__(schedule=schedule)
        self.kind = kind
        self.save_path = None if save_path is None else Path(save_path)
        self.save_kwargs = dict(save_kwargs or {})
        self.close = close
        self.animation_kwargs = animation_kwargs
        self.frames = []
        self.frame_steps = []
        self.animation = None

    def setup(self, lgca, runner) -> None:
        _validate_observer_backend(lgca)
        self.frames = []
        self.frame_steps = []
        self.animation = None

    def on_step(self, lgca, step: int) -> None:
        self.frames.append(_capture_frame(lgca, self.kind))
        self.frame_steps.append(step)

    def finalize(self, lgca, runner) -> None:
        data = _frames_to_array(self.frames)
        self.animation = animate(lgca, kind=self.kind, data=data, **self.animation_kwargs)
        self.frames = []
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.animation.save(str(self.save_path), **self.save_kwargs)
        if self.close:
            fig = getattr(self.animation, "_fig", None)
            if fig is not None:
                _close_figure(fig)


MovieObserver = AnimationObserver


def _resolve_kind(kind: str, mapping: dict[str, str]) -> str:
    key = kind.replace("_", "").lower()
    aliases = {name.replace("_", "").lower(): value for name, value in mapping.items()}
    try:
        return aliases[key]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown plot kind {kind!r}. Use one of: {valid}.") from exc


def _resolve_animation(kind: str) -> tuple[str, str]:
    key = kind.replace("_", "").lower()
    aliases = {name.replace("_", "").lower(): value for name, value in _ANIMATION_METHODS.items()}
    try:
        return aliases[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_ANIMATION_METHODS))
        raise ValueError(f"Unknown animation kind {kind!r}. Use one of: {valid}.") from exc


def _figure_from_result(result):
    try:
        return result[0]
    except (TypeError, IndexError) as exc:
        raise TypeError("Plot functions used with PlotSnapshotObserver must return a figure as their first item.") from exc


def _close_figure(fig) -> None:
    from matplotlib import pyplot as plt

    plt.close(fig)


def _capture_frame(lgca, kind: str):
    method_name, data_argument = _resolve_animation(kind)
    if data_argument == "density_t":
        if hasattr(lgca, "species_density"):
            return np.array(lgca.species_density[lgca.nonborder], copy=True)
        return np.array(lgca.cell_density[lgca.nonborder], copy=True)
    if data_argument == "nodes_t":
        nodes = lgca.nodes[lgca.nonborder]
        if lgca.nodes.dtype == object:
            return _copy_arr_of_lists(nodes)
        return np.array(nodes, copy=True)
    raise ValueError(f"Unsupported animation data source for {method_name!r}.")


def _frames_to_array(frames):
    if not frames:
        raise RuntimeError("Cannot build an animation without recorded frames.")
    first = frames[0]
    if getattr(first, "dtype", None) == object:
        arr = get_arr_of_empty_lists((len(frames),) + first.shape)
        for i, frame in enumerate(frames):
            arr[i, ...] = frame
        return arr
    return np.asarray(frames)


def _validate_observer_backend(lgca) -> None:
    if getattr(lgca, "geometry", None) in {"cubic", "moore"}:
        raise NotImplementedError(
            "Plotting observers do not yet support the 3-D Mayavi lifecycle; "
            "call the cubic plotting method directly."
        )
