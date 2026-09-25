"""Plotting and animation observers for LGCA simulations."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .list_utils import _copy_arr_of_lists, get_arr_of_empty_lists
from .plot_data import resolve_animation_history
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
    "density_cubes": "plot_density_cubes",
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
    "scalarfield": ("animate_scalarfield", "field_t"),
}


def plot(lgca, kind: str = "density", **kwargs):
    """Create one plot for the current state using the model's renderer."""

    method = _model_method(lgca, kind, _resolve_kind(kind, _PLOT_METHODS))
    return method(**kwargs)


def animate(lgca, kind: str = "density", data=None, steps=None, **kwargs):
    """Animate frames with paired simulation ``steps`` (dense if unspecified)."""

    method_name, data_argument = _resolve_animation(kind)
    method = _model_method(lgca, kind, method_name)
    channels = kwargs.pop("channels", slice(None)) if data_argument == "density_t" else slice(None)
    data, times = resolve_animation_history(lgca, data_argument, data, steps, channels)
    return method(**{data_argument: data}, steps=times, **kwargs)


class PlotSnapshotObserver(Observer):
    """Create static plot snapshots during a simulation run.

    Snapshots of 3D (cubic and Moore) models are rendered with Mayavi. When they
    are closed after saving (the default with `output_dir`), they are rendered
    offscreen without opening windows.
    """

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
        _model_method(lgca, self.kind, _resolve_kind(self.kind, _PLOT_METHODS))
        self.results = []
        self.paths = []
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, lgca, step: int) -> None:
        with _render_context(lgca, offscreen=self.close):
            result = plot(lgca, kind=self.kind, **self.plot_kwargs)
            fig = _figure_from_result(result)
            if self.retain_results:
                self.results.append((step, result))

            if self.output_dir is not None:
                path = self.output_dir / self.filename.format(kind=self.kind, step=step)
                _save_figure(fig, path)
                self.paths.append(path)

            if self.close:
                _close_figure(fig)


class AnimationObserver(Observer):
    """Collect simulation frames and build an animation at the end.

    For 2D models, :attr:`animation` is a Matplotlib animation. For 3D (cubic
    and Moore) models, the frames are rendered with Mayavi without blocking the
    run: with `save_path`, every frame is rendered offscreen, the movie is
    written and :attr:`animation` is its path; otherwise :attr:`animation` is a
    Mayavi ``Animator`` that plays after :func:`mayavi.mlab.show` is called.
    """

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
        _model_method(lgca, self.kind, _resolve_animation(self.kind)[0])
        if self.schedule.steps is not None and not any(step <= runner.timesteps for step in self.schedule.steps):
            raise ValueError("Animation schedule selects no frames in this run; "
                             f"include a local step between 0 and {runner.timesteps}")
        self.frames = []
        self.frame_steps = []
        self.animation = None

    def on_step(self, lgca, step: int) -> None:
        self.frames.append(_capture_frame(lgca, self.kind, self.animation_kwargs.get("channels", slice(None))))
        self.frame_steps.append(step)

    def finalize(self, lgca, runner) -> None:
        data = _frames_to_array(self.frames)
        kwargs = dict(self.animation_kwargs)
        if _resolve_animation(self.kind)[1] == "density_t":
            kwargs.pop("channels", None)  # Frame capture already selected the channels.
        if _uses_mayavi(lgca):
            self.frames = []
            kwargs.setdefault("show", False)
            if self.save_path is not None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                kwargs.update(save_path=self.save_path, save_kwargs=self.save_kwargs)
            self.animation = animate(lgca, kind=self.kind, data=data, steps=self.frame_steps, **kwargs)
            return
        self.animation = animate(lgca, kind=self.kind, data=data, steps=self.frame_steps,
                                 **kwargs)
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


def _model_method(lgca, kind: str, method_name: str):
    method = getattr(lgca, method_name, None)
    if method is None:
        raise ValueError(f"{type(lgca).__name__} does not provide {kind!r} plots (no method {method_name}).")
    return method


def _uses_mayavi(lgca) -> bool:
    return getattr(lgca, "geometry", None) in {"cubic", "moore"}


def _render_context(lgca, offscreen: bool):
    if not _uses_mayavi(lgca):
        return nullcontext()
    from .mayavi_style import offscreen as mayavi_offscreen

    return mayavi_offscreen(offscreen)


def _is_matplotlib_figure(fig) -> bool:
    from matplotlib.figure import Figure

    return isinstance(fig, Figure)


def _save_figure(fig, path) -> None:
    if _is_matplotlib_figure(fig):
        fig.savefig(path)
    else:
        from .mayavi_style import save_figure

        save_figure(fig, path)


def _close_figure(fig) -> None:
    if _is_matplotlib_figure(fig):
        from matplotlib import pyplot as plt

        plt.close(fig)
    else:
        from .mayavi_style import mlab

        mlab.close(fig)


def _capture_frame(lgca, kind: str, channels=slice(None)):
    method_name, data_argument = _resolve_animation(kind)
    if data_argument == "density_t":
        if channels != slice(None):
            nodes = lgca.nodes[lgca.nonborder][..., channels]
            if nodes.dtype == object:
                nodes = lgca.length_checker(nodes)
            elif hasattr(lgca, "occupied"):
                nodes = nodes > 0
            return np.array(nodes.sum(-1), copy=True)
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
