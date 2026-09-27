"""Explore a model live in a notebook: sliders for its parameters and a view that updates as it runs.

:func:`explore` shows a model of a :class:`~lgca.model.ModelSpec` running in a Jupyter notebook
(JupyterLab, Colab, VS Code), with play/pause, a step button, the number of steps per frame and a
choice of view: the density, the flux on 2D lattices, or a field. A panel beside it plots the
population, or other measures, over time::

    import lgca

    explorer = lgca.explore(spec, {"beta": (0, 10), "density": (0.05, 1)})
    explorer  # the last line of a cell shows the controls

Sliders are named like the paths of :func:`lgca.study.vary`: full paths, or short names such as
``"beta"`` when only one place of the model has that name. A change of a parameter of the dynamics
applies to the running lattice, which keeps its cells, so you can watch the transition from one
regime to another. A change of the lattice, the state or the time (``density``, ``dims``,
``restchannels``, ``seed``) builds the model again from a new initial state. ``explorer.spec`` is
the model with the current values, ready for :func:`~lgca.model.run_model` or
:func:`~lgca.study.sweep`.
"""

from __future__ import annotations

import io
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import numpy as np

from ._warnings import warn_user

__all__ = ["Explorer", "explore"]

_PLAYING: weakref.WeakSet = weakref.WeakSet()  # explorers whose model is running


def explore(spec, controls: Mapping[str, Any] | None = None, *, view: str = "density",
            measure: str | Sequence[str] | Mapping[str, Any] | None = "population", steps_per_frame: int = 1,
            interval: float = 0.1, window: int = 100, figsize: tuple[float, float] | None = None) -> Explorer:
    """Run a model live in a notebook, with sliders for some of its parameters.

    Parameters
    ----------
    spec : ModelSpec
        The model. Its observers and ``time.steps`` are not used: the model runs until paused.
    controls : mapping, optional
        Path or short name (as in :func:`lgca.study.vary`) -> control:

        - ``(min, max)`` or ``(min, max, step)``: a slider, of integers if all three are integers;
        - a list of values: a dropdown, e.g. ``{"geometry": ["square", "hex"]}``;
        - an ipywidgets widget with a ``value``, e.g. ``ipywidgets.FloatLogSlider(min=-3, max=0)``.

        Each control starts at the model's value (the parameter's default if the model leaves it out).
        Parameters of the dynamics change the running model; other values build it again.
    view : str, default="density"
        What the lattice panel shows first: ``"density"``, ``"density: species i"`` (several
        species), ``"flux"`` (square and hexagonal lattices), the name of a field, or, in
        identity-based models, ``"mean <trait>"``, the mean trait of the cells at each node. A
        dropdown changes it. 1D lattices show the last ``window`` steps as a kymograph, time running down.
    measure : str, list of str, mapping or None, default="population"
        Quantities plotted over time beside the lattice, evaluated after every step:
        ``"population"``, the name of a field (its mean), a trait of an identity-based model (its
        mean over the cells), or a mapping of labels to functions of the LGCA object that return a
        number. ``None`` shows the lattice only.
    steps_per_frame : int, default=1
        Model steps between two frames; a slider changes it.
    interval : float, default=0.1
        Shortest time between two frames, in seconds. Drawing a frame takes about 0.06-0.1 s.
    window : int, default=100
        Steps shown by the kymograph of a 1D lattice.
    figsize : (float, float), optional
        Size of the figure in inches.

    Returns
    -------
    Explorer
        Shows its controls when it is the last line of a notebook cell (or with ``display``).

    Examples
    --------
    >>> from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec
    >>> from lgca.pipeline import InteractionPipelineSpec
    >>> spec = ModelSpec(space=SpaceSpec(geometry="hex", dims=(30, 30)),
    ...                  state=StateSpec(density=0.2, restchannels=1), time=TimeSpec(seed=1),
    ...                  dynamics=InteractionPipelineSpec(operators=[
    ...                      {"name": "polar_alignment", "parameters": {"beta": 2.0}}]))
    >>> explorer = explore(spec, {"beta": (0.0, 5.0), "density": (0.05, 1.0)})
    >>> explorer.set(beta=4.0)  # as if the slider moved: the model keeps its cells
    >>> explorer.advance(10)
    >>> explorer.step, explorer.spec.dynamics.operators[0]["parameters"]["beta"]
    (10, 4.0)
    >>> explorer.close()
    """
    return Explorer(spec, controls, view=view, measure=measure, steps_per_frame=steps_per_frame,
                    interval=interval, window=window, figsize=figsize)


@dataclass
class _Control:
    name: str
    path: str
    widget: Any
    live: bool  # a parameter of the dynamics, applied to the running model


class Explorer:
    """A model running live in a notebook; made by :func:`explore`.

    Attributes
    ----------
    spec : ModelSpec
        The model with the current values of the controls.
    model : CompiledModel
        The running model; ``explorer.model.lgca`` is its LGCA object.
    step : int
        Steps since the model was last built.
    widget : ipywidgets.VBox
        The controls and the view.
    """

    def __init__(self, spec, controls=None, *, view="density", measure="population", steps_per_frame=1,
                 interval=0.1, window=100, figsize=None):
        import ipywidgets as widgets

        from .model import ModelSpec

        if not isinstance(spec, ModelSpec):
            raise TypeError(f"explore takes a ModelSpec, got {type(spec).__name__}; models of get_lgca are "
                            "explored by writing them as a ModelSpec (tutorial 3)")
        if int(steps_per_frame) < 1:
            raise ValueError(f"steps_per_frame must be a positive integer, got {steps_per_frame!r}")
        if not interval >= 0:
            raise ValueError(f"interval must be a non-negative number of seconds, got {interval!r}")
        if int(window) < 2:
            raise ValueError(f"window must be at least 2 steps, got {window!r}")
        self.interval = float(interval)
        self.window = int(window)
        self.figsize = figsize
        self._measures = _measures(measure, spec)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._quiet = False  # set while the explorer itself moves a widget

        from .study import _column_names

        self._controls = [_control(spec, name, value) for name, value in (controls or {}).items()]
        paths = [control.path for control in self._controls]
        if len(set(paths)) < len(paths):
            raise ValueError(f"several controls set the same place of the model: {paths}")
        labels = _column_names(paths)
        for control in self._controls:
            control.widget.description = labels[control.path]  # e.g. "chemotaxis.beta"
        values = {control.path: control.widget.value for control in self._controls}
        for control in self._controls:
            start = _value_at(spec, control.path)
            if isinstance(start, Real) and not isinstance(start, bool) and start != control.widget.value:
                warn_user(f"{control.name} = {start!r} lies outside its control; the explorer starts at "
                          f"{control.widget.value!r}")
        from .study import vary

        self.spec = vary(spec, values)

        self._play = widgets.ToggleButton(value=False, description="Play", icon="play", tooltip="Run the model")
        self._next = widgets.Button(description="Step", icon="step-forward", tooltip="One frame")
        self._reset = widgets.Button(description="Reset", icon="undo", tooltip="Build the model again")
        self._speed = widgets.IntSlider(value=int(steps_per_frame), min=1, max=max(50, int(steps_per_frame)),
                                        description="steps/frame", continuous_update=False)
        self._label = widgets.Label()
        self._status = widgets.HTML()
        self._image = widgets.Image(format="png")
        self._view = widgets.Dropdown(description="view")
        self._build()  # also fills the options of the view
        if view not in self._view.options:
            raise ValueError(f"view {view!r} is not available for this model; choose one of "
                             f"{list(self._view.options)}")
        with self._lock:
            self._set_view(view)
        self._view.value = view

        self._play.observe(self._toggled, "value")
        self._next.on_click(lambda _: self._guarded(self.advance))
        self._reset.on_click(lambda _: self._guarded(self.reset))
        self._view.observe(lambda change: self._guarded(self._set_view, change["new"], draw=True), "value")
        for control in self._controls:
            control.widget.observe(self._changed(control), "value")
        buttons = widgets.HBox([self._play, self._next, self._reset, self._speed, self._label])
        sliders = [control.widget for control in self._controls]
        self.widget = widgets.VBox([buttons, widgets.HBox([self._view]), *sliders, self._image, self._status])
        self._draw()

    # ---------------------------------------------------------------- public

    @property
    def lgca(self):
        """The LGCA object of the running model."""
        return self.model.lgca

    def advance(self, steps: int | None = None) -> None:
        """Run ``steps`` model steps (default: the steps per frame) and draw the new frame."""
        steps = self._speed.value if steps is None else int(steps)
        with self._lock:
            for _ in range(steps):
                self.model.step()
                self.step += 1
                self._measure()
            self._update()
        self._draw()

    def set(self, **values) -> None:
        """Set controls by their names or labels, as if their sliders moved, e.g. ``explorer.set(beta=4.0)``.

        Names that are not Python identifiers go in a mapping: ``explorer.set(**{"chemotaxis.beta": 4.0})``.
        """
        by_name = {control.name: control for control in self._controls}
        by_name.update({control.widget.description: control for control in self._controls})
        unknown = [name for name in values if name not in by_name]
        if unknown:
            raise KeyError(f"no control named {unknown}; the controls are {list(by_name)}")
        for name, value in values.items():
            self._error = None
            by_name[name].widget.value = value
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            if by_name[name].widget.value != value:
                raise ValueError(f"{name} = {value!r} lies outside its control")

    def play(self) -> None:
        """Run frames until :meth:`pause`; other explorers pause."""
        self._play.value = True

    def pause(self) -> None:
        """Stop after the current frame."""
        self._play.value = False

    def reset(self) -> None:
        """Build the model again from its initial state, with the current values."""
        with self._lock:
            self._build()
        self._draw()

    def frame(self) -> bytes:
        """The current frame as PNG."""
        with self._lock:
            return self._render()

    def close(self) -> None:
        """Stop the model and close the widgets."""
        self.pause()
        if self._thread is not None:
            self._thread.join()
        import matplotlib.pyplot as plt

        plt.close(self._figure)
        self.widget.close()

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)

    def __repr__(self):
        values = ", ".join(f"{control.widget.description}={control.widget.value!r}" for control in self._controls)
        return f"<Explorer at step {self.step}{': ' + values if values else ''}>"

    # ---------------------------------------------------------------- model

    def _build(self, model=None):
        from .model import build_model

        self.model = model if model is not None else build_model(self.spec)
        self.step = 0
        self._series = {label: [] for label in self._measures}
        self._scales = {}
        options = _views(self.model.lgca, self.spec)
        self._histories = ({view: np.full((self.window, self.model.lgca.dims[0]), np.nan) for view in options}
                           if len(self.model.lgca.dims) == 1 else {})
        self._steps = []
        self._changes = []
        self._error = None
        current = self._view.value if self._view.value in options else options[0]
        self._quiet, self._view.options, self._view.value, self._quiet = True, options, current, False
        self._measure()
        if hasattr(self, "_figure"):  # not while the explorer is made: the view is set afterwards
            self._set_view(current)

    def _apply(self, control, value):
        from .model import _normalize_and_validate_spec, build_model
        from .study import vary

        spec = vary(self.spec, {control.path: value})
        with self._lock:
            if not control.live:
                model = build_model(spec)
                _trial(model, model.spec)
                self.spec = spec
                self._build(model)
                self._draw_locked()
                return
            compiled = self.model
            spec = _normalize_and_validate_spec(spec)
            running = replace(spec, time=replace(spec.time, seed=compiled.spec.time.seed))
            _trial(compiled, running)
            context = compiled.context
            context.spec = running
            compiled.pipeline, compiled.spec, self.spec = _compile(context), running, spec
            compiled.lgca.enable_propagation = spec.dynamics.propagation not in (False, None, "none", "disabled")
            self._changes.append(self.step)
            self._update()
            self._draw_locked()

    def _measure(self):
        lgca = self.model.lgca
        self._steps.append(self.step)
        for label, function in self._measures.items():
            self._series[label].append(float(function(lgca)))
        for view, history in self._histories.items():  # 1D: every view, so that switching keeps them
            history[:] = np.roll(history, -1, axis=0)
            history[-1] = _values(lgca, view)

    # ---------------------------------------------------------------- widgets

    def _changed(self, control):
        def callback(change):
            if self._quiet:
                return
            try:
                self._apply(control, change["new"])
                self._status.value = ""
            except Exception as exc:  # noqa: BLE001 - any error of the model: show it and put the control back
                self._error = exc
                self._status.value = _message(exc)
                self._quiet = True
                try:
                    control.widget.value = change["old"]
                finally:
                    self._quiet = False
        return callback

    def _guarded(self, function, *args, **kwargs):
        if self._quiet:
            return
        try:
            function(*args, **kwargs)
            self._status.value = ""
        except Exception as exc:  # noqa: BLE001 - shown in the widget, not lost in the kernel's log
            self._status.value = _message(exc)
            self.pause()

    def _toggled(self, change):
        self._play.description, self._play.icon = ("Pause", "pause") if change["new"] else ("Play", "play")
        if not change["new"]:
            self._stop.set()
            _PLAYING.discard(self)
            return
        for other in list(_PLAYING):  # one model runs at a time, also after a cell ran again
            if other is not self:
                other.pause()
        _PLAYING.add(self)
        if self._thread is not None:
            self._thread.join()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            start = time.perf_counter()
            try:
                self.advance()
            except Exception as exc:  # noqa: BLE001 - shown in the widget; the thread ends
                self._status.value = _message(exc)
                self._stop.set()
                self._play.value = False
                return
            self._stop.wait(max(self.interval - (time.perf_counter() - start), 0.0))

    # ---------------------------------------------------------------- drawing

    def _set_view(self, view, draw=False):
        import matplotlib.pyplot as plt

        with self._lock:
            if hasattr(self, "_figure"):
                plt.close(self._figure)
            lgca = self.model.lgca
            self._figure = figure = plt.figure(figsize=self.figsize or _figsize(lgca, bool(self._measures)))
            plt.close(figure)  # drawn into the widget, not shown by the notebook
            grid = figure.add_gridspec(1, 2 if self._measures else 1, width_ratios=[1.3, 1][:1 + bool(self._measures)],
                                       wspace=0.45)
            axes = figure.add_subplot(grid[0])
            self._panel = (_Kymograph if len(lgca.dims) == 1 else _Lattice)(lgca, view, axes, self)
            self._series_axes = figure.add_subplot(grid[1]) if self._measures else None
            self._lines = {}
            if self._series_axes is not None:
                for label in self._measures:
                    (self._lines[label],) = self._series_axes.plot([], [], label=label)
                self._series_axes.set_xlabel("Time step $k$")
                if len(self._measures) > 1:
                    self._series_axes.legend(loc="upper left", frameon=False)
                else:
                    self._series_axes.set_ylabel(next(iter(self._measures)))
            self._update()
        if draw:
            self._draw()

    def _update(self):
        if self._panel.update(self) is False:
            self._set_view(self._panel.view)
            return
        self._label.value = f"step {self.step}"
        axes = self._series_axes
        if axes is None:
            return
        for label, line in self._lines.items():
            line.set_data(self._steps, self._series[label])
        for line in list(axes.lines)[len(self._lines):]:
            line.remove()
        for step in self._changes:
            axes.axvline(step, color="0.6", linestyle=":", linewidth=1)
        axes.relim()
        axes.autoscale_view()
        axes.set_xlim(0, max(self.step, 1))

    def _render(self):
        buffer = io.BytesIO()
        self._figure.savefig(buffer, format="png", dpi=80)
        return buffer.getvalue()

    def _draw(self):
        with self._lock:
            self._draw_locked()

    def _draw_locked(self):
        self._image.value = self._render()


# -------------------------------------------------------------------- panels

class _Lattice:
    """The density, flux, a field or a mean trait on a square or hexagonal lattice."""

    def __init__(self, lgca, view, axes, explorer):
        self.view = view
        self.scales = explorer._scales  # tops of the density scales, kept when the figure is drawn again
        kind, index = _kind(view, lgca)
        if kind == "density":
            density = _density(lgca, index)
            self.vmax = self.scales[view] = max(self.scales.get(view) or _density_scale(lgca, index),
                                                int(np.ceil(density.max())))
            label = "Cells $n$" if index is None else f"Cells of species {index}"
            _, self.artist, self.mappable = lgca.plot_density(density=density, ax=axes, vmax=self.vmax,
                                                              tight_layout=False, cbarlabel=label)
        elif kind == "flux":
            _, self.artist, self.mappable = lgca.plot_flux(ax=axes, tight_layout=False)
        else:  # a field, or the mean trait of the cells at each node (clear where there are none)
            values = _scalar(lgca, kind, index)
            label, cmap = (index, "cividis") if kind == "field" else (f"mean {index}", "viridis")
            _, self.artist, self.mappable = lgca.plot_scalarfield(values, ax=axes, tight_layout=False, cmap=cmap,
                                                                  cbarlabel=label, mask=np.isnan(values),
                                                                  **_limits(values))

    def update(self, explorer):
        """Show the current state; False if the figure must be drawn again."""
        lgca = explorer.model.lgca
        kind, index = _kind(self.view, lgca)
        if kind == "density":
            values = _density(lgca, index)
            if values.max() > self.vmax:  # more cells than the colour scale has colours
                self.scales[self.view] = max(2 * self.vmax, int(np.ceil(values.max())))
                return False
            self._show(values)
        elif kind == "flux":
            self.artist.set(facecolor=_flux_colours(lgca, self.mappable))
        else:
            values = _scalar(lgca, kind, index)
            self.mappable.set_clim(**_limits(values))
            self._show(values)

    def _show(self, values):
        if hasattr(self.artist, "set_data"):
            self.artist.set_data(values.T)
        else:
            self.artist.set(facecolor=self.mappable.to_rgba(values.ravel()))


class _Kymograph:
    """The last steps of a 1D lattice, time running down."""

    def __init__(self, lgca, view, axes, explorer):
        from .plots import colorbar_axes

        self.view = view
        self.axes = axes
        self.kind, index = _kind(view, lgca)
        self.history = explorer._histories[view]
        cmap = {"density": "hot_r", "flux": "RdBu_r", "trait": "viridis"}.get(self.kind, "cividis")
        self.image = axes.imshow(self.history, aspect="auto", interpolation="nearest", cmap=cmap)
        label = {"density": "Cells $n$" if index is None else f"Cells of species {index}",
                 "flux": "Flux $j_x$", "trait": f"mean {index}"}.get(self.kind, index)
        colorbar = axes.figure.colorbar(self.image, cax=colorbar_axes(axes))
        colorbar.set_label(label)
        axes.set_xlabel("$x$")
        axes.set_ylabel("Time step $k$")
        self.scale = _density_scale(lgca, index) if self.kind == "density" else None

    def update(self, explorer):
        rows, size = self.history.shape
        last = explorer.step
        top = last - rows + 1
        self.image.set_data(self.history)
        self.image.set_extent((-0.5, size - 0.5, last + 0.5, top - 0.5))
        self.axes.set_ylim(last + 0.5, max(top, 0) - 0.5)
        seen = self.history[max(rows - 1 - last, 0):]
        if self.kind == "density":
            self.scale = max(self.scale, float(np.nanmax(seen)))
            self.image.set_clim(0, self.scale)
        elif self.kind == "flux":
            largest = float(np.nanmax(np.abs(seen))) or 1.0
            self.image.set_clim(-largest, largest)
        else:
            self.image.set_clim(**_limits(seen))


# -------------------------------------------------------------------- helpers

def _control(spec, name, control):
    import ipywidgets as widgets
    from traitlets import TraitError

    from .study import resolve_path

    path = resolve_path(spec, name)
    value = _value_at(spec, path)
    live = path.startswith("dynamics.")
    if isinstance(control, widgets.ValueWidget):
        widget = control
        if value is not None:
            try:
                widget.value = value
            except (TraitError, TypeError, ValueError):
                warn_user(f"the widget of {name!r} cannot show the model's value {value!r}; the explorer starts "
                          f"at the widget's value {widget.value!r}")
    elif isinstance(control, list):
        if not control:
            raise ValueError(f"the control of {name!r} lists no values")
        # (label, value) pairs: a tuple such as dims would otherwise be read as one
        widget = widgets.Dropdown(options=[(str(option), option) for option in control],
                                  value=value if value in control else control[0])
    elif isinstance(control, tuple) and len(control) in (2, 3) and all(
            isinstance(bound, Real) and not isinstance(bound, bool) for bound in control):
        low, high, *step = control
        if not low < high:
            raise ValueError(f"the range of {name!r} must have min < max, got {control!r}")
        if all(isinstance(bound, (int, np.integer)) for bound in control):
            widget = widgets.IntSlider(min=low, max=high, step=step[0] if step else 1)
        else:
            widget = widgets.FloatSlider(min=low, max=high, step=step[0] if step else (high - low) / 100,
                                         readout_format=".3g")
        if isinstance(value, Real) and not isinstance(value, bool):
            widget.value = value
    else:
        raise TypeError(f"the control of {name!r} must be (min, max), (min, max, step), a list of values or an "
                        f"ipywidgets widget, got {control!r}")
    if widget.has_trait("continuous_update"):
        widget.continuous_update = live  # dragging a slider would build the model at every position
    if widget.has_trait("style") and hasattr(widget.style, "description_width"):
        widget.style.description_width = "initial"
    return _Control(name=name, path=path, widget=widget, live=live)


def _compile(context):
    """The pipeline of ``context.spec`` for the running lattice; a steady field is solved again."""
    from .pipeline import compile_pipeline

    pipeline = compile_pipeline(context.spec.dynamics, context)
    for operator in pipeline.operators:
        attach = getattr(operator, "attach_field", None)
        if attach is not None:
            attach(context.lgca)
    return pipeline


def _trial(compiled, spec):
    """Run one step of a copy of the model with ``spec``, so that errors show before a change is made.

    Some values are checked only when the model steps, e.g. a rate that must be a probability.
    """
    from .model import ModelContext

    try:  # the copy leaves out the running model, whose solvers may not be copied
        lattice = deepcopy(compiled.lgca, {id(compiled): None})
        metadata = deepcopy(compiled.metadata)
    except TypeError:
        return
    context = ModelContext(lgca=lattice, spec=spec, fields=dict(compiled.context.fields), metadata=metadata)
    _compile(context).execute_step(context, compiled._step + 1)


def _value_at(spec, path):
    from .study import _get, _tokens

    return _get(spec, _tokens(path))


def _measures(measure, spec) -> dict[str, Callable]:
    from .simulation import _total_population

    if measure is None:
        return {}
    if isinstance(measure, str):
        measure = [measure]
    if isinstance(measure, Mapping):
        for label, function in measure.items():
            if not callable(function):
                raise TypeError(f"measure {label!r} must be a function of the LGCA object, got {function!r}")
        return dict(measure)
    fields = spec.state.fields or {}
    functions = {}
    for name in measure:
        if name == "population":
            functions[name] = _total_population
        elif name in fields:
            functions[f"mean {name}"] = lambda lgca, name=name: float(np.mean(_field(lgca, name)))
        elif spec.state.identity_based:  # a trait; an unknown one fails when the model is built
            functions[f"mean {name}"] = lambda lgca, name=name: _cells_mean(lgca, name)
        else:
            raise ValueError(f"unknown measure {name!r}; give 'population', a field "
                             f"({', '.join(map(repr, fields)) or 'the model has none'}), a trait of an "
                             "identity-based model, or a mapping of labels to functions of the LGCA object")
    return functions


def _cells_mean(lgca, name):
    """Mean trait of all cells; NaN without cells."""
    from .lattice_state import LatticeState

    cells = LatticeState(lgca).cells
    if name not in cells.traits:
        raise KeyError(f"the cells have no trait {name!r}; their traits are {list(cells.traits)}")
    values = np.asarray(cells[name], dtype=float)
    return float(values.mean()) if len(values) else float("nan")


def _views(lgca, spec):
    if len(lgca.dims) > 2:
        raise ValueError(f"explore draws 1D, square and hexagonal lattices, not {lgca.geometry!r}")
    views = ["density"]
    n_species = 1 if spec.state.identity_based else int(spec.state.n_species)
    if n_species > 1:
        views += [f"density: species {index}" for index in range(n_species)]
    views.append("flux")
    views += list(spec.state.fields or {})
    if spec.state.identity_based:  # e.g. "mean kappa": the mean trait of the cells at each node
        from .lattice_state import LatticeState

        views += [f"mean {name}" for name in LatticeState(lgca).cells.traits if name != "family"]
    return views


def _kind(view, lgca):
    if view == "density" or view == "flux":
        return view, None
    if view.startswith("density: species "):
        return "density", int(view.rsplit(" ", 1)[1])
    if view.startswith("mean ") and view[5:] in getattr(lgca, "props", {}):
        return "trait", view[5:]
    return "field", view


def _values(lgca, view):
    """One value per node for a view of a 1D lattice."""
    kind, index = _kind(view, lgca)
    if kind == "density":
        return _density(lgca, index)
    if kind == "flux":
        return lgca.calc_flux(_channel_counts(lgca).astype(float))[..., 0]
    return _scalar(lgca, kind, index)


def _channel_counts(lgca):
    """Cells per channel of the interior nodes.

    Identity-based models without volume exclusion hold them without label lists; reading
    ``lgca.nodes`` would build the lists and change the order of the cells in later steps.
    """
    populations = getattr(lgca, "_channel_populations", None)
    nodes = populations() if populations is not None else lgca.nodes
    return lgca._channel_counts(nodes[lgca.nonborder])


def _density(lgca, species=None):
    if species is None:
        return np.asarray(lgca.cell_density[lgca.nonborder])
    return np.asarray(lgca.species_density[lgca.nonborder][..., species])


def _density_scale(lgca, species=None):
    """Top of the colour scale: the channels (volume exclusion) or the capacity, and the current density."""
    capacity = getattr(lgca, "capacity", None)
    if species is None and capacity is not None and not _volume_exclusion(lgca):
        scale = capacity
    elif species is None and hasattr(lgca, "species_density"):
        scale = lgca.K * lgca.species_density.shape[-1]
    else:
        scale = lgca.K
    return max(int(np.ceil(scale)), int(np.ceil(_density(lgca, species).max())), 1)


def _volume_exclusion(lgca):
    from .nove_base import NoVE_LGCA_base

    return not isinstance(lgca, NoVE_LGCA_base)


def _scalar(lgca, kind, name):
    """The values of a field, or the mean trait of the cells at each node (NaN where there are none)."""
    if kind == "trait":
        from .plot_data import mean_trait

        return mean_trait(lgca, name)
    return _field(lgca, name)


def _field(lgca, name):
    from .plot_data import select_scalar_field

    return np.asarray(select_scalar_field(lgca, getattr(lgca, name)), dtype=float)


def _limits(values):
    if not np.isfinite(values).any():  # e.g. a trait without cells
        return {"vmin": 0.0, "vmax": 1.0}
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if not high > low:
        high = low + (abs(low) or 1.0) * 1e-3
    return {"vmin": low, "vmax": high}


def _flux_colours(lgca, mappable):
    """Colours of :meth:`plot_flux`: the direction of the flux, grey where it vanishes, clear where empty."""
    counts = _channel_counts(lgca)
    jx, jy = np.moveaxis(lgca.calc_flux(counts), -1, 0)
    density = counts.sum(-1)
    colours = mappable.to_rgba(np.angle(jx + 1j * jy, deg=True) % 360.)
    colours[..., -1] = np.sign(density)
    colours[(jx ** 2 + jy ** 2) < 1e-6, :3] = 0.5
    return colours.reshape(-1, 4)


def _figsize(lgca, series):
    if len(lgca.dims) == 1:
        return (9.5, 3.6) if series else (5.5, 3.6)
    return (10, 4.4) if series else (5.8, 4.4)


def _message(exc):
    from html import escape

    return f"<span style='color:#b00020'>{escape(type(exc).__name__)}: {escape(str(exc))}</span>"
