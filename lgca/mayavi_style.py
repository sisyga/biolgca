# biolgca is a Python package for simulating different kinds of lattice-gas
# cellular automata (LGCA) in the biological context.
# Copyright (C) 2018-2026 Technische Universität Dresden, Germany.
# The full license notice is found in the file lgca/__init__.py.
"""Shared Mayavi figure style for three-dimensional lattice plots.

All 3D plots use the same publication-oriented look: a white background, a
thin box around the lattice domain with sparse tick labels, a fixed oblique
camera, anti-aliased rendering, correctly blended translucent surfaces and a
compact colour bar. Lattice node ``i`` occupies the unit cell ``[i, i + 1]``,
so the box spans ``[0, L]`` along each axis and glyphs sit at cell centres.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from lgca.base import MaxNLocator, _MissingPlotLib

try:  # optional plotting dependency
    from mayavi import mlab
    from tvtk.api import tvtk
except ImportError:  # pragma: no cover - handled at runtime
    mlab = tvtk = _MissingPlotLib(
        "mayavi", "Install the 3D plotting extra with 'uv sync --extra plot3d' or "
        "'python -m pip install -e \".[plot3d]\"'.")

INK = (0.1, 0.1, 0.1)
MUTED = (0.6, 0.6, 0.6)
FONT = "arial"
DEFAULT_SIZE = (760, 620)
DEFAULT_VIEW = {"azimuth": -55.0, "elevation": 65.0}
VIEW_ANGLE = 20.0

# Box edges of the unit cube as pairs of corner indices (corner bits = x, y, z).
_CUBE_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
               (0, 4), (1, 5), (2, 6), (3, 7)]


def style_text(prop, size, italic=False):
    """Apply the shared font to a VTK text property."""
    prop.color = INK
    prop.font_family = FONT
    prop.bold = False
    prop.italic = italic
    prop.shadow = False
    prop.font_size = size


def style_surface(actor, opacity=1.0):
    """Give a surface or glyph actor soft, matte shading."""
    prop = actor.property
    prop.opacity = opacity
    prop.ambient = 0.15
    prop.diffuse = 0.85
    prop.specular = 0.15
    prop.specular_power = 20.0


def style_arrows(quiver, opacity=1.0):
    """Use bold arrow glyphs that stay legible when many arrows are drawn."""
    arrow = quiver.glyph.glyph_source.glyph_source
    arrow.tip_length, arrow.tip_radius, arrow.shaft_radius = 0.38, 0.16, 0.06
    arrow.tip_resolution = arrow.shaft_resolution = 16
    style_surface(quiver.actor, opacity)


def new_figure(size=None):
    """Create a figure with a white background and high-quality rendering.

    Depth peeling blends nested translucent surfaces in the correct order; FXAA
    smooths edges because depth peeling cannot be combined with multisampling.
    """
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=INK, size=size or DEFAULT_SIZE)
    scene = fig.scene
    scene.disable_render = True
    scene.render_window.multi_samples = 0
    scene.render_window.alpha_bit_planes = 1
    scene.renderer.use_depth_peeling = True
    scene.renderer.maximum_number_of_peels = 32
    scene.renderer.occlusion_ratio = 0.0
    scene.renderer.use_fxaa = True
    return fig


def decorate_domain(fig, dims, axes=True, view=None):
    """Draw the lattice domain box with sparse ticks and set the camera.

    Parameters
    ----------
    fig : mayavi.core.scene.Scene
        Figure created by :func:`new_figure`.
    dims : tuple of int
        Lattice extent ``(lx, ly, lz)``.
    axes : bool, default=True
        Whether to label the box edges with ``x``, ``y`` and ``z`` and tick
        values at 0, the middle and the end of each axis.
    view : dict, optional
        Keyword arguments for :func:`mayavi.mlab.view`; the default is an
        oblique view onto the domain centre.
    """
    lx, ly, lz = dims
    corners = np.array([[(i & 1) * lx, (i >> 1 & 1) * ly, (i >> 2 & 1) * lz] for i in range(8)], dtype=float)
    # Draw all edges as one polyline with NaN breaks so the box needs no data source.
    points = []
    for a, b in _CUBE_EDGES:
        points.extend([corners[a], corners[b], [np.nan] * 3])
    x, y, z = np.array(points).T
    box = mlab.plot3d(x, y, z, color=INK, tube_radius=None, line_width=1.0, figure=fig)
    box.actor.property.opacity = 0.8

    if axes:
        extent = [0, lx, 0, ly, 0, lz]
        cube_axes = mlab.axes(box, extent=extent, ranges=extent, nb_labels=3, xlabel="x", ylabel="y", zlabel="z",
                              color=INK, figure=fig)
        cube_axes.axes.label_format = "%g"
        cube_axes.axes.fly_mode = "outer_edges"
        cube_axes.axes.font_factor = 1.0
        cube_axes.axes.corner_offset = 0.04
        cube_axes.property.line_width = 0.0
        cube_axes.property.opacity = 0.0
        style_text(cube_axes.label_text_property, 12)
        style_text(cube_axes.title_text_property, 14, italic=True)

    # A narrow view angle keeps perspective distortion small; the distance fits the box diagonal
    # into the view.
    camera = fig.scene.camera
    camera.view_angle = VIEW_ANGLE
    diagonal = float(np.linalg.norm(dims))
    view = {**DEFAULT_VIEW, **(view or {})}
    view.setdefault("focalpoint", (lx / 2, ly / 2, lz / 2))
    view.setdefault("distance", 0.5 * diagonal / np.tan(np.radians(VIEW_ANGLE / 2)))
    mlab.view(figure=fig, **view)
    fig.scene.disable_render = False
    return box


def add_colorbar(obj, title, vmin, vmax, integer=False, discrete=None):
    """Attach a compact vertical colour bar to the right of the scene.

    Parameters
    ----------
    obj : mayavi module
        The coloured glyphs or surfaces.
    title : str
        Colour bar title.
    vmin, vmax : float
        Colour scale limits.
    integer : bool, default=False
        Whether the values are integers; ticks are then placed at integers.
    discrete : bool, optional
        Whether every integer gets its own colour bin with the label at its
        centre. Default: for integer values with at most 12 distinct values.
    """
    if discrete is None:
        discrete = integer
    discrete = discrete and 0 < vmax - vmin <= 11
    fmt = "%.0f" if integer else "%g"
    # mlab.colorbar returns the scalar or vector lookup-table manager of `obj`.
    colorbar = mlab.colorbar(object=obj, title=title, orientation="vertical", nb_labels=5, label_fmt=fmt)
    colorbar.use_default_range = False
    if discrete:
        # One colour bin per integer, centred on it, with a label in the middle of each bin.
        ticks = np.arange(vmin, vmax + 1, dtype=float)
        colorbar.data_range = (vmin - 0.5, vmax + 0.5)
        colorbar.number_of_colors = len(ticks)
    else:
        ticks = MaxNLocator(nbins=5, integer=integer).tick_values(vmin, vmax)
        ticks = ticks[(ticks >= vmin - 1e-9 * abs(vmax - vmin)) & (ticks <= vmax + 1e-9 * abs(vmax - vmin))]
        colorbar.data_range = (vmin, vmax)
    labels = tvtk.DoubleArray()
    labels.from_array(np.asarray(ticks, dtype=float))
    colorbar.scalar_bar.custom_labels = labels
    colorbar.scalar_bar.use_custom_labels = True
    colorbar.scalar_bar.vertical_title_separation = 12
    colorbar.scalar_bar.unconstrained_font_size = True
    colorbar.scalar_bar.bar_ratio = 0.3
    style_text(colorbar.label_text_property, 12)
    style_text(colorbar.title_text_property, 13)
    colorbar.scalar_bar_representation.position = [0.84, 0.25]
    colorbar.scalar_bar_representation.position2 = [0.1, 0.5]
    # Shift the scene left to make room for the colour bar.
    obj.scene.camera.window_center = (0.12, 0.0)
    return colorbar


def time_label(fig, text):
    """Place a fixed-size time label in the upper-left corner."""
    label = mlab.text(0.03, 0.92, text, figure=fig, width=0.2)
    label.actor.text_scale_mode = "none"
    style_text(label.property, 15)
    return label


@contextmanager
def offscreen(enabled=True):
    """Create and render figures without opening a window while the context is active.

    Mayavi chooses its rendering engine per call, so keep every call that builds
    or renders an offscreen figure inside the context. With ``enabled=False`` the
    context does nothing, so it does not require Mayavi.
    """
    if not enabled:
        yield
        return
    previous = mlab.options.offscreen
    mlab.options.offscreen = previous or enabled
    try:
        yield
    finally:
        mlab.options.offscreen = previous


def save_figure(fig, path):
    """Save a Mayavi figure as an image; the format follows the file extension."""
    mlab.savefig(str(path), figure=fig)


def movie_writer(path, fps, save_kwargs=None):
    """Return a Matplotlib movie writer for ``path``.

    ``save_kwargs`` takes the options of :meth:`matplotlib.animation.Animation.save`
    except ``dpi``: ``writer`` (name or instance), ``fps``, ``codec``,
    ``bitrate``, ``extra_args`` and ``metadata``. GIF files default to the
    Pillow writer, all other files to ``rcParams['animation.writer']`` (ffmpeg).
    """
    from matplotlib import animation, rcParams

    kwargs = dict(save_kwargs or {})
    if "dpi" in kwargs:
        raise ValueError("3D movies are recorded at the figure size in pixels; pass size=(width, height) "
                         "to the plotting method instead of dpi")
    writer = kwargs.pop("writer", None)
    fps = kwargs.pop("fps", fps)
    if writer is None:
        writer = "pillow" if Path(path).suffix.lower() == ".gif" else rcParams["animation.writer"]
    if isinstance(writer, str):
        if not animation.writers.is_available(writer):
            raise RuntimeError(f"Movie writer {writer!r} is not available. Install ffmpeg to write video files, "
                               "or save a .gif, which uses the Pillow writer.")
        writer = animation.writers[writer](fps=fps, **kwargs)
    return writer


def save_movie(fig, update, n_frames, label, path, fps=10, save_kwargs=None):
    """Render ``update(frame)`` for every frame and write the frames to a movie file.

    Parameters
    ----------
    fig : mayavi.core.scene.Scene
        Figure to record; render it offscreen (see :func:`offscreen`) to avoid
        opening a window.
    update : callable
        Called with the frame index; updates the plotted data.
    n_frames : int
        Number of frames.
    label : callable or None
        Maps the frame index to the text of the time label.
    path : str or pathlib.Path
        Output file, e.g. ``density.mp4`` or ``density.gif``.
    fps : float, default=10
        Frames per second, unless ``save_kwargs`` sets ``fps``.
    save_kwargs : dict, optional
        Writer options; see :func:`movie_writer`.

    Returns
    -------
    pathlib.Path
        The movie file.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(path)
    writer = movie_writer(path, fps, save_kwargs)
    text = time_label(fig, label(0)) if label is not None else None

    def render(frame):
        update(frame)
        if text is not None:
            text.text = label(frame)
        fig.scene.render()
        image = mlab.screenshot(figure=fig, mode="rgb", antialiased=False)
        # Video codecs such as H.264 require even frame dimensions.
        return image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]

    first = render(0)
    height, width = first.shape[:2]
    canvas = Figure(figsize=(width / 100, height / 100), dpi=100)
    FigureCanvasAgg(canvas)
    axes = canvas.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    image = axes.imshow(first, interpolation="nearest")
    with writer.saving(canvas, str(path), dpi=100):
        writer.grab_frame()
        for frame in range(1, n_frames):
            image.set_data(render(frame))
            writer.grab_frame()
    return path


def play(fig, update, n_frames=None, label=None, interval=100, show=True, save_path=None, save_kwargs=None):
    """Animate ``update(frame)`` in the figure window, or record it as a movie.

    Parameters
    ----------
    fig : mayavi.core.scene.Scene
        Figure to animate.
    update : callable
        Called with the frame index; updates the plotted data.
    n_frames : int, optional
        Number of frames. None animates until the window is closed.
    label : callable, optional
        Maps the frame index to the text of the time label.
    interval : int, default=100
        Delay between frames in milliseconds; a movie plays at ``1000 / interval``
        frames per second unless ``save_kwargs`` sets ``fps``.
    show : bool, default=True
        Whether to enter the GUI event loop with :func:`mayavi.mlab.show`. Pass
        False inside an application that already runs an event loop.
    save_path : str or pathlib.Path, optional
        Record all frames to this movie file instead of animating in a window,
        then close the figure. Requires `n_frames`.
    save_kwargs : dict, optional
        Writer options; see :func:`movie_writer`.

    Returns
    -------
    mayavi.tools.animator.Animator or pathlib.Path
        The animation controller, or the movie file if `save_path` is given.
    """
    if save_path is not None:
        if n_frames is None:
            raise ValueError("Recording a movie requires a finite number of frames")
        try:
            return save_movie(fig, update, n_frames, label, save_path, 1000 / interval, save_kwargs)
        finally:
            mlab.close(fig)

    text = time_label(fig, label(0)) if label is not None else None

    @mlab.animate(delay=interval)
    def frames():
        frame = 0
        while n_frames is None or frame < n_frames:
            fig.scene.disable_render = True
            update(frame)
            if text is not None:
                text.text = label(frame)
            fig.scene.disable_render = False
            frame += 1
            yield

    animator = frames()
    if show:
        mlab.show()
    return animator
