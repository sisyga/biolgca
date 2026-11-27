"""Make the figure of the JOSS paper: four typical biolgca simulations.

Run from the repository root (takes about a minute)::

    uv run python paper/make_figure.py

Every panel runs a model that ships with biolgca and plots its final state:
(a) ``lgca.examples.aggregation``, (b) ``lgca.examples.alignment``,
(c) ``lgca.examples.tumour_spheroid`` and (d) ``lgca.zoo.excitable_media``.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from lgca.examples import aggregation, alignment, tumour_spheroid
from lgca.model import run_model
from lgca.zoo import excitable_media

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def resized(spec, dims, steps):
    """The example ``spec`` on a larger lattice and for more steps."""
    return replace(spec, space=replace(spec.space, dims=dims), time=replace(spec.time, steps=steps))


def bare(axis, title):
    axis.set(title=title, xticks=[], yticks=[], xlabel="", ylabel="", aspect="equal")
    axis.set_anchor("N")
    for spine in axis.spines.values():
        spine.set_visible(False)


def colorbar(fig, axis, cmap, norm, label, **kwargs):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=axis, shrink=0.85, pad=0.03, **kwargs)
    bar.set_label(label)
    return bar


fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), constrained_layout=True)

# (a) aggregation: cells per node on a hexagonal lattice, a third of all channels occupied
spec = aggregation.build_spec()
spec = replace(spec, space=replace(spec.space, geometry="hex", dims=(60, 70)), state=replace(spec.state, density=3.0),
               time=replace(spec.time, steps=300))
run = run_model(spec, showprogress=False)
lattice = run.lgca
density = lattice.nodes[lattice.nonborder].sum(-1)
axis = axes[0, 0]
norm = Normalize(0, density.max())
lattice.plot_scalarfield(density.astype(float), ax=axis, cmap="viridis", vmin=0, vmax=density.max(), cbar=False,
                         edgecolor="none")
bare(axis, "(a) aggregation")
colorbar(fig, axis, "viridis", norm, "cells per node", ticks=range(0, int(density.max()) + 1, 3))

# (b) alignment: direction of the local flux on a hexagonal lattice
run = run_model(resized(alignment.build_spec(), (100, 115), 300), showprogress=False)
lattice = run.lgca
flux = lattice.calc_flux(lattice.nodes[lattice.nonborder])
angle = np.degrees(np.arctan2(flux[..., 1], flux[..., 0]))
axis = axes[0, 1]
norm = Normalize(-180, 180)
lattice.plot_scalarfield(angle, ax=axis, mask=np.hypot(flux[..., 0], flux[..., 1]) == 0, cmap="hsv", vmin=-180,
                         vmax=180, cbar=False, edgecolor="none")
bare(axis, "(b) alignment")
colorbar(fig, axis, "hsv", norm, "direction of the local flux", ticks=[-180, -90, 0, 90, 180], format="%d°")

# (c) tumor spheroid: the most frequent cell class of every node after ten days
run = run_model(tumour_spheroid.build_spec(), showprogress=False)
lattice = run.lgca
classes = np.asarray(run.data["density"])[-1]  # nodes, class: proliferating, quiescent, necrotic
dominant = classes.argmax(-1).astype(float)
middle = (lattice.xcoords.max() / 2, lattice.ycoords.max() / 2)
axis = axes[1, 0]
cmap = ListedColormap(["#d95f02", "#7570b3", "#4d4d4d"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
lattice.plot_scalarfield(dominant, ax=axis, mask=classes.sum(-1) == 0, cmap=cmap, vmin=-0.5, vmax=2.5, cbar=False,
                         edgecolor="none")
bare(axis, "(c) tumor spheroid")
axis.set(xlim=(middle[0] - 27, middle[0] + 27), ylim=(middle[1] - 27, middle[1] + 27))
bar = colorbar(fig, axis, cmap, norm, "most frequent cell class", ticks=[0, 1, 2])
bar.ax.set_yticklabels(["proliferating", "quiescent", "necrotic"], rotation=90, va="center")

# (d) excitable medium: density of the excited species, a spiral wave
run = run_model(excitable_media.build_spec(K=40, size=120), showprogress=False)
lattice = run.lgca
rho_x, _ = excitable_media.fractions(np.asarray(run.data["nodes"])[-1])
axis = axes[1, 1]
norm = Normalize(0, 1)
lattice.plot_scalarfield(rho_x, ax=axis, cmap="magma", vmin=0, vmax=1, cbar=False, edgecolor="none")
bare(axis, "(d) excitable medium")
middle = lattice.xcoords.max() / 2  # a square part of the lattice, which is wider than it is high
half = (lattice.ycoords.max() - lattice.ycoords.min()) / 2
axis.set(xlim=(middle - half, middle + half))
colorbar(fig, axis, "magma", norm, "density of the excited species")

fig.savefig(Path(__file__).with_name("figure.png"), dpi=300)
