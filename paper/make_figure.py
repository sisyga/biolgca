"""Compose the figure of the JOSS paper from the saved outputs of zoo and example notebooks.

Run from the repository root: ``uv run python paper/make_figure.py``.
"""

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "source"


def notebook_image(notebook, index=0):
    """The index-th PNG output of a notebook, as a PIL image."""
    cells = json.loads((DOCS / notebook).read_text(encoding="utf-8"))["cells"]
    images = [output["data"]["image/png"] for cell in cells for output in cell.get("outputs", [])
              if "image/png" in output.get("data", {})]
    return Image.open(io.BytesIO(base64.b64decode(images[index])))


excitable = notebook_image("zoo/excitable_media.ipynb")
front = notebook_image("zoo/evolving_front.ipynb")
spheroid = notebook_image("examples/tumour_spheroid.ipynb")

# (box in pixels of the saved outputs, panel title)
panels = [
    (excitable.crop((810, 30, 1205, 375)), "(a) excitable medium, LGCA"),
    (excitable.crop((810, 455, 1205, 795)), "(b) its mean-field equations"),
    (spheroid.crop((948, 30, 1178, 262)), "(c) tumour spheroid in oxygen"),
    (front.crop((0, 42, 628, 432)), "(d) evolving invasion front"),
    (front.crop((660, 42, 1211, 432)), "(e) front speed and theory"),
]

fig = plt.figure(figsize=(10, 6.4), layout="constrained")
grid = fig.add_gridspec(2, 6)
axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])]
for ax, (image, title) in zip(axes, panels):
    ax.imshow(image)
    ax.set_title(title, fontsize=11, loc="left")
    ax.axis("off")
fig.savefig(Path(__file__).with_name("figure.png"), dpi=150)
