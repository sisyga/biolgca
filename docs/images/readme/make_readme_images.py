"""Regenerate the README images by running the code blocks in README.md.

Run from the repository root (needs ffmpeg for the animated GIF):

    uv run python docs/images/readme/make_readme_images.py

The first three Python blocks of the README produce the three figures, so the
images always show what the README code produces.
"""

import re
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

OUT = Path(__file__).parent
README = OUT.parents[2] / "README.md"


def readme_blocks():
    text = README.read_text(encoding="utf-8")
    return [textwrap.dedent(block) for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL)]


def run_block(block, namespace):
    exec(compile(block, str(README), "exec"), namespace)  # noqa: S102 - trusted README code


quick_start, mechanisms, growth = readme_blocks()[:3]
namespace = {}

run_block(quick_start, namespace)
plt.close("all")
lgca = namespace["lgca"]
# Show only the lattice: no axes, colour bar or title, and the lattice fills the frame.
movie = lgca.animate_flux(interval=50, cbar=False, figsize=(8, 8 * np.ptp(lgca.ycoords) / np.ptp(lgca.xcoords)))
axis = movie._fig.axes[0]
axis.set_axis_off()
axis.title.set_visible(False)
for collection in axis.collections:
    collection.set_antialiased(False)  # flat colours compress far better in a GIF
movie._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
movie._fig.set_layout_engine(None)
frames = OUT / "alignment_flux.mkv"
movie.save(frames, writer=FFMpegWriter(fps=20, codec="png"), dpi=120)  # lossless intermediate
plt.close("all")
# Encode with an optimized palette that only redraws the changed regions of each frame.
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", frames, "-filter_complex",
     ("fps=20,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
      "[b][p]paletteuse=dither=none:diff_mode=rectangle"),
     OUT / "alignment_flux.gif"],
    check=True,
)
frames.unlink()

for block, name in ((mechanisms, "chemotaxis_density.png"), (growth, "go_or_grow_density.png")):
    run_block(block, namespace)
    plt.savefig(OUT / name, dpi=80)
    plt.close("all")
