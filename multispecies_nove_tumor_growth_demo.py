"""Run multispecies NoVE tumor-growth demos for hex and 3D Moore lattices.

The script exercises the multispecies NoVE ``birth``, ``birthdeath``, and
``go_or_grow`` interactions from a single initially occupied node. Occupied
nodes are colored by the local mean class property: ``r_b`` for birth-based
interactions and ``kappa`` for go-or-grow.

Examples
--------
Run the default demo and write figures to ``tumor_growth_outputs``::

    python multispecies_nove_tumor_growth_demo.py

Fast smoke run without plotting::

    python multispecies_nove_tumor_growth_demo.py --max-steps 2 --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lgca import get_lgca


INTERACTION_CONFIGS = {
    "birth": {
        "property": "r_b",
        "values": np.array([0.12, 0.18, 0.26, 0.36]),
        "kwargs": {"std": 0.08},
    },
    "birthdeath": {
        "property": "r_b",
        "values": np.array([0.14, 0.20, 0.28, 0.38]),
        "kwargs": {"r_d": 0.035, "std": 0.08},
    },
    "go_or_grow": {
        "property": "kappa",
        "values": np.array([0.5, 1.5, 3.0, 5.0]),
        "kwargs": {"r_b": 0.5, "r_d": 0.0, "theta": 0.05, "kappa_std": 1.0},
    },
}

GEOMETRY_CONFIGS = {
    "hex": {"dims": (46, 46), "capacity": 48, "target_cells": 4500},
    "moore": {"dims": (35, 35, 35), "capacity": 64, "target_cells": 6000},
}


def centered_single_seed(geometry: str, dims: tuple[int, ...], n_species: int, channels: int) -> np.ndarray:
    """Create a NoVE multispecies lattice with one resting seed at the center."""
    nodes = np.zeros(dims + (n_species, channels), dtype=np.uint)
    center = tuple(dim // 2 for dim in dims)
    nodes[center + (0, channels - 1)] = 1
    return nodes


def make_lgca(geometry: str, interaction: str, seed: int):
    """Construct one multispecies NoVE LGCA for the requested demo."""
    cfg = GEOMETRY_CONFIGS[geometry]
    interaction_cfg = INTERACTION_CONFIGS[interaction]
    n_species = len(interaction_cfg["values"])

    probe = get_lgca(
        geometry=geometry,
        n_species=n_species,
        ve=False,
        dims=cfg["dims"],
        density=0,
        restchannels=1,
        interaction="only_propagation",
    )
    nodes = centered_single_seed(geometry, cfg["dims"], n_species, probe.K)

    kwargs = dict(interaction_cfg["kwargs"])
    prop = interaction_cfg["property"]
    kwargs[prop] = interaction_cfg["values"]

    return get_lgca(
        geometry=geometry,
        n_species=n_species,
        ve=False,
        nodes=nodes,
        interaction=interaction,
        capacity=cfg["capacity"],
        seed=seed,
        **kwargs,
    )


def run_until_large_tumor(lgca, target_cells: int, max_steps: int) -> int:
    """Advance the model until the tumor is large enough or max_steps is reached."""
    for step in range(1, max_steps + 1):
        lgca.timestep()
        if int(lgca.cell_density[lgca.nonborder].sum()) >= target_cells:
            return step
    return max_steps


def local_mean_property(lgca, property_values: np.ndarray) -> np.ma.MaskedArray:
    """Compute mean class property per lattice node from species counts."""
    species_counts = lgca.nodes[lgca.nonborder].sum(axis=-1)
    density = species_counts.sum(axis=-1)
    weighted_sum = np.tensordot(species_counts, property_values, axes=([-1], [0]))

    mean_property = np.ma.masked_all(density.shape, dtype=float)
    occupied = density > 0
    mean_property[occupied] = weighted_sum[occupied] / density[occupied]
    return mean_property


def plot_hex_property(lgca, mean_property, label: str, outfile: Path) -> None:
    """Plot a hex lattice scalar field with the existing hex scalar plotter."""
    import matplotlib.pyplot as plt

    lgca.plot_scalarfield(
        mean_property.filled(0),
        mask=mean_property.mask,
        cmap="viridis",
        cbarlabel=f"Mean {label}",
        edgecolor="0.82",
        vmin=float(mean_property.min()),
        vmax=float(mean_property.max()),
    )
    plt.title(f"Hex tumor colored by mean {label}")
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close()


def plot_moore_property(lgca, mean_property, label: str, outfile: Path) -> None:
    """Plot a 3D Moore scalar field with the existing cubic scalar plotter."""
    field = mean_property.filled(np.nan)
    mask = np.isfinite(field)
    if not mask.any():
        raise RuntimeError("Cannot plot Moore tumor: no occupied voxels found.")

    try:
        from lgca.lgca_cubic import mlab

        fig, _, _ = lgca.plot_scalarfield(
            mean_property.filled(0),
            mask=~mask,
            colormap="viridis",
            opacity=0.85,
            cbarlabel=f"Mean {label}",
            vmin=float(np.nanmin(field)),
            vmax=float(np.nanmax(field)),
        )
        mlab.title(f"Moore tumor colored by mean {label}", size=0.35, color=(0, 0, 0))
        mlab.savefig(str(outfile), figure=fig)
        mlab.close(fig)
    except ImportError:
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt

        norm = colors.Normalize(vmin=float(np.nanmin(field)), vmax=float(np.nanmax(field)))
        cmap = plt.get_cmap("viridis")
        facecolors = np.zeros(field.shape + (4,), dtype=float)
        facecolors[mask] = cmap(norm(field[mask]))
        facecolors[..., 3] = np.where(mask, 0.82, 0.0)

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(projection="3d")
        ax.voxels(mask, facecolors=facecolors, edgecolor="0.55", linewidth=0.08)
        ax.set_box_aspect(field.shape)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Moore tumor colored by mean {label}")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            shrink=0.72,
            pad=0.08,
            label=f"Mean {label}",
        )
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_property(lgca, geometry: str, interaction: str, mean_property, label: str, outdir: Path) -> None:
    """Dispatch to the plotting backend for each geometry."""
    suffix = "png"
    outfile = outdir / f"{geometry}_{interaction}_mean_{label}.{suffix}"
    if geometry == "hex":
        plot_hex_property(lgca, mean_property, label, outfile)
    elif geometry == "moore":
        plot_moore_property(lgca, mean_property, label, outfile)
    else:
        raise ValueError(f"Unsupported geometry {geometry!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--outdir", type=Path, default=Path("tumor_growth_outputs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--geometries", nargs="+", default=tuple(GEOMETRY_CONFIGS))
    parser.add_argument("--interactions", nargs="+", default=tuple(INTERACTION_CONFIGS))
    args = parser.parse_args()

    if not args.no_plot:
        args.outdir.mkdir(parents=True, exist_ok=True)

    for geometry in args.geometries:
        for interaction in args.interactions:
            cfg = INTERACTION_CONFIGS[interaction]
            lgca = make_lgca(geometry, interaction, seed=args.seed)
            target_cells = GEOMETRY_CONFIGS[geometry]["target_cells"]
            steps = run_until_large_tumor(
                lgca,
                target_cells=target_cells,
                max_steps=args.max_steps,
            )
            total_cells = int(lgca.cell_density[lgca.nonborder].sum())
            mean_property = local_mean_property(lgca, cfg["values"])

            print(
                f"{geometry:>5s} {interaction:>10s}: "
                f"{total_cells} cells after {steps} steps; "
                f"mean {cfg['property']} range "
                f"{mean_property.min():.3g}..{mean_property.max():.3g}"
            )

            if not args.no_plot:
                plot_property(lgca, geometry, interaction, mean_property, cfg["property"], args.outdir)


if __name__ == "__main__":
    main()
