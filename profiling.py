import argparse
import cProfile
import csv
import io
import pstats
import time
from typing import Dict, Iterable, List, Tuple

from lgca import get_lgca


def profile_timeevo(config: Dict, timesteps: int = 100) -> Tuple[float, int, str]:
    """Profile ``timeevo`` for a single LGCA configuration.

    Parameters
    ----------
    config : dict
        Keyword arguments passed to :func:`lgca.get_lgca`.
    timesteps : int, optional
        Number of timesteps for the simulation. ``100`` by default.

    Returns
    -------
    float
        Wall clock time spent in :meth:`timeevo`.
    int
        Number of particles in the lattice after initialization.
    str
        Formatted profiling information for the 10 most costly functions.
    """
    lgca = get_lgca(**config)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    lgca.timeevo(timesteps=timesteps, recorddens=False, showprogress=False)
    profiler.disable()
    runtime = time.perf_counter() - start

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(10)
    particle_count = int(lgca.nodes.sum())
    return runtime, particle_count, stream.getvalue()


def parse_comma_separated(value: str, cast=str) -> List:
    """Return a list from a comma-separated command line argument."""
    return [cast(v) for v in value.split(',') if v]


def run_benchmarks(args: argparse.Namespace) -> None:
    """Run profiling for all parameter combinations."""
    dim_map = {"lin": 1, "square": 2, "hex": 2, "cubic": 3}
    sizes = parse_comma_separated(args.sizes, int)
    densities = parse_comma_separated(args.densities, float)
    geometries = parse_comma_separated(args.geometries)
    interactions = parse_comma_separated(args.interactions)

    results = []
    for geom in geometries:
        for interaction in interactions:
            for size in sizes:
                dims = (size,) * dim_map[geom]
                for dens in densities:
                    cfg = dict(
                        geometry=geom,
                        ib=args.ib,
                        ve=args.ve,
                        interaction=interaction,
                        density=dens,
                        dims=dims,
                        restchannels=args.restchannels,
                    )
                    print(f"Profiling {cfg}")
                    runtimes = []
                    particle_count = None
                    for _ in range(args.repeats):
                        runtime, pcount, stats = profile_timeevo(
                            cfg, timesteps=args.timesteps
                        )
                        runtimes.append(runtime)
                        particle_count = pcount
                        print(stats)
                    avg_runtime = sum(runtimes) / len(runtimes)
                    results.append(
                        (
                            geom,
                            interaction,
                            dims,
                            dens,
                            particle_count,
                            avg_runtime,
                        )
                    )

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "geometry",
                    "interaction",
                    "dims",
                    "density",
                    "particles",
                    "time_sec",
                ]
            )
            for row in results:
                writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile LGCA timeevo")
    parser.add_argument(
        "--geometries",
        default="square",
        help="Comma separated list of lattice geometries",
    )
    parser.add_argument(
        "--interactions",
        default="random_walk",
        help="Comma separated list of interaction rules",
    )
    parser.add_argument("--sizes", default="20,40", help="Comma separated lattice sizes")
    parser.add_argument("--densities", default="0.5,0.8", help="Comma separated densities")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--restchannels", type=int, default=1, help="Number of rest channels")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each configuration")
    parser.add_argument("--ib", action="store_true", help="Use identity based LGCA")
    parser.add_argument("--ve", action="store_true", default=False, help="Enable volume exclusion")
    parser.add_argument("--output", help="File to write summary results")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    run_benchmarks(parser.parse_args())
