"""Time the field solvers against one LGCA step.

A disc of cells (a quarter of the lattice's width in radius) takes up a field that is produced
everywhere, diffuses and decays (D = 1, production and decay 10⁻³, so 1 far from the cells;
uptake 0.05 per cell, saturating at 0.1 in one route) on a periodic square lattice, in some routes
carried by a uniform flow. The cells divide and die every step, so the matrix of a steady field
changes every step. For every solver the time of the ``pde`` operator alone is taken from the
pipeline's timings; ``go_or_grow`` is the time of one step of a go-or-grow model (interactions and
propagation) on the same lattice, for comparison. Run from the repository root:

    uv run python benchmarks/fields.py [--sizes 100 200 400] [--steps 10] [--repeats 3] [--output results.json]

Every measurement runs in a fresh process. Timings are diagnostics for one machine, not portable
performance gates.
"""

import argparse
import json
import multiprocessing
import sys
import warnings
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgca.fields import PDESpec
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec

UPTAKE = {"uptake": 0.05}
SATURATING = {"uptake": 0.05, "saturation": 0.1}
SLOW, FAST = (0.5, 0.25), (5.0, 2.5)  # advection velocities: Péclet numbers |v| / D of about 0.6 and 6
# route: (solver, backend, cell term, advection), or None for the go-or-grow step
ROUTES = {
    "go_or_grow": None,
    "steady amg": ("steady", "amg", UPTAKE, None),
    "steady cg": ("steady", "cg", UPTAKE, None),
    "steady direct": ("steady", "direct", UPTAKE, None),
    "steady amg saturating": ("steady", "amg", SATURATING, None),
    "implicit cg": ("implicit", "cg", UPTAKE, None),
    "steady amg v=0.5": ("steady", "amg", UPTAKE, SLOW),
    "steady cg v=0.5": ("steady", "cg", UPTAKE, SLOW),
    "steady direct v=0.5": ("steady", "direct", UPTAKE, SLOW),
    "steady amg v=5": ("steady", "amg", UPTAKE, FAST),
    "steady cg v=5": ("steady", "cg", UPTAKE, FAST),
    "steady direct v=5": ("steady", "direct", UPTAKE, FAST),
    "implicit cg v=5": ("implicit", "cg", UPTAKE, FAST),
}


def _disc(size):
    nodes = np.zeros((size, size, 5), dtype=bool)
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    nodes[(x - size / 2) ** 2 + (y - size / 2) ** 2 < (size / 4) ** 2] = True
    return nodes


def _measure(route, size, steps, repeat, warm_up=3):
    """Seconds per step of one route, in a fresh process (see time_route)."""
    warnings.simplefilter("ignore")
    solver = ROUTES[route]
    if solver is None:
        operators = [{"name": "go_or_rest", "parameters": {"kappa": 5.0, "theta": 0.75}},
                     {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01}},
                     {"name": "random_walk", "parameters": {"channels": "velocity"}}]
    else:
        name, backend, term, advection = solver
        operators = [{"name": "birth_death", "parameters": {"birth_rate": 0.05, "death_rate": 0.05}},
                     PDESpec(field="u", diffusion=1.0, decay=1e-3, production=1e-3, cells=[term], solver=name,
                             advection=advection, solver_options={"backend": backend})]
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(size, size), boundary="periodic"),
        state=StateSpec(nodes=_disc(size), restchannels=1, fields={"u": 1.0}),
        time=TimeSpec(steps=steps, seed=repeat),
        dynamics=InteractionPipelineSpec(operators=operators, propagation=solver is None)))
    for _ in range(warm_up):
        model.step()
    if solver is None:
        start = perf_counter()
        for _ in range(steps):
            model.step()
        return (perf_counter() - start) / steps, {}
    timing = {}
    for _ in range(steps):
        model.step(timing=timing)
    return timing[("pde", "field")]["total_seconds"] / steps, model.metadata["fields"]["u"]


def time_route(route, size, steps, repeats):
    """Median seconds per step. Every measurement runs in a fresh process: the memory allocator's
    state after other models changes timings noticeably."""
    context = multiprocessing.get_context("spawn")
    with context.Pool(1, maxtasksperchild=1) as pool:
        results = pool.starmap(_measure, [(route, size, steps, repeat) for repeat in range(repeats)])
    return median(seconds for seconds, _ in results), results[0][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("routes", nargs="*", help=f"routes to time (default: all of {', '.join(ROUTES)})")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="write the results as JSON to this file")
    arguments = parser.parse_args()
    routes = arguments.routes or list(ROUTES)
    results = {}
    print(f"{'route':24s} " + " ".join(f"{f'{size}²':>10s}" for size in arguments.sizes) + "   (ms per step)")
    for route in routes:
        results[route] = {}
        cells = []
        for size in arguments.sizes:
            seconds, statistics = time_route(route, size, arguments.steps, arguments.repeats)
            results[route][size] = {"seconds": seconds, "statistics": statistics}
            cells.append(f"{seconds * 1e3:10.1f}")
        print(f"{route:24s} " + " ".join(cells))
    if "go_or_grow" in results:
        print("\nper go-or-grow step (x):")
        for route in routes:
            if route != "go_or_grow":
                ratios = [results[route][size]["seconds"] / results["go_or_grow"][size]["seconds"]
                          for size in arguments.sizes]
                print(f"{route:24s} " + " ".join(f"{ratio:10.2f}" for ratio in ratios))
    if "steady direct" in results:
        print("\nspeed-up over steady direct:")
        for route in routes:
            if route.startswith("steady") and route != "steady direct" and "v=" not in route:
                ratios = [results["steady direct"][size]["seconds"] / results[route][size]["seconds"]
                          for size in arguments.sizes]
                print(f"{route:24s} " + " ".join(f"{ratio:9.1f}x" for ratio in ratios))
    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump({"steps": arguments.steps, "repeats": arguments.repeats, "results": results}, file,
                      indent=2)


if __name__ == "__main__":
    main()
