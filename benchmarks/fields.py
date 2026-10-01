"""Time the field solvers against one LGCA step.

A disc of cells (a quarter of the lattice's width in radius) takes up a field that is produced
everywhere, diffuses and decays (D = 1, production and decay 10⁻³, so 1 far from the cells;
uptake 0.05 per cell, saturating at 0.1 in some routes) on a periodic square lattice, in some routes
carried by a uniform flow. The cells divide and die every step, so the matrix of a steady field
changes every step. For every solver the time of the ``pde`` operator alone is taken from the
pipeline's timings; ``go_or_grow`` is the time of one step of a go-or-grow model (interactions and
propagation) on the same lattice, for comparison.

The routes with terms that depend on the field compare Newton's method (the default) with Picard
iteration (``"picard"``, the iteration of earlier versions): saturating uptake with Hill exponents
``n = 1``, ``2`` and ``4``; a reaction in which every cell produces a signal that amplifies itself
(``rate c² / (K² + c²)`` per cell, so the field is bistable where the cells are dense), lost by
decay; and the saturating uptake of ``n = 1`` written as a reaction. Routes that fail record why; ``iterations`` is the largest number of iterations that one
update used. Run from the repository root:

    uv run python benchmarks/fields.py [routes] [--sizes 100 200 400] [--steps 10] [--repeats 3] [--output results.json]

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
from lgca.fields import PDESpec, reaction
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec

UPTAKE = {"uptake": 0.05}
SATURATING = {"uptake": 0.05, "saturation": 0.1}
HILL2 = {"uptake": 0.05, "saturation": 0.1, "n": 2}
HILL4 = {"uptake": 0.05, "saturation": 0.1, "n": 4}
SLOW, FAST = (0.5, 0.25), (5.0, 2.5)  # advection velocities: Péclet numbers |v| / D of about 0.6 and 6
PICARD = {"nonlinear": "picard"}
SIGNAL = {"reactions": [{"name": "benchmark_amplification", "rate": 0.05, "K": 0.5}], "decay": 0.02}
CONSUMPTION = {"reactions": [{"name": "benchmark_consumption", "rate": 0.05, "K": 0.1}]}


@reaction(name="benchmark_amplification")
def _amplification(state, c, rate=0.05, K=0.5):
    """Production by the cells that amplifies itself and saturates."""
    return rate * state.density * c**2 / (K**2 + c**2), 0.0


@reaction(name="benchmark_consumption")
def _consumption(state, c, rate=0.05, K=0.1):
    """Michaelis-Menten uptake by the cells written as a reaction: the cell term SATURATING."""
    return 0.0, rate * state.density / (K + c)


# route: (solver, backend, cell term, advection, further parameters and solver options), or None for the
# go-or-grow step
ROUTES = {
    "go_or_grow": None,
    "steady amg": ("steady", "amg", UPTAKE, None, {}),
    "steady cg": ("steady", "cg", UPTAKE, None, {}),
    "steady direct": ("steady", "direct", UPTAKE, None, {}),
    "steady amg saturating": ("steady", "amg", SATURATING, None, {}),
    "implicit cg": ("implicit", "cg", UPTAKE, None, {}),
    "steady amg v=0.5": ("steady", "amg", UPTAKE, SLOW, {}),
    "steady cg v=0.5": ("steady", "cg", UPTAKE, SLOW, {}),
    "steady direct v=0.5": ("steady", "direct", UPTAKE, SLOW, {}),
    "steady amg v=5": ("steady", "amg", UPTAKE, FAST, {}),
    "steady cg v=5": ("steady", "cg", UPTAKE, FAST, {}),
    "steady direct v=5": ("steady", "direct", UPTAKE, FAST, {}),
    "implicit cg v=5": ("implicit", "cg", UPTAKE, FAST, {}),
    "steady amg saturating picard": ("steady", "amg", SATURATING, None, PICARD),
    "steady amg hill n=2": ("steady", "amg", HILL2, None, {}),
    "steady amg hill n=2 picard": ("steady", "amg", HILL2, None, PICARD),
    "steady amg hill n=4": ("steady", "amg", HILL4, None, {}),
    "steady amg hill n=4 picard": ("steady", "amg", HILL4, None, PICARD),
    "steady direct hill n=2": ("steady", "direct", HILL2, None, {}),
    "steady direct hill n=2 picard": ("steady", "direct", HILL2, None, PICARD),
    "implicit cg hill n=4": ("implicit", "cg", HILL4, None, {}),
    "implicit cg hill n=4 picard": ("implicit", "cg", HILL4, None, PICARD),
    "implicit cg reaction": ("implicit", "cg", None, None, SIGNAL),
    "implicit cg reaction picard": ("implicit", "cg", None, None, {**SIGNAL, **PICARD}),
    "steady amg reaction": ("steady", "amg", None, None, SIGNAL),
    "steady amg reaction picard": ("steady", "amg", None, None, {**SIGNAL, **PICARD}),
    "steady amg consumption reaction": ("steady", "amg", None, None, CONSUMPTION),
    "steady amg consumption reaction picard": ("steady", "amg", None, None, {**CONSUMPTION, **PICARD}),
}
_OPTIONS = ("nonlinear",)  # keys of the further parameters that are solver options


def _disc(size):
    nodes = np.zeros((size, size, 5), dtype=bool)
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    nodes[(x - size / 2) ** 2 + (y - size / 2) ** 2 < (size / 4) ** 2] = True
    return nodes


def _measure(route, size, steps, repeat, warm_up=3):
    """Seconds per step of one route and the field's statistics, in a fresh process (see time_route); the
    seconds are None if the field solver failed."""
    warnings.simplefilter("ignore")
    solver = ROUTES[route]
    if solver is None:
        operators = [{"name": "go_or_rest", "parameters": {"kappa": 5.0, "theta": 0.75}},
                     {"name": "go_or_grow.growth", "parameters": {"r_b": 0.2, "r_d": 0.01}},
                     {"name": "random_walk", "parameters": {"channels": "velocity"}}]
    else:
        name, backend, term, advection, further = solver
        options = {"backend": backend, **{key: value for key, value in further.items() if key in _OPTIONS}}
        parameters = {"diffusion": 1.0, "decay": 1e-3, "production": 1e-3, "cells": [term] if term else [],
                      **{key: value for key, value in further.items() if key not in _OPTIONS}}
        operators = [{"name": "birth_death", "parameters": {"birth_rate": 0.05, "death_rate": 0.05}},
                     PDESpec(field="u", solver=name, advection=advection, solver_options=options, **parameters)]
    try:
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
    except RuntimeError as error:  # the field solver failed
        return None, {"error": f"{type(error).__name__}: {error}"}
    return timing[("pde", "field")]["total_seconds"] / steps, model.metadata["fields"]["u"]


def time_route(route, size, steps, repeats):
    """Median seconds per step, or None if a run failed. Every measurement runs in a fresh process: the
    memory allocator's state after other models changes timings noticeably."""
    context = multiprocessing.get_context("spawn")
    with context.Pool(1, maxtasksperchild=1) as pool:
        results = pool.starmap(_measure, [(route, size, steps, repeat) for repeat in range(repeats)])
    for seconds, statistics in results:
        if seconds is None:
            return None, statistics
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
    unknown = sorted(set(routes) - set(ROUTES))
    if unknown:
        parser.error(f"unknown routes {unknown}; choose from {', '.join(ROUTES)}")
    results = {}
    width = max(len(route) for route in routes) + 1
    print(f"{'route':{width}s} " + " ".join(f"{f'{size}²':>10s}" for size in arguments.sizes)
          + "   (ms per step; iterations)")
    for route in routes:
        results[route] = {}
        cells, iterations = [], []
        for size in arguments.sizes:
            seconds, statistics = time_route(route, size, arguments.steps, arguments.repeats)
            results[route][size] = {"seconds": seconds, "statistics": statistics}
            cells.append(f"{'failed':>10s}" if seconds is None else f"{seconds * 1e3:10.1f}")
            if "max_iterations_used" in statistics:
                iterations.append(str(statistics["max_iterations_used"]))
        print(f"{route:{width}s} " + " ".join(cells) + (f"   {'/'.join(iterations)}" if iterations else ""))
        for size in arguments.sizes:
            error = results[route][size]["statistics"].get("error")
            if error:
                print(f"{'':{width}s} {size}²: {error[:150]}")

    def seconds(route, size):
        return results[route][size]["seconds"]

    if "go_or_grow" in results:
        print("\nper go-or-grow step (x):")
        for route in routes:
            if route != "go_or_grow":
                ratios = [seconds(route, size) / seconds("go_or_grow", size)
                          if seconds(route, size) is not None else np.nan for size in arguments.sizes]
                print(f"{route:{width}s} " + " ".join(f"{ratio:10.2f}" for ratio in ratios))
    if "steady direct" in results:
        print("\nspeed-up over steady direct:")
        for route in routes:
            if (route.startswith("steady") and route != "steady direct" and "v=" not in route
                    and ROUTES[route][2] in (UPTAKE, SATURATING) and not ROUTES[route][4]):
                ratios = [seconds("steady direct", size) / seconds(route, size) for size in arguments.sizes]
                print(f"{route:{width}s} " + " ".join(f"{ratio:9.1f}x" for ratio in ratios))
    compared = [route for route in routes if route.endswith(" picard") and route[:-len(" picard")] in results]
    if compared:
        print("\nNewton's method against Picard iteration (Picard's time / Newton's):")
        for route in compared:
            newton = route[:-len(" picard")]
            ratios = [seconds(route, size) / seconds(newton, size)
                      if seconds(route, size) is not None and seconds(newton, size) is not None else np.nan
                      for size in arguments.sizes]
            print(f"{newton:{width}s} " + " ".join(f"{ratio:9.2f}x" for ratio in ratios))
    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump({"steps": arguments.steps, "repeats": arguments.repeats, "results": results}, file,
                      indent=2, default=str)


if __name__ == "__main__":
    main()
