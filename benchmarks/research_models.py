"""Time the rules that replace legacy interactions against the legacy interactions.

The legacy interaction functions are kept in ``tests/legacy`` and run under the names
``legacy.<family>.<name>``.

Every pair runs the same seeded model for some steps, propagation included, and reports the
median time per step over repeats, after warm-up steps (the first steps of a model are slower:
growth has not settled and the memory allocator adapts). The legacy growth rules also move the
cells, so they are compared with ``birth_death`` followed by ``random_walk``. Run from the
repository root:

    uv run python benchmarks/research_models.py [--repeats 3] [--output results.json] [NAME ...]

Timings are diagnostics for one machine, not portable performance gates.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for tests.legacy
import tests.legacy  # noqa: E402,F401  (registers "legacy.<family>.<name>")
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec

DIMS = (100, 100)

# name: (legacy operator, new operator(s), parameters (legacy, new), state)
_IB = {"restchannels": 1, "identity_based": True, "density": 0.3}
_NOVE_IB = {"restchannels": 1, "identity_based": True, "volume_exclusion": False, "capacity": 8, "density": 2.0}
PAIRS = {
    "birth_death (classical)": ("legacy.classical.birthdeath", ("birth_death", "random_walk"),
                                ({"r_b": 0.2, "r_d": 0.05}, {"birth_rate": 0.2, "death_rate": 0.05}),
                                {"restchannels": 1, "density": 0.3}),
    "birth_death (species)": ("legacy.multispecies.birthdeath", ("birth_death", "random_walk"),
                              ({"r_b": [0.2, 0.1], "r_d": 0.05}, {"birth_rate": [0.2, 0.1], "death_rate": 0.05}),
                              {"restchannels": 1, "n_species": 2, "volume_exclusion": False, "capacity": 8,
                               "density": 2.0}),
    "birth_death (ib)": ("legacy.ib.birthdeath", ("birth_death", "random_walk"),
                         ({"r_b": 0.2, "r_d": 0.05, "std": 0.01, "a_max": 1.0},
                          {"birth_rate": "r_b", "death_rate": 0.05, "mutation": {"r_b": {
                              "distribution": "normal", "scale": 0.01, "bounds": [0, 1], "at_bounds": "redraw"}}}),
                         {**_IB, "traits": {"r_b": 0.2}}),
    "birth_death (nove_ib)": ("legacy.nove_ib.birthdeath", ("birth_death", "random_walk"),
                              ({"r_b": 0.2, "r_d": 0.05, "std": 0.01, "a_max": 1.0},
                               {"birth_rate": "r_b", "death_rate": 0.05, "mutation": {"r_b": {
                                   "distribution": "normal", "scale": 0.01, "bounds": [0, 1],
                                   "at_bounds": "redraw"}}}),
                              {**_NOVE_IB, "traits": {"r_b": 0.2}}),
    "birthdeath_cancerdfe": ("legacy.nove_ib.birthdeath_cancerdfe", "birthdeath_cancerdfe",
                             {"r_b": 0.3, "p_d": 0.05, "p_p": 0.2}, _NOVE_IB),
    "go_or_grow_kappa": ("legacy.nove_ib.go_or_grow_kappa", "go_or_grow_kappa", {}, _NOVE_IB),
    "go_or_grow_kappa_chemo": ("legacy.nove_ib.go_or_grow_kappa_chemo", "go_or_grow_kappa_chemo", {}, _NOVE_IB),
    "go_or_grow_glioblastoma": ("legacy.nove_ib.go_or_grow_glioblastoma", "go_or_grow_glioblastoma", {"r_m": 0.05},
                                _NOVE_IB),
    "evo_steric": ("legacy.nove_ib.evo_steric", "evo_steric", {"r_m": 0.05}, {**_NOVE_IB, "capacity": 50}),
    "go_and_grow_mutations": ("legacy.ib.go_and_grow_mutations", "go_and_grow_mutations",
                              {"r_m": 0.05, "effect": "driver_mutation"}, _IB),
    "birthdeath_discrete": ("legacy.ib.birthdeath_discrete", "birthdeath_discrete", {}, _IB),
    "excitable_medium": ("legacy.classical.excitable_medium", "excitable_medium", {"N": 10},
                         {"restchannels": 4, "density": 0.3, "geometry": "square"}),
    "excitable_medium (species)": ("legacy.multispecies.excitable_medium_ms", "excitable_medium", {"N": 10},
                                   {"restchannels": 4, "n_species": 2, "geometry": "square"}),
}


def _state(state, seed):
    state = dict(state)
    geometry = state.pop("geometry", "hex")
    if state.get("n_species") == 2 and geometry == "square":  # excitable medium: inhibitors rest, activators move
        nodes = np.random.default_rng(seed).random(DIMS + (2, 8)) < 0.3
        nodes[..., 0, :4] = nodes[..., 1, 4:] = False
        state.pop("density", None)
        state["nodes"] = nodes
    return geometry, StateSpec(**state)


def _operators(names, parameters):
    names = (names,) if isinstance(names, str) else names
    return [{"name": name, "parameters": parameters if index == 0 else {}} for index, name in enumerate(names)]


def _measure(name, route, repeat, steps, warm_up):
    """Seconds per step of one route of a pair, in a fresh process (see time_pair)."""
    warnings.simplefilter("ignore")
    legacy, new, parameters, state = PAIRS[name]
    legacy_parameters, new_parameters = parameters if isinstance(parameters, tuple) else (parameters,) * 2
    operator, values = (legacy, legacy_parameters) if route == "legacy" else (new, new_parameters)
    geometry, state_spec = _state(state, repeat)
    if route == "legacy":
        state_spec = StateSpec(**{**state_spec.__dict__, "traits": {}})
    model = build_model(ModelSpec(
        space=SpaceSpec(geometry=geometry, dims=DIMS, boundary="periodic"), state=state_spec,
        time=TimeSpec(steps=steps, seed=repeat),
        dynamics=InteractionPipelineSpec(operators=_operators(operator, values))))
    for _ in range(warm_up):
        model.step()
    start = perf_counter()
    for _ in range(steps):
        model.step()
    return (perf_counter() - start) / steps


def time_pair(name, steps, repeats, warm_up=50):
    """Median seconds per step of both routes. Every measurement runs in a fresh process: the
    memory allocator's state after other models changes the timings by up to a factor of two."""
    context = multiprocessing.get_context("spawn")
    result = {}
    with context.Pool(1, maxtasksperchild=1) as pool:
        for route in ("legacy", "new"):
            result[route] = median(pool.starmap(_measure, [(name, route, repeat, steps, warm_up)
                                                           for repeat in range(repeats)]))
    result["speed_up"] = result["legacy"] / result["new"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"pairs to time (default: all of {', '.join(PAIRS)})")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="write the results as JSON to this file")
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")
    results = {}
    print(f"{'rule':30s} {'legacy':>10s} {'new':>10s} {'speed-up':>9s}   (ms per step, {DIMS[0]} x {DIMS[1]})")
    for name in arguments.names or PAIRS:
        results[name] = time_pair(name, arguments.steps, arguments.repeats)
        timing = results[name]
        print(f"{name:30s} {timing['legacy'] * 1e3:10.1f} {timing['new'] * 1e3:10.1f} {timing['speed_up']:8.1f}x")
    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump({"dims": DIMS, "steps": arguments.steps, "repeats": arguments.repeats, "results": results},
                      file, indent=2)


if __name__ == "__main__":
    main()
