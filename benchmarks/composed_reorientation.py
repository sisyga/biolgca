"""Repeatable diagnostic for the recommended composed reorientation route.

Compares the legacy aggregation function (``tests/legacy``, route "legacy") with the
aggregation cue alone and combined with nematic alignment. Run from the repository root,
``uv run python benchmarks/composed_reorientation.py OUTPUT``. Timings are diagnostics, not
portable performance gates. Each repeat constructs a fresh seeded model.
"""

import argparse
import cProfile
import io
import json
import pstats
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import (
    InteractionPipelineSpec,
    ReorientationSpec,
    ReorientationTermSpec,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for tests.legacy
import tests.legacy


def make_model(size, occupancy, route, steps):
    nodes = np.zeros((size, size, 4), dtype=bool)
    nodes[..., :occupancy] = True
    if route == "legacy":
        operator = {"name": "legacy.classical.aggregation", "parameters": {"beta": 2.0}}
    else:
        terms = [ReorientationTermSpec("aggregation", beta=2.0)]
        if route == "combined":
            terms.append(ReorientationTermSpec("nematic_alignment", beta=1.0))
        operator = ReorientationSpec(terms=terms)
    return build_model(ModelSpec(
        space=SpaceSpec(geometry="square", dims=(size, size)),
        state=StateSpec(nodes=nodes), time=TimeSpec(steps=steps, seed=143),
        dynamics=InteractionPipelineSpec(operators=[operator]),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    results = []
    for size in (64, 128):
        for occupancy in (1, 2):
            for route in ("legacy", "composed", "combined"):
                elapsed = []
                for _ in range(args.repeats):
                    model = make_model(size, occupancy, route, args.steps)
                    start = perf_counter()
                    model.run(False)
                    elapsed.append(perf_counter() - start)
                results.append(dict(size=size, initial_occupancy=occupancy, route=route,
                                    steps=args.steps, seconds=elapsed, median_seconds=median(elapsed)))
    profiler = cProfile.Profile()
    model = make_model(64, 1, "combined", 1)
    profiler.runcall(model.run, False)
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(18)
    args.output.write_text(json.dumps({"results": results, "profile": output.getvalue()}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
