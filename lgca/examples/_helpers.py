"""Small helpers shared by the example scripts."""

from __future__ import annotations

import argparse
from dataclasses import replace


def run_spec(build_spec, steps: int | None = None, showprogress: bool = False):
    """Build a spec, optionally override steps, and run it."""

    from lgca.model import run_model

    spec = build_spec()
    if steps is not None:
        spec = replace(spec, time=replace(spec.time, steps=int(steps)))
    return run_model(spec, showprogress=showprogress)


def print_result_summary(result) -> None:
    """Print a compact summary useful for command-line example runs."""

    fields = ", ".join(sorted(result.context.fields)) or "-"
    print(f"title: {result.metadata['title']}")
    print(f"steps: {result.metadata['steps']}")
    print(f"operators: {', '.join(result.metadata['operator_names'])}")
    print(f"observers: {', '.join(result.metadata['observer_names'])}")
    print(f"fields: {fields}")
    print(f"population: {result.lgca.total_population()}")


def main(run_func) -> None:
    """Run an example module from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--showprogress", action="store_true")
    args = parser.parse_args()
    print_result_summary(run_func(steps=args.steps, showprogress=args.showprogress))
