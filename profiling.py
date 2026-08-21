"""Reproducible end-to-end BioLGCA benchmark harness."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lgca import get_lgca
from lgca.model import ModelSpec, SpaceSpec, StateSpec, TimeSpec, build_model
from lgca.pipeline import InteractionPipelineSpec
from lgca.simulation import SimulationRunner


@dataclass(frozen=True)
class BenchmarkScenario:
    """One seeded end-to-end simulation benchmark."""

    name: str
    family: str
    geometry: str
    dims: tuple[int, ...]
    interaction: str
    density: float
    steps: int
    seed: int
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    model_spec_operator: Mapping[str, Any] | None = None


def default_scenarios() -> list[BenchmarkScenario]:
    """Return small representative scenarios for maintained state families."""

    common = {"density": 0.35, "steps": 20, "seed": 7}
    return [
        BenchmarkScenario(
            "classical_1d_propagation", "classical_ve", "lin", (256,),
            "only_propagation", kwargs={"ve": True}, **common,
        ),
        BenchmarkScenario(
            "classical_square_random_walk", "classical_ve", "square", (64, 64),
            "random_walk", kwargs={"ve": True, "restchannels": 1}, **common,
        ),
        BenchmarkScenario(
            "classical_square_native_birth_death", "classical_ve", "square", (64, 64),
            "birth_death", kwargs={"restchannels": 1},
            model_spec_operator={
                "name": "birth_death",
                "parameters": {"birth_rate": 0.2, "death_rate": 0.05, "capacity": 5},
            },
            **common,
        ),
        BenchmarkScenario(
            "classical_hex_random_walk", "classical_ve", "hex", (64, 64),
            "random_walk", kwargs={"ve": True}, **common,
        ),
        BenchmarkScenario(
            "classical_cubic_propagation", "classical_ve", "cubic", (16, 16, 16),
            "only_propagation", kwargs={"ve": True}, **common,
        ),
        BenchmarkScenario(
            "classical_moore_propagation", "classical_ve", "moore", (12, 12, 12),
            "only_propagation", kwargs={"ve": True}, **common,
        ),
        BenchmarkScenario(
            "nove_square_random_walk", "nove", "square", (64, 64),
            "random_walk", kwargs={"ve": False, "restchannels": 1}, **common,
        ),
        BenchmarkScenario(
            "identity_square_random_walk", "identity_ve", "square", (64, 64),
            "random_walk", kwargs={"ve": True, "ib": True, "restchannels": 1}, **common,
        ),
        BenchmarkScenario(
            "identity_nove_square_random_walk", "identity_nove", "square", (32, 32),
            "random_walk", kwargs={"ve": False, "ib": True, "restchannels": 1}, **common,
        ),
        BenchmarkScenario(
            "multispecies_square_birth", "multispecies", "square", (64, 64),
            "birth", kwargs={"ve": False, "n_species": 3, "restchannels": 1,
                              "capacity": 12, "r_b": [0.2, 0.2, 0.2]}, **common,
        ),
    ]


def run_scenario(scenario: BenchmarkScenario, repeats: int = 3) -> dict[str, Any]:
    """Run a scenario repeatedly and return median time and peak memory."""

    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    runtimes = []
    peak_memory = 0
    particles_final = 0
    for _ in range(int(repeats)):
        if scenario.model_spec_operator is None:
            lgca = get_lgca(
                geometry=scenario.geometry,
                dims=scenario.dims,
                density=scenario.density,
                interaction=scenario.interaction,
                seed=scenario.seed,
                **dict(scenario.kwargs),
            )
            run = lambda: SimulationRunner(
                lgca,
                timesteps=scenario.steps,
                observers=(),
                showprogress=False,
            ).run()
        else:
            state_kwargs = dict(scenario.kwargs)
            restchannels = int(state_kwargs.pop("restchannels", 0))
            if state_kwargs:
                raise ValueError(
                    "ModelSpec benchmark kwargs only support restchannels; "
                    f"got {sorted(state_kwargs)}"
                )
            compiled = build_model(
                ModelSpec(
                    space=SpaceSpec(
                        geometry=scenario.geometry,
                        dims=scenario.dims,
                        boundary="periodic",
                    ),
                    state=StateSpec(
                        density=scenario.density,
                        restchannels=restchannels,
                    ),
                    time=TimeSpec(steps=scenario.steps, seed=scenario.seed),
                    dynamics=InteractionPipelineSpec(
                        operators=[dict(scenario.model_spec_operator)]
                    ),
                )
            )
            lgca = compiled.lgca
            run = lambda: compiled.run(showprogress=False)
        tracemalloc.start()
        start = time.perf_counter()
        run()
        runtimes.append(time.perf_counter() - start)
        _, repeat_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory = max(peak_memory, int(repeat_peak))
        particles_final = int(np.asarray(lgca.cell_density[lgca.nonborder]).sum())

    wall_seconds = float(statistics.median(runtimes))
    try:
        package_version = importlib.metadata.version("biolgca")
    except importlib.metadata.PackageNotFoundError:
        package_version = "0.1.0"
    return {
        "scenario": scenario.name,
        "family": scenario.family,
        "geometry": scenario.geometry,
        "interaction": scenario.interaction,
        "dims": list(scenario.dims),
        "density": scenario.density,
        "seed": scenario.seed,
        "steps": scenario.steps,
        "repeats": int(repeats),
        "wall_seconds": wall_seconds,
        "wall_seconds_per_step": wall_seconds / max(scenario.steps, 1),
        "peak_memory_bytes": peak_memory,
        "particles_final": particles_final,
        "observer_policy": "none",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "biolgca_version": package_version,
    }


def write_results(results: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write benchmark results as JSON or CSV according to the suffix."""

    path = Path(path)
    rows = [dict(result) for result in results]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    if path.suffix.lower() != ".csv":
        raise ValueError("benchmark output must end in .json or .csv")
    if not rows:
        raise ValueError("cannot write an empty CSV benchmark result")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", help="Scenario name; repeat to select several")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenarios = default_scenarios()
    if args.scenario:
        selected = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario.name in selected]
        unknown = selected - {scenario.name for scenario in scenarios}
        if unknown:
            raise SystemExit(f"unknown benchmark scenario(s): {', '.join(sorted(unknown))}")
    results = [run_scenario(scenario, repeats=args.repeats) for scenario in scenarios]
    write_results(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
